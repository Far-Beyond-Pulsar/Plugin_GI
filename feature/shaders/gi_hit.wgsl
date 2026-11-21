// How GI lights a ray's hit and a ray's miss (`//!use gi_hit`, with
// `frame_sky`, `sky_table`, `frame_ray_stream`, `frame_ray_surface`,
// `frame_surface`, `frame_radiance_cache`, `gi_common`, `gi_sample` and
// `gi_rays`): the one place GI's readers of a stream (the probes' and the
// cache's rays in `gi_light`, the pixels' rays in `gi_gather`) turn a hit
// into the light it sends back along its ray. A shader that opts in binds
// `frame_gi`, `gi_probes`, `gi_trace`, `sky_frame`, `sky_table` and the
// radiance cache under their names.

const PI: f32 = 3.14159265;

// The sky's radiance along unit direction `d` from eye-relative point
// `from_camera` (km): the atmosphere's sky table there (irradiance / pi in
// L2 harmonics, cosine-convolved) back to radiance, band by band; without
// an atmosphere, the flat sky and ground.
fn gi_sky_radiance(from_camera: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    if sky_frame.planet.w < 0.5 { return select(frame_gi.ground.rgb, frame_gi.sky.rgb, d.y >= 0.0); }
    // The basis over each band's convolution (1, 2/3, 1/4).
    return sky_table_eval(from_camera, d, vec3<f32>(1.0, 1.5, 4.0));
}

// How many levels coarser than its own cell a GI ray's hit may read while
// its cell has no value: one, a cell twice as wide. Coarser ones (a mark's
// ancestor is 8 times wider) span thin walls and floors under them, so a
// probe's first update would take both sides' light into its history.
const GI_CACHE_REACH: u32 = 1u;

// The irradiance / pi the cache holds at eye-relative hit `at` on the side of
// unit `side` for a ray of footprint `footprint` in space `space` (its
// indirect part alone for `indirect_only`), marking its cell, and 1 in w;
// where it holds none within GI_CACHE_REACH, the probes' at the hit toward
// `n` (their own previous state, as a bounce took it before the cache), and
// 0 in w.
fn gi_hit_irradiance(at: vec3<f32>, side: vec3<f32>, n: vec3<f32>, footprint: f32, space: u32, indirect_only: bool) -> vec4<f32> {
    let p = at + gi_trace.eye.xyz;
    let c = radiance_cache_read(p, side, footprint, space, GI_CACHE_REACH);
    radiance_cache_mark(p, side, footprint, space);
    if !c.found { return vec4<f32>(gi_irradiance(at, n).rgb, 0.0); }
    return vec4<f32>((c.indirect + select(c.direct, vec3<f32>(0.0), indirect_only)) / PI, 1.0);
}

// The light a hit at eye-relative `at` sends back along its ray's unit
// direction `dir` (rgb): a Lambertian surface `surface` (a stream's row) lit
// by the cache on the side the ray met (its whole irradiance, or its
// indirect part for `indirect_only`), plus its transmission share of the far
// side's cell (two-sided leaves) and what it emits; in w, 1 where the cache
// lit every side that reflects or lets light through, 0 where the probes
// stood in.
fn gi_hit_light(at: vec3<f32>, dir: vec3<f32>, surface: SceneSurface, footprint: f32, space: u32, indirect_only: bool) -> vec4<f32> {
    let s = scene_surface_unpack(surface);
    let face = scene_surface_face(surface);
    let side = select(face, -face, dot(face, dir) > 0.0);
    let through = surface_transmission(s);
    let front = gi_hit_irradiance(at, side, s.normal, footprint, space, indirect_only);
    var light = s.albedo.rgb * front.rgb + s.emissive;
    var answered = front.w > 0.0 || all(s.albedo.rgb == vec3<f32>(0.0));
    if any(through > vec3<f32>(0.0)) {
        let back = gi_hit_irradiance(at, -side, -s.normal, footprint, space, indirect_only);
        light += through * back.rgb;
        answered = answered && back.w > 0.0;
    }
    return vec4<f32>(light, select(0.0, 1.0, answered));
}
