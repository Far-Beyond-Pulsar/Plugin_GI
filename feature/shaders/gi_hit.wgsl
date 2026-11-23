// How GI lights a ray's hit and a ray's miss (`//!use gi_hit`, with
// `frame_sky`, `sky_table`, `frame_ray_stream`, `frame_ray_surface`,
// `frame_surface`, `frame_light_samples`, `frame_radiance_cache`,
// `gi_common`, `gi_sample` and `gi_rays`): the one place GI's readers of a stream (the probes' and the
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
// indirect part alone for `indirect_only`), marking its cell where `mark`
// (a reader that does not trust the cell reads it without claiming it), and
// 1 in w;
// where it holds none within GI_CACHE_REACH, the probes' at the hit toward
// `n` (their own previous state, as a bounce took it before the cache), and
// 0 in w.
fn gi_hit_irradiance(at: vec3<f32>, side: vec3<f32>, n: vec3<f32>, footprint: f32, space: u32, indirect_only: bool, mark: bool) -> vec4<f32> {
    let p = at + gi_trace.eye.xyz;
    let c = radiance_cache_read(p, side, footprint, space, GI_CACHE_REACH);
    if mark { radiance_cache_mark(p, side, footprint, space); }
    return gi_hit_cached(c, at, n, indirect_only);
}

// What `gi_hit_irradiance` makes of read `c` at eye-relative `at` (normal
// `n`): the cache's irradiance / pi and 1 in w, or the probes' at the hit
// and 0 where it found none.
fn gi_hit_cached(c: RadianceCacheIrradiance, at: vec3<f32>, n: vec3<f32>, indirect_only: bool) -> vec4<f32> {
    if !c.found { return vec4<f32>(gi_irradiance(at, n).rgb, 0.0); }
    return vec4<f32>((c.indirect + select(c.direct, vec3<f32>(0.0), indirect_only)) / PI, 1.0);
}

// The light a hit at eye-relative `at` sends back along its ray's unit
// direction `dir` (rgb): a Lambertian surface `surface` (a stream's row) lit
// by the cache on the side the ray met (its whole irradiance, or its
// indirect part for `indirect_only`), plus its transmission share of the far
// side's cell (two-sided leaves) and what it emits; in w, 1 where the cache
// lit every side that reflects or lets light through, 0 where the probes
// stood in. The cells read are marked where `mark`.
fn gi_hit_light(at: vec3<f32>, dir: vec3<f32>, surface: SceneSurface, footprint: f32, space: u32, indirect_only: bool, mark: bool) -> vec4<f32> {
    let s = scene_surface_unpack(surface);
    let face = scene_surface_face(surface);
    let side = select(face, -face, dot(face, dir) > 0.0);
    let through = surface_transmission(s);
    let front = gi_hit_irradiance(at, side, s.normal, footprint, space, indirect_only, mark);
    var light = s.albedo.rgb * front.rgb + s.emissive;
    var answered = front.w > 0.0 || all(s.albedo.rgb == vec3<f32>(0.0));
    if any(through > vec3<f32>(0.0)) {
        let back = gi_hit_irradiance(at, -side, -s.normal, footprint, space, indirect_only, mark);
        light += through * back.rgb;
        answered = answered && back.w > 0.0;
    }
    return vec4<f32>(light, select(0.0, 1.0, answered));
}

// A hit's direct light from one light sample (`frame_light_samples`):
// Lambertian on the side the light is on; from behind, the surface's share
// of it through (two-sided leaves). `light` before its shadow ray, which
// leaves from `side` (the face turned toward the light's side); `lit` false
// where the sample gives the surface nothing.
struct GiHitDirect {
    light: vec3<f32>,
    side: vec3<f32>,
    lit: bool,
}

fn gi_hit_direct(surface: SceneSurface, sample: LightSample) -> GiHitDirect {
    var d = GiHitDirect(vec3<f32>(0.0), vec3<f32>(0.0), false);
    if !light_sample_found(sample) { return d; }
    let s = scene_surface_unpack(surface);
    let face = scene_surface_face(surface);
    let towards = light_sample_dir(sample);
    let radiance = light_sample_radiance(sample);
    let through = surface_transmission(s);
    let cosine = dot(s.normal, towards);
    if cosine > 0.0 {
        d.light = s.albedo.rgb * radiance * cosine / PI;
        d.side = face;
    } else if any(through > vec3<f32>(0.0)) && cosine < 0.0 {
        d.light = through * radiance * -cosine / PI;
        d.side = -face;
    }
    d.lit = any(d.light > vec3<f32>(0.0));
    return d;
}

// A shadow ray from frame point `p` (already off its surface) toward a light
// sample: `distance` away (a directional light: as far as GI's rays reach),
// as wide as `width` with no spread (it stands for that patch of surface).
fn gi_shadow_ray(p: vec3<f32>, sample: LightSample, width: f32, space: u32) -> SceneRay {
    let off = gi_ray_offset(p);
    let distance = light_sample_distance(sample);
    let reach = select(max(distance - 2.0 * off, 1e-4), GI_RAY_REACH, distance >= LIGHT_SAMPLE_FAR);
    let cone = scene_cone_in_space(scene_ray_cone(width, 0.0, RAY_MASK_ALL), space);
    return SceneRay(p, reach, light_sample_dir(sample), cone);
}
