// How GI lights a ray's hit and a ray's miss (`//!use gi_hit`, with
// `frame_sky`, `sky_radiance`, `frame_ray_stream`, `frame_ray_surface`,
// `frame_surface`, `frame_light_samples`, `frame_radiance_cache`,
// `gi_common`, `gi_sample` and `gi_rays`): the one place GI's readers of a
// stream (the probes' and the cache's rays in `gi_light`, the pixels' rays
// in `gi_gather`) turn a hit into the light it sends back along its ray. A
// shader that opts in binds `frame_gi`, `gi_probes`, `gi_trace`,
// `sky_frame`, `sky_radiance`, `sky_radiance_sampler` and the radiance cache
// under their names.

const PI: f32 = 3.14159265;

// The sky's radiance along unit direction `d` from eye-relative point
// `from_camera` (km): the atmosphere's radiance table there
// (`sky_radiance_eval`, from the eye the sky the sky pass draws); without an
// atmosphere, the flat sky and ground.
fn gi_sky_radiance(from_camera: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    if sky_frame.planet.w < 0.5 { return select(frame_gi.ground.rgb, frame_gi.sky.rgb, d.y >= 0.0); }
    return sky_radiance_eval(from_camera, d);
}

// A side of a hit as the cache holds it (`gi_hit_cell`): its irradiance / pi
// from the lights (`direct`) and from everything else (`indirect`); `found`
// where the cache has light for the hit; elsewhere `indirect` is the probes'
// irradiance at the hit (the probes take in what their rays meet, never the
// lights' own light: DDGI's recursion), and `waiting` says the surface's
// cell there has no value yet (it is new: a scene starting, the eye
// somewhere new), as against a cell whose light does not hold at the hit
// (an occluder or the reader's reach between).
struct GiHitCell {
    direct: vec3<f32>,
    indirect: vec3<f32>,
    found: bool,
    waiting: bool,
}

// The cache at eye-relative hit `at` on the side of unit `side` (normal `n`)
// in space `space` for a reader of reach `reach` (`radiance_cache_read`: the
// ray's length for a ray that left a surface, RADIANCE_CACHE_FAR for a
// probe's), marking its cell when `mark`.
fn gi_hit_cell(at: vec3<f32>, side: vec3<f32>, n: vec3<f32>, space: u32, reach: f32, mark: bool) -> GiHitCell {
    let p = at + gi_trace.eye.xyz;
    let c = radiance_cache_read(p, side, space, reach);
    if mark { radiance_cache_mark(p, side, space, reach); }
    if c.found { return GiHitCell(c.direct / PI, c.indirect / PI, true, false); }
    var waiting = false;
    if radiance_cache_present() {
        let s = radiance_cache_find(radiance_cache_key(p, side, radiance_cache_level(p), space));
        waiting = s == RADIANCE_CACHE_NONE || !radiance_cache_has_value(radiance_cache_entries[s]);
    }
    return GiHitCell(vec3<f32>(0.0), gi_irradiance(at, n).rgb, false, waiting);
}

// The sides of a hit (`GiHitLight::own`, `gi_hit_direct`): the one its ray
// met (the surface's normal turned toward the ray) and the far one.
const GI_SIDE_NEAR: u32 = 1u;
const GI_SIDE_FAR: u32 = 2u;

// What a hit sends back along its ray of the cache's light (`gi_hit_light`):
// `reflected`, what it reflects of the irradiance on the side its ray met and
// lets through of its far side's; `emitted`, what it emits; `answered`, the
// cache lit every side that reflects or lets light through; `waiting`, a
// side that does took the probes' while its cell is new (`GiHitCell`);
// `own`, the sides whose direct light the reader takes from a light sample
// of its own (`gi_hit_direct`; GI_SIDE_*).
struct GiHitLight {
    reflected: vec3<f32>,
    emitted: vec3<f32>,
    answered: bool,
    waiting: bool,
    own: u32,
}

// The light a hit at eye-relative `at` sends back along its ray's unit
// direction `dir`: a Lambertian surface `surface` (a stream's row) lit by
// the cache on the side the ray met, plus its transmission share of the far
// side's cell (two-sided leaves), and what it emits; `reach` the reader's
// (`gi_hit_cell`). Where the cache has no light for a side, the hit is lit
// on the spot: the probes' irradiance there for its indirect light and, for
// a reader that lights such a side itself (`own_light`), its own light
// sample for the direct (the side is in `own`). A side's direct light is
// none for `indirect_only` (the reader lights every side itself). The near
// side's cell is marked where `mark`; the far side's is read, never marked:
// the hit is on the side its ray met, and behind it may be the inside of
// what lets light through (a voxel of leaves: a cell seated there would
// start its rays in it). The far side's cells are the ones rays from that
// side make.
fn gi_hit_light(at: vec3<f32>, dir: vec3<f32>, surface: SceneSurface, space: u32, reach: f32, indirect_only: bool, own_light: bool, mark: bool) -> GiHitLight {
    let s = scene_surface_unpack(surface);
    let face = scene_surface_face(surface);
    let side = select(face, -face, dot(face, dir) > 0.0);
    let through = surface_transmission(s);
    let front = gi_hit_cell(at, side, s.normal, space, reach, mark);
    let dark = all(s.albedo.rgb == vec3<f32>(0.0));
    var lit = GiHitLight(vec3<f32>(0.0), s.emissive, front.found || dark, front.waiting && !dark, 0u);
    if !dark {
        lit.reflected = s.albedo.rgb * (front.indirect + select(vec3<f32>(0.0), front.direct, !indirect_only));
        if own_light && !front.found { lit.own |= GI_SIDE_NEAR; }
    }
    if any(through > vec3<f32>(0.0)) {
        // A far side the cache has no light for passes on nothing of the
        // probes' (at a point on a thin surface they hold the near side's
        // light too, both sides' probes seeing the point: the surface's own
        // light would come back through it).
        let back = gi_hit_cell(at, -side, -s.normal, space, reach, false);
        if back.found {
            lit.reflected += through * (back.indirect + select(vec3<f32>(0.0), back.direct, !indirect_only));
        }
        lit.answered = lit.answered && back.found;
        lit.waiting = lit.waiting || back.waiting;
        if own_light && !back.found { lit.own |= GI_SIDE_FAR; }
    }
    return lit;
}

// A hit's direct light from one light sample (`frame_light_samples`) on its
// sides `sides` (GI_SIDE_*): Lambertian on the side its ray met (its normal's,
// turned toward the ray); from behind, the surface's share of it through
// (two-sided leaves). `light` before its shadow ray, which leaves from `side`
// (the face turned toward the light's side); `lit` false where the sample
// gives those sides nothing.
struct GiHitDirect {
    light: vec3<f32>,
    side: vec3<f32>,
    lit: bool,
}

fn gi_hit_direct(surface: SceneSurface, sample: LightSample, sides: u32) -> GiHitDirect {
    var d = GiHitDirect(vec3<f32>(0.0), vec3<f32>(0.0), false);
    if !light_sample_found(sample) { return d; }
    let s = scene_surface_unpack(surface);
    let face = scene_surface_face(surface);
    let towards = light_sample_dir(sample);
    let radiance = light_sample_radiance(sample);
    let through = surface_transmission(s);
    let cosine = dot(s.normal, towards);
    if cosine > 0.0 && (sides & GI_SIDE_NEAR) != 0u {
        d.light = s.albedo.rgb * radiance * cosine / PI;
        d.side = face;
    } else if any(through > vec3<f32>(0.0)) && cosine < 0.0 && (sides & GI_SIDE_FAR) != 0u {
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
