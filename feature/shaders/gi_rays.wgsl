// GI's rays as it writes them into its ray streams (README.md, "The frame";
// `//!use gi_rays`, with `gi_common` and `frame_ray_stream`): the frame's
// place of the eye, how far rays start off what they leave, where the
// radiance cache's rays follow the probes', and what the lighting of a ray's
// hit leaves for its integration.

// A probe ray's cone: no width at the probe, and the width of its footprint
// a metre out the side of the square of one of its probe's GI_RAYS rays'
// solid angle (sqrt(4 pi / 64)). A shadow ray from its hit starts as wide as
// its footprint there and keeps that width.
const GI_RAY_SPREAD: f32 = 0.443;
// The most distance a probe ray or a shadow ray toward the sun travels:
// past the far side of an Earth-sized planet.
const GI_RAY_REACH: f32 = 1e9;
// A pending ray light that waits on no shadow ray.
const GI_NO_ROW: u32 = 0xffffffffu;
// A probe ray's `GiPending::light`: its hit's light was not in the radiance
// cache yet (the probes' own stood in).
const GI_UNLIT: u32 = 1u;

// The radiance cache's rays in GI's stream (`gi_cache.wgsl`, `cache.rs`):
// CACHE_RAYS a lit cell, its cells' rays together, after the probes' rays
// (row `frame_gi.updated * GI_RAYS` on). `cache_plan` (CACHE_PLAN_WORDS
// words) holds in CACHE_PLAN_CELLS the cells listed this frame
// (`cache_list`), of which the pipelines' CACHE_CELLS are lit.
const CACHE_RAYS: u32 = 4u;
const CACHE_PLAN_WORDS: u32 = 80u;
const CACHE_PLAN_CELLS: u32 = 68u;

// The frame's rays (`GiTrace`, `helio-pass-gi`): the eye in the frame's space
// (rays are in frame space, the probes eye-relative); x the frame's seed, y
// the coordinate space every probe ray and shadow ray travels in
// (`SceneRay::space`: the camera's `view_info.w`, as the probes light what it
// sees).
struct GiTrace {
    eye: vec4<f32>,
    info: vec4<u32>,
}

// What lighting a ray's hit leaves for its integration (`gi_pending`, a row
// a ray of GI's stream): for a cache ray, the light of the hit's light
// sample (`scene_pack_radiance`: the light the frame's samplers chose for
// it, off the surface), waiting on `row` (`gi_light`: the hit's row of the
// light batch; once `gi_shadow` ran, its shadow ray's row of the shadow
// stream; GI_NO_ROW: none, and always for a probe's ray, whose `light` is
// GI_UNLIT where the cache could not light its hit); the rest of its
// light (from the radiance cache at the hit, through it from its far side,
// what it emits; the sky on a miss); and the ray's distance from its
// origin (a probe's: from the probe): above 0 a hit, 0 a miss, below 0 a hit
// on an inside at minus that.
struct GiPending {
    light: u32,
    rest: u32,
    t: f32,
    row: u32,
}

// How far a ray leaving frame point `p` starts off its surface: a few ulps
// of f32 there, and 2 mm.
fn gi_ray_offset(p: vec3<f32>) -> f32 {
    return 0.002 + 4e-6 * max(abs(p.x), max(abs(p.y), abs(p.z)));
}

fn gi_hash(x: u32) -> u32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}
