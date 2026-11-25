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
// A pending ray light that waits on no shadow ray (a cache ray's), or a
// probe's hit the cache lit whole.
const GI_NO_ROW: u32 = 0xffffffffu;
// A probe's hit lit on the spot (`GiPending::row` other than GI_NO_ROW): the
// row of what its light waits on in the low bits (GI_ROW_INDEX all ones:
// nothing, its light sample gave it nothing), and GI_ROW_WAITING where the
// cell it had no light from was new (`gi_integrate`'s readiness).
const GI_ROW_INDEX: u32 = 0x3fffffffu;
const GI_ROW_WAITING: u32 = 0x40000000u;

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
// a ray of GI's stream): the light of the hit's own light sample where it
// takes one (`scene_pack_radiance`: the light the frame's samplers chose for
// it, off the surface; every cache ray's hit, a probe's hit lit on the
// spot), waiting on `row` (`gi_light`: the hit's row of the light batch,
// `light` the sides that take it, GI_SIDE_*; once `gi_shadow` ran, its
// shadow ray's row of the shadow stream; GI_NO_ROW: none; a probe's as
// GI_ROW_INDEX and GI_ROW_WAITING say); the rest of its light (from the
// radiance cache at the hit or the probes' where it has none, through it
// from its far side, what it emits; the sky on a miss); and the ray's
// distance from its origin (a probe's: from the probe): above 0 a hit, 0 a
// miss, below 0 a hit on an inside at minus that.
struct GiPending {
    light: u32,
    rest: u32,
    t: f32,
    row: u32,
}

// What lighting a cache ray leaves for its cell (`gi_cell_pending`, a row a
// cache ray): the irradiance at the cell of the ray's light sample there
// (`light`, `scene_pack_radiance`), waiting on `row` as `GiPending`'s; the
// light the ray's hit reflects of the cache's own (`bounced`: the hit's
// cell's indirect light, and what passes through it from its far side),
// which the cell folds apart from what the hit sends as it meets it
// (`GiPending::rest` and `light`); `unlit` 1 where the cache had no light
// for that hit (the bounced part leaves the ray out: the probes' irradiance
// there would carry light across what the cells keep apart, a thin card's
// lit top into the shade under it).
struct CachePending {
    light: u32,
    row: u32,
    bounced: u32,
    unlit: u32,
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
