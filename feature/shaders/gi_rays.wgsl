// The probes' rays as GI writes them into its ray streams (README.md, "The
// frame"; `//!use gi_rays`, with `gi_common` and `frame_ray_stream`): the
// frame's place of the eye, how far rays start off what they leave, and
// what the lighting of a probe ray's hit leaves for its integration.

// A probe ray's cone: the width of its footprint a metre out, the side of
// the square of one of its probe's GI_RAYS rays' solid angle
// (sqrt(4 pi / 64)). Shadow rays from its hit keep it.
const GI_RAY_SPREAD: f32 = 0.443;
// The most distance a probe ray or a shadow ray toward the sun travels:
// past the far side of an Earth-sized planet.
const GI_RAY_REACH: f32 = 1e9;
// A pending ray light that waits on no shadow ray.
const GI_NO_ROW: u32 = 0xffffffffu;

// The frame's rays (`GiTrace`, `helio-pass-gi`): the eye in the frame's space
// (rays are in frame space, the probes eye-relative); x the frame's seed.
struct GiTrace {
    eye: vec4<f32>,
    info: vec4<u32>,
}

// What lighting a probe ray's hit leaves for its integration (`gi_pending`,
// a row a ray): the sun's light off the hit and the chosen local light's
// (`scene_pack_radiance`), each waiting on its shadow ray's row of the
// shadow stream (GI_NO_ROW: none); the rest of its light (its irradiance
// from the probes, through it from its far side, what it emits; the sky on
// a miss); and the ray's distance from its probe: above 0 a hit, 0 a miss,
// below 0 a hit on an inside at minus that.
struct GiPending {
    sun: u32,
    local: u32,
    rest: u32,
    t: f32,
    sun_row: u32,
    local_row: u32,
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
