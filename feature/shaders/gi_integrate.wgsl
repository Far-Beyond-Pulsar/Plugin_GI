//!use frame_ray_stream
//!use gi_common
//!use gi_rays
// This frame's probe rays into their probes (README.md, "The frame"): a
// workgroup a probe, a lane a ray. A ray's radiance is what `gi_light` left
// it (`gi_pending`): its hit lit from the radiance cache, or lit on the spot
// where the cache has no light for it (the probes' irradiance there and its
// own light sample through its shadow ray), or the sky; nothing from an
// inside. Every ray is projected, on L1 harmonics over the whole sphere, and
// blended into the probe; its distances into the probe's moments (a lane a
// texel). The one hold is readiness: a probe never updated more than 1/32
// of whose hits met cells still new (a scene loading, the eye somewhere
// new) holds and goes on the held list (`gi_held`), which brings it back to
// the schedule's front once the cache has lit those cells (`integrate.rs`),
// so its first update does not take the probes' own empty state for the
// light there. A probe whose rays mostly meet insides is inside the ground
// (or a wall). On a frame where no probe ray met anything the probes hold.

@group(0) @binding(0) var<uniform> frame_gi: GiFrame;
@group(0) @binding(1) var<storage, read_write> gi_probes: array<u32>;
@group(0) @binding(2) var<storage, read> gi_lists: array<u32>;
@group(0) @binding(3) var<storage, read> gi_pending: array<GiPending>;
// The shadow stream (`radiance_cache_shadow`) as its tracers answered it:
// what a probe's own light sample waits on.
@group(0) @binding(4) var<storage, read> shadow_hits: array<SceneHit>;
@group(0) @binding(5) var<storage, read> gi_answered: array<u32, 4>;
// The probes held since the list was last read back: their count, then
// their indices (GI_HELD places; past them a held probe waits its turn).
@group(0) @binding(6) var<storage, read_write> gi_held: array<atomic<u32>>;

override GI_HELD: u32 = 0u;

const PI: f32 = 3.14159265;
const LUMINANCE: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

var<workgroup> w_radiance: array<vec3<f32>, 64>;
var<workgroup> w_dir: array<vec3<f32>, 64>;
var<workgroup> w_t: array<f32, 64>;
var<workgroup> w_inside: atomic<u32>;
var<workgroup> w_hits: atomic<u32>;
var<workgroup> w_waiting: atomic<u32>;
var<workgroup> w_hold: u32;
var<workgroup> w_answered: u32;

@compute @workgroup_size(64)
fn gi_integrate(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if wg.x >= frame_gi.updated { return; }
    if li == 0u {
        w_answered = gi_answered[0];
        atomicStore(&w_inside, 0u);
        atomicStore(&w_hits, 0u);
        atomicStore(&w_waiting, 0u);
    }
    // Nothing answered (no tracer has its geometry yet: a scene loading):
    // the probes keep what they hold rather than take the sky everywhere.
    if workgroupUniformLoad(&w_answered) == 0u { return; }
    let index = gi_lists[wg.x];
    let spacing = frame_gi.levels[gi_probe_cell(index).w].origin.w;
    // Moments past a few spacings tell nothing more.
    let far = 3.5 * spacing;
    let dir = gi_ray_dir(li, frame_gi.rotation);
    let pending = gi_pending[wg.x * GI_RAYS + li];
    // A hit lit on the spot: its own light sample's light through its shadow
    // ray, where it has one.
    let spot = pending.t > 0.0 && pending.row != GI_NO_ROW;
    if pending.t > 0.0 {
        atomicAdd(&w_hits, 1u);
        if spot && (pending.row & GI_ROW_WAITING) != 0u { atomicAdd(&w_waiting, 1u); }
    }
    let at = index * GI_PROBE_WORDS;
    let state = gi_probes[at + GI_STATE_WORD];
    let updates = state >> 8u;
    workgroupBarrier();
    if li == 0u {
        w_hold = u32(updates == 0u && atomicLoad(&w_waiting) * 32u > atomicLoad(&w_hits));
        if w_hold != 0u {
            let k = atomicAdd(&gi_held[0], 1u);
            if k < GI_HELD { atomicStore(&gi_held[1u + k], index); }
        }
    }
    if workgroupUniformLoad(&w_hold) != 0u { return; }
    var radiance = vec3<f32>(0.0);
    var t = far;
    if pending.t < 0.0 {
        atomicAdd(&w_inside, 1u);
        t = min(-pending.t, far);
    } else {
        if pending.t > 0.0 { t = min(pending.t, far); }
        radiance = scene_unpack_radiance(pending.rest);
        if spot && (pending.row & GI_ROW_INDEX) != GI_ROW_INDEX {
            radiance += scene_unpack_radiance(pending.light) * scene_occlusion_through(shadow_hits[pending.row & GI_ROW_INDEX]);
        }
    }
    // Fireflies: a ray meeting a surface a few centimetres from a point
    // light (its fitting) carries tens of times the room's light, and would
    // dominate its probe for many updates. Once the probe holds light, a
    // ray is capped at 16 times its mean irradiance / pi, plus 2 (so a dark
    // probe still sees a sunlit opening).
    let held = gi_probes[index * GI_PROBE_WORDS + GI_STATE_WORD] >> 8u;
    if held > 0u {
        let dc = vec3<f32>(gi_half2(gi_probes[index * GI_PROBE_WORDS]), gi_half2(gi_probes[index * GI_PROBE_WORDS + 1u]).x) * 0.282095;
        let cap = 16.0 * dot(max(dc, vec3<f32>(0.0)), LUMINANCE) + 2.0;
        let l = dot(radiance, LUMINANCE);
        if l > cap { radiance *= cap / l; }
    }
    w_radiance[li] = radiance;
    w_dir[li] = dir;
    w_t[li] = t;
    workgroupBarrier();
    // Moments: a lane a texel, its nearby rays' distances.
    {
        let texel = vec2<f32>(f32(li % GI_MOMENTS_SIDE), f32(li / GI_MOMENTS_SIDE));
        let d = gi_oct_decode((texel + 0.5) / f32(GI_MOMENTS_SIDE));
        var sum = vec2<f32>(0.0);
        var weights = 0.0;
        for (var k = 0u; k < GI_RAYS; k++) {
            let w = pow(max(dot(d, w_dir[k]), 0.0), 16.0);
            sum += vec2<f32>(w_t[k], w_t[k] * w_t[k]) * w;
            weights += w;
        }
        if weights > 1e-6 {
            let fresh = sum / weights;
            let old = gi_half2(gi_probes[at + GI_MOMENTS_WORD + li]);
            let alpha = select(max(1.0 / f32(updates + 1u), 0.05), 1.0, updates == 0u);
            gi_probes[at + GI_MOMENTS_WORD + li] = pack2x16float(mix(old, fresh, alpha));
        }
    }
    if li != 0u { return; }
    // Irradiance / pi: radiance on L1 harmonics, convolved (bands 1, 2/3).
    var sh = array<vec3<f32>, 4>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    for (var k = 0u; k < GI_RAYS; k++) {
        let b = gi_sh_basis(w_dir[k]);
        sh[0] += w_radiance[k] * b.x;
        sh[1] += w_radiance[k] * b.y;
        sh[2] += w_radiance[k] * b.z;
        sh[3] += w_radiance[k] * b.w;
    }
    // Radiance's coefficients by Monte Carlo over the sphere; irradiance /
    // pi's are theirs times 1 and 2/3 (uniform radiance 1 gives 1).
    let scale = 4.0 * PI / f32(GI_RAYS);
    sh[0] *= scale;
    sh[1] *= scale * (2.0 / 3.0);
    sh[2] *= scale * (2.0 / 3.0);
    sh[3] *= scale * (2.0 / 3.0);
    let alpha = select(max(1.0 / f32(updates + 1u), 0.03), 1.0, updates == 0u);
    // Words: (r0 g0) (b0 r1) (g1 b1) (r2 g2) (b2 r3) (g3 b3).
    let fresh = array<vec2<f32>, 6>(vec2<f32>(sh[0].r, sh[0].g), vec2<f32>(sh[0].b, sh[1].r), vec2<f32>(sh[1].g, sh[1].b), vec2<f32>(sh[2].r, sh[2].g), vec2<f32>(sh[2].b, sh[3].r), vec2<f32>(sh[3].g, sh[3].b));
    for (var w = 0u; w < 6u; w++) {
        gi_probes[at + w] = pack2x16float(mix(gi_half2(gi_probes[at + w]), fresh[w], alpha));
    }
    let inside = atomicLoad(&w_inside) * 4u > GI_RAYS;
    gi_probes[at + GI_STATE_WORD] = (min(updates + 1u, 255u) << 8u) | select(GI_ACTIVE, 0u, inside);
}
