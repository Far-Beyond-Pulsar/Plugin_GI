//!use frame_ray_stream
//!use frame_radiance_cache
//!use frame_surface
//!use frame_bsdf
//!use gi_common
//!use gi_rays
// GI's radiance cache (README.md, "The radiance cache"): the frame's
// `frame_radiance_cache` contract, published, kept and lit here.
//
// At the start of GI's update (`RadianceCache`):
// - `cache_scan`, a thread a slot: how long ago each cell was used and lit,
//   into histograms of powers of two of frames.
// - `cache_plan`, one thread: which cells go (unused for CACHE_EVICT_AGE
//   frames, or the least recently used past half the slots) and which are
//   lit this frame: CACHE_CELLS at most, new ones first, then the longest
//   unlit.
// - `cache_select`, a thread a slot: evicts and lists.
// - `cache_rays`, a thread a ray: CACHE_RAYS rays a listed cell, cosine
//   distributed about its normal from its point, into GI's stream after the
//   probes' rays.
// - `cache_header`: the stream's count with them, the frame's statistics.
//
// After GI's light and shadow passes lit the rays' hits (`gi_light.wgsl`)
// and the shadow stream's tracers answered (`RadianceCacheFold`):
// - `cache_fold`, a thread a listed cell: its rays' irradiance folded into
//   it, the direct and the indirect part each with a history that restarts
//   when the estimate moves.

@group(0) @binding(0) var<uniform> radiance_cache_frame: RadianceCacheFrame;
@group(0) @binding(1) var<storage, read_write> radiance_cache_slots: array<RadianceCacheSlot>;
@group(0) @binding(2) var<storage, read_write> radiance_cache_entries: array<RadianceCacheEntry>;
@group(0) @binding(3) var<storage, read_write> cache_life: array<CacheLife>;
@group(0) @binding(4) var<storage, read_write> cache_plan: array<atomic<u32>, CACHE_PLAN_WORDS>;
@group(0) @binding(5) var<storage, read_write> cache_list: array<u32>;
@group(0) @binding(6) var<uniform> frame_gi: GiFrame;
// GI's stream (`gi`): the probes' rays from row 0 (frame_gi.updated *
// GI_RAYS, the count the CPU wrote into the header), the cache's after.
@group(0) @binding(7) var<storage, read_write> gi_rays: array<SceneRay>;
@group(0) @binding(8) var<storage, read_write> gi_hits: array<SceneHit>;
@group(0) @binding(9) var<storage, read_write> gi_header: array<u32, 4>;
@group(0) @binding(10) var<storage, read_write> stats: array<atomic<u32>>;
// The fold's inputs: what GI's light and shadow passes left a cache ray.
@group(0) @binding(12) var<storage, read> gi_pending: array<GiPending>;
@group(0) @binding(13) var<storage, read> cell_pending: array<vec2<u32>>;
@group(0) @binding(14) var<storage, read> shadow_hits: array<SceneHit>;
@group(0) @binding(15) var<storage, read> gi_answered: array<u32, 4>;

// Cells lit a frame at most (the ray budget over CACHE_RAYS).
override CACHE_CELLS: u32 = 8192u;
// The frame's statistics.
override CACHE_STAT_CELLS: u32 = 0u;
override CACHE_STAT_NEW: u32 = 0u;
override CACHE_STAT_EVICTED: u32 = 0u;
override CACHE_STAT_LIT: u32 = 0u;
override CACHE_STAT_RAYS: u32 = 0u;
override CACHE_STAT_STREAM: u32 = 0u;

// Updates a part's history holds at most (`cache::HISTORY`).
const CACHE_HISTORY: u32 = 32u;
// Frames unused before a cell goes (`cache::EVICT_AGE`), as a histogram
// bin: ages 128 to 255.
const CACHE_EVICT_BIN: u32 = 8u;
// The fold's change test (`cache.rs`): a fast (CACHE_FAST updates) and a
// slow (the history's) mean of each update's luminance, clamped from above at
// the fast mean plus CACHE_CLAMP mean absolute deviations, apart by more than
// CACHE_LAG deviations (and CACHE_FLOOR of the slow mean) once CACHE_DETECT
// updates measured them, restart the part at the next update.
const CACHE_FAST: u32 = 4u;
const CACHE_CLAMP: f32 = 3.0;
const CACHE_LAG: f32 = 1.2;
const CACHE_FLOOR: f32 = 0.05;
const CACHE_DETECT: u32 = 8u;
// The spread of a cell's ray (`cache::SPREAD`): the side of the solid angle
// one of its history's rays stands for, sqrt(2 pi / (CACHE_RAYS *
// CACHE_HISTORY)).
const CACHE_SPREAD: f32 = 0.2216;

// `cache_plan`'s words: the histograms of lit age (0..32; 31: never lit)
// and used age (32..64), then the plan.
const PLAN_EVICT: u32 = 64u;
const PLAN_TAKE: u32 = 65u;
const PLAN_LEFT: u32 = 66u;
const PLAN_TAKEN: u32 = 67u;
const PLAN_OCCUPIED: u32 = 69u;
const PLAN_EVICTED: u32 = 70u;


// A cell's life beside its entry (the cache's own), 32 bytes: the frame it
// was last lit plus one (0: never, a cell without a value: the scan and the
// selection read lives and slots alone, never an entry's line); a part
// restarts at its next update (bit 0 direct, bit 1 indirect); each part's
// change test (its fast and slow mean of clamped luminance and their mean
// absolute deviation). Evicting a cell clears its life.
struct CacheLife {
    lit: u32,
    restart: u32,
    direct: array<f32, 3>,
    indirect: array<f32, 3>,
}

fn life_test(t: array<f32, 3>) -> vec3<f32> {
    return vec3<f32>(t[0], t[1], t[2]);
}

fn life_row(t: vec3<f32>) -> array<f32, 3> {
    return array<f32, 3>(t.x, t.y, t.z);
}

// Frames `age` as a histogram bin: 0 for 0, else 1 + its highest bit.
fn age_bin(age: u32) -> u32 {
    return select(0u, min(firstLeadingBit(age) + 1u, 30u), age > 0u);
}

// The bin of cell `s` by how long ago it was lit (31: never).
fn lit_bin(s: u32) -> u32 {
    let lit = cache_life[s].lit;
    if lit == 0u { return 31u; }
    return age_bin(radiance_cache_frame.origin_z.w + 1u - lit);
}

// A workgroup's histograms (lit age, used age) and counts, added to the
// plan's once a workgroup: a cell an atomic on 64 shared words would
// serialise the scan.
var<workgroup> w_bins: array<atomic<u32>, 64>;
var<workgroup> w_count: array<atomic<u32>, 2>;
var<workgroup> w_base: array<u32, 2>;

@compute @workgroup_size(64)
fn cache_scan(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    atomicStore(&w_bins[li], 0u);
    workgroupBarrier();
    let s = id.x;
    if s < radiance_cache_frame.origin_z.z && atomicLoad(&radiance_cache_slots[s].state) != 0u {
        let used = age_bin(radiance_cache_frame.origin_z.w - atomicLoad(&radiance_cache_slots[s].used));
        atomicAdd(&w_bins[32u + used], 1u);
        // A cell that goes by age is not lit.
        if used < CACHE_EVICT_BIN { atomicAdd(&w_bins[lit_bin(s)], 1u); }
    }
    workgroupBarrier();
    let here = atomicLoad(&w_bins[li]);
    if here > 0u { atomicAdd(&cache_plan[li], here); }
}

@compute @workgroup_size(1)
fn cache_plan_main() {
    // What goes: unused for CACHE_EVICT_AGE frames; past half the slots, the
    // least recently used too (whole bins, never those used this frame or
    // the last), so a new cell's bucket keeps room.
    var evict = CACHE_EVICT_BIN;
    var kept = 0u;
    for (var b = 0u; b < CACHE_EVICT_BIN; b++) { kept += atomicLoad(&cache_plan[32u + b]); }
    let load = radiance_cache_frame.origin_z.z / 2u;
    while kept > load && evict > 2u {
        evict -= 1u;
        kept -= atomicLoad(&cache_plan[32u + evict]);
    }
    var evicted = 0u;
    for (var b = evict; b < 32u; b++) { evicted += atomicLoad(&cache_plan[32u + b]); }
    // What is lit: whole bins from the never lit down, then a share of the
    // bin the budget ends in.
    var left = CACHE_CELLS;
    var take = 0u;
    for (var b = 31u; b > 0u; b--) {
        let here = atomicLoad(&cache_plan[b]);
        if here > left { take = b; break; }
        left -= here;
    }
    atomicStore(&cache_plan[PLAN_EVICT], evict);
    atomicStore(&cache_plan[PLAN_TAKE], take);
    atomicStore(&cache_plan[PLAN_LEFT], left);
    atomicStore(&cache_plan[PLAN_OCCUPIED], kept);
    atomicStore(&cache_plan[PLAN_EVICTED], evicted);
    atomicAdd(&stats[CACHE_STAT_CELLS], kept);
    atomicAdd(&stats[CACHE_STAT_NEW], atomicLoad(&cache_plan[31]));
    atomicAdd(&stats[CACHE_STAT_EVICTED], evicted);
}

@compute @workgroup_size(64)
fn cache_select(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if li < 2u { atomicStore(&w_count[li], 0u); }
    workgroupBarrier();
    let s = id.x;
    let take = atomicLoad(&cache_plan[PLAN_TAKE]);
    // What this slot's cell is: 0 nothing (empty, evicted), 1 lit, 2 lit if
    // the budget's share of the bin it ends in reaches it.
    var kind = 0u;
    if s < radiance_cache_frame.origin_z.z && atomicLoad(&radiance_cache_slots[s].state) != 0u {
        let used = age_bin(radiance_cache_frame.origin_z.w - atomicLoad(&radiance_cache_slots[s].used));
        if used >= atomicLoad(&cache_plan[PLAN_EVICT]) {
            atomicStore(&radiance_cache_slots[s].state, 0u);
            cache_life[s] = CacheLife(0u, 0u, array<f32, 3>(0.0, 0.0, 0.0), array<f32, 3>(0.0, 0.0, 0.0));
        } else {
            let bin = lit_bin(s);
            kind = select(select(0u, 2u, bin == take), 1u, bin > take);
        }
    }
    // The bin's share, in a workgroup's block of its count.
    var rank = 0u;
    if kind == 2u { rank = atomicAdd(&w_count[0], 1u); }
    workgroupBarrier();
    if li == 0u {
        let n = atomicLoad(&w_count[0]);
        w_base[0] = select(0u, atomicAdd(&cache_plan[PLAN_TAKEN], n), n > 0u);
    }
    workgroupBarrier();
    if kind == 2u && w_base[0] + rank >= atomicLoad(&cache_plan[PLAN_LEFT]) { kind = 0u; }
    // The lit cells, in a workgroup's block of the list.
    if kind != 0u { rank = atomicAdd(&w_count[1], 1u); }
    workgroupBarrier();
    if li == 0u {
        let n = atomicLoad(&w_count[1]);
        w_base[1] = select(0u, atomicAdd(&cache_plan[CACHE_PLAN_CELLS], n), n > 0u);
    }
    workgroupBarrier();
    let at = w_base[1] + rank;
    if kind != 0u && at < CACHE_CELLS { cache_list[at] = s; }
}

// Cell `slot`'s ray `k` of this frame: cosine distributed about `n` (the
// frame's BSDF's frame and cosine direction), one of CACHE_RAYS strata,
// jittered by the cell and the frame.
fn cell_dir(slot: u32, k: u32, n: vec3<f32>) -> vec3<f32> {
    let seed = gi_hash(slot * 0x9e3779b9u ^ radiance_cache_frame.origin_z.w * 0x85ebca6bu);
    let j = vec2<f32>(f32(gi_hash(seed ^ 0x68e31da4u) >> 8u), f32(gi_hash(seed ^ 0xb5297a4du) >> 8u)) / 16777216.0;
    // Strata of a 2x2 grid, the jitter turned a stratum each frame.
    let u = fract((vec2<f32>(f32(k & 1u), f32(k >> 1u)) + j) * 0.5 + j * 0.5);
    let frame = bsdf_frame(surface_data(vec3<f32>(1.0), n, 1.0, 0.0, vec3<f32>(0.0)));
    return normalize(bsdf_to_world(frame, bsdf_cosine_direction(u)));
}

@compute @workgroup_size(64)
fn cache_rays(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x;
    let cells = min(atomicLoad(&cache_plan[CACHE_PLAN_CELLS]), CACHE_CELLS);
    if r >= cells * CACHE_RAYS { return; }
    let slot = cache_list[r / CACHE_RAYS];
    let e = radiance_cache_entries[slot];
    let n = scene_unpack_unit(e.normal);
    let p = radiance_cache_entry_point(e);
    let dir = cell_dir(slot, r % CACHE_RAYS, n);
    let space = (e.key.tag >> 8u) & 0x1ffu;
    // From the cell's point, spreading by its share of the cell's history:
    // its hit reads the cache at the detail the cell's estimate resolves
    // there (and at least the eye angle's), not at the cell's own width,
    // which would key its hits' cells as coarse as itself and mix a
    // surface's parallel layers within them.
    let cone = scene_cone_in_space(scene_ray_cone(0.0, CACHE_SPREAD, RAY_MASK_ALL), space);
    let row = frame_gi.updated * GI_RAYS + r;
    // Off its surface by f32's rounding at the point and by the point's own
    // precision (a 32768th of the cell an axis: up to sqrt(3) / 65536 of the
    // cell from where the hit was), so no ray starts behind it.
    let off = gi_ray_offset(p) + radiance_cache_size(e.key.tag & 31u) * 2.65e-5;
    gi_rays[row] = SceneRay(p + n * off, GI_RAY_REACH, dir, cone);
    gi_hits[row] = SceneHit(0.0, 0u, 0u, 0u);
}

@compute @workgroup_size(1)
fn cache_header() {
    let rays = min(atomicLoad(&cache_plan[CACHE_PLAN_CELLS]), CACHE_CELLS) * CACHE_RAYS;
    let count = frame_gi.updated * GI_RAYS + rays;
    gi_header[0] = (count + RAY_STREAM_GROUP - 1u) / RAY_STREAM_GROUP;
    gi_header[1] = 1u;
    gi_header[2] = 1u;
    gi_header[3] = count;
    atomicAdd(&stats[CACHE_STAT_RAYS], rays);
    atomicAdd(&stats[CACHE_STAT_STREAM], rays);
}

// One part's fold: its value over a history `n` updates long (0: no value
// yet), its change test (fast, slow, deviation: `CacheLife`), whether it
// restarts at this update; this frame's update `s`.
//
// The value is the plain mean of the updates since its last restart (at
// most CACHE_HISTORY of them): weights that depend on the samples bias it
// (a history cut short after a bright update keeps that update and weighs
// it more; one cut after it forgets it), by a factor 2 on the heavy tails
// of one light sample among many lamps. The change test runs on luminance
// clamped from above (one bright update cannot move it) and only chooses to
// restart at the next update, whose sample the test has not seen.
struct Part {
    v: vec3<f32>,
    n: u32,
    test: vec3<f32>,
    restart: bool,
}

fn fold(p: Part, s: vec3<f32>) -> Part {
    let l = bsdf_luminance(s);
    if p.n == 0u {
        return Part(s, 1u, vec3<f32>(l, l, 0.0), false);
    }
    if p.restart {
        return Part(s, 1u, p.test, false);
    }
    let n = min(p.n + 1u, CACHE_HISTORY);
    let v = p.v + (s - p.v) / f32(n);
    let fast = p.test.x;
    let slow = p.test.y;
    var dev = p.test.z;
    let w = select(l, min(l, fast + CACHE_CLAMP * dev), n > CACHE_DETECT);
    dev += (abs(w - fast) - dev) / f32(n);
    let f = fast + (w - fast) / f32(min(n, CACHE_FAST));
    let sl = slow + (w - slow) / f32(n);
    if n > CACHE_DETECT && abs(f - sl) > CACHE_LAG * dev + CACHE_FLOOR * sl + 1e-4 {
        // The level moved: the slow mean starts again at the fast one.
        return Part(v, n, vec3<f32>(f, f, dev), true);
    }
    return Part(v, n, vec3<f32>(f, sl, dev), false);
}

@compute @workgroup_size(64)
fn cache_fold(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    // Nothing answered (a scene loading): the cells keep what they hold.
    if gi_answered[0] == 0u { return; }
    if i >= min(atomicLoad(&cache_plan[CACHE_PLAN_CELLS]), CACHE_CELLS) { return; }
    let slot = cache_list[i];
    let e = radiance_cache_entries[slot];
    let life = cache_life[slot];
    var direct = vec3<f32>(0.0);
    var indirect = vec3<f32>(0.0);
    for (var k = 0u; k < CACHE_RAYS; k++) {
        let r = i * CACHE_RAYS + k;
        let pending = gi_pending[frame_gi.updated * GI_RAYS + r];
        var l = scene_unpack_radiance(pending.rest);
        if pending.row != GI_NO_ROW {
            l += scene_unpack_radiance(pending.light) * scene_occlusion_through(shadow_hits[pending.row]);
        }
        indirect += l;
        let cell = cell_pending[r];
        if cell.y != GI_NO_ROW {
            direct += scene_unpack_radiance(cell.x) * scene_occlusion_through(shadow_hits[cell.y]);
        }
    }
    // Cosine-distributed rays: irradiance is pi times their mean radiance;
    // the light samples' parts are irradiance already.
    indirect *= BSDF_PI / f32(CACHE_RAYS);
    direct /= f32(CACHE_RAYS);
    let fresh = !radiance_cache_has_value(e);
    let d = fold(Part(vec3<f32>(e.direct[0], e.direct[1], e.direct[2]), select(e.history & 0xffu, 0u, fresh), life_test(life.direct), (life.restart & 1u) != 0u), direct);
    let n = fold(Part(vec3<f32>(e.indirect[0], e.indirect[1], e.indirect[2]), select((e.history >> 8u) & 0xffu, 0u, fresh), life_test(life.indirect), (life.restart & 2u) != 0u), indirect);
    radiance_cache_entries[slot].history = (radiance_cache_entries[slot].history & 0xffff0000u) | d.n | (n.n << 8u);
    radiance_cache_entries[slot].direct = array<f32, 3>(d.v.x, d.v.y, d.v.z);
    radiance_cache_entries[slot].indirect = array<f32, 3>(n.v.x, n.v.y, n.v.z);
    cache_life[slot] = CacheLife(radiance_cache_frame.origin_z.w + 1u, u32(d.restart) | (u32(n.restart) << 1u), life_row(d.test), life_row(n.test));
    atomicAdd(&stats[CACHE_STAT_LIT], 1u);
}
