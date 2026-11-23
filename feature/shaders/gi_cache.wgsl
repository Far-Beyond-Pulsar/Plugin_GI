//!use frame_ray_stream
//!use frame_ray_surface
//!use frame_radiance_cache
//!use frame_surface
//!use frame_bsdf
//!use gi_common
//!use gi_rays
// GI's radiance cache (README.md, "The radiance cache"): the frame's
// `frame_radiance_cache` contract, published, kept and lit here.
//
// The cells in use are a list (`cache_cells`, two halves, one current):
// the cells the last frame kept, then the new cells' queue the readers' marks
// filled (`frame_radiance_cache`). Every pass walks that list (indirect
// dispatches, `cache_args`), never the hash's slots: a frame's work follows
// the cells in use and those lit, not the cache's size.
//
// At the start of GI's update (`RadianceCache`):
// - `cache_begin`, one thread: the walk's length (the cells kept, the new
//   ones queued) and its dispatch.
// - `cache_scan`, a thread a cell: how long ago it was used and how much
//   it needs light (`need_bin`), into histograms.
// - `cache_plan`, one thread: which cells go (unused for CACHE_EVICT_AGE
//   frames, or the least recently used past half the slots) and which are
//   lit (CACHE_CELLS at most, the most in need first).
// - `cache_select`, a thread a cell: evicts, keeps the others in the other
//   half of the list, lists those lit.
// - `cache_end`, one thread: the kept list current, the queue emptied, the
//   rays' dispatch, GI's stream count, the frame's statistics.
// - `cache_rays`, a thread a ray: CACHE_RAYS rays a lit cell, cosine
//   distributed about its normal from its point, into GI's stream after the
//   probes' rays.
//
// After GI's light and shadow passes lit the rays' hits (`gi_light.wgsl`)
// and the shadow stream's tracers answered (`RadianceCacheFold`):
// - `cache_fold`, a thread a lit cell: its rays folded into its three parts.

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
// The cells in use: words 0 (the current half) and 1 (its length), then two
// halves of `radiance_cache_frame.origin_z.z` words.
@group(0) @binding(11) var<storage, read_write> cache_cells: array<u32>;
// The indirect dispatches: the walks' (0..3), the rays' (4..7). In a group
// of its own, bound only to `cache_begin` and `cache_end`, which write it: a
// dispatch may not read its arguments from a buffer it binds for writing.
@group(1) @binding(0) var<storage, read_write> cache_args: array<u32, 8>;
// The fold's inputs: what GI's light and shadow passes left a cache ray.
@group(0) @binding(12) var<storage, read> gi_pending: array<GiPending>;
@group(0) @binding(13) var<storage, read> cell_pending: array<CachePending>;
@group(0) @binding(14) var<storage, read> shadow_hits: array<SceneHit>;
@group(0) @binding(15) var<storage, read> gi_answered: array<u32, 4>;
// GI's stream's surfaces (the fold's: a cache ray's hit's face, for its
// cell's clear distances).
@group(0) @binding(16) var<storage, read> ray_surfaces: array<SceneSurface>;

// Cells lit a frame at most (the ray budget over CACHE_RAYS).
override CACHE_CELLS: u32 = 8192u;
// Frames between two updates of a settled cell (`cache::refresh_period`).
override CACHE_REFRESH: u32 = 21u;
// The frame's statistics.
override CACHE_STAT_CELLS: u32 = 0u;
override CACHE_STAT_NEW: u32 = 0u;
override CACHE_STAT_EVICTED: u32 = 0u;
override CACHE_STAT_LIT: u32 = 0u;
override CACHE_STAT_SETTLED: u32 = 0u;
override CACHE_STAT_RAYS: u32 = 0u;
override CACHE_STAT_STREAM: u32 = 0u;

// A part's history (`cache.rs`): at least CACHE_HISTORY_MIN updates, at most
// CACHE_HISTORY_MAX, between them as many as bring its value's relative
// standard error to CACHE_ERROR; the bounced part's CACHE_BOUNCED.
const CACHE_HISTORY_MIN: u32 = 16u;
const CACHE_HISTORY_MAX: u32 = 64u;
const CACHE_ERROR: f32 = 0.03;
const CACHE_BOUNCED: u32 = 16u;
// Frames unused before a cell goes (`cache::EVICT_AGE`), as a histogram
// bin: ages 128 to 255. Past half the slots, the least recently used go too,
// but never a cell marked within half that (ages under 64, bin 7 and up go):
// longer than any reader goes between two reads of a cell it keeps reading.
const CACHE_EVICT_BIN: u32 = 8u;
const CACHE_REVISIT_BIN: u32 = 7u;
// The fold's change test (`cache.rs`): a fast (CACHE_FAST updates) and a
// slow (the history's) mean of each update's luminance, clamped from above at
// the fast mean plus CACHE_CLAMP mean absolute deviations, apart by more than
// CACHE_LAG deviations (and CACHE_FLOOR of the slow mean) once CACHE_DETECT
// updates measured them, restart the part at the next update.
const CACHE_FAST: u32 = 4u;
const CACHE_CLAMP: f32 = 3.0;
const CACHE_LAG: f32 = 1.74;
const CACHE_FLOOR: f32 = 0.01;
const CACHE_DETECT: u32 = 8u;
// The spread of a cell's ray (`cache::SPREAD`).
const CACHE_SPREAD: f32 = 0.2216;
// How much a cell's clear distance grows an update its rays meet nothing
// nearer that way (`cache::CLEAR_GROWTH`: twice in eight updates).
const CACHE_CLEAR_GROWTH: f32 = 1.0905;

// `cache_plan`'s words: the histograms of need (0..32; 0: no light) and of
// used age (32..64), then the plan.
const PLAN_EVICT: u32 = 64u;
const PLAN_TAKE: u32 = 65u;
const PLAN_LEFT: u32 = 66u;
const PLAN_TAKEN: u32 = 67u;
const PLAN_OCCUPIED: u32 = 69u;
const PLAN_EVICTED: u32 = 70u;
const PLAN_ITEMS: u32 = 71u;
const PLAN_KEPT_BEFORE: u32 = 72u;
const PLAN_KEPT: u32 = 73u;
const PLAN_NEW: u32 = 74u;
const PLAN_SETTLED: u32 = 75u;

// The need bins (`need_bin`).
const NEED_NEW: u32 = 31u;
const NEED_RESTART: u32 = 30u;
const NEED_REFRESH: u32 = 29u;
const NEED_CONVERGING: u32 = 28u;

// A cell's life beside its entry (the cache's own), 48 bytes: the frame it
// was last lit plus one (0: never); `counts`: the updates folded into its
// direct (0..8), indirect (8..16) and bounced (16..24) parts, and the parts
// that restart at their next update (24: direct, 25: indirect and bounced,
// 26: bounced); each part's change test (the fast and slow mean of its
// clamped luminance, their mean absolute deviation). The walks read
// lives and slots alone, never an entry's line. Evicting a cell clears its
// life.
struct CacheLife {
    lit: u32,
    counts: u32,
    direct: array<f32, 3>,
    indirect: array<f32, 3>,
    bounced: array<f32, 3>,
    pad: u32,
}

fn life_test(t: array<f32, 3>) -> vec3<f32> {
    return vec3<f32>(t[0], t[1], t[2]);
}

fn life_row(t: vec3<f32>) -> array<f32, 3> {
    return array<f32, 3>(t.x, t.y, t.z);
}

// A part's history from its change test (fast, slow, deviation): as many
// updates as bring its value's relative standard error to CACHE_ERROR (the
// standard deviation of an update 1.2533 mean absolute deviations), within
// CACHE_HISTORY_MIN..CACHE_HISTORY_MAX. A part without light: the least.
fn part_history(t: vec3<f32>) -> u32 {
    if !(t.y > 0.0) { return CACHE_HISTORY_MIN; }
    let relative = 1.2533 * t.z / t.y / CACHE_ERROR;
    return u32(clamp(ceil(relative * relative), f32(CACHE_HISTORY_MIN), f32(CACHE_HISTORY_MAX)));
}

// Frames `age` as a histogram bin: 0 for 0, else 1 + its highest bit.
fn age_bin(age: u32) -> u32 {
    return select(0u, min(firstLeadingBit(age) + 1u, 30u), age > 0u);
}

// The share of its history a cell holds: the least of its parts' (1: they
// hold the histories their noise asks for).
fn life_held(life: CacheLife) -> f32 {
    return min(
        min(f32(life.counts & 0xffu) / f32(part_history(life_test(life.direct))),
            f32((life.counts >> 8u) & 0xffu) / f32(part_history(life_test(life.indirect)))),
        f32((life.counts >> 16u) & 0xffu) / f32(CACHE_BOUNCED));
}

// A part holds its history at CACHE_SETTLED of it: the history its noise
// asks for is measured by the part's own deviation, itself uncertain by
// about that much (its square's error over a hundred updates), so a cell
// whose history wobbles by that does not count as converging again.
const CACHE_SETTLED: f32 = 0.875;

// Whether a cell is settled: lit, no part restarting, its histories held.
fn life_settled(life: CacheLife) -> bool {
    return life.lit != 0u && (life.counts >> 24u) == 0u && life_held(life) >= CACHE_SETTLED;
}

// How much cell `s` (its life `life`) needs light this frame, a bin (high
// first): never lit; a part restarts (its light changed); its refresh
// (every CACHE_REFRESH frames, spread over the frames by its slot, settled
// or not: a change no reader's frame shows is found, and a cell the
// converging ones crowd out still takes an update); a cell still
// converging, by the share of its history it holds (least first); 0: none.
fn need_bin(s: u32, life: CacheLife) -> u32 {
    // A cell never lit, unless it waits for a mark to seat it
    // (`RADIANCE_CACHE_UNSEATED`: its rays have nowhere to start from).
    if life.lit == 0u { return select(NEED_NEW, 0u, radiance_cache_entries[s].point == RADIANCE_CACHE_UNSEATED); }
    if (life.counts >> 24u) != 0u { return NEED_RESTART; }
    let turn = (radiance_cache_frame.origin_z.w + radiance_cache_mix(s)) % CACHE_REFRESH;
    if turn == 0u { return NEED_REFRESH; }
    let held = life_held(life);
    if held >= CACHE_SETTLED { return 0u; }
    return NEED_CONVERGING - u32(clamp(held / CACHE_SETTLED * 8.0, 0.0, 7.0));
}

// The walk's cells: the current half's, then the queue's.
fn cache_cell(i: u32) -> u32 {
    let kept = atomicLoad(&cache_plan[PLAN_KEPT_BEFORE]);
    if i < kept { return cache_cells[2u + cache_cells[0] * radiance_cache_frame.origin_z.z + i]; }
    return atomicLoad(&radiance_cache_slots[radiance_cache_frame.origin_z.z + 1u + (i - kept)].state);
}

@compute @workgroup_size(1)
fn cache_begin() {
    let slots = radiance_cache_frame.origin_z.z;
    let kept = cache_cells[1];
    let queued = min(atomicLoad(&radiance_cache_slots[slots].state), slots / RADIANCE_CACHE_QUEUE_SHARE);
    let items = kept + queued;
    atomicStore(&cache_plan[PLAN_ITEMS], items);
    atomicStore(&cache_plan[PLAN_KEPT_BEFORE], kept);
    atomicStore(&cache_plan[PLAN_NEW], queued);
    cache_args[0] = (items + 63u) / 64u;
    cache_args[1] = 1u;
    cache_args[2] = 1u;
}

// A workgroup's histograms (need, used age) and counts, added to the plan's
// once a workgroup: a cell an atomic on 64 shared words would serialise the
// walk.
var<workgroup> w_bins: array<atomic<u32>, 64>;
var<workgroup> w_count: array<atomic<u32>, 3>;
var<workgroup> w_base: array<u32, 3>;

@compute @workgroup_size(64)
fn cache_scan(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    atomicStore(&w_bins[li], 0u);
    if li == 0u { atomicStore(&w_count[0], 0u); }
    workgroupBarrier();
    var s = RADIANCE_CACHE_NONE;
    if id.x < atomicLoad(&cache_plan[PLAN_ITEMS]) { s = cache_cell(id.x); }
    if s != RADIANCE_CACHE_NONE {
        let used = age_bin(radiance_cache_frame.origin_z.w - atomicLoad(&radiance_cache_slots[s].used));
        atomicAdd(&w_bins[32u + used], 1u);
        // A cell that goes by age is not lit.
        if used < CACHE_EVICT_BIN {
            let need = need_bin(s, cache_life[s]);
            atomicAdd(&w_bins[need], 1u);
            if need == 0u || (need == NEED_REFRESH && life_settled(cache_life[s])) { atomicAdd(&w_count[0], 1u); }
        }
    }
    workgroupBarrier();
    let here = atomicLoad(&w_bins[li]);
    if here > 0u { atomicAdd(&cache_plan[li], here); }
    if li == 0u {
        let settled = atomicLoad(&w_count[0]);
        if settled > 0u { atomicAdd(&cache_plan[PLAN_SETTLED], settled); }
    }
}

@compute @workgroup_size(1)
fn cache_plan_main() {
    // What goes: unused for CACHE_EVICT_AGE frames; past half the slots, the
    // least recently used too (whole bins, never those a reader may still
    // come back to: CACHE_REVISIT_BIN), so a new cell's bucket keeps room.
    var evict = CACHE_EVICT_BIN;
    var kept = 0u;
    for (var b = 0u; b < CACHE_EVICT_BIN; b++) { kept += atomicLoad(&cache_plan[32u + b]); }
    let load = radiance_cache_frame.origin_z.z / 2u;
    while kept > load && evict > CACHE_REVISIT_BIN {
        evict -= 1u;
        kept -= atomicLoad(&cache_plan[32u + evict]);
    }
    var evicted = 0u;
    for (var b = evict; b < 32u; b++) { evicted += atomicLoad(&cache_plan[32u + b]); }
    // What is lit: whole bins from the most in need down, then a share of
    // the bin the budget ends in; bin 0 never.
    var left = CACHE_CELLS;
    var take = 0u;
    for (var b = 31u; b > 0u; b--) {
        let here = atomicLoad(&cache_plan[b]);
        if here > left { take = b; break; }
        left -= here;
    }
    // A budget that covers every bin: the share is the whole of bin 0's,
    // which `cache_select` never lists.
    atomicStore(&cache_plan[PLAN_EVICT], evict);
    atomicStore(&cache_plan[PLAN_TAKE], take);
    atomicStore(&cache_plan[PLAN_LEFT], left);
    atomicStore(&cache_plan[PLAN_OCCUPIED], kept);
    atomicStore(&cache_plan[PLAN_EVICTED], evicted);
}

@compute @workgroup_size(64)
fn cache_select(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if li < 3u { atomicStore(&w_count[li], 0u); }
    workgroupBarrier();
    let take = atomicLoad(&cache_plan[PLAN_TAKE]);
    var s = RADIANCE_CACHE_NONE;
    if id.x < atomicLoad(&cache_plan[PLAN_ITEMS]) { s = cache_cell(id.x); }
    // What this cell is: 0 nothing (evicted, or not lit), 1 lit, 2 lit if
    // the budget's share of the bin it ends in reaches it; and whether kept.
    var kind = 0u;
    var keep = false;
    if s != RADIANCE_CACHE_NONE {
        let used = age_bin(radiance_cache_frame.origin_z.w - atomicLoad(&radiance_cache_slots[s].used));
        if used >= atomicLoad(&cache_plan[PLAN_EVICT]) {
            atomicStore(&radiance_cache_slots[s].state, 0u);
            cache_life[s] = CacheLife(0u, 0u, array<f32, 3>(0.0, 0.0, 0.0), array<f32, 3>(0.0, 0.0, 0.0), array<f32, 3>(0.0, 0.0, 0.0), 0u);
        } else {
            keep = true;
            let bin = need_bin(s, cache_life[s]);
            if bin > 0u { kind = select(select(0u, 2u, bin == take), 1u, bin > take); }
        }
    }
    // The kept cells, in a workgroup's block of the other half.
    var at_kept = 0u;
    if keep { at_kept = atomicAdd(&w_count[2], 1u); }
    // The bin's share, in a workgroup's block of its count.
    var rank = 0u;
    if kind == 2u { rank = atomicAdd(&w_count[0], 1u); }
    workgroupBarrier();
    if li == 0u {
        let n = atomicLoad(&w_count[0]);
        w_base[0] = select(0u, atomicAdd(&cache_plan[PLAN_TAKEN], n), n > 0u);
        let k = atomicLoad(&w_count[2]);
        w_base[2] = select(0u, atomicAdd(&cache_plan[PLAN_KEPT], k), k > 0u);
    }
    workgroupBarrier();
    if kind == 2u && w_base[0] + rank >= atomicLoad(&cache_plan[PLAN_LEFT]) { kind = 0u; }
    if keep {
        let other = 1u - cache_cells[0];
        cache_cells[2u + other * radiance_cache_frame.origin_z.z + w_base[2] + at_kept] = s;
    }
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

@compute @workgroup_size(1)
fn cache_end() {
    let slots = radiance_cache_frame.origin_z.z;
    // The kept cells are the list now; the queue is taken in.
    cache_cells[0] = 1u - cache_cells[0];
    cache_cells[1] = atomicLoad(&cache_plan[PLAN_KEPT]);
    atomicStore(&radiance_cache_slots[slots].state, 0u);
    let cells = min(atomicLoad(&cache_plan[CACHE_PLAN_CELLS]), CACHE_CELLS);
    let rays = cells * CACHE_RAYS;
    cache_args[4] = (rays + 63u) / 64u;
    cache_args[5] = 1u;
    cache_args[6] = 1u;
    let count = frame_gi.updated * GI_RAYS + rays;
    gi_header[0] = (count + RAY_STREAM_GROUP - 1u) / RAY_STREAM_GROUP;
    gi_header[1] = 1u;
    gi_header[2] = 1u;
    gi_header[3] = count;
    atomicAdd(&stats[CACHE_STAT_CELLS], atomicLoad(&cache_plan[PLAN_KEPT]));
    atomicAdd(&stats[CACHE_STAT_NEW], atomicLoad(&cache_plan[PLAN_NEW]));
    atomicAdd(&stats[CACHE_STAT_EVICTED], atomicLoad(&cache_plan[PLAN_EVICTED]));
    atomicAdd(&stats[CACHE_STAT_SETTLED], atomicLoad(&cache_plan[PLAN_SETTLED]));
    atomicAdd(&stats[CACHE_STAT_RAYS], rays);
    atomicAdd(&stats[CACHE_STAT_STREAM], rays);
}

// Cell `slot`'s ray `k` of this frame: cosine distributed about `n`
// (`bsdf_cosine_direction` in the cell's own frame, `radiance_cache_tangents`),
// stratified by the cosine's measure into two elevation halves (the outer:
// below 45 degrees over the surface) and in azimuth. Rays 0 and 1 are low,
// in the two opposite ways (`radiance_cache_along`'s eighths) that this
// frame's turn names (`frame % 4` and 4 more), so every way of a cell's
// surface is explored once in four consecutive frames' updates (its clear
// distances, `radiance_cache_holds`), and over those the low half's azimuth
// is covered evenly; rays 2 and 3 are high, an azimuth half each. Jittered
// within its stratum by the cell and the frame.
fn cell_dir(slot: u32, k: u32, n: vec3<f32>) -> vec3<f32> {
    let frame_number = radiance_cache_frame.origin_z.w;
    let seed = gi_hash(slot * 0x9e3779b9u ^ frame_number * 0x85ebca6bu);
    let j = vec2<f32>(f32(gi_hash(seed ^ (k * 0x68e31da4u)) >> 8u), f32(gi_hash(seed ^ (k * 0xb5297a4du + 1u)) >> 8u)) / 16777216.0;
    var u: vec2<f32>;
    if k < 2u {
        // The way's eighth, centred on its axis (`radiance_cache_along`).
        let way = (frame_number % 4u) + 4u * k;
        u = vec2<f32>(0.5 + 0.5 * j.x, (f32(way) - 0.5 + j.y) / f32(RADIANCE_CACHE_WAYS));
    } else {
        u = vec2<f32>(0.5 * j.x, (f32(k - 2u) + j.y) * 0.5);
    }
    let l = bsdf_cosine_direction(vec2<f32>(u.x, fract(u.y)));
    let t = radiance_cache_tangents(n);
    return normalize(t[0] * l.x + t[1] * l.y + n * l.z);
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

// One part's fold: its value over a history `n` updates long (0: no value
// yet), its change test (fast, slow, deviation: `CacheLife`), whether it
// restarts at this update; this frame's update `s`.
//
// The value is the plain mean of the updates since its last restart (at
// most the part's history, `part_history`, of them): weights that depend on
// the samples bias it (a history cut short after a bright update keeps that
// update and weighs it more; one cut after it forgets it). The change test
// runs on luminance clamped from above (one bright update cannot move it)
// and only chooses to restart at the next update, whose sample the test has
// not seen.
struct Part {
    v: vec3<f32>,
    n: u32,
    test: vec3<f32>,
    restart: bool,
}

// `history`: the part's history, or 0 for the one its noise asks for
// (`part_history`).
fn fold(p: Part, s: vec3<f32>, history: u32) -> Part {
    let l = bsdf_luminance(s);
    if p.n == 0u {
        return Part(s, 1u, vec3<f32>(l, l, 0.0), false);
    }
    if p.restart {
        return Part(s, 1u, p.test, false);
    }
    // The test's means over the updates so far (the history's, at most).
    let m = min(p.n + 1u, select(history, part_history(p.test), history == 0u));
    let fast = p.test.x;
    let slow = p.test.y;
    var dev = p.test.z;
    let w = select(l, min(l, fast + CACHE_CLAMP * dev), m > CACHE_DETECT);
    dev += (abs(w - fast) - dev) / f32(m);
    let f = fast + (w - fast) / f32(min(m, CACHE_FAST));
    let sl = slow + (w - slow) / f32(m);
    let moved = m > CACHE_DETECT && abs(f - sl) > CACHE_LAG * dev + CACHE_FLOOR * sl + 1e-6;
    // The level moved: the slow mean starts again at the fast one.
    let test = select(vec3<f32>(f, sl, dev), vec3<f32>(f, f, dev), moved);
    // The value's history is the one this test asks for, so the walk's need
    // (`need_bin`, from the same test) finds it full once it is.
    let n = min(p.n + 1u, select(history, part_history(test), history == 0u));
    return Part(p.v + (s - p.v) / f32(n), n, test, moved);
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
    var bounced = vec3<f32>(0.0);
    // How far along the cell's surface its rays found nothing in the way,
    // this update, along each way (`radiance_cache_holds`).
    let normal = scene_unpack_unit(e.normal);
    // (0: no ray went low that way this update.)
    var clear = array<f32, RADIANCE_CACHE_WAYS>();
    // A ray that met an inside within a few of its start's offsets started
    // in a solid: the cell's point, kept to a 32768th of its cell, fell behind
    // a surface next to the one it is on (a corner's). Neither it nor its
    // light sample at that point is light arriving at the surface: the update
    // is the mean of the other rays. An inside further out is a surface's
    // back seen from in front of it: dark, and counted.
    let p0 = radiance_cache_entry_point(e);
    let behind = 4.0 * (gi_ray_offset(p0) + radiance_cache_size(e.key.tag & 31u) * 2.65e-5);
    var rays = 0u;
    var bounced_rays = 0u;
    for (var k = 0u; k < CACHE_RAYS; k++) {
        let r = i * CACHE_RAYS + k;
        let pending = gi_pending[frame_gi.updated * GI_RAYS + r];
        if pending.t < 0.0 && -pending.t < behind { continue; }
        rays += 1u;
        // A ray that goes low (below 45 degrees over the cell's surface)
        // explores its way: a hit bounds it (a wall standing there), a miss
        // leaves it open. A ray higher up says nothing of the surface (a
        // ceiling over a floor is no occluder along it). The bound is where
        // the plane of the face it met crosses the cell's surface, no farther
        // than the hit: a way is 45 degrees wide, and the hit of a ray off
        // its middle lies past the face's foot (up to 8% for a wall square
        // to the way), where a reader just behind a thin wall stands.
        let dir = cell_dir(slot, k, normal);
        let flat = radiance_cache_along(normal, dir);
        if dot(dir, normal) <= flat.y {
            let w = u32(flat.x);
            var reach = RADIANCE_CACHE_FAR;
            if pending.t > 0.0 {
                reach = flat.y * pending.t;
                // The face's distance from the point, over how fast that
                // distance changes along the cell's surface.
                let face = scene_surface_face(ray_surfaces[frame_gi.updated * GI_RAYS + r]);
                let across = length(face - normal * dot(face, normal));
                if across > 1e-3 { reach = min(reach, abs(dot(dir, face)) * pending.t / across); }
            }
            clear[w] = select(min(clear[w], reach), reach, clear[w] == 0.0);
        }
        var l = scene_unpack_radiance(pending.rest);
        if pending.row != GI_NO_ROW {
            l += scene_unpack_radiance(pending.light) * scene_occlusion_through(shadow_hits[pending.row]);
        }
        indirect += l;
        let cell = cell_pending[r];
        if cell.unlit == 0u {
            bounced += scene_unpack_radiance(cell.bounced);
            bounced_rays += 1u;
        }
        if cell.row != GI_NO_ROW {
            direct += scene_unpack_radiance(cell.light) * scene_occlusion_through(shadow_hits[cell.row]);
        }
    }
    // Every ray started in a solid: the cell's point is no longer on a free
    // surface (the surface it was taken on was replaced: terrain streamed in
    // finer, an edit, a finer level of detail). The cell starts again,
    // unseated: no value, its life cleared, its next mark its point.
    if rays == 0u {
        radiance_cache_entries[slot].history = 0u;
        radiance_cache_entries[slot].point = RADIANCE_CACHE_UNSEATED;
        cache_life[slot] = CacheLife(0u, 0u, array<f32, 3>(0.0, 0.0, 0.0), array<f32, 3>(0.0, 0.0, 0.0), array<f32, 3>(0.0, 0.0, 0.0), 0u);
        return;
    }
    // Cosine-distributed rays: irradiance is pi times their mean radiance;
    // the light samples' parts are irradiance already.
    indirect *= BSDF_PI / f32(rays);
    bounced *= BSDF_PI / f32(max(bounced_rays, 1u));
    direct /= f32(rays);
    let fresh = life.lit == 0u;
    let counts = select(life.counts, 0u, fresh);
    let d = fold(Part(vec3<f32>(e.direct[0], e.direct[1], e.direct[2]), counts & 0xffu, life_test(life.direct), ((counts >> 24u) & 1u) != 0u), direct, 0u);
    let restart = ((counts >> 25u) & 1u) != 0u;
    let n = fold(Part(vec3<f32>(e.indirect[0], e.indirect[1], e.indirect[2]), (counts >> 8u) & 0xffu, life_test(life.indirect), restart), indirect, 0u);
    // The bounced part: the light the hits reflect of the cache's own, which
    // follows the cells it reads; a plain mean over CACHE_BOUNCED updates,
    // restarting with the indirect part (what lights those cells changed).
    // The bounced part: the plain mean of its last CACHE_BOUNCED updates,
    // restarting with the indirect part (what lights the cells it reads
    // changed) or with its own change test (those cells still converging:
    // a part whose level moves is not settled).
    var bp = Part(vec3<f32>(e.bounced[0], e.bounced[1], e.bounced[2]), select((counts >> 16u) & 0xffu, 0u, restart), life_test(life.bounced), ((counts >> 26u) & 1u) != 0u);
    if bounced_rays > 0u { bp = fold(bp, bounced, CACHE_BOUNCED); }
    let nb = bp.n;
    let b = bp.v;
    // The clear distances: the nearest a ray of this update met along each
    // way, or what the cell held grown by CACHE_CLEAR_GROWTH (a way that
    // opened is found clear again within a few updates).
    // A way explored this update takes its nearest bound (the nearer of it
    // and what it held, grown); one not, what it held, grown; one never
    // explored stays unexplored (0: a reader's light holds there only near
    // the cell's point, `radiance_cache_holds`).
    var kept = array<f32, RADIANCE_CACHE_WAYS>();
    for (var w = 0u; w < RADIANCE_CACHE_WAYS; w++) {
        let old = select(radiance_cache_clear(e, w), 0.0, fresh);
        let grown = min(old * CACHE_CLEAR_GROWTH, RADIANCE_CACHE_FAR);
        if clear[w] == 0.0 {
            kept[w] = grown;
        } else if old == 0.0 {
            kept[w] = clear[w];
        } else {
            kept[w] = min(grown, clear[w]);
        }
    }
    radiance_cache_entries[slot].history = (radiance_cache_entries[slot].history & 0xffff0000u) | d.n | (n.n << 8u);
    radiance_cache_entries[slot].direct = array<f32, 3>(d.v.x, d.v.y, d.v.z);
    radiance_cache_entries[slot].indirect = array<f32, 3>(n.v.x, n.v.y, n.v.z);
    radiance_cache_entries[slot].bounced = array<f32, 3>(b.x, b.y, b.z);
    radiance_cache_entries[slot].bounced_history = nb;
    radiance_cache_entries[slot].clear = array<u32, 4>(
        pack2x16float(vec2<f32>(kept[0], kept[1])),
        pack2x16float(vec2<f32>(kept[2], kept[3])),
        pack2x16float(vec2<f32>(kept[4], kept[5])),
        pack2x16float(vec2<f32>(kept[6], kept[7])),
    );
    let flags = (u32(d.restart) << 24u) | (u32(n.restart) << 25u) | (u32(bp.restart) << 26u);
    cache_life[slot] = CacheLife(radiance_cache_frame.origin_z.w + 1u, d.n | (n.n << 8u) | (nb << 16u) | flags, life_row(d.test), life_row(n.test), life_row(bp.test), 0u);
    atomicAdd(&stats[CACHE_STAT_LIT], 1u);
}
