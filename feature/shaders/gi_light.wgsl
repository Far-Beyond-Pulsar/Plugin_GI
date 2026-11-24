//!use frame_sky
//!use sky_radiance
//!use frame_ray_stream
//!use frame_ray_surface
//!use frame_surface
//!use frame_bsdf
//!use frame_light_samples
//!use frame_radiance_cache
//!use frame_surface_stops
//!use gi_common
//!use gi_sample
//!use gi_rays
//!use gi_hit
// The light of GI's rays' hits (README.md, "The frame"), once every tracer
// answered GI's stream: the probes' rays (a workgroup a probe, a lane a
// ray), then the radiance cache's (a lane a ray).
//
// `gi_light`: a hit's light as GI lights a hit (`gi_hit`: the radiance
// cache's irradiance at its cell, through it from its far side, what it
// emits), or the sky on a miss, into `gi_pending`; every hit marks its cell
// in use (the side its ray met; the far side is read, not marked). A probe's
// hit takes the cache's whole irradiance where the cache has light for it,
// and is lit on the spot where it has none (the probes' irradiance there
// and a light sample of its own). A cache ray's hit takes the cache's
// indirect part and a light sample for its direct light, with one more
// query a cache ray for its cell's own direct light. The queries of the
// frame's light samples (`LightQuery`) are compacted into the light batch.
//
// `gi_shadow`: each light sample's light, the hit's off its surface on the
// sides that take it (`gi_hit_direct`) and a cell's as irradiance there,
// into `gi_pending` and `cell_pending`, waiting on shadow rays toward the
// light, compacted into the shadow stream.

@group(0) @binding(0) var<uniform> frame_gi: GiFrame;
@group(0) @binding(1) var<storage, read> gi_probes: array<u32>;
@group(0) @binding(2) var<storage, read> gi_lists: array<u32>;
@group(0) @binding(3) var<uniform> sky_frame: SkyFrame;
@group(0) @binding(4) var sky_radiance: texture_3d<f32>;
@group(0) @binding(24) var sky_radiance_sampler: sampler;
@group(0) @binding(5) var<uniform> gi_trace: GiTrace;
// GI's stream (`gi`) as the tracers left it: the probes' rays, then the
// cache's.
@group(0) @binding(6) var<storage, read> ray_hits: array<SceneHit>;
@group(0) @binding(7) var<storage, read> ray_surfaces: array<SceneSurface>;
// The shadow stream (`radiance_cache_shadow`): its rows written from 0 up,
// its count in its header (cleared before `gi_shadow`; `gi_shadow_header`
// writes the dispatch).
@group(0) @binding(8) var<storage, read_write> shadow_rays: array<SceneRay>;
@group(0) @binding(9) var<storage, read_write> shadow_hits: array<SceneHit>;
@group(0) @binding(10) var<storage, read_write> shadow_header: array<atomic<u32>, 4>;
@group(0) @binding(11) var<storage, read_write> gi_pending: array<GiPending>;
// Word 0: set once any ray met anything this frame (`gi_integrate` and the
// cache's fold hold on a frame where none did).
@group(0) @binding(12) var<storage, read_write> gi_answered: array<atomic<u32>, 4>;
// The light batch (`radiance_cache`): its queries written from 0 up, its
// count in its header (cleared before `gi_light`; `gi_light_header` writes
// the dispatch); the samples the samplers left.
@group(0) @binding(13) var<storage, read_write> light_header: array<atomic<u32>, 4>;
@group(0) @binding(14) var<storage, read_write> light_queries: array<LightQuery>;
@group(0) @binding(15) var<storage, read> light_samples: array<LightSample>;
// The frame's statistics (`helio_core::stats`): the shadow rays a frame.
@group(0) @binding(16) var<storage, read_write> stats: array<atomic<u32>>;
// The frame's radiance cache (GI's own, `gi_cache.wgsl`).
@group(0) @binding(17) var<uniform> radiance_cache_frame: RadianceCacheFrame;
@group(0) @binding(18) var<storage, read_write> radiance_cache_slots: array<RadianceCacheSlot>;
@group(0) @binding(19) var<storage, read_write> radiance_cache_entries: array<RadianceCacheEntry>;
@group(0) @binding(20) var<storage, read> cache_plan: array<u32, CACHE_PLAN_WORDS>;
@group(0) @binding(21) var<storage, read> cache_list: array<u32>;
// What a cache ray leaves its cell (`CachePending`), a row a cache ray.
@group(0) @binding(22) var<storage, read_write> cell_pending: array<CachePending>;
@group(0) @binding(23) var<storage, read> gi_rays: array<SceneRay>;

override GI_SHADOW_STAT: u32 = 0u;
// The hits' states (`gi.hits.probes.*`, `gi.hits.cache.*`: GI_HIT_STATES).
override PROBE_HIT_STAT_0: u32 = 0u;
override PROBE_HIT_STAT_1: u32 = 0u;
override PROBE_HIT_STAT_2: u32 = 0u;
override PROBE_HIT_STAT_3: u32 = 0u;
override PROBE_HIT_STAT_4: u32 = 0u;
override PROBE_HIT_STAT_5: u32 = 0u;
override PROBE_HIT_STAT_6: u32 = 0u;
override CACHE_HIT_STAT_0: u32 = 0u;
override CACHE_HIT_STAT_1: u32 = 0u;
override CACHE_HIT_STAT_2: u32 = 0u;
override CACHE_HIT_STAT_3: u32 = 0u;
override CACHE_HIT_STAT_4: u32 = 0u;
override CACHE_HIT_STAT_5: u32 = 0u;
override CACHE_HIT_STAT_6: u32 = 0u;

fn hit_stat(probe: bool, k: u32) -> u32 {
    switch k {
        case 0u: { return select(CACHE_HIT_STAT_0, PROBE_HIT_STAT_0, probe); }
        case 1u: { return select(CACHE_HIT_STAT_1, PROBE_HIT_STAT_1, probe); }
        case 2u: { return select(CACHE_HIT_STAT_2, PROBE_HIT_STAT_2, probe); }
        case 3u: { return select(CACHE_HIT_STAT_3, PROBE_HIT_STAT_3, probe); }
        case 4u: { return select(CACHE_HIT_STAT_4, PROBE_HIT_STAT_4, probe); }
        case 5u: { return select(CACHE_HIT_STAT_5, PROBE_HIT_STAT_5, probe); }
        default: { return select(CACHE_HIT_STAT_6, PROBE_HIT_STAT_6, probe); }
    }
}

var<workgroup> w_states: array<atomic<u32>, GI_HIT_STATES>;
// Cells the cache lights a frame at most (`CacheConfig::cells`).
override CACHE_CELLS: u32 = 8192u;

var<workgroup> w_rows: atomic<u32>;
var<workgroup> w_hit: atomic<u32>;
var<workgroup> w_base: u32;
var<workgroup> w_cache: u32;

// The cache's rays this frame.
fn cache_rays_count() -> u32 {
    return min(cache_plan[CACHE_PLAN_CELLS], CACHE_CELLS) * CACHE_RAYS;
}

@compute @workgroup_size(64)
fn gi_light(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let probe = wg.x < GI_MAX_UPDATES;
    if probe && wg.x >= frame_gi.updated { return; }
    if li == 0u {
        atomicStore(&w_rows, 0u);
        atomicStore(&w_hit, 0u);
        w_cache = cache_rays_count();
    }
    if li < GI_HIT_STATES { atomicStore(&w_states[li], 0u); }
    let cached = workgroupUniformLoad(&w_cache);
    let first = (wg.x - GI_MAX_UPDATES) * 64u;
    if !probe && first >= cached { return; }
    let eye = gi_trace.eye.xyz;
    // The ray: its row of GI's stream, where it starts (eye-relative), its
    // direction, how far behind its origin it starts (a probe's), the
    // footprint's spread and width, its space.
    var row = 0u;
    var r = 0u;
    var live = true;
    var origin = vec3<f32>(0.0);
    var dir = vec3<f32>(0.0, 1.0, 0.0);
    var behind = 0.0;
    var width = 0.0;
    var spread = GI_RAY_SPREAD;
    var space = gi_trace.info.y;
    if probe {
        row = wg.x * GI_RAYS + li;
        origin = gi_probe_position(gi_lists[wg.x]);
        dir = gi_ray_dir(li, frame_gi.rotation);
        behind = gi_ray_offset(origin + eye);
    } else {
        r = first + li;
        live = r < cached;
        row = frame_gi.updated * GI_RAYS + r;
        if live {
            let ray = gi_rays[row];
            origin = ray.origin - eye;
            dir = ray.dir;
            width = scene_ray_width(ray);
            spread = scene_ray_spread(ray);
            space = scene_ray_space(ray);
        }
    }
    var pending = GiPending(0u, 0u, 0.0, GI_NO_ROW);
    var cell = CachePending(0u, GI_NO_ROW, 0u, 0u);
    var rest = vec3<f32>(0.0);
    // A cache ray's hit: the light it reflects of the cache's own.
    var bounced = vec3<f32>(0.0);
    var hit_query: LightQuery;
    var cell_query: LightQuery;
    var want_hit = false;
    // A probe's hit lit on the spot while its cell is new (`gi_integrate`'s
    // readiness).
    var waiting = false;
    let seed = gi_hash(row ^ gi_trace.info.x);
    if live {
        let hit = ray_hits[row];
        if hit.t > 0.0 {
            atomicOr(&w_hit, 1u);
            let t = hit.t - behind;
            let surface = ray_surfaces[row];
            let s = scene_surface_unpack(surface);
            let at = origin + dir * t;
            // Its inside: a back face (the ray starts in a wall or a solid);
            // and a probe on a surface (within `behind` of it): inside it, so
            // probes on a floor are dropped alike, not some of them.
            if scene_hit_inside(hit) || (probe && t < behind) {
                pending.t = -max(t, 1e-3);
            } else {
                pending.t = t;
                // The hit's surface: the frame's `SurfaceData`, as a G-buffer
                // texel holds it. GI's stream asks for
                // `SurfaceDetail::Diffuse`: lit as Lambertian (`gi_hit`), on
                // the side the ray met.
                let n = s.normal;
                let footprint = width + spread * hit.t;
                let through = surface_transmission(s);
                // The reach (`radiance_cache_read`): any. A probe is in the
                // air; a cache ray's hit's cell is the bounced part's, a
                // bounce on from the cell's own rays, whose error a cell's
                // point a little past the ray's length moves by a second
                // order (as SHaRC's and GI-1.0's secondary hits read their
                // grids), where keeping finer cells at every contact the
                // cache's rays meet would multiply its working set.
                let reach = RADIANCE_CACHE_FAR;
                let lit = gi_hit_light(at, dir, surface, space, reach, !probe, probe, true);
                atomicAdd(&w_states[lit.miss], 1u);
                if (lit.own & GI_SIDE_FAR) != 0u { atomicAdd(&w_states[GI_HIT_FAR_OWN], 1u); }
                bounced = lit.reflected;
                rest = lit.emitted;
                // A probe takes the hit's whole light; a cache ray's cell
                // folds what the hit reflects of the cache apart.
                if probe { rest += bounced; }
                // A cache ray's hit the cache has no light for: no bounced
                // light from it (the probes' irradiance there would carry
                // light into the cache across what its cells keep apart, and
                // the cache's light feeds itself).
                if !probe && !lit.answered { cell.unlit = 1u; }
                // The sides that take a light sample of their own: every side
                // of a cache ray's hit (its direct light is its own), a
                // probe's that the cache has no light for (lit on the spot).
                let sides = select(lit.own, GI_SIDE_NEAR | GI_SIDE_FAR, !probe);
                if sides != 0u {
                    // One light sample at the hit, as wide as the ray's
                    // footprint there, from both sides where its far side
                    // takes it and light passes through.
                    let u = vec2<f32>(f32(gi_hash(seed ^ 0x68e31da4u) >> 8u), f32(gi_hash(seed ^ 0xb5297a4du) >> 8u)) / 16777216.0;
                    let both = (sides & GI_SIDE_FAR) != 0u && any(through > vec3<f32>(0.0));
                    hit_query = light_query_point(at + eye, footprint, n, both, u, gi_hash(seed), space);
                    pending.light = sides;
                    waiting = probe && lit.waiting;
                    want_hit = true;
                }
            }
        } else {
            rest = gi_sky_radiance(origin * 0.001, dir);
        }
        if !probe {
            // The cell's own direct light: one light sample at its point (the
            // ray's origin), as wide as the cell, on its side.
            let e = radiance_cache_entries[cache_list[r / CACHE_RAYS]];
            let cs = gi_hash(seed ^ 0x1b873593u);
            let u = vec2<f32>(f32(gi_hash(cs ^ 0x68e31da4u) >> 8u), f32(gi_hash(cs ^ 0xb5297a4du) >> 8u)) / 16777216.0;
            cell_query = light_query_point(origin + eye, radiance_cache_size(e.key.tag & 31u), scene_unpack_unit(e.normal), false, u, gi_hash(cs), space);
        }
    }
    pending.rest = scene_pack_radiance(rest);
    // The workgroup's queries in one block of the batch.
    let want_cell = !probe && live;
    let rows = u32(want_hit) + u32(want_cell);
    var slot = 0u;
    if rows > 0u { slot = atomicAdd(&w_rows, rows); }
    workgroupBarrier();
    if li == 0u {
        let here = atomicLoad(&w_rows);
        w_base = select(0u, atomicAdd(&light_header[3], here), here > 0u);
        if atomicLoad(&w_hit) != 0u { atomicOr(&gi_answered[0], 1u); }
    }
    if li < GI_HIT_STATES {
        let n = atomicLoad(&w_states[li]);
        if n > 0u { atomicAdd(&stats[hit_stat(probe, li)], n); }
    }
    workgroupBarrier();
    var at = w_base + slot;
    if want_hit {
        pending.row = at | select(0u, GI_ROW_WAITING, waiting);
        light_queries[at] = hit_query;
        at += 1u;
    }
    if want_cell {
        cell.row = at;
        cell.bounced = scene_pack_radiance(bounced);
        light_queries[at] = cell_query;
        cell_pending[r] = cell;
    }
    if live { gi_pending[row] = pending; }
}

// The light batch's dispatch for the samplers, from its count.
@compute @workgroup_size(1)
fn gi_light_header() {
    let count = atomicLoad(&light_header[3]);
    atomicStore(&light_header[0], (count + RAY_STREAM_GROUP - 1u) / RAY_STREAM_GROUP);
    atomicStore(&light_header[1], 1u);
    atomicStore(&light_header[2], 1u);
}

// A workgroup a probe's rays, then one for 64 cache rays.
@compute @workgroup_size(64)
fn gi_shadow(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let probe = wg.x < GI_MAX_UPDATES;
    if probe && wg.x >= frame_gi.updated { return; }
    if li == 0u {
        atomicStore(&w_rows, 0u);
        w_cache = cache_rays_count();
    }
    let cached = workgroupUniformLoad(&w_cache);
    let first = (wg.x - GI_MAX_UPDATES) * 64u;
    if !probe && first >= cached { return; }
    let r = first + li;
    let live = probe || r < cached;
    let row = select(frame_gi.updated * GI_RAYS + r, wg.x * GI_RAYS + li, probe);
    var pending = GiPending(0u, 0u, 0.0, GI_NO_ROW);
    var cell = CachePending(0u, GI_NO_ROW, 0u, 0u);
    // Up to two shadow rays: the hit's, then the cell's.
    var first_ray: SceneRay;
    var second_ray: SceneRay;
    var count = 0u;
    if live {
        pending = gi_pending[row];
        if !probe { cell = cell_pending[r]; }
        let ray = gi_rays[row];
        let space = scene_ray_space(ray);
        if pending.row != GI_NO_ROW {
            let sample = light_samples[pending.row & GI_ROW_INDEX];
            let waiting = pending.row & GI_ROW_WAITING;
            // A cache ray's light waits on no shadow ray; a probe's hit lit
            // on the spot keeps that it was, and whether its cell was new.
            pending.row = select(GI_NO_ROW, GI_ROW_INDEX | waiting, probe);
            let direct = gi_hit_direct(ray_surfaces[row], sample, pending.light);
            pending.light = 0u;
            if direct.lit {
                let p = ray.origin + ray.dir * ray_hits[row].t;
                first_ray = gi_shadow_ray(p + direct.side * gi_ray_offset(p), sample, scene_ray_footprint(ray, ray_hits[row].t), space);
                pending.light = scene_pack_radiance(direct.light);
                pending.row = waiting;
                count = 1u;
            }
        }
        if cell.row != GI_NO_ROW {
            let sample = light_samples[cell.row];
            cell.row = GI_NO_ROW;
            if light_sample_found(sample) {
                let entry = radiance_cache_entries[cache_list[r / CACHE_RAYS]];
                let n = scene_unpack_unit(entry.normal);
                let e = light_sample_radiance(sample) * max(dot(n, light_sample_dir(sample)), 0.0);
                if any(e > vec3<f32>(0.0)) {
                    // As wide as the cell it stands for.
                    let shadow = gi_shadow_ray(ray.origin, sample, radiance_cache_size(entry.key.tag & 31u), space);
                    if count == 0u { first_ray = shadow; } else { second_ray = shadow; }
                    cell.light = scene_pack_radiance(e);
                    cell.row = count;
                    count += 1u;
                }
            }
        }
    }
    // The workgroup's shadow rays in one block of the stream.
    var slot = 0u;
    if count > 0u { slot = atomicAdd(&w_rows, count); }
    workgroupBarrier();
    if li == 0u {
        let here = atomicLoad(&w_rows);
        w_base = select(0u, atomicAdd(&shadow_header[3], here), here > 0u);
    }
    workgroupBarrier();
    let base = w_base + slot;
    if count > 0u {
        shadow_rays[base] = first_ray;
        shadow_hits[base] = SceneHit(0.0, 0u, 0u, 0u);
    }
    if count > 1u {
        shadow_rays[base + 1u] = second_ray;
        shadow_hits[base + 1u] = SceneHit(0.0, 0u, 0u, 0u);
    }
    if pending.row != GI_NO_ROW && (pending.row & GI_ROW_INDEX) != GI_ROW_INDEX { pending.row += base; }
    if cell.row != GI_NO_ROW { cell.row += base; }
    if live {
        gi_pending[row] = pending;
        if !probe { cell_pending[r] = cell; }
    }
}

// The shadow stream's dispatch for the tracers, from its count; and the
// count into the frame's statistics.
@compute @workgroup_size(1)
fn gi_shadow_header() {
    let count = atomicLoad(&shadow_header[3]);
    atomicStore(&shadow_header[0], (count + RAY_STREAM_GROUP - 1u) / RAY_STREAM_GROUP);
    atomicStore(&shadow_header[1], 1u);
    atomicStore(&shadow_header[2], 1u);
    atomicAdd(&stats[GI_SHADOW_STAT], count);
}
