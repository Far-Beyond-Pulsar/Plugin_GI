//!use frame_sky
//!use sky_table
//!use frame_ray_stream
//!use frame_ray_surface
//!use frame_surface
//!use frame_light_samples
//!use frame_radiance_cache
//!use gi_common
//!use gi_sample
//!use gi_rays
//!use gi_hit
// The light of GI's rays' hits (README.md, "The frame"), once every tracer
// answered GI's stream: the probes' rays (a workgroup a probe, a lane a
// ray), then the radiance cache's (a lane a ray).
//
// `gi_light`: a hit's light from the radiance cache (`frame_radiance_cache`:
// the irradiance arriving at the hit's cell), through it from its far side
// and what it emits, or the sky on a miss, into `gi_pending`; every hit marks
// its cell in use. A probe's hit takes the cache's whole irradiance: lighting
// a surface for indirect purposes is the cache's alone. A cache ray's hit
// takes its indirect part, and a query of the frame's light samples
// (`LightQuery`) for its direct light, with one more query a cache ray for
// its cell's own direct light, compacted into the light batch.
//
// `gi_shadow`: each cache ray's light samples, its hit's off the hit's
// surface (Lambertian on the side the light is on; through it by its
// transmission share from behind) and its cell's as irradiance there, into
// `gi_pending` and `cell_pending`, waiting on shadow rays toward the light,
// compacted into the shadow stream.

@group(0) @binding(0) var<uniform> frame_gi: GiFrame;
@group(0) @binding(1) var<storage, read> gi_probes: array<u32>;
@group(0) @binding(2) var<storage, read> gi_lists: array<u32>;
@group(0) @binding(3) var<uniform> sky_frame: SkyFrame;
@group(0) @binding(4) var<uniform> sky_table: array<vec4<f32>, 2304>;
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
// A cache ray's cell's own direct light (x: a ray's light sample's
// irradiance there, waiting on row y as `GiPending`'s), a row a cache ray.
@group(0) @binding(22) var<storage, read_write> cell_pending: array<vec2<u32>>;
@group(0) @binding(23) var<storage, read> gi_rays: array<SceneRay>;

override GI_SHADOW_STAT: u32 = 0u;
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
    var cell = vec2<u32>(0u, GI_NO_ROW);
    var rest = vec3<f32>(0.0);
    var hit_query: LightQuery;
    var cell_query: LightQuery;
    var want_hit = false;
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
                // the side the ray met. A probe's hit the cache could not
                // light yet, where it matters (a surface that reflects or lets
                // light through): `gi_integrate` holds a probe with many.
                let n = s.normal;
                let footprint = width + spread * hit.t;
                let through = surface_transmission(s);
                let lit = gi_hit_light(at, dir, surface, footprint, space, !probe);
                rest = lit.rgb;
                let answered = lit.w > 0.0;
                if probe && !answered { pending.light = GI_UNLIT; }
                if !probe {
                    // Its direct light: one light sample at the hit, as wide
                    // as the ray's footprint there, lit from both sides when
                    // light passes through it.
                    let u = vec2<f32>(f32(gi_hash(seed ^ 0x68e31da4u) >> 8u), f32(gi_hash(seed ^ 0xb5297a4du) >> 8u)) / 16777216.0;
                    hit_query = light_query_point(at + eye, footprint, n, any(through > vec3<f32>(0.0)), u, gi_hash(seed), space);
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
    workgroupBarrier();
    var at = w_base + slot;
    if want_hit {
        pending.row = at;
        light_queries[at] = hit_query;
        at += 1u;
    }
    if want_cell {
        cell.y = at;
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

// A shadow ray from frame point `p` (already off its surface) toward a light
// sample: `distance` away (a directional light: as far as GI's rays reach),
// as wide as `width` with no spread (it stands for that patch of surface).
fn shadow_ray(p: vec3<f32>, sample: LightSample, width: f32, space: u32) -> SceneRay {
    let off = gi_ray_offset(p);
    let distance = light_sample_distance(sample);
    let reach = select(max(distance - 2.0 * off, 1e-4), GI_RAY_REACH, distance >= LIGHT_SAMPLE_FAR);
    let cone = scene_cone_in_space(scene_ray_cone(width, 0.0, RAY_MASK_ALL), space);
    return SceneRay(p, reach, light_sample_dir(sample), cone);
}

// A workgroup of 64 cache rays.
@compute @workgroup_size(64)
fn gi_shadow(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if li == 0u {
        atomicStore(&w_rows, 0u);
        w_cache = cache_rays_count();
    }
    let cached = workgroupUniformLoad(&w_cache);
    if wg.x * 64u >= cached { return; }
    let r = wg.x * 64u + li;
    let live = r < cached;
    let row = frame_gi.updated * GI_RAYS + r;
    var pending = GiPending(0u, 0u, 0.0, GI_NO_ROW);
    var cell = vec2<u32>(0u, GI_NO_ROW);
    // Up to two shadow rays: the hit's, then the cell's.
    var first_ray: SceneRay;
    var second_ray: SceneRay;
    var count = 0u;
    if live {
        pending = gi_pending[row];
        cell = cell_pending[r];
        let ray = gi_rays[row];
        let space = scene_ray_space(ray);
        if pending.row != GI_NO_ROW {
            let sample = light_samples[pending.row];
            pending.row = GI_NO_ROW;
            if light_sample_found(sample) {
                let hit = ray_hits[row];
                let surface = ray_surfaces[row];
                let s = scene_surface_unpack(surface);
                let face = scene_surface_face(surface);
                let towards = light_sample_dir(sample);
                let radiance = light_sample_radiance(sample);
                let through = surface_transmission(s);
                // Lambertian on the side the light is on; from behind, the
                // surface's share of it through.
                let cosine = dot(s.normal, towards);
                var light = vec3<f32>(0.0);
                var side = face;
                if cosine > 0.0 {
                    light = s.albedo.rgb * radiance * cosine / PI;
                } else if any(through > vec3<f32>(0.0)) && cosine < 0.0 {
                    light = through * radiance * -cosine / PI;
                    side = -face;
                }
                if any(light > vec3<f32>(0.0)) {
                    let p = ray.origin + ray.dir * hit.t;
                    first_ray = shadow_ray(p + side * gi_ray_offset(p), sample, scene_ray_footprint(ray, hit.t), space);
                    pending.light = scene_pack_radiance(light);
                    pending.row = 0u;
                    count = 1u;
                }
            }
        }
        if cell.y != GI_NO_ROW {
            let sample = light_samples[cell.y];
            cell.y = GI_NO_ROW;
            if light_sample_found(sample) {
                let entry = radiance_cache_entries[cache_list[r / CACHE_RAYS]];
                let n = scene_unpack_unit(entry.normal);
                let e = light_sample_radiance(sample) * max(dot(n, light_sample_dir(sample)), 0.0);
                if any(e > vec3<f32>(0.0)) {
                    // As wide as the cell it stands for.
                    let shadow = shadow_ray(ray.origin, sample, radiance_cache_size(entry.key.tag & 31u), space);
                    if count == 0u { first_ray = shadow; } else { second_ray = shadow; }
                    cell.x = scene_pack_radiance(e);
                    cell.y = count;
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
    if pending.row != GI_NO_ROW { pending.row += base; }
    if cell.y != GI_NO_ROW { cell.y += base; }
    if live {
        gi_pending[row] = pending;
        cell_pending[r] = cell;
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
