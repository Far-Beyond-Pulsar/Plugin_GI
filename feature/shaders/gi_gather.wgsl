//!use helio_prelude
//!use frame_sky
//!use sky_table
//!use frame_ray_stream
//!use frame_ray_surface
//!use frame_surface
//!use frame_bsdf
//!use frame_radiance_cache
//!use frame_history
//!use frame_denoise_irradiance
//!use gi_common
//!use gi_sample
//!use gi_rays
//!use gi_hit
// Indirect diffuse at a pixel (README.md, "The pixels' gather"): one ray a
// block of internal pixels, its hit lit from the radiance cache (`gi_hit`),
// kept in a reservoir a traced texel and reused across frames and
// neighbours (ReSTIR GI, Ouyang et al. 2021), into the frame's denoiser's
// irradiance signal, demodulated by the near field.
//
// - `gather_near` (a pixel): the near field. Occlusion nearer than a block's
//   footprint is not in one ray a block; a horizon search over that
//   footprint and no farther gives it at every pixel (V), and the signal is
//   divided by V at its traced pixel and multiplied by it at each pixel
//   (`gi_resolve_pixels`), so the rays' estimate is exact where it was traced
//   and the near field moves it between the pixels of a block. A surface
//   whose occlusion is its own (`gbuffer_own_occlusion`: a traced terrain's
//   short sky rays, a meadow's) brings its V in its material occlusion, which
//   the lighting multiplies in itself.
// - `gather_rays` (a texel): its traced pixel's ray (cosine about the
//   pixel's normal) into the stream `gi_pixels`, and for one reservoir in
//   GATHER period, a ray re-tracing its kept sample.
// - `gather_validate` (a texel): a re-traced sample's light now; a sample
//   whose light changed lowers the confidence written for the denoiser.
// - `gather_temporal` (a texel): this frame's sample combined with the
//   reservoir where the traced pixel was last frame (`frame_history`).
// - `gather_spatial` (a texel): that combined with neighbours' (not kept:
//   only the temporal reservoirs carry over), resolved to the signal.

// The frame (`GatherFrame`, gather.rs).
struct GatherFrame {
    // x, y: the internal size; z, w: the traced size (a texel a block).
    size: vec4<u32>,
    // x: the block's side; y: the frame (the graph's frame number, truncated:
    // the denoiser's); z: GATHER_* flags; w: the camera's coordinate space
    // (`SceneRay::space`).
    info: vec4<u32>,
    // xyz: what moves a point of the previous frame's space into this
    // frame's (the previous origin minus this one); w: a pixel's width a
    // metre of view depth.
    rebase: vec4<f32>,
    // x: the most samples' worth (M) a reservoir keeps; y: one reservoir in
    // this many is re-traced a frame; z: this frame's re-traced phase; w:
    // the spatial neighbours' radius in texels.
    reuse: vec4<f32>,
    // The camera's forward axis (view depth is along it).
    forward: vec4<f32>,
}

const GATHER_HAS_PREVIOUS: u32 = 1u;
const GATHER_REUSE: u32 = 2u;
const GATHER_TAGS: u32 = 4u;

// A traced texel's reservoir, 48 bytes: the receiver (the traced pixel's
// surface, frame space) and its normal; the kept sample (a hit's point and
// the normal of the side the ray met, or for the sky a unit direction), the
// light it sends to the receiver (`scene_pack_radiance`), its unbiased
// contribution weight W (area measure for a point, solid angle for the
// sky) and how many samples' worth it holds (M).
struct GatherReservoir {
    receiver: vec3<f32>,
    receiver_normal: u32,
    sample: vec3<f32>,
    sample_normal: u32,
    radiance: u32,
    weight: f32,
    m: f32,
    flags: u32,
}

const RESERVOIR_VALID: u32 = 1u;
const RESERVOIR_SKY: u32 = 2u;

// A hit distance the denoiser takes as none (a half float's largest).
const GATHER_NO_HIT: f32 = 65504.0;
// Spatial neighbours a texel takes: with the temporal reservoirs' M cap,
// GATHER_NEIGHBOURS * M covers the texels within the radius (about 28 at 3
// texels) over the frames one sample stays.
const GATHER_NEIGHBOURS: u32 = 4u;
// The relative change a re-traced hit's light may show from the cache's own
// noise: a settled cell's light is the mean of its last 32 updates (the
// cache's history), and one update varies by about its own size (one light
// sample among several, shadowed or not), so the mean by 1 / sqrt(32). A
// cell that restarted because its light changed has fewer updates and moves
// more, which is the change this detects. The sky's light is exact (0).
const CACHE_NOISE: f32 = 0.17677670;
// The horizon search: slices across the pixel (two sides each).
const NEAR_SLICES: u32 = 4u;

@group(0) @binding(0) var<uniform> gather: GatherFrame;
@group(0) @binding(1) var<storage, read> cameras: array<Camera>;
@group(0) @binding(2) var gbuf_depth: texture_depth_2d;
@group(0) @binding(3) var gbuf_normal: texture_2d<f32>;
@group(0) @binding(4) var surface_tags: texture_2d<f32>;
@group(0) @binding(5) var gbuf_orm: texture_2d<f32>;
// The near field (`gather_near`): x V (what the signal is divided by at its
// traced pixel), y what a pixel's resolved irradiance is multiplied by (V,
// or V over the material occlusion the lighting multiplies in itself).
@group(0) @binding(6) var near_out: texture_storage_2d<rgba16float, write>;
// The stream `gi_pixels`: a texel's ray at its index, then the re-traced
// samples' rays.
@group(0) @binding(7) var<storage, read_write> rays: array<SceneRay>;
// The reservoirs: the previous frame's (re-traced in place) and this
// frame's temporal ones.
@group(0) @binding(8) var<storage, read_write> previous: array<GatherReservoir>;
@group(0) @binding(9) var<storage, read_write> reservoirs: array<GatherReservoir>;
// A texel's: x this frame's ray's hit distance (GATHER_NO_HIT: none), y the
// confidence its previous reservoir's re-trace left (1 where none ran), z
// the confidence of what its temporal reservoir took.
@group(0) @binding(10) var<storage, read_write> texels: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> hits: array<SceneHit>;
@group(0) @binding(12) var<storage, read> surfaces: array<SceneSurface>;
@group(0) @binding(13) var near_field: texture_2d<f32>;
@group(0) @binding(14) var history_pixels: texture_2d<f32>;
@group(0) @binding(15) var signal0: texture_storage_2d<rgba16float, write>;
@group(0) @binding(16) var signal1: texture_storage_2d<rgba16float, write>;
// x: every tracer answered the stream in full this frame.
@group(0) @binding(17) var<uniform> gather_answer: vec4<u32>;
@group(0) @binding(18) var<uniform> frame_gi: GiFrame;
@group(0) @binding(19) var<storage, read> gi_probes: array<u32>;
@group(0) @binding(20) var<uniform> gi_trace: GiTrace;
@group(0) @binding(21) var<uniform> sky_frame: SkyFrame;
@group(0) @binding(22) var<uniform> sky_table: array<vec4<f32>, 2304>;
@group(0) @binding(23) var<uniform> radiance_cache_frame: RadianceCacheFrame;
@group(0) @binding(24) var<storage, read_write> radiance_cache_slots: array<RadianceCacheSlot>;
@group(0) @binding(25) var<storage, read_write> radiance_cache_entries: array<RadianceCacheEntry>;
@group(0) @binding(26) var<storage, read> rays_in: array<SceneRay>;

fn internal_size() -> vec2<u32> {
    return gather.size.xy;
}

fn traced_size() -> vec2<u32> {
    return gather.size.zw;
}

fn side() -> u32 {
    return gather.info.x;
}

// The frame point at pixel `q` and whether it holds a surface.
fn position_at(q: vec2<i32>) -> vec4<f32> {
    let depth = textureLoad(gbuf_depth, q, 0);
    if helio_is_sky(depth) { return vec4<f32>(0.0); }
    let uv = (vec2<f32>(q) + 0.5) / vec2<f32>(internal_size());
    let h = cameras[0].view_proj_inv * vec4<f32>(helio_uv_to_ndc(uv), depth, 1.0);
    return vec4<f32>(h.xyz / h.w, 1.0);
}

// A pixel's width at frame point `x` (its view depth times the pixel's
// angle).
fn footprint_at(x: vec3<f32>) -> f32 {
    return gather.rebase.w * max(dot(x - cameras[0].position_near.xyz, gather.forward.xyz), 0.0);
}

// Whether the surface at pixel `q` was drawn through a remap (a portal's
// chain): its light is the probes' (the resolve's own path).
fn remapped(q: vec2<i32>) -> bool {
    if (gather.info.z & GATHER_TAGS) == 0u { return false; }
    return gbuffer_remap_of(textureLoad(surface_tags, q, 0).xy) >= 0;
}

// The texel traced at internal pixel `p`'s block this frame.
fn traced_pixel(b: vec2<u32>) -> vec2<u32> {
    return denoise_traced_pixel(b, gather.info.y, side(), internal_size());
}

// ── The near field ──────────────────────────────────────────────────────────

// The largest block side (`TraceBlock::Sixteen`): the near field's search
// reaches this many pixels from its own.
const NEAR_REACH: i32 = 4;
// A workgroup's 8x8 pixels and NEAR_REACH around them: their frame points
// (w 0: no surface or past the image), read once a workgroup.
const NEAR_TILE: i32 = 16;
var<workgroup> near_tile: array<vec4<f32>, 256>;
var<workgroup> near_origin: vec2<i32>;

// The frame point at pixel `q` (within NEAR_REACH of the workgroup's
// pixels), from the tile.
fn tile_at(q: vec2<i32>) -> vec4<f32> {
    let t = q - near_origin + NEAR_REACH;
    return near_tile[t.y * NEAR_TILE + t.x];
}

// The surface's geometric normal at pixel `q` (frame point `x`), from its
// neighbours' points: on each axis the nearer of the two one-sided
// differences (a depth edge's far side says nothing of this surface),
// turned toward the eye. A shading normal (normal maps) would count the
// surface's own plane as an occluder.
fn geometric_normal(q: vec2<i32>, x: vec3<f32>) -> vec3<f32> {
    var d: array<vec3<f32>, 2>;
    for (var axis = 0u; axis < 2u; axis++) {
        let step = select(vec2<i32>(0, 1), vec2<i32>(1, 0), axis == 0u);
        let a = tile_at(q + step);
        let b = tile_at(q - step);
        let da = select(vec3<f32>(1e30), a.xyz - x, a.w > 0.0);
        let db = select(vec3<f32>(1e30), x - b.xyz, b.w > 0.0);
        d[axis] = select(db, da, dot(da, da) < dot(db, db));
    }
    var n = cross(d[1], d[0]);
    if dot(n, n) < 1e-30 || any(abs(n) > vec3<f32>(1e20)) {
        return vec3<f32>(0.0);
    }
    n = normalize(n);
    return select(-n, n, dot(n, cameras[0].position_near.xyz - x) >= 0.0);
}

// The share of the cosine-weighted hemisphere about `n` at pixel `p`
// (frame point `x`) that the depth buffer leaves open within distance `r`:
// along each of NEAR_SLICES directions across the screen and both their
// sides, the highest occluder within `r` over the tangent plane (elevation
// e) closes the elevations below it, a share sin^2 e of that azimuth's
// cosine-weighted measure; the mean over the 2 NEAR_SLICES azimuths. Every
// pixel out to `steps` along each direction is searched (no stochastic
// steps: the term needs no history of its own).
fn horizon(p: vec2<i32>, x: vec3<f32>, n: vec3<f32>, r: f32, steps: i32) -> f32 {
    let size = vec2<i32>(internal_size());
    var open = 0.0;
    for (var k = 0u; k < NEAR_SLICES; k++) {
        let angle = (f32(k) + 0.5) * PI / f32(NEAR_SLICES);
        let dir = vec2<f32>(cos(angle), sin(angle));
        for (var s = -1; s <= 1; s += 2) {
            var highest = 0.0;
            for (var j = 1; j <= steps; j++) {
                let q = p + vec2<i32>(round(dir * f32(s * j)));
                if any(q < vec2<i32>(0)) || any(q >= size) { break; }
                let o = tile_at(q);
                if o.w == 0.0 { continue; }
                let v = o.xyz - x;
                let distance = length(v);
                if distance > r || distance < 1e-6 { continue; }
                highest = max(highest, dot(v, n) / distance);
            }
            open += 1.0 - highest * highest;
        }
    }
    return open / f32(2u * NEAR_SLICES);
}

// The least V the near field takes: the search cannot resolve an opening
// narrower than a pixel, and a pit a pixel wide and a block deep leaves
// (1 / (2 side))^2 of the hemisphere open; below that the division would
// only amplify what the search cannot measure.
fn near_floor() -> f32 {
    let s = 2.0 * f32(side());
    return 1.0 / (s * s);
}

@compute @workgroup_size(8, 8)
fn gather_near(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    // The tile: four points a lane.
    let origin = vec2<i32>(group.xy * 8u);
    let size = vec2<i32>(internal_size());
    if li == 0u { near_origin = origin; }
    for (var k = 0u; k < 4u; k++) {
        let t = i32(li + 64u * k);
        let q = origin - NEAR_REACH + vec2<i32>(t % NEAR_TILE, t / NEAR_TILE);
        var point = vec4<f32>(0.0);
        if all(q >= vec2<i32>(0)) && all(q < size) { point = position_at(q); }
        near_tile[t] = point;
    }
    workgroupBarrier();
    if any(id.xy >= internal_size()) { return; }
    let p = vec2<i32>(id.xy);
    let at = tile_at(p);
    if at.w == 0.0 {
        textureStore(near_out, p, vec4<f32>(1.0, 1.0, 0.0, 0.0));
        return;
    }
    let floor_v = near_floor();
    if (gather.info.z & GATHER_TAGS) != 0u && gbuffer_own_occlusion(textureLoad(surface_tags, p, 0).xy) {
        // Its occlusion is its own: the material occlusion the lighting
        // multiplies in itself is its V.
        let ao = textureLoad(gbuf_orm, p, 0).r;
        let v = max(ao, floor_v);
        textureStore(near_out, p, vec4<f32>(v, v / max(ao, 1e-4), 0.0, 0.0));
        return;
    }
    var v = 1.0;
    let n = geometric_normal(p, at.xyz);
    if any(n != vec3<f32>(0.0)) {
        // A block's footprint at the pixel: what one ray a block misses.
        let r = f32(side()) * footprint_at(at.xyz);
        v = max(horizon(p, at.xyz, n, r, i32(side())), floor_v);
    }
    textureStore(near_out, p, vec4<f32>(v, v, 0.0, 0.0));
}

// ── Rays ────────────────────────────────────────────────────────────────────

// A texel's unit square sample on frame `frame`: Roberts' R2 (the plastic
// number's powers, a rank-1 lattice over frames) shifted by a hash of the
// texel, in 32-bit fixed point (exact for any frame number): each texel's
// directions over successive frames cover its hemisphere evenly, while
// neighbouring texels stay decorrelated.
fn texel_sample(texel: u32, frame: u32, stream: u32) -> vec2<f32> {
    let shift = vec2<u32>(gi_hash(texel ^ stream), gi_hash(texel ^ stream ^ 0x9e3779b9u));
    let at = shift + frame * vec2<u32>(3242174890u, 2447445414u);
    return vec2<f32>(at >> vec2<u32>(8u)) / 16777216.0;
}

// A unit direction cosine-distributed about `n` (the frame BSDF's frame).
fn cosine_about(n: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
    let frame = bsdf_frame(surface_data(vec3<f32>(1.0), n, 1.0, 0.0, vec3<f32>(0.0)));
    return normalize(bsdf_to_world(frame, bsdf_cosine_direction(u)));
}

// A ray leaving frame point `x` (normal `n`, a pixel `width` wide there)
// along `dir`: off the surface by `gi_ray_offset`, as wide as the pixel,
// spreading as one of the rays the denoiser's history accumulates
// (GI_RAY_SPREAD: sqrt(2 pi / 32), a 32-frame history's share of the
// hemisphere).
fn leaving(x: vec3<f32>, n: vec3<f32>, width: f32, dir: vec3<f32>) -> SceneRay {
    let cone = scene_cone_in_space(scene_ray_cone(width, GI_RAY_SPREAD, RAY_MASK_ALL), gather.info.w);
    return SceneRay(x + n * gi_ray_offset(x), GI_RAY_REACH, dir, cone);
}

fn no_ray() -> SceneRay {
    return SceneRay(vec3<f32>(0.0), 0.0, vec3<f32>(0.0, 1.0, 0.0), 0u);
}

// Whether texel `i`'s previous reservoir is re-traced this frame, and its
// row: one texel in GATHER period, rows after every texel's own.
fn retraced(i: u32) -> bool {
    let period = u32(gather.reuse.y);
    return (gather.info.z & GATHER_REUSE) != 0u && period > 0u && i % period == u32(gather.reuse.z);
}

fn retrace_row(i: u32) -> u32 {
    let traced = traced_size();
    return traced.x * traced.y + i / u32(gather.reuse.y);
}

@compute @workgroup_size(8, 8)
fn gather_rays(@builtin(global_invocation_id) id: vec3<u32>) {
    let traced = traced_size();
    if any(id.xy >= traced) { return; }
    let i = id.y * traced.x + id.x;
    let p = vec2<i32>(traced_pixel(id.xy));
    var ray = no_ray();
    let at = position_at(p);
    if at.w > 0.0 && !remapped(p) {
        let n = helio_gbuffer_normal(textureLoad(gbuf_normal, p, 0).xyz);
        let dir = cosine_about(n, texel_sample(i, gather.info.y, 0u));
        ray = leaving(at.xyz, n, footprint_at(at.xyz), dir);
    }
    rays[i] = ray;
    if retraced(i) {
        var check = no_ray();
        let r = previous[i];
        if (gather.info.z & GATHER_HAS_PREVIOUS) != 0u && (r.flags & RESERVOIR_VALID) != 0u {
            // Along the kept sample again, from where it was taken.
            let x = r.receiver + gather.rebase.xyz;
            let n = scene_unpack_unit(r.receiver_normal);
            let o = x + n * gi_ray_offset(x);
            var dir = r.sample;
            if (r.flags & RESERVOIR_SKY) == 0u {
                dir = normalize(r.sample + gather.rebase.xyz - o);
            }
            check = leaving(x, n, footprint_at(x), dir);
        }
        rays[retrace_row(i)] = check;
    }
}

// ── Shading ─────────────────────────────────────────────────────────────────

fn luminance(c: vec3<f32>) -> f32 {
    return bsdf_luminance(c);
}

// A ray's answer: the light back along it (rgb), and in w the hit distance
// (0: a miss, below 0: an inside); `point` and `normal` the hit's (the side
// the ray met).
struct Answer {
    light: vec3<f32>,
    t: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
}

fn answer(row: u32) -> Answer {
    let ray = rays_in[row];
    let hit = hits[row];
    let eye = gi_trace.eye.xyz;
    var a: Answer;
    if hit.t <= 0.0 {
        a.light = gi_sky_radiance((ray.origin - eye) * 0.001, ray.dir);
        a.normal = ray.dir;
        return a;
    }
    a.point = ray.origin + ray.dir * hit.t;
    if scene_hit_inside(hit) {
        // The ray started inside something: no light reaches the receiver
        // that way.
        a.t = -hit.t;
        a.normal = -ray.dir;
        return a;
    }
    a.t = hit.t;
    let surface = surfaces[row];
    let face = scene_surface_face(surface);
    a.normal = select(face, -face, dot(face, ray.dir) > 0.0);
    let footprint = scene_ray_footprint(ray, hit.t);
    let space = scene_ray_space(ray);
    a.light = gi_hit_light(a.point - eye, ray.dir, surface, footprint, space, false).rgb;
    return a;
}

// The geometry term of `r`'s sample seen from a receiver at `x` with normal
// `n`: for a point, the cosines at both ends over the distance squared (area
// measure); for the sky, the receiver's cosine (solid angle).
fn geometry(x: vec3<f32>, n: vec3<f32>, r: GatherReservoir) -> f32 {
    return point_geometry(x, n, sample_point(r));
}

// A sample unpacked: its point (or the sky's direction) and normal (zero
// for the sky).
struct SamplePoint {
    at: vec3<f32>,
    normal: vec3<f32>,
}

fn sample_point(r: GatherReservoir) -> SamplePoint {
    if (r.flags & RESERVOIR_SKY) != 0u { return SamplePoint(r.sample, vec3<f32>(0.0)); }
    return SamplePoint(r.sample, scene_unpack_unit(r.sample_normal));
}

// `geometry` of an unpacked sample.
fn point_geometry(x: vec3<f32>, n: vec3<f32>, y: SamplePoint) -> f32 {
    if all(y.normal == vec3<f32>(0.0)) { return max(dot(n, y.at), 0.0); }
    let v = y.at - x;
    let d2 = max(dot(v, v), 1e-12);
    let w = v * inverseSqrt(d2);
    let cos_x = dot(n, w);
    let cos_y = -dot(y.normal, w);
    if cos_x <= 0.0 || cos_y <= 0.0 { return 0.0; }
    return cos_x * cos_y / d2;
}

// The unit direction from `x` toward `r`'s sample.
fn toward(x: vec3<f32>, r: GatherReservoir) -> vec3<f32> {
    if (r.flags & RESERVOIR_SKY) != 0u { return r.sample; }
    return normalize(r.sample - x);
}

// `r` with its points moved into this frame's space.
fn rebased(r: GatherReservoir) -> GatherReservoir {
    var out = r;
    out.receiver += gather.rebase.xyz;
    if (r.flags & RESERVOIR_SKY) == 0u { out.sample += gather.rebase.xyz; }
    return out;
}

fn invalid_reservoir() -> GatherReservoir {
    return GatherReservoir(vec3<f32>(0.0), 0u, vec3<f32>(0.0), 0u, 0u, 0.0, 0.0, 0u);
}

// Whether a reservoir received at `q` (normal `nq`) serves a receiver at `x`
// (normal `n`, a pixel `footprint` wide there) whose samples are within
// `reach`: the denoiser's stops (`helio-pass-denoise`'s geometry weight), a
// normal within 41 degrees (a share of 1/8 of the cosine-weighted
// hemisphere unseen) and a point within 0.378 of the separation off the
// tangent plane (the chord of a surface turning by that much), plus a
// pixel's footprint.
fn same_surface(x: vec3<f32>, n: vec3<f32>, q: vec3<f32>, nq: vec3<f32>, footprint: f32, reach: f32) -> bool {
    let d = distance(x, q);
    return d <= reach && dot(n, nq) >= 0.75 && abs(dot(n, q - x)) <= 0.378 * d + footprint;
}

// How far from its receiver a reservoir's receiver may be and still be the
// same place's: a block's diagonal (the traced pixel anywhere in the
// previous block) and a pixel (the reprojection's rounding).
fn block_reach(footprint: f32) -> f32 {
    return (1.41421356 * f32(side()) + 1.0) * footprint;
}

// Texel `i`'s re-traced sample (one texel in GATHER period, `gather_rays`):
// its light now replaces the kept light; a sample whose light changed
// beyond the cache's own noise leaves a confidence below 1 (and fewer
// samples' worth in its reservoir), one whose ray now meets something else
// leaves none.
@compute @workgroup_size(8, 8)
fn gather_validate(@builtin(global_invocation_id) id: vec3<u32>) {
    let traced = traced_size();
    if any(id.xy >= traced) { return; }
    let i = id.y * traced.x + id.x;
    texels[i].y = 1.0;
    if !retraced(i) || gather_answer.x == 0u { return; }
    var r = previous[i];
    let row = retrace_row(i);
    if (gather.info.z & GATHER_HAS_PREVIOUS) == 0u || (r.flags & RESERVOIR_VALID) == 0u || rays_in[row].t_max <= 0.0 {
        return;
    }
    let ray = rays_in[row];
    let a = answer(row);
    var same = false;
    var noise = 0.0;
    if (r.flags & RESERVOIR_SKY) != 0u {
        same = a.t == 0.0;
    } else if a.t > 0.0 {
        // The same point: the ray left the same place along the same line,
        // so a tracer finds it at the same distance but for its own rounding
        // (2^-8 of the distance) and the start's offset.
        let expected = distance(r.sample + gather.rebase.xyz, ray.origin);
        same = abs(a.t - expected) <= 2.0 * gi_ray_offset(a.point) + expected / 256.0;
        noise = CACHE_NOISE;
    }
    var confidence = 0.0;
    if same {
        let before = luminance(scene_unpack_radiance(r.radiance));
        let now = luminance(a.light);
        let most = max(before, now);
        let change = select(0.0, abs(now - before) / most, most > 1e-20);
        confidence = select(1.0 - (change - noise) / (1.0 - noise), 1.0, change <= noise);
        r.radiance = scene_pack_radiance(a.light);
        r.m *= confidence;
    } else {
        r = invalid_reservoir();
    }
    previous[i] = r;
    texels[i].y = saturate(confidence);
}

// Texel `i`'s ray as a reservoir (one sample, M 1, W 1 / pdf: with the
// cosine density, pi over the geometry term in the sample's measure).
fn fresh_reservoir(i: u32, x: vec3<f32>, n: vec3<f32>) -> GatherReservoir {
    let a = answer(i);
    var r: GatherReservoir;
    r.receiver = x;
    r.receiver_normal = scene_pack_unit(n);
    r.radiance = scene_pack_radiance(a.light);
    r.m = 1.0;
    r.flags = RESERVOIR_VALID;
    if a.t == 0.0 {
        r.flags |= RESERVOIR_SKY;
        r.sample = a.normal;
    } else {
        r.sample = a.point;
        r.sample_normal = scene_pack_unit(a.normal);
    }
    let g = geometry(x, n, r);
    r.weight = select(0.0, PI / g, g > 0.0);
    return r;
}

// A random number in [0, 1) for texel `i`'s choice `k` this frame.
fn choice(i: u32, k: u32) -> f32 {
    return f32(gi_hash(gi_hash(i ^ (k * 0x85ebca6bu)) ^ gather.info.y * 0x27d4eb2fu) >> 8u) / 16777216.0;
}

@compute @workgroup_size(8, 8)
fn gather_temporal(@builtin(global_invocation_id) id: vec3<u32>) {
    let traced = traced_size();
    if any(id.xy >= traced) { return; }
    let i = id.y * traced.x + id.x;
    let p = traced_pixel(id.xy);
    let at = position_at(vec2<i32>(p));
    texels[i].x = GATHER_NO_HIT;
    texels[i].z = 1.0;
    if at.w == 0.0 || remapped(vec2<i32>(p)) {
        reservoirs[i] = invalid_reservoir();
        return;
    }
    let x = at.xyz;
    let n = helio_gbuffer_normal(textureLoad(gbuf_normal, vec2<i32>(p), 0).xyz);
    let answered = gather_answer.x != 0u;
    var f = invalid_reservoir();
    if answered {
        f = fresh_reservoir(i, x, n);
        let t = hits[i].t;
        texels[i].x = select(GATHER_NO_HIT, t, t > 0.0);
    }
    let reuse = (gather.info.z & GATHER_REUSE) != 0u;
    // Where the traced pixel was: the reservoir of the previous block its
    // surface's footprint held most of.
    var prev = invalid_reservoir();
    if reuse && (gather.info.z & GATHER_HAS_PREVIOUS) != 0u {
        let hp = history_pixel(textureLoad(history_pixels, vec2<i32>(p), 0), p);
        let taps = history_taps(hp);
        var best = 0u;
        for (var k = 1u; k < 4u; k++) {
            if taps.weights[k] > taps.weights[best] { best = k; }
        }
        if taps.weights[best] > 0.0 {
            let q = clamp(history_tap(taps, best), vec2<i32>(0), vec2<i32>(internal_size()) - 1);
            let b = vec2<u32>(q) / side();
            if all(b < traced) {
                let j = b.y * traced.x + b.x;
                let r = rebased(previous[j]);
                let footprint = footprint_at(x);
                if (r.flags & RESERVOIR_VALID) != 0u && r.m > 0.0
                    && same_surface(x, n, r.receiver, scene_unpack_unit(r.receiver_normal), footprint, block_reach(footprint)) {
                    prev = r;
                    prev.m = min(prev.m, gather.reuse.x);
                    texels[i].z = texels[j].y;
                }
            }
        }
    }
    if !answered && (prev.flags & RESERVOIR_VALID) == 0u {
        reservoirs[i] = invalid_reservoir();
        return;
    }
    // Generalised RIS over the two domains (this frame's sample at this
    // receiver, M 1; the previous reservoir at its own receiver, M its own),
    // each sample weighted by the balance heuristic over both receivers'
    // target functions (no visibility between them: the receivers are one
    // surface within a block).
    let xp = prev.receiver;
    let np = scene_unpack_unit(prev.receiver_normal);
    let mf = select(0.0, 1.0, answered);
    var sum = 0.0;
    var chosen = f;
    var aim = 0.0;
    if answered {
        let y = luminance(scene_unpack_radiance(f.radiance));
        let pc = y * geometry(x, n, f);
        let pp = select(0.0, y * geometry(xp, np, f), prev.m > 0.0);
        let w = select(0.0, pc / (pc + prev.m * pp), pc > 0.0) * pc * f.weight;
        sum += w;
        aim = pc;
    }
    if (prev.flags & RESERVOIR_VALID) != 0u {
        let y = luminance(scene_unpack_radiance(prev.radiance));
        let pc = y * geometry(x, n, prev);
        let pp = y * geometry(xp, np, prev);
        let denominator = mf * pc + prev.m * pp;
        let w = select(0.0, prev.m * pp / denominator, denominator > 0.0) * pc * prev.weight;
        sum += w;
        if choice(i, 0u) * sum < w {
            chosen = prev;
            aim = pc;
        }
    }
    chosen.receiver = x;
    chosen.receiver_normal = scene_pack_unit(n);
    chosen.weight = select(0.0, sum / aim, aim > 0.0);
    chosen.m = min(mf + prev.m, gather.reuse.x);
    chosen.flags |= RESERVOIR_VALID;
    reservoirs[i] = chosen;
}

// The texel `b + offset`'s spatial neighbour `k` this frame: points of a
// disc of `gather.reuse.w` texels, turned each frame.
fn neighbour(i: u32, b: vec2<u32>, k: u32) -> vec2<i32> {
    let turn = choice(i, 7u);
    let angle = (f32(k) + turn) * (2.0 * PI / f32(GATHER_NEIGHBOURS));
    let radius = gather.reuse.w * sqrt((f32(k) + choice(i, k + 1u)) / f32(GATHER_NEIGHBOURS));
    return vec2<i32>(b) + vec2<i32>(round(vec2<f32>(cos(angle), sin(angle)) * max(radius, 1.0)));
}

@compute @workgroup_size(8, 8)
fn gather_spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    let traced = traced_size();
    if any(id.xy >= traced) { return; }
    let i = id.y * traced.x + id.x;
    let b = vec2<i32>(id.xy);
    let own = reservoirs[i];
    let t = texels[i];
    if (own.flags & RESERVOIR_VALID) == 0u {
        textureStore(signal0, b, vec4<f32>(0.0, 0.0, 0.0, GATHER_NO_HIT));
        textureStore(signal1, b, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let x = own.receiver;
    let n = scene_unpack_unit(own.receiver_normal);
    let footprint = footprint_at(x);
    // The domains: this texel's temporal reservoir, then its neighbours'
    // that are the same surface within the radius; their receivers' normals
    // and their samples' unpacked once.
    var domains: array<GatherReservoir, 5>;
    var normals: array<vec3<f32>, 5>;
    var count = 1u;
    domains[0] = own;
    normals[0] = n;
    if (gather.info.z & GATHER_REUSE) != 0u {
        let reach = (gather.reuse.w * f32(side()) * 1.41421356 + 1.0) * footprint;
        for (var k = 0u; k < GATHER_NEIGHBOURS; k++) {
            let q = neighbour(i, id.xy, k);
            if any(q < vec2<i32>(0)) || any(q >= vec2<i32>(traced)) || all(q == b) { continue; }
            let r = reservoirs[u32(q.y) * traced.x + u32(q.x)];
            if (r.flags & RESERVOIR_VALID) == 0u || r.m <= 0.0 { continue; }
            let nq = scene_unpack_unit(r.receiver_normal);
            if !same_surface(x, n, r.receiver, nq, footprint, reach) { continue; }
            domains[count] = r;
            normals[count] = nq;
            count += 1u;
        }
    }
    // Generalised RIS over the domains, each sample weighted by the balance
    // heuristic over every domain's target function at its own receiver
    // (luminance cancels in it: the geometry terms alone); no visibility
    // ray between a neighbour's sample and this receiver (rendering.md:
    // its bias measured against the reference).
    var sum = 0.0;
    var chosen = 0u;
    var aim = 0.0;
    for (var k = 0u; k < count; k++) {
        let r = domains[k];
        let y = sample_point(r);
        var own_g = 0.0;
        var all_g = 0.0;
        for (var j = 0u; j < count; j++) {
            let g = point_geometry(domains[j].receiver, normals[j], y);
            all_g += domains[j].m * g;
            if j == k { own_g = g; }
        }
        let pc = luminance(scene_unpack_radiance(r.radiance)) * point_geometry(x, n, y);
        let w = select(0.0, r.m * own_g / all_g, all_g > 0.0) * pc * r.weight;
        sum += w;
        if choice(i, 11u + k) * sum < w {
            chosen = k;
            aim = pc;
        }
    }
    let r = domains[chosen];
    let weight = select(0.0, sum / aim, aim > 0.0);
    // The irradiance estimate: the sample's light times its geometry term
    // and weight, divided by the near field at the traced pixel.
    let e = scene_unpack_radiance(r.radiance) * geometry(x, n, r) * weight;
    let p = traced_pixel(id.xy);
    let v = max(textureLoad(near_field, vec2<i32>(p), 0).x, 1e-4);
    // The hit distance the denoiser holds a record over: this frame's ray's,
    // no nearer than a block's footprint (what is nearer is the near
    // field's).
    let hit = max(t.x, f32(side()) * footprint);
    var input = DenoiseIrradianceInput(vec4<f32>(0.0, 0.0, 0.0, hit), vec4<f32>(0.0, 0.0, 0.0, saturate(t.z)));
    input = denoise_irradiance_add(input, e / v, toward(x, r));
    textureStore(signal0, b, input.signal0);
    textureStore(signal1, b, input.signal1);
}
