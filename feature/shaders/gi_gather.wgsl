//!use helio_prelude
//!use frame_sky
//!use sky_radiance
//!use frame_ray_stream
//!use frame_ray_surface
//!use frame_surface
//!use frame_bsdf
//!use frame_light_samples
//!use frame_radiance_cache
//!use frame_history
//!use frame_surface_stops
//!use frame_surface_seen
//!use frame_denoise_irradiance
//!use gi_common
//!use gi_sample
//!use gi_rays
//!use gi_hit
// Indirect diffuse at a pixel (README.md, "The pixels' gather"): one ray a
// block of internal pixels, its hit lit as GI lights a hit (`gi_hit`), kept
// in a reservoir a traced texel and reused across frames and neighbours
// (ReSTIR GI, Ouyang et al. 2021), into the frame's denoiser's irradiance
// signal, demodulated by the near field.
//
// - `gather_near` (a pixel): the near field. Occlusion nearer than a block's
//   footprint is not in one ray a block; a horizon search over that
//   footprint and no farther gives it at every pixel (V), and the signal is
//   divided by V at its traced pixel and multiplied by the same V at each
//   pixel (`gi_resolve_pixels`), so the rays' estimate is exact where it was
//   traced and the near field moves it between the pixels of a block. A
//   surface whose occlusion is its own (`gbuffer_own_occlusion`: a traced
//   terrain's short sky rays, a meadow's) brings its V in its material
//   occlusion, which the lighting multiplies in itself.
// - `gather_rays` (a texel): its traced pixel's ray (cosine about the
//   pixel's normal) into the stream `gi_pixels`, and for one reservoir in
//   GATHER period, a ray re-tracing its kept sample.
// - `gather_hits` (a texel): its ray's hit lit as GI lights a hit
//   (`gi_hit_light`, read at the ray's footprint there: the cell's
//   prefiltered indirect light; a side the cache has no cell for, the
//   probes'), its direct light a light query of its own (`gi_pixels` light
//   batch), exact at the hit: a cell's direct light is its point's, which
//   near a light differs across the cell far more than its indirect does.
//   Its mark claims the cell: the gather is the cache's one claimer.
// - `gather_shadow` (a query): its light sample's light at the hit, waiting
//   on a shadow ray (`gi_pixels_shadow`).
// - `gather_validate` (a texel): a re-traced sample's light now; a sample
//   whose light changed lowers the confidence written for the denoiser.
// - `gather_temporal` (a texel): this frame's sample combined with the
//   reservoir where the traced pixel was last frame (`frame_history`).
// - `gather_spatial` (a texel): that combined with neighbours' (not kept:
//   only the temporal reservoirs carry over); a visibility ray from the
//   receiver to the sample it chose, where that sample came from another
//   receiver (`gi_pixels_visibility`).
// - `gather_signal` (a texel): the chosen sample's estimate where the
//   receiver sees it (this frame's own sample where it does not), into the
//   signal.

// The frame (`GatherFrame`, gather.rs).
struct GatherFrame {
    // x, y: the internal size; z, w: the traced size (a texel a block).
    size: vec4<u32>,
    // x: the block's side; y: the frame (the graph's frame number, truncated:
    // the denoiser's); z: GATHER_* flags; w: the camera's coordinate space
    // (`SceneRay::space`).
    info: vec4<u32>,
    // x: the most samples' worth (M) a reservoir keeps; y: one reservoir in
    // this many is re-traced a frame; z: this frame's re-traced phase; w:
    // the spatial neighbours' radius in texels.
    reuse: vec4<f32>,
    // xyz: the camera's forward axis (view depth is along it); w: a pixel's
    // width a metre of view depth.
    forward: vec4<f32>,
    // xyz: this frame's eye (frame space; reservoirs hold their receivers
    // from it).
    eye: vec4<f32>,
    // xyz: the previous frame's eye in this frame's space (moved by the
    // origin's change).
    previous_eye: vec4<f32>,
}

const GATHER_HAS_PREVIOUS: u32 = 1u;
const GATHER_REUSE: u32 = 2u;
const GATHER_TAGS: u32 = 4u;

// A traced texel's reservoir, 36 bytes (`RESERVOIR_BYTES`): the receiver
// (the traced pixel's surface) from the eye at the frame's own precision
// (zero: none; `res_unpack` says why), its normal (8:8) and the luminance of the
// cache's light at the sample when it was taken (a half: what a re-trace
// compares); the kept sample as a direction from the receiver (16:16) and a
// distance (below 0: the sky), its normal (12:12, the side the ray met) and
// how many samples' worth the reservoir holds (M, 8 bits in 16ths); the light it
// sends to the receiver (`scene_pack_radiance`) and its unbiased
// contribution weight W (area measure for a point, solid angle for the sky).
struct GatherReservoir {
    receiver_x: f32,
    receiver_y: f32,
    receiver_z: f32,
    receiver_normal_y: u32,
    sample_dir: u32,
    sample_t: f32,
    sample_normal_m: u32,
    radiance: u32,
    weight: f32,
}

// A reservoir unpacked.
struct Res {
    valid: bool,
    sky: bool,
    receiver: vec3<f32>,
    normal: vec3<f32>,
    // The sample's point, or the sky's direction.
    sample: vec3<f32>,
    sample_normal: vec3<f32>,
    m: f32,
    cache_y: f32,
    radiance: u32,
    weight: f32,
}

// A texel's light at its ray's hit (`gather_hits`, `gather_shadow`): the
// light it sends back but for its direct light where it has its own
// (`rest`, packed), that direct light waiting on a shadow ray (`light`,
// packed; `row` the shadow stream's row, GATHER_NONE: none; between the two
// passes `light` holds the sides that light themselves, `GiHitLight::own`,
// and `row` the light query's), the luminance of the cache's whole light at
// the hit (`cache_y`, what a re-trace compares).
struct GatherHitLight {
    rest: u32,
    light: u32,
    row: u32,
    cache_y: f32,
}

const GATHER_NONE: u32 = 0xffffffffu;

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
// the confidence of what its temporal reservoir took, w unused.
@group(0) @binding(10) var<storage, read_write> texels: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> hits: array<SceneHit>;
@group(0) @binding(12) var<storage, read> surfaces: array<SceneSurface>;
@group(0) @binding(13) var near_field: texture_2d<f32>;
@group(0) @binding(14) var history_pixels: texture_2d<f32>;
@group(0) @binding(15) var signal0: texture_storage_2d<rgba16float, write>;
@group(0) @binding(16) var signal1: texture_storage_2d<rgba16float, write>;
// Whether every tracer answered in full this frame: x the stream
// `gi_pixels`, y its shadow rays, z its visibility rays.
@group(0) @binding(17) var<uniform> gather_answer: vec4<u32>;
@group(0) @binding(18) var<uniform> frame_gi: GiFrame;
@group(0) @binding(19) var<storage, read> gi_probes: array<u32>;
@group(0) @binding(20) var<uniform> gi_trace: GiTrace;
@group(0) @binding(21) var<uniform> sky_frame: SkyFrame;
@group(0) @binding(22) var sky_radiance: texture_3d<f32>;
@group(0) @binding(39) var sky_radiance_sampler: sampler;
@group(0) @binding(23) var<uniform> radiance_cache_frame: RadianceCacheFrame;
@group(0) @binding(24) var<storage, read_write> radiance_cache_slots: array<RadianceCacheSlot>;
@group(0) @binding(25) var<storage, read_write> radiance_cache_entries: array<RadianceCacheEntry>;
@group(0) @binding(26) var<storage, read> rays_in: array<SceneRay>;
// The light batch `gi_pixels`: its count (from 0), its queries, the
// samples the samplers left.
@group(0) @binding(27) var<storage, read_write> light_header: array<atomic<u32>, 4>;
@group(0) @binding(28) var<storage, read_write> light_queries: array<LightQuery>;
@group(0) @binding(29) var<storage, read> light_samples: array<LightSample>;
@group(0) @binding(30) var<storage, read_write> hit_light: array<GatherHitLight>;
// The shadow stream `gi_pixels_shadow`: rows from 0, the count in its
// header; read back after its tracers (34).
@group(0) @binding(31) var<storage, read_write> shadow_rays: array<SceneRay>;
@group(0) @binding(32) var<storage, read_write> shadow_hits: array<SceneHit>;
@group(0) @binding(33) var<storage, read_write> shadow_header: array<atomic<u32>, 4>;
@group(0) @binding(34) var<storage, read> shadow_answers: array<SceneHit>;
// A texel's chosen sample (`gather_spatial`): its irradiance estimate (rgb)
// and its direction (w, a packed unit).
@group(0) @binding(35) var<storage, read_write> picks: array<vec4<f32>>;
// The visibility stream `gi_pixels_visibility`: a texel's ray at its index
// (no ray where its chosen sample is its own), read back after its tracers
// (37).
@group(0) @binding(36) var<storage, read_write> visibility_rays: array<SceneRay>;
@group(0) @binding(37) var<storage, read> visibility_answers: array<SceneHit>;
// The frame's statistics: the shadow stream's count (`rays.gi_pixels_shadow`).
@group(0) @binding(38) var<storage, read_write> stats: array<atomic<u32>>;
override GATHER_SHADOW_STAT: u32 = 0u;
// The hits' states (`gi.hits.pixels.*`: GI_HIT_STATES).
override PIXEL_HIT_STAT_0: u32 = 0u;
override PIXEL_HIT_STAT_1: u32 = 0u;
override PIXEL_HIT_STAT_2: u32 = 0u;
override PIXEL_HIT_STAT_3: u32 = 0u;
override PIXEL_HIT_STAT_4: u32 = 0u;
override PIXEL_HIT_STAT_5: u32 = 0u;
override PIXEL_HIT_STAT_6: u32 = 0u;
override PIXEL_HIT_STAT_7: u32 = 0u;
override PIXEL_HIT_STAT_8: u32 = 0u;

fn pixel_hit_stat(k: u32) -> u32 {
    switch k {
        case 0u: { return PIXEL_HIT_STAT_0; }
        case 1u: { return PIXEL_HIT_STAT_1; }
        case 2u: { return PIXEL_HIT_STAT_2; }
        case 3u: { return PIXEL_HIT_STAT_3; }
        case 4u: { return PIXEL_HIT_STAT_4; }
        case 5u: { return PIXEL_HIT_STAT_5; }
        case 6u: { return PIXEL_HIT_STAT_6; }
        case 7u: { return PIXEL_HIT_STAT_7; }
        default: { return PIXEL_HIT_STAT_8; }
    }
}

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
    return gather.forward.w * max(dot(x - gather.eye.xyz, gather.forward.xyz), 0.0);
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

// ── Reservoirs ──────────────────────────────────────────────────────────────

// A unit vector in 16 bits (8:8): the top 8 bits an axis of
// `scene_pack_unit`'s, rounded.
fn pack_unit8(n: vec3<f32>) -> u32 {
    let p = scene_pack_unit(n);
    let x = min(((p & 0xffffu) + 128u) >> 8u, 255u);
    let y = min(((p >> 16u) + 128u) >> 8u, 255u);
    return x | (y << 8u);
}

// A unit vector in 24 bits (12:12, the low 24 bits of the word): the top 12
// bits an axis of `scene_pack_unit`'s, rounded.
fn pack_unit12(n: vec3<f32>) -> u32 {
    let p = scene_pack_unit(n);
    let x = min(((p & 0xffffu) + 8u) >> 4u, 4095u);
    let y = min(((p >> 16u) + 8u) >> 4u, 4095u);
    return x | (y << 12u);
}

fn unpack_unit12(p: u32) -> vec3<f32> {
    let x = p & 0xfffu;
    let y = (p >> 12u) & 0xfffu;
    return scene_unpack_unit(((x << 4u) | (x >> 8u)) | (((y << 4u) | (y >> 8u)) << 16u));
}

// M in 8 bits (16ths, up to 15.94: `M_CAP` is 8).
fn m_bits(m: f32) -> u32 {
    return u32(clamp(round(m * 16.0), 0.0, 255.0));
}

fn unpack_unit8(p: u32) -> vec3<f32> {
    return scene_unpack_unit(((p & 0xffu) * 257u) | ((((p >> 8u) & 0xffu) * 257u) << 16u));
}

fn half_bits(v: f32) -> u32 {
    return pack2x16float(vec2<f32>(v, 0.0)) & 0xffffu;
}

fn half_value(b: u32) -> f32 {
    return unpack2x16float(b & 0xffffu).x;
}

// `r` unpacked, its receiver held from eye `eye`. The receiver is kept as
// the frame holds it (f32, from the eye), not as a direction and distance:
// a sample's weight is taken at its receiver's exact point, and for a hit a
// few of GI's ray offsets away (`gi_ray_offset`: 2 mm and 4e-6 of the
// distance; a voxel step's riser, a corner) any receiver off by more than a
// share of that offset changes the sample's geometry term there by orders
// of magnitude. A direction at 16 bits an axis is up to 3e-5 of the
// distance off, past the offset from some tens of metres: the canonical
// estimate, the history's and a re-trace from the receiver went wrong there
// (bright discs on voxel ground).
fn res_unpack(r: GatherReservoir, eye: vec3<f32>) -> Res {
    var x: Res;
    let v = vec3<f32>(r.receiver_x, r.receiver_y, r.receiver_z);
    // A surface point is never the eye itself (the near plane).
    x.valid = any(v != vec3<f32>(0.0));
    if !x.valid { return x; }
    x.receiver = eye + v;
    x.normal = unpack_unit8(r.receiver_normal_y);
    x.cache_y = half_value(r.receiver_normal_y >> 16u);
    let d = scene_unpack_unit(r.sample_dir);
    x.sky = r.sample_t < 0.0;
    x.sample = select(x.receiver + d * r.sample_t, d, x.sky);
    x.sample_normal = select(unpack_unit12(r.sample_normal_m), vec3<f32>(0.0), x.sky);
    x.m = f32(r.sample_normal_m >> 24u) / 16.0;
    x.radiance = r.radiance;
    x.weight = r.weight;
    return x;
}

// `x` packed, its receiver held from eye `eye`.
fn res_pack(x: Res, eye: vec3<f32>) -> GatherReservoir {
    var r: GatherReservoir;
    if !x.valid { return r; }
    let v = x.receiver - eye;
    r.receiver_x = v.x;
    r.receiver_y = v.y;
    r.receiver_z = v.z;
    r.receiver_normal_y = pack_unit8(x.normal) | (half_bits(x.cache_y) << 16u);
    if x.sky {
        r.sample_dir = scene_pack_unit(x.sample);
        r.sample_t = -1.0;
    } else {
        let s = x.sample - x.receiver;
        let t = length(s);
        r.sample_dir = scene_pack_unit(s / max(t, 1e-30));
        r.sample_t = t;
    }
    r.sample_normal_m = select(pack_unit12(x.sample_normal), 0u, x.sky) | (m_bits(x.m) << 24u);
    r.radiance = x.radiance;
    r.weight = x.weight;
    return r;
}

fn no_res() -> Res {
    var x: Res;
    return x;
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

// A workgroup's pixels' V (below 0: no surface), for their blocks' means
// (a block's side, 1, 2 or 4, divides the workgroup's 8).
var<workgroup> near_v: array<f32, 64>;

@compute @workgroup_size(8, 8)
fn gather_near(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(local_invocation_id) local: vec3<u32>) {
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
    let p = vec2<i32>(id.xy);
    let inside = all(id.xy < internal_size());
    var at = vec4<f32>(0.0);
    if inside { at = tile_at(p); }
    // This pixel's V, and what the lighting multiplies in itself (the
    // material occlusion where the occlusion is the surface's own: a traced
    // terrain's short sky rays, a meadow's).
    var v = -1.0;
    var own = 1.0;
    if at.w > 0.0 {
        v = 1.0;
        if (gather.info.z & GATHER_TAGS) != 0u && gbuffer_own_occlusion(textureLoad(surface_tags, p, 0).xy) {
            own = textureLoad(gbuf_orm, p, 0).r;
            v = own;
        } else {
            let n = geometric_normal(p, at.xyz);
            if any(n != vec3<f32>(0.0)) {
                // A block's footprint at the pixel: what one ray a block
                // misses.
                let r = f32(side()) * footprint_at(at.xyz);
                v = horizon(p, at.xyz, n, r, i32(side()));
            }
        }
    }
    near_v[li] = v;
    workgroupBarrier();
    if !inside { return; }
    if at.w == 0.0 {
        textureStore(near_out, p, vec4<f32>(1.0, 1.0, 0.0, 0.0));
        return;
    }
    // The block's mean V over its pixels that hold a surface: the rays'
    // estimate (its traced pixel any of them, `denoise_traced_pixel`) is the
    // block's mean light, and each pixel takes its V's share of it, so the
    // near field moves light between a block's pixels and never changes
    // their mean (a ray that meets a near occluder sends its light, which a
    // division by the traced pixel's V would count again).
    let b = (local.xy / side()) * side();
    var sum = 0.0;
    var count = 0.0;
    for (var y = 0u; y < side(); y++) {
        for (var x = 0u; x < side(); x++) {
            let w = near_v[(b.y + y) * 8u + b.x + x];
            if w >= 0.0 {
                sum += w;
                count += 1.0;
            }
        }
    }
    let mean = sum / max(count, 1.0);
    let share = select(1.0, v / mean, mean > 0.0);
    textureStore(near_out, p, vec4<f32>(mean, share / max(own, 1e-4), 0.0, 0.0));
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

// A ray leaving frame point `x` (normal `n`) along `dir` for the block of
// pixels it stands for: off the surface by `gi_ray_offset`, as wide as the
// block's footprint there (`block_width`), spreading as one of the rays the
// denoiser's history accumulates (GI_RAY_SPREAD: sqrt(2 pi / 32), a 32-frame
// history's share of the hemisphere). Its footprint at a hit,
// `width + spread t`, is the patch the estimate stands for: the cache's cell
// it reads and the terrain's level a tracer may stop at.
fn block_width(x: vec3<f32>) -> f32 {
    return f32(side()) * footprint_at(x);
}

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
        ray = leaving(at.xyz, n, block_width(at.xyz), dir);
    }
    rays[i] = ray;
    if retraced(i) {
        var check = no_ray();
        let r = res_unpack(previous[i], gather.previous_eye.xyz);
        if (gather.info.z & GATHER_HAS_PREVIOUS) != 0u && r.valid {
            // Along the kept sample again, from where it was taken.
            let o = r.receiver + r.normal * gi_ray_offset(r.receiver);
            check = leaving(r.receiver, r.normal, block_width(r.receiver), toward(o, r));
        }
        rays[retrace_row(i)] = check;
    }
}

// ── Shading ─────────────────────────────────────────────────────────────────

fn luminance(c: vec3<f32>) -> f32 {
    return bsdf_luminance(c);
}

// A sample unpacked: its point (or the sky's direction) and normal (zero
// for the sky).
struct SamplePoint {
    at: vec3<f32>,
    normal: vec3<f32>,
}

fn sample_point(r: Res) -> SamplePoint {
    return SamplePoint(r.sample, select(r.sample_normal, vec3<f32>(0.0), r.sky));
}

// The geometry term of a sample `y` seen from a receiver at `x` with normal
// `n`: for a point, the cosines at both ends over the distance squared (area
// measure); for the sky, the receiver's cosine (solid angle).
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

fn geometry(x: vec3<f32>, n: vec3<f32>, r: Res) -> f32 {
    return point_geometry(x, n, sample_point(r));
}

// The unit direction from `x` toward `r`'s sample.
fn toward(x: vec3<f32>, r: Res) -> vec3<f32> {
    if r.sky { return r.sample; }
    return normalize(r.sample - x);
}

// Whether a reservoir received at `q` (normal `nq`) serves a receiver at `x`
// (normal `n`, a pixel `footprint` wide there) whose samples are within
// `reach`: the frame's surface stops, the denoiser's (`surface_stops_weight`).
fn same_surface(x: vec3<f32>, n: vec3<f32>, q: vec3<f32>, nq: vec3<f32>, footprint: f32, reach: f32) -> bool {
    return distance(x, q) <= reach && surface_stops_weight(x, n, q, nq, footprint) > 0.0;
}

// The internal pixel frame point `x` projects to this frame.
fn screen_pixel(x: vec3<f32>) -> vec2<i32> {
    let clip = cameras[0].view_proj * vec4<f32>(x, 1.0);
    let uv = helio_ndc_to_uv(clip.xy / clip.w);
    return clamp(vec2<i32>(floor(uv * vec2<f32>(internal_size()))), vec2<i32>(0), vec2<i32>(internal_size()) - 1);
}

// The frame point at internal pixel `q` and a pixel's footprint there (0:
// the sky), as the surface stops' walk reads the depth buffer
// (`surface_stops_seen`: a neighbour on the same plane across an occluder, a
// floor's two sides of a wall, is not taken; its samples are what the
// occluder hides from this receiver, which the resampling's weights,
// geometry terms with no visibility, do not see).
fn surface_stops_point(q: vec2<i32>) -> vec4<f32> {
    let o = position_at(q);
    if o.w == 0.0 { return vec4<f32>(0.0); }
    return vec4<f32>(o.xyz, footprint_at(o.xyz));
}

// How far from its receiver a reservoir's receiver may be and still be the
// same place's: a block's diagonal (the traced pixel anywhere in the
// previous block) and a pixel (the reprojection's rounding).
fn block_reach(footprint: f32) -> f32 {
    return (1.41421356 * f32(side()) + 1.0) * footprint;
}

// The luminance of the cache's whole light at a hit (the front side's cell,
// read without claiming it, as a reader in the air: a light that changed
// shows wherever the cell holds), what a re-trace compares; 0 where none.
fn cache_luminance(point: vec3<f32>, side: vec3<f32>, footprint: f32, surface: SceneSurface, space: u32) -> f32 {
    let c = radiance_cache_read(point, side, space, footprint);
    if !c.found { return 0.0; }
    let s = scene_surface_unpack(surface);
    return luminance(s.albedo.rgb * (c.direct + c.indirect) / PI + s.emissive);
}

// What a ray of row `row` met: the hit's point, the normal of the side the
// ray met and its distance (0: a miss, below 0: an inside).
struct Met {
    t: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
}

fn met(row: u32) -> Met {
    let ray = rays_in[row];
    let hit = hits[row];
    var m: Met;
    if hit.t <= 0.0 {
        m.normal = ray.dir;
        return m;
    }
    m.point = ray.origin + ray.dir * hit.t;
    if scene_hit_inside(hit) {
        m.t = -hit.t;
        m.normal = -ray.dir;
        return m;
    }
    m.t = hit.t;
    let face = scene_surface_face(surfaces[row]);
    m.normal = select(face, -face, dot(face, ray.dir) > 0.0);
    return m;
}

var<workgroup> w_rows: atomic<u32>;
var<workgroup> w_base: u32;
var<workgroup> w_states: array<atomic<u32>, GI_HIT_STATES>;

// A workgroup of 64 texels: each one's hit's light (module docs), its light
// queries in one block of the batch.
@compute @workgroup_size(64)
fn gather_hits(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if li == 0u { atomicStore(&w_rows, 0u); }
    if li < GI_HIT_STATES { atomicStore(&w_states[li], 0u); }
    workgroupBarrier();
    let traced = traced_size();
    let n = traced.x * traced.y;
    let i = id.x;
    var light = GatherHitLight(0u, 0u, GATHER_NONE, 0.0);
    var query: LightQuery;
    var want = false;
    if i < n && rays_in[i].t_max > 0.0 && gather_answer.x != 0u {
        let ray = rays_in[i];
        let m = met(i);
        let eye = gi_trace.eye.xyz;
        if m.t == 0.0 {
            let sky = gi_sky_radiance((ray.origin - eye) * 0.001, ray.dir);
            light.rest = scene_pack_radiance(sky);
            light.cache_y = luminance(sky);
        } else if m.t > 0.0 {
            let surface = surfaces[i];
            let footprint = scene_ray_footprint(ray, m.t);
            let space = scene_ray_space(ray);
            let lit = gi_hit_light(m.point - eye, ray.dir, footprint, surface, space, true, false, true);
            atomicAdd(&w_states[lit.miss], 1u);
            if lit.finer { atomicAdd(&w_states[GI_HIT_FINER], 1u); }
            if lit.claimed { atomicAdd(&w_states[select(GI_HIT_CLAIMED, GI_HIT_CLAIMED_FINER, lit.finer)], 1u); }
            light.rest = scene_pack_radiance(lit.reflected + lit.emitted);
            light.cache_y = cache_luminance(m.point, m.normal, footprint, surface, space);
            // Its direct light: a light sample of its own, two-sided where
            // light passes through it.
            let s = scene_surface_unpack(surface);
            let seed = gi_hash(i ^ gather.info.y * 0x9e3779b9u);
            let u = texel_sample(i, gather.info.y, 1u);
            let both = any(surface_transmission(s) > vec3<f32>(0.0));
            query = light_query_point(m.point, footprint, s.normal, both, u, seed, space);
            light.light = GI_SIDE_NEAR | GI_SIDE_FAR;
            want = true;
        }
    }
    var slot = 0u;
    if want { slot = atomicAdd(&w_rows, 1u); }
    workgroupBarrier();
    if li == 0u {
        let here = atomicLoad(&w_rows);
        w_base = select(0u, atomicAdd(&light_header[3], here), here > 0u);
    }
    if li < GI_HIT_STATES {
        let k = atomicLoad(&w_states[li]);
        if k > 0u { atomicAdd(&stats[pixel_hit_stat(li)], k); }
    }
    workgroupBarrier();
    if want {
        light.row = w_base + slot;
        light_queries[light.row] = query;
    }
    if i < n { hit_light[i] = light; }
}

// The light batch's dispatch for the samplers, from its count.
@compute @workgroup_size(1)
fn gather_hits_header() {
    let count = atomicLoad(&light_header[3]);
    atomicStore(&light_header[0], (count + RAY_STREAM_GROUP - 1u) / RAY_STREAM_GROUP);
    atomicStore(&light_header[1], 1u);
    atomicStore(&light_header[2], 1u);
}

// A workgroup of 64 texels: each one's light sample's light at its hit,
// waiting on a shadow ray from the side the light is on, the shadow rays in
// one block of the stream.
@compute @workgroup_size(64)
fn gather_shadow(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if li == 0u { atomicStore(&w_rows, 0u); }
    workgroupBarrier();
    let traced = traced_size();
    let i = id.x;
    var light = GatherHitLight(0u, 0u, GATHER_NONE, 0.0);
    var shadow: SceneRay;
    var want = false;
    if i < traced.x * traced.y {
        light = hit_light[i];
        if light.row != GATHER_NONE {
            let sample = light_samples[light.row];
            light.row = GATHER_NONE;
            let direct = gi_hit_direct(surfaces[i], sample, light.light);
            light.light = 0u;
            if direct.lit {
                let ray = rays_in[i];
                let t = hits[i].t;
                let p = ray.origin + ray.dir * t;
                shadow = gi_shadow_ray(p + direct.side * gi_ray_offset(p), sample, scene_ray_footprint(ray, t), scene_ray_space(ray));
                light.light = scene_pack_radiance(direct.light);
                want = true;
            }
        }
    }
    var slot = 0u;
    if want { slot = atomicAdd(&w_rows, 1u); }
    workgroupBarrier();
    if li == 0u {
        let here = atomicLoad(&w_rows);
        w_base = select(0u, atomicAdd(&shadow_header[3], here), here > 0u);
    }
    workgroupBarrier();
    if want {
        light.row = w_base + slot;
        shadow_rays[light.row] = shadow;
        shadow_hits[light.row] = SceneHit(0.0, 0u, 0u, 0u);
    }
    if i < traced.x * traced.y { hit_light[i] = light; }
}

// The shadow stream's dispatch for the tracers, from its count.
@compute @workgroup_size(1)
fn gather_shadow_header() {
    let count = atomicLoad(&shadow_header[3]);
    atomicStore(&shadow_header[0], (count + RAY_STREAM_GROUP - 1u) / RAY_STREAM_GROUP);
    atomicStore(&shadow_header[1], 1u);
    atomicStore(&shadow_header[2], 1u);
    atomicAdd(&stats[GATHER_SHADOW_STAT], count);
}

// Texel `i`'s ray's light: the rest and its direct light through its shadow
// ray (none on a frame the shadow rays were answered short of: the sample is
// not taken).
fn hit_radiance(i: u32) -> vec3<f32> {
    let l = hit_light[i];
    var c = scene_unpack_radiance(l.rest);
    if l.row != GATHER_NONE {
        c += scene_unpack_radiance(l.light) * scene_occlusion_through(shadow_answers[l.row]);
    }
    return c;
}

// Texel `i`'s re-traced sample (one texel in GATHER period, `gather_rays`):
// a sample whose cache light changed beyond the cache's own noise leaves a
// confidence below 1 (and fewer samples' worth in its reservoir, its light
// scaled by the change), one whose ray now meets something else leaves none.
@compute @workgroup_size(8, 8)
fn gather_validate(@builtin(global_invocation_id) id: vec3<u32>) {
    let traced = traced_size();
    if any(id.xy >= traced) { return; }
    let i = id.y * traced.x + id.x;
    texels[i].y = 1.0;
    if !retraced(i) || gather_answer.x == 0u { return; }
    let row = retrace_row(i);
    var r = res_unpack(previous[i], gather.previous_eye.xyz);
    if (gather.info.z & GATHER_HAS_PREVIOUS) == 0u || !r.valid || rays_in[row].t_max <= 0.0 {
        return;
    }
    let ray = rays_in[row];
    let m = met(row);
    var same = false;
    var noise = 0.0;
    var now = 0.0;
    if r.sky {
        same = m.t == 0.0;
        now = luminance(gi_sky_radiance((ray.origin - gi_trace.eye.xyz) * 0.001, ray.dir));
    } else if m.t > 0.0 {
        // The same point: the ray left the same place along the same line,
        // so a tracer finds it at the same distance but for its own rounding
        // (2^-8 of the distance) and the start's offset.
        let expected = distance(r.sample, ray.origin);
        same = abs(m.t - expected) <= 2.0 * gi_ray_offset(m.point) + expected / 256.0;
        noise = CACHE_NOISE;
        now = cache_luminance(m.point, m.normal, scene_ray_footprint(ray, m.t), surfaces[row], scene_ray_space(ray));
    }
    var confidence = 0.0;
    if same {
        let before = r.cache_y;
        let most = max(before, now);
        let change = select(0.0, abs(now - before) / most, most > 1e-20);
        confidence = select(1.0 - (change - noise) / (1.0 - noise), 1.0, change <= noise);
        if before > 1e-20 && now > 0.0 {
            r.radiance = scene_pack_radiance(scene_unpack_radiance(r.radiance) * (now / before));
        }
        r.cache_y = now;
        r.m *= confidence;
    } else {
        r = no_res();
    }
    previous[i] = res_pack(r, gather.previous_eye.xyz);
    texels[i].y = saturate(confidence);
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
    texels[i] = vec4<f32>(GATHER_NO_HIT, texels[i].y, 1.0, 0.0);
    if at.w == 0.0 || remapped(vec2<i32>(p)) {
        reservoirs[i] = res_pack(no_res(), gather.eye.xyz);
        return;
    }
    let x = at.xyz;
    let n = helio_gbuffer_normal(textureLoad(gbuf_normal, vec2<i32>(p), 0).xyz);
    let answered = gather_answer.x != 0u && gather_answer.y != 0u && rays_in[i].t_max > 0.0;
    // This frame's sample (one sample, M 1, W 1 / pdf: with the cosine
    // density, pi over the geometry term in the sample's measure).
    var f = no_res();
    if answered {
        let m = met(i);
        let l = hit_radiance(i);
        f.valid = true;
        f.receiver = x;
        f.normal = n;
        f.radiance = scene_pack_radiance(l);
        f.cache_y = hit_light[i].cache_y;
        f.m = 1.0;
        f.sky = m.t == 0.0;
        f.sample = select(m.point, m.normal, f.sky);
        f.sample_normal = m.normal;
        let g = geometry(x, n, f);
        f.weight = select(0.0, PI / g, g > 0.0);
        texels[i].x = select(GATHER_NO_HIT, m.t, m.t > 0.0);
    }
    // Where the traced pixel was: the reservoir of the previous block its
    // surface's footprint held most of.
    var prev = no_res();
    if (gather.info.z & GATHER_REUSE) != 0u && (gather.info.z & GATHER_HAS_PREVIOUS) != 0u {
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
                let r = res_unpack(previous[j], gather.previous_eye.xyz);
                let footprint = footprint_at(x);
                // The history's receiver must be this one's surface point
                // seen from here: within the block's reach on the same
                // surface, and nothing the depth buffer shows between them
                // (a block across an occluder's foot: the reprojection's
                // rejection, which leaves a temporal sample traced from the
                // same surface; its validity over time is the re-trace's).
                if r.valid && r.m > 0.0 && same_surface(x, n, r.receiver, r.normal, footprint, block_reach(footprint))
                    && surface_stops_seen(vec2<i32>(p), screen_pixel(r.receiver), x, n, r.receiver, r.normal) {
                    prev = r;
                    prev.m = min(prev.m, gather.reuse.x);
                    texels[i].z = texels[j].y;
                }
            }
        }
    }
    if !f.valid && !prev.valid {
        reservoirs[i] = res_pack(no_res(), gather.eye.xyz);
        return;
    }
    // Generalised RIS over the two domains (this frame's sample at this
    // receiver, M 1; the previous reservoir at its own receiver, M its own),
    // each sample weighted by the balance heuristic over both receivers'
    // target functions (luminance times the geometry term). No visibility
    // ray: the previous receiver is this surface point's (above).
    let xp = prev.receiver;
    let np = prev.normal;
    let mf = select(0.0, 1.0, f.valid);
    var sum = 0.0;
    var chosen = f;
    var aim = 0.0;
    if f.valid {
        let y = luminance(scene_unpack_radiance(f.radiance));
        let pc = y * geometry(x, n, f);
        let pp = select(0.0, y * geometry(xp, np, f), prev.m > 0.0);
        let w = select(0.0, pc / (pc + prev.m * pp), pc > 0.0) * pc * f.weight;
        sum += w;
        aim = pc;
    }
    if prev.valid {
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
    chosen.normal = n;
    chosen.weight = select(0.0, sum / aim, aim > 0.0);
    chosen.m = min(mf + prev.m, gather.reuse.x);
    chosen.valid = true;
    reservoirs[i] = res_pack(chosen, gather.eye.xyz);
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
    let own = res_unpack(reservoirs[i], gather.eye.xyz);
    visibility_rays[i] = no_ray();
    if !own.valid { return; }
    // The receiver's exact surface (the G-buffer's, as `gather_temporal`
    // took it): the integrand's cosine at the receiver is the sampling's,
    // where the reservoir's normal is 8 bits an axis (enough for the
    // resampling's weights, not for an integrand a grazing sample's W
    // multiplies).
    let traced_at = vec2<i32>(traced_pixel(id.xy));
    let x = position_at(traced_at).xyz;
    let n = helio_gbuffer_normal(textureLoad(gbuf_normal, traced_at, 0).xyz);
    let footprint = footprint_at(x);
    // The domains: this texel's temporal reservoir, then its neighbours'
    // that are the same surface within the radius, with nothing the depth
    // buffer shows between their receiver and this one.
    var domains: array<Res, 5>;
    var count = 1u;
    domains[0] = own;
    if (gather.info.z & GATHER_REUSE) != 0u {
        let reach = (gather.reuse.w * f32(side()) * 1.41421356 + 1.0) * footprint;
        for (var k = 0u; k < GATHER_NEIGHBOURS; k++) {
            let q = neighbour(i, id.xy, k);
            if any(q < vec2<i32>(0)) || any(q >= vec2<i32>(traced)) || all(q == b) { continue; }
            let r = res_unpack(reservoirs[u32(q.y) * traced.x + u32(q.x)], gather.eye.xyz);
            if !r.valid || r.m <= 0.0 { continue; }
            if !same_surface(x, n, r.receiver, r.normal, footprint, reach) { continue; }
            if !surface_stops_seen(vec2<i32>(traced_pixel(id.xy)), vec2<i32>(traced_pixel(vec2<u32>(q))), x, n, r.receiver, r.normal) { continue; }
            domains[count] = r;
            count += 1u;
        }
    }
    // Generalised RIS over the domains, each sample weighted by the balance
    // heuristic over every domain's target function at its own receiver
    // (luminance cancels in it: the geometry terms alone).
    var sum = 0.0;
    var chosen = 0u;
    var aim = 0.0;
    for (var k = 0u; k < count; k++) {
        let r = domains[k];
        let y = sample_point(r);
        var own_g = 0.0;
        var all_g = 0.0;
        for (var j = 0u; j < count; j++) {
            let g = point_geometry(domains[j].receiver, domains[j].normal, y);
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
    let dir = toward(x, r);
    picks[i] = vec4<f32>(scene_unpack_radiance(r.radiance) * geometry(x, n, r) * weight, bitcast<f32>(scene_pack_unit(dir)));
    // A neighbour's sample is assumed visible by the resampling: a
    // visibility ray from this receiver to it decides (`gather_signal`).
    // The texel's own temporal reservoir was traced from this surface point
    // (`gather_temporal`'s rejection), as ReSTIR GI takes it.
    if chosen != 0u {
        let o = x + n * gi_ray_offset(x);
        var reach = GI_RAY_REACH;
        if !r.sky {
            reach = max(distance(r.sample, o) - 2.0 * gi_ray_offset(r.sample), 0.0);
        }
        let cone = scene_cone_in_space(scene_ray_cone(footprint, 0.0, RAY_MASK_ALL), gather.info.w);
        visibility_rays[i] = SceneRay(o, max(reach, 1e-4), dir, cone);
    }
}

@compute @workgroup_size(8, 8)
fn gather_signal(@builtin(global_invocation_id) id: vec3<u32>) {
    let traced = traced_size();
    if any(id.xy >= traced) { return; }
    let i = id.y * traced.x + id.x;
    let b = vec2<i32>(id.xy);
    let t = texels[i];
    let p = vec2<i32>(traced_pixel(id.xy));
    var input = DenoiseIrradianceInput(vec4<f32>(0.0, 0.0, 0.0, GATHER_NO_HIT), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    let at = position_at(p);
    if at.w == 0.0 || remapped(p) {
        // The sky and remapped content: no sample (the denoiser skips the
        // sky; the resolve takes the probes for remapped pixels).
        textureStore(signal0, b, input.signal0);
        textureStore(signal1, b, input.signal1);
        return;
    }
    let footprint = footprint_at(at.xyz);
    // The estimate: the chosen sample's times what its visibility ray lets
    // through (none where the receiver does not see it: the resampling gave
    // it its share assuming it did, so its share of the light is none, as
    // ReSTIR GI discards an occluded reuse; this frame's own sample in its
    // place measured no better behind an occluder). Where the texel has no sample, or the
    // visibility rays were answered short, the probes' irradiance at the
    // traced pixel (a sample of no light would be taken as dark), its
    // history kept.
    var e = vec3<f32>(0.0);
    var dir = vec3<f32>(0.0);
    var have = false;
    let tested = visibility_rays[i].t_max > 0.0;
    if res_unpack(reservoirs[i], gather.eye.xyz).valid && (!tested || gather_answer.z != 0u) {
        let chosen = picks[i];
        var through = vec3<f32>(1.0);
        if tested {
            through = scene_occlusion_through(visibility_answers[i]);
        }
        e = chosen.xyz * through;
        dir = scene_unpack_unit(bitcast<u32>(chosen.w));
        have = true;
    }
    let n = helio_gbuffer_normal(textureLoad(gbuf_normal, p, 0).xyz);
    if !have {
        e = PI * gi_irradiance(at.xyz - gi_trace.eye.xyz, n).rgb;
        dir = n;
        input.signal0.w = f32(side()) * footprint;
    } else {
        // The hit distance the denoiser holds a record over: this frame's
        // ray's, no nearer than a block's footprint (what is nearer is the
        // near field's); the confidence the reused reservoir's re-trace left.
        input.signal0.w = max(t.x, f32(side()) * footprint);
        input.signal1.w = saturate(t.z);
    }
    input = denoise_irradiance_add(input, e, dir);
    textureStore(signal0, b, input.signal0);
    textureStore(signal1, b, input.signal1);
}
