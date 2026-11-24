//!use helio_prelude
//!use frame_ray_stream
//!use frame_radiance_cache
//!use frame_surface_stops
//!use gi_common
//!use gi_sample
//!use frame_denoise_irradiance
// The frame's indirect diffuse at every pixel (README.md, "Deferred
// lighting"): a pixel's word pair of `gi_irradiance` (rows of the frame)
// holds its irradiance / pi (rgb, half floats) and how much it holds the
// pixel (the fourth half: 0 past the clipmap, or for the sky).
// `gi_resolve`: the probes' irradiance. `gi_resolve_pixels`: the pixels'
// gather (`gi_gather.wgsl`) as the frame's denoiser filtered it, times the
// near field, the probes' where it has no light yet or a remap put the
// pixel in another space. Its debug views show the probes (GI_PROBES_VIEW)
// or the radiance cache (RADIANCE_CACHE_VIEW).

@group(0) @binding(0) var<uniform> frame_gi: GiFrame;
@group(0) @binding(1) var<storage, read> gi_probes: array<u32>;
@group(0) @binding(2) var<storage, read> cameras: array<Camera>;
@group(0) @binding(3) var gbuf_normal: texture_2d<f32>;
@group(0) @binding(4) var gbuf_depth: texture_depth_2d;
@group(0) @binding(5) var<storage, read_write> out_irradiance: array<vec2<u32>>;
// x: debug mode (GI_PROBES_VIEW, RADIANCE_CACHE_VIEW: those instead), y:
// remaps published.
@group(0) @binding(6) var<uniform> resolve_view: vec4<u32>;
@group(0) @binding(7) var surface_tags: texture_2d<f32>;
@group(0) @binding(8) var<uniform> view_remaps: HelioViewRemaps;
@group(0) @binding(9) var<uniform> remap_spaces: HelioViewRemapSpaces;
// The frame's radiance cache, read for its view (a read marks nothing).
@group(0) @binding(10) var<uniform> radiance_cache_frame: RadianceCacheFrame;
@group(0) @binding(11) var<storage, read_write> radiance_cache_slots: array<RadianceCacheSlot>;
@group(0) @binding(12) var<storage, read_write> radiance_cache_entries: array<RadianceCacheEntry>;
// The pixels' gather (`gi_resolve_pixels` only): x the traced block's side,
// y the frame (the denoiser's), z 1 when `gathered` is the denoiser's output
// (else the gather's own signal, a texel a block).
@group(0) @binding(13) var<uniform> resolve_gather: vec4<u32>;
@group(0) @binding(14) var gathered0: texture_2d<f32>;
@group(0) @binding(15) var gathered1: texture_2d<f32>;
// The near field (`gather_near`): y what the gathered irradiance is
// multiplied by at the pixel.
@group(0) @binding(16) var near_field: texture_2d<f32>;

const GI_PROBES_VIEW: u32 = 36u;
const RADIANCE_CACHE_VIEW: u32 = 39u;
const GI_NEAR_FIELD_VIEW: u32 = 37u;
const PI: f32 = 3.14159265;

// The cache's cell a ray meeting frame point `p` with normal `n` reads (in
// the camera's space, any reach), as `gi_irradiance` holds light: its
// irradiance / pi, weight 0 where no cell holds a value; darker toward its
// cell's edges (along the two axes across its side).
fn cache_view(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    let c = radiance_cache_read(p, n, cameras[0].view_info.w, 0.0);
    if !c.found { return vec4<f32>(0.0); }
    let q = radiance_cache_cells(p, c.level);
    let f = abs(q - floor(q) - 0.5);
    let axis = radiance_cache_side(n) / 2u;
    let across = select(select(max(f.x, f.y), max(f.x, f.z), axis == 1u), max(f.y, f.z), axis == 0u);
    let edge = mix(1.0, 0.4, smoothstep(0.42, 0.5, across));
    return vec4<f32>((c.direct + c.indirect) / PI * edge, 1.0);
}

// The probe nearest eye-relative point `p` in the finest level holding it,
// and its look: its irradiance toward `n`; dark red inside the ground,
// dark purple not yet updated; darker toward its cell's edges, so the
// probes' cells show.
fn probe_view(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    for (var level = 0u; level < GI_LEVELS; level++) {
        let grid = frame_gi.levels[level];
        let local = (p - grid.origin.xyz) / grid.origin.w;
        if any(local < vec3<f32>(0.0)) || any(local > vec3<f32>(f32(GI_SIDE) - 1.0)) { continue; }
        let index = gi_probe_index(level, vec3<u32>(round(local)));
        let state = gi_probes[index * GI_PROBE_WORDS + GI_STATE_WORD];
        var colour = gi_probe_irradiance(index, n);
        if (state & GI_ACTIVE) == 0u { colour = vec3<f32>(0.35, 0.0, 0.0); }
        else if (state >> 8u) == 0u { colour = vec3<f32>(0.25, 0.0, 0.25); }
        let edge = max(abs(local - round(local)).x, max(abs(local - round(local)).y, abs(local - round(local)).z));
        return colour * mix(1.0, 0.4, smoothstep(0.42, 0.5, edge));
    }
    return vec3<f32>(0.0);
}

// A pixel's surface: its frame point and normal (in the space a remap put
// its content in), and whether it holds one the probes can light (none: the
// sky, or content in another coordinate space than the clipmap's).
struct Pixel {
    position: vec3<f32>,
    normal: vec3<f32>,
    remapped: bool,
    lit: bool,
}

fn pixel_at(id: vec2<u32>, size: vec2<u32>) -> Pixel {
    var px: Pixel;
    let depth = textureLoad(gbuf_depth, vec2<i32>(id), 0);
    if helio_is_sky(depth) { return px; }
    let uv = (vec2<f32>(id) + 0.5) / vec2<f32>(size);
    let h = cameras[0].view_proj_inv * vec4<f32>(helio_uv_to_ndc(uv), depth, 1.0);
    px.position = h.xyz / h.w;
    px.normal = normalize(textureLoad(gbuf_normal, vec2<i32>(id), 0).xyz);
    px.lit = true;
    if resolve_view.y != 0u {
        let remap = gbuffer_remap_of(textureLoad(surface_tags, vec2<i32>(id), 0).xy);
        if remap >= 0 {
            let r = u32(remap);
            px.remapped = true;
            // The active probe clipmap is built for the unpartitioned world.
            if remap_spaces[r / 4u][r % 4u] != 0u {
                px.lit = false;
                return px;
            }
            let unmap = view_remaps[r];
            px.position = (unmap * vec4<f32>(px.position, 1.0)).xyz;
            px.normal = normalize((unmap * vec4<f32>(px.normal, 0.0)).xyz);
        }
    }
    return px;
}

// The probes' irradiance / pi at a pixel, or its debug views'.
fn probes_at(px: Pixel) -> vec4<f32> {
    let p = px.position - cameras[0].position_near.xyz;
    var e = gi_irradiance(p, px.normal);
    if resolve_view.x == GI_PROBES_VIEW { e = vec4<f32>(probe_view(p, px.normal), 1.0); }
    if resolve_view.x == RADIANCE_CACHE_VIEW { e = cache_view(px.position, px.normal); }
    return e;
}

fn store(at: u32, e: vec4<f32>) {
    out_irradiance[at] = vec2<u32>(pack2x16float(e.rg), pack2x16float(vec2<f32>(e.b, e.a)));
}

@compute @workgroup_size(8, 8)
fn gi_resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(gbuf_depth);
    if id.x >= size.x || id.y >= size.y { return; }
    let at = id.y * size.x + id.x;
    let px = pixel_at(id.xy, size);
    if !px.lit {
        out_irradiance[at] = vec2<u32>(0u);
        return;
    }
    store(at, probes_at(px));
}

// The gathered irradiance at pixel `id` (normal `n`), or below 0 where the
// gather has no light there yet.
fn gathered_at(id: vec2<u32>, size: vec2<u32>, n: vec3<f32>) -> vec4<f32> {
    if resolve_gather.z != 0u {
        let o0 = textureLoad(gathered0, vec2<i32>(id), 0);
        if o0.w < 0.0 { return vec4<f32>(-1.0); }
        return vec4<f32>(denoise_irradiance(o0, textureLoad(gathered1, vec2<i32>(id), 0), n), 1.0);
    }
    // The gather's own signal, a texel a block, at its traced pixel's normal.
    let side = resolve_gather.x;
    let block = id / side;
    let traced = denoise_traced_pixel(block, resolve_gather.y, side, size);
    let n0 = normalize(textureLoad(gbuf_normal, vec2<i32>(traced), 0).xyz);
    let s0 = textureLoad(gathered0, vec2<i32>(block), 0);
    let s1 = textureLoad(gathered1, vec2<i32>(block), 0);
    return vec4<f32>(denoise_irradiance_input(s0, s1, n0, n), 1.0);
}

@compute @workgroup_size(8, 8)
fn gi_resolve_pixels(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(gbuf_depth);
    if id.x >= size.x || id.y >= size.y { return; }
    let at = id.y * size.x + id.x;
    let px = pixel_at(id.xy, size);
    if !px.lit {
        out_irradiance[at] = vec2<u32>(0u);
        return;
    }
    var e = vec4<f32>(-1.0);
    if !px.remapped && resolve_view.x != GI_PROBES_VIEW && resolve_view.x != RADIANCE_CACHE_VIEW {
        e = gathered_at(id.xy, size, px.normal);
    }
    if e.w < 0.0 {
        store(at, probes_at(px));
        return;
    }
    let near = textureLoad(near_field, vec2<i32>(id.xy), 0).y;
    if resolve_view.x == GI_NEAR_FIELD_VIEW {
        store(at, vec4<f32>(vec3<f32>(near), 1.0));
        return;
    }
    store(at, vec4<f32>(e.rgb * near / PI, 1.0));
}
