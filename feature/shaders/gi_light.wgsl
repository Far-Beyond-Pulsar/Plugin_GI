//!use frame_sky
//!use sky_table
//!use frame_ray_stream
//!use gi_common
//!use gi_sample
//!use gi_rays
//!use gi_local_lights
//!use sun_transmittance
// The light of the probe rays' hits (README.md, "The frame"), once every
// tracer answered the probe stream: a workgroup a probe, a lane a ray. From
// the hit's surface (`SceneSurface`, whichever kind's it is) and place, the
// light waiting on a shadow ray (the sun's off its sunlit side, through the
// surface for one that lets light through; one local light's), and the rest
// (the probes' irradiance there, from its far side through it, what it
// emits; the sky on a miss) into `gi_pending`; the shadow rays, compacted,
// into the shadow stream.

@group(0) @binding(0) var<uniform> frame_gi: GiFrame;
@group(0) @binding(1) var<storage, read> gi_probes: array<u32>;
@group(0) @binding(2) var<storage, read> gi_lists: array<u32>;
@group(0) @binding(3) var<storage, read> gi_sun: array<vec4<f32>, 2>;
@group(0) @binding(4) var<uniform> sky_frame: SkyFrame;
@group(0) @binding(5) var<uniform> sky_table: array<vec4<f32>, 2304>;
@group(0) @binding(6) var<uniform> gi_trace: GiTrace;
// The probe stream as the tracers left it (`gi_probe`).
@group(0) @binding(7) var<storage, read> probe_hits: array<SceneHit>;
@group(0) @binding(8) var<storage, read> probe_surfaces: array<SceneSurface>;
// The shadow stream (`gi_shadow`): its rows written from 0 up, its count
// in its header (cleared before this pass; `gi_shadow_header` writes the
// dispatch).
@group(0) @binding(9) var<storage, read_write> shadow_rays: array<SceneRay>;
@group(0) @binding(10) var<storage, read_write> shadow_hits: array<SceneHit>;
@group(0) @binding(11) var<storage, read_write> shadow_header: array<atomic<u32>, 4>;
@group(0) @binding(12) var<storage, read_write> gi_pending: array<GiPending>;
// Word 0: set once any probe ray met anything this frame (`gi_integrate`
// holds the probes on a frame where none did).
@group(0) @binding(13) var<storage, read_write> gi_answered: array<atomic<u32>, 4>;
@group(0) @binding(14) var<storage, read> local_lights: GiLocalLights;
// The sun's transmittance through the media above (the frame's
// `sun_transmittance`: the clouds'): the sun at a hit under a cloud is the
// cloud's share of it, so the shade under a cloud is not lit by bounce from
// ground the cloud shades.
@group(0) @binding(15) var sun_transmittance_map: texture_2d<f32>;
@group(0) @binding(16) var<uniform> sun_transmittance_frame: SunTransmittanceFrame;
// The frame's statistics (`helio_core::stats`): the shadow rays a frame.
@group(0) @binding(17) var<storage, read_write> stats: array<atomic<u32>>;

override GI_SHADOW_STAT: u32 = 0u;

const PI: f32 = 3.14159265;

// The sky's radiance along unit direction `d` from eye-relative point
// `from_camera` (km): the atmosphere's sky table there (irradiance / pi in
// L2 harmonics, cosine-convolved) back to radiance, band by band; without
// an atmosphere, the flat sky and ground.
fn sky_radiance(from_camera: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    if sky_frame.planet.w < 0.5 { return select(frame_gi.ground.rgb, frame_gi.sky.rgb, d.y >= 0.0); }
    // The basis over each band's convolution (1, 2/3, 1/4).
    return sky_table_eval(from_camera, d, vec3<f32>(1.0, 1.5, 4.0));
}

var<workgroup> w_sun: atomic<u32>;
var<workgroup> w_local: atomic<u32>;
var<workgroup> w_hit: atomic<u32>;
var<workgroup> w_base: u32;
var<workgroup> w_suns: u32;

@compute @workgroup_size(64)
fn gi_light(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if wg.x >= frame_gi.counts.x { return; }
    if li == 0u {
        atomicStore(&w_sun, 0u);
        atomicStore(&w_local, 0u);
        atomicStore(&w_hit, 0u);
    }
    workgroupBarrier();
    let i = wg.x * GI_RAYS + li;
    let probe = gi_probe_position(gi_lists[wg.x]);
    let dir = gi_ray_dir(li, frame_gi.rotation);
    let eye = gi_trace.eye.xyz;
    let behind = gi_ray_offset(probe + eye);
    let hit = probe_hits[i];
    let cone = scene_ray_cone(GI_RAY_SPREAD, RAY_MASK_ALL);
    var pending = GiPending(0u, 0u, 0u, 0.0, GI_NO_ROW, GI_NO_ROW);
    var rest = vec3<f32>(0.0);
    var sun_ray = SceneRay(vec3<f32>(0.0), 0.0, vec3<f32>(0.0, 1.0, 0.0), cone);
    var local_ray = sun_ray;
    var want_sun = false;
    var want_local = false;
    if hit.t > 0.0 {
        atomicOr(&w_hit, 1u);
        let t = hit.t - behind;
        // Its inside: a back face (the probe is in a wall or a solid); and a
        // probe on a surface (within `behind` of it): inside it, so probes on
        // a floor are dropped alike, not some of them.
        if scene_hit_inside(hit) || t < behind {
            pending.t = -max(t, 1e-3);
        } else {
            pending.t = t;
            let s = probe_surfaces[i];
            let n = scene_unpack_unit(s.normal);
            let face = scene_unpack_unit(s.face);
            let base = scene_surface_base_color(s);
            let through = scene_surface_transmission(s);
            // Eye-relative, as the probes are; and in the frame.
            let at = probe + dir * t;
            let p = at + eye;
            let off = gi_ray_offset(p);
            // Lambertian on the side the ray meets, plus the surface's
            // share of its far side's light.
            rest = base * gi_irradiance(at, n).rgb + scene_unpack_radiance(s.emissive);
            if through > 0.0 {
                rest += base * through * gi_irradiance(at, -n).rgb;
            }
            // The sun, from the side of the surface that faces it.
            let sun = gi_sun[1].rgb * sun_transmittance_at(at);
            let towards = gi_sun[0].xyz;
            let cosine = dot(n, towards);
            var sun_light = vec3<f32>(0.0);
            var side = face;
            if cosine > 0.0 {
                sun_light = base * sun * cosine / PI;
            } else if through > 0.0 && cosine < 0.0 {
                sun_light = base * through * sun * -cosine / PI;
                side = -face;
            }
            if any(sun_light > vec3<f32>(0.0)) {
                want_sun = true;
                pending.sun = scene_pack_radiance(sun_light);
                sun_ray = SceneRay(p + side * off, GI_RAY_REACH, towards, cone);
            }
            if local_lights.count.x > 0u {
                let pick = gi_local_light(p, n, gi_hash(i ^ gi_trace.info.x));
                if any(pick.irradiance > vec3<f32>(0.0)) {
                    want_local = true;
                    pending.local = scene_pack_radiance(base / PI * pick.irradiance);
                    local_ray = SceneRay(p + face * off, max(pick.distance - 2.0 * off, 1e-4), pick.dir, cone);
                }
            }
        }
    } else {
        rest = sky_radiance(probe * 0.001, dir);
    }
    pending.rest = scene_pack_radiance(rest);
    // The workgroup's shadow rays in one block of the stream: its sun rays,
    // then its local lights'.
    var sun_slot = 0u;
    var local_slot = 0u;
    if want_sun { sun_slot = atomicAdd(&w_sun, 1u); }
    if want_local { local_slot = atomicAdd(&w_local, 1u); }
    workgroupBarrier();
    if li == 0u {
        let suns = atomicLoad(&w_sun);
        let rows = suns + atomicLoad(&w_local);
        w_suns = suns;
        w_base = select(0u, atomicAdd(&shadow_header[3], rows), rows > 0u);
        if atomicLoad(&w_hit) != 0u { atomicOr(&gi_answered[0], 1u); }
    }
    workgroupBarrier();
    if want_sun {
        let row = w_base + sun_slot;
        shadow_rays[row] = sun_ray;
        shadow_hits[row] = SceneHit(0.0, 0u, 0u, 0u);
        pending.sun_row = row;
    }
    if want_local {
        let row = w_base + w_suns + local_slot;
        shadow_rays[row] = local_ray;
        shadow_hits[row] = SceneHit(0.0, 0u, 0u, 0u);
        pending.local_row = row;
    }
    gi_pending[i] = pending;
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
