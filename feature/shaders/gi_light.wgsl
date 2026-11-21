//!use frame_sky
//!use sky_radiance
//!use frame_ray_stream
//!use frame_ray_surface
//!use frame_surface
//!use frame_light_samples
//!use gi_common
//!use gi_sample
//!use gi_rays
// The light of the probe rays' hits (README.md, "The frame"), once every
// tracer answered the probe stream: a workgroup a probe, a lane a ray.
//
// `gi_light`: from the hit's surface (`SceneSurface`, whichever kind's it
// is) and place, the light that needs no shadow ray (the probes'
// irradiance there, from its far side through it, what it emits; the sky on
// a miss) into `gi_pending`, and a query of the frame's light samples
// (`LightQuery`) a hit, compacted into the light batch. The frame's samplers
// (every kind of light's) then choose one light a hit.
//
// `gi_shadow`: each hit's light sample off its surface (Lambertian on the
// side the light is on; through it by its transmission share from behind)
// into `gi_pending`, waiting on a shadow ray toward the light, compacted
// into the shadow stream.

@group(0) @binding(0) var<uniform> frame_gi: GiFrame;
@group(0) @binding(1) var<storage, read> gi_probes: array<u32>;
@group(0) @binding(2) var<storage, read> gi_lists: array<u32>;
@group(0) @binding(3) var<uniform> sky_frame: SkyFrame;
@group(0) @binding(4) var sky_radiance: texture_3d<f32>;
@group(0) @binding(17) var sky_radiance_sampler: sampler;
@group(0) @binding(5) var<uniform> gi_trace: GiTrace;
// The probe stream as the tracers left it (`gi_probe`).
@group(0) @binding(6) var<storage, read> probe_hits: array<SceneHit>;
@group(0) @binding(7) var<storage, read> probe_surfaces: array<SceneSurface>;
// The shadow stream (`gi_shadow`): its rows written from 0 up, its count
// in its header (cleared before `gi_shadow`; `gi_shadow_header` writes the
// dispatch).
@group(0) @binding(8) var<storage, read_write> shadow_rays: array<SceneRay>;
@group(0) @binding(9) var<storage, read_write> shadow_hits: array<SceneHit>;
@group(0) @binding(10) var<storage, read_write> shadow_header: array<atomic<u32>, 4>;
@group(0) @binding(11) var<storage, read_write> gi_pending: array<GiPending>;
// Word 0: set once any probe ray met anything this frame (`gi_integrate`
// holds the probes on a frame where none did).
@group(0) @binding(12) var<storage, read_write> gi_answered: array<atomic<u32>, 4>;
// The light batch (`gi_probe`): its queries written from 0 up, its count in
// its header (cleared before `gi_light`; `gi_light_header` writes the
// dispatch); the samples the samplers left.
@group(0) @binding(13) var<storage, read_write> light_header: array<atomic<u32>, 4>;
@group(0) @binding(14) var<storage, read_write> light_queries: array<LightQuery>;
@group(0) @binding(15) var<storage, read> light_samples: array<LightSample>;
// The frame's statistics (`helio_core::stats`): the shadow rays a frame.
@group(0) @binding(16) var<storage, read_write> stats: array<atomic<u32>>;

override GI_SHADOW_STAT: u32 = 0u;

const PI: f32 = 3.14159265;

// The sky's radiance along unit direction `d` from eye-relative point
// `from_camera` (km): the atmosphere's radiance table there
// (`sky_radiance_eval`, from the eye the sky the sky pass draws); without
// an atmosphere, the flat sky and ground.
fn sky_radiance_at(from_camera: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    if sky_frame.planet.w < 0.5 { return select(frame_gi.ground.rgb, frame_gi.sky.rgb, d.y >= 0.0); }
    return sky_radiance_eval(from_camera, d);
}

var<workgroup> w_rows: atomic<u32>;
var<workgroup> w_hit: atomic<u32>;
var<workgroup> w_base: u32;

// The hit of probe ray `i` (its ray `dir` from probe `probe`, eye-relative):
// where it is (eye-relative), the surface and its face.
struct Hit {
    at: vec3<f32>,
    t: f32,
    s: SurfaceData,
    face: vec3<f32>,
}

fn hit_of(i: u32, probe: vec3<f32>, dir: vec3<f32>, behind: f32) -> Hit {
    let t = probe_hits[i].t - behind;
    let row = probe_surfaces[i];
    return Hit(probe + dir * t, t, scene_surface_unpack(row), scene_surface_face(row));
}

@compute @workgroup_size(64)
fn gi_light(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if wg.x >= frame_gi.updated { return; }
    if li == 0u {
        atomicStore(&w_rows, 0u);
        atomicStore(&w_hit, 0u);
    }
    workgroupBarrier();
    let i = wg.x * GI_RAYS + li;
    let probe = gi_probe_position(gi_lists[wg.x]);
    let dir = gi_ray_dir(li, frame_gi.rotation);
    let eye = gi_trace.eye.xyz;
    let behind = gi_ray_offset(probe + eye);
    let hit = probe_hits[i];
    var pending = GiPending(0u, 0u, 0.0, GI_NO_ROW);
    var rest = vec3<f32>(0.0);
    var query: LightQuery;
    var want = false;
    if hit.t > 0.0 {
        atomicOr(&w_hit, 1u);
        let h = hit_of(i, probe, dir, behind);
        // Its inside: a back face (the probe is in a wall or a solid); and a
        // probe on a surface (within `behind` of it): inside it, so probes on
        // a floor are dropped alike, not some of them.
        if scene_hit_inside(hit) || h.t < behind {
            pending.t = -max(h.t, 1e-3);
        } else {
            pending.t = h.t;
            // The hit's surface: the frame's `SurfaceData`, as a G-buffer
            // texel holds it. The probe stream asks for
            // `SurfaceDetail::Diffuse`: lit as Lambertian here.
            let n = h.s.normal;
            // The share of the light behind it that passes through, by
            // colour (two-sided leaves).
            let through = surface_transmission(h.s);
            rest = h.s.albedo.rgb * gi_irradiance(h.at, n).rgb + h.s.emissive;
            if any(through > vec3<f32>(0.0)) {
                rest += through * gi_irradiance(h.at, -n).rgb;
            }
            // Its direct light: one light sample at the hit, in the probes'
            // coordinate space (`GiTrace`), as wide as the probe ray's
            // footprint there (`scene_ray_footprint` of a ray with no width,
            // GI_RAY_SPREAD a metre from its origin), lit from both sides
            // when light passes through it.
            let seed = gi_hash(i ^ gi_trace.info.x);
            let u = vec2<f32>(f32(gi_hash(seed ^ 0x68e31da4u) >> 8u), f32(gi_hash(seed ^ 0xb5297a4du) >> 8u)) / 16777216.0;
            query = light_query_point(h.at + eye, GI_RAY_SPREAD * h.t, n, any(through > vec3<f32>(0.0)), u, gi_hash(seed), gi_trace.info.y);
            want = true;
        }
    } else {
        rest = sky_radiance_at(probe * 0.001, dir);
    }
    pending.rest = scene_pack_radiance(rest);
    // The workgroup's queries in one block of the batch.
    var slot = 0u;
    if want { slot = atomicAdd(&w_rows, 1u); }
    workgroupBarrier();
    if li == 0u {
        let rows = atomicLoad(&w_rows);
        w_base = select(0u, atomicAdd(&light_header[3], rows), rows > 0u);
        if atomicLoad(&w_hit) != 0u { atomicOr(&gi_answered[0], 1u); }
    }
    workgroupBarrier();
    if want {
        pending.row = w_base + slot;
        light_queries[pending.row] = query;
    }
    gi_pending[i] = pending;
}

// The light batch's dispatch for the samplers, from its count.
@compute @workgroup_size(1)
fn gi_light_header() {
    let count = atomicLoad(&light_header[3]);
    atomicStore(&light_header[0], (count + RAY_STREAM_GROUP - 1u) / RAY_STREAM_GROUP);
    atomicStore(&light_header[1], 1u);
    atomicStore(&light_header[2], 1u);
}

@compute @workgroup_size(64)
fn gi_shadow(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if wg.x >= frame_gi.updated { return; }
    if li == 0u { atomicStore(&w_rows, 0u); }
    workgroupBarrier();
    let i = wg.x * GI_RAYS + li;
    var pending = gi_pending[i];
    var ray: SceneRay;
    var want = false;
    if pending.row != GI_NO_ROW {
        let sample = light_samples[pending.row];
        pending.row = GI_NO_ROW;
        if light_sample_found(sample) {
            let probe = gi_probe_position(gi_lists[wg.x]);
            let dir = gi_ray_dir(li, frame_gi.rotation);
            let eye = gi_trace.eye.xyz;
            let h = hit_of(i, probe, dir, gi_ray_offset(probe + eye));
            let towards = light_sample_dir(sample);
            let radiance = light_sample_radiance(sample);
            let through = surface_transmission(h.s);
            // Lambertian on the side the light is on; from behind, the
            // surface's share of it through.
            let cosine = dot(h.s.normal, towards);
            var light = vec3<f32>(0.0);
            var side = h.face;
            if cosine > 0.0 {
                light = h.s.albedo.rgb * radiance * cosine / PI;
            } else if any(through > vec3<f32>(0.0)) && cosine < 0.0 {
                light = through * radiance * -cosine / PI;
                side = -h.face;
            }
            if any(light > vec3<f32>(0.0)) {
                let p = h.at + eye;
                let off = gi_ray_offset(p);
                // A shadow ray starts as wide as the probe ray's footprint at
                // the hit and keeps that width (no spread: it stands for the
                // probe ray's patch of surface, not a cone from a point).
                let cone = scene_cone_in_space(scene_ray_cone(GI_RAY_SPREAD * h.t, 0.0, RAY_MASK_ALL), gi_trace.info.y);
                let distance = light_sample_distance(sample);
                let reach = select(max(distance - 2.0 * off, 1e-4), GI_RAY_REACH, distance >= LIGHT_SAMPLE_FAR);
                ray = SceneRay(p + side * off, reach, towards, cone);
                pending.light = scene_pack_radiance(light);
                want = true;
            }
        }
    }
    // The workgroup's shadow rays in one block of the stream.
    var slot = 0u;
    if want { slot = atomicAdd(&w_rows, 1u); }
    workgroupBarrier();
    if li == 0u {
        let rows = atomicLoad(&w_rows);
        w_base = select(0u, atomicAdd(&shadow_header[3], rows), rows > 0u);
    }
    workgroupBarrier();
    if want {
        pending.row = w_base + slot;
        shadow_rays[pending.row] = ray;
        shadow_hits[pending.row] = SceneHit(0.0, 0u, 0u, 0u);
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
