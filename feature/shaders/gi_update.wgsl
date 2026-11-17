//!use light_list
//!use frame_sky
//!use sky_table
//!use frame_ray_stream
//!use gi_common
//!use gi_rays
// The start of the probes' update (README.md, "The frame"): the sun found,
// probes that came into the clipmap reset, this frame's probe rays written
// into the probe stream for the tracers.

@group(0) @binding(0) var<uniform> frame_gi: GiFrame;
@group(0) @binding(1) var<storage, read_write> gi_probes: array<u32>;
// The probe stream's rays and hits (`gi_probe`): a row a ray, the hits
// cleared for the tracers.
@group(0) @binding(2) var<storage, read_write> probe_rays: array<SceneRay>;
@group(0) @binding(3) var<storage, read_write> probe_hits: array<SceneHit>;
// This frame's update list (GI_MAX_UPDATES probes), then the probes to
// reset.
@group(0) @binding(4) var<storage, read> gi_lists: array<u32>;
// The sun: towards it (xyz), its illuminance at the eye (rgb).
@group(0) @binding(5) var<storage, read_write> gi_sun: array<vec4<f32>, 2>;
// The lights' directional lights (`helio_light_api::DIRECTIONAL_LIGHTS`),
// the first the primary sun.
@group(0) @binding(6) var<storage, read> directional: DirectionalLights;
@group(0) @binding(7) var<uniform> sky_frame: SkyFrame;
@group(0) @binding(8) var<uniform> sky_table: array<vec4<f32>, 2304>;
@group(0) @binding(9) var<uniform> gi_trace: GiTrace;

// The sun: the atmosphere's, through the air to the eye; without one, the
// lights' primary sun.
@compute @workgroup_size(1)
fn gi_sun_main() {
    if sky_frame.planet.w > 0.5 {
        let towards = sky_frame.sun.xyz;
        gi_sun[0] = vec4<f32>(towards, 0.0);
        gi_sun[1] = vec4<f32>(sky_frame.sun_illuminance.rgb * sky_transmittance_towards(sky_frame, vec3<f32>(0.0), towards), 0.0);
        return;
    }
    if directional.count.x > 0u {
        gi_sun[0] = vec4<f32>(directional.lights[0].toward.xyz, 0.0);
        gi_sun[1] = vec4<f32>(directional.lights[0].radiance.rgb, 0.0);
        return;
    }
    gi_sun[0] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    gi_sun[1] = vec4<f32>(0.0);
}

// Probes that came into the clipmap: no light, no distances, no updates
// (before this frame's integration, which may update them at once).
@compute @workgroup_size(64)
fn gi_reset(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame_gi.counts.z { return; }
    let at = gi_lists[GI_MAX_UPDATES + id.x] * GI_PROBE_WORDS;
    for (var w = 0u; w < GI_PROBE_WORDS; w++) { gi_probes[at + w] = 0u; }
    gi_probes[at + GI_STATE_WORD] = GI_ACTIVE;
}

// This frame's rays: from each probe to update, its set turned by this
// frame's rotation, in the frame's space. A ray starts a little behind its
// probe, so a surface the probe lies on is met whichever side f32 rounding
// put the probe (`gi_light` counts it as inside).
@compute @workgroup_size(64)
fn gi_rays_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame_gi.counts.x * GI_RAYS { return; }
    let index = gi_lists[id.x / GI_RAYS];
    let probe = gi_probe_position(index) + gi_trace.eye.xyz;
    let dir = gi_ray_dir(id.x % GI_RAYS, frame_gi.rotation);
    let behind = gi_ray_offset(probe);
    probe_rays[id.x] = SceneRay(probe - dir * behind, GI_RAY_REACH, dir, scene_ray_cone(0.0, GI_RAY_SPREAD, RAY_MASK_ALL));
    probe_hits[id.x] = SceneHit(0.0, 0u, 0u, 0u);
}
