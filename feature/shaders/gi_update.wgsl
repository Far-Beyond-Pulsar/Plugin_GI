//!use frame_ray_stream
//!use gi_common
//!use gi_rays
// The start of the probes' update (README.md, "The frame"): probes that
// came into the clipmap reset, this frame's probe rays written into the probe
// stream for the tracers.

@group(0) @binding(0) var<uniform> frame_gi: GiFrame;
@group(0) @binding(1) var<storage, read_write> gi_probes: array<u32>;
// The probe stream's rays and hits (`gi_probe`): a row a ray, the hits
// cleared for the tracers.
@group(0) @binding(2) var<storage, read_write> probe_rays: array<SceneRay>;
@group(0) @binding(3) var<storage, read_write> probe_hits: array<SceneHit>;
// This frame's update list (GI_MAX_UPDATES probes), then the probes to
// reset.
@group(0) @binding(4) var<storage, read> gi_lists: array<u32>;
@group(0) @binding(5) var<uniform> gi_trace: GiTrace;

// Probes that came into the clipmap: no light, no distances, no updates
// (before this frame's integration, which may update them at once).
@compute @workgroup_size(64)
fn gi_reset(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame_gi.reset { return; }
    let at = gi_lists[GI_MAX_UPDATES + id.x] * GI_PROBE_WORDS;
    for (var w = 0u; w < GI_PROBE_WORDS; w++) { gi_probes[at + w] = 0u; }
    gi_probes[at + GI_STATE_WORD] = GI_ACTIVE;
}

// This frame's rays: from each probe to update, its set turned by this
// frame's rotation, in the frame's space. A ray starts a little behind its
// probe, along its own direction, so every ray of a probe that lies on a
// surface crosses it, whichever side f32 rounding put the probe (`gi_light`
// counts it as inside). That is why a probe's rays are not a run
// (`Stream::with_run`): they share no origin. From the probe itself, only
// the rays toward the surface meet it, the probe stays active and passes
// light through the surface (a solid card let 1% through).
@compute @workgroup_size(64)
fn gi_rays_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame_gi.updated * GI_RAYS { return; }
    let index = gi_lists[id.x / GI_RAYS];
    let probe = gi_probe_position(index) + gi_trace.eye.xyz;
    let dir = gi_ray_dir(id.x % GI_RAYS, frame_gi.rotation);
    let behind = gi_ray_offset(probe);
    probe_rays[id.x] = SceneRay(probe - dir * behind, GI_RAY_REACH, dir, scene_cone_in_space(scene_ray_cone(0.0, GI_RAY_SPREAD, RAY_MASK_ALL), gi_trace.info.y));
    probe_hits[id.x] = SceneHit(0.0, 0u, 0u, 0u);
}
