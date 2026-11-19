//!use light_list
//!use gi_rays
//!use gi_local_lights
// The scene's live point and spot lights, in row order, the first
// GI_MAX_LOCAL_LIGHTS of them (`helio_pass_gi::local_lights`), once a frame.

@group(0) @binding(0) var<storage, read> scene_lights: array<GpuLight>;
@group(0) @binding(1) var<storage, read_write> local_lights: GiLocalLights;

@compute @workgroup_size(1)
fn gi_gather_lights() {
    var count = 0u;
    for (var i = 0u; i < arrayLength(&scene_lights) && count < GI_MAX_LOCAL_LIGHTS; i++) {
        let light = scene_lights[i];
        if light.light_type == LIGHT_DIRECTIONAL || light.color_intensity.w <= 0.0 || light.position_range.w <= 0.0 { continue; }
        local_lights.lights[count] = GiLocalLight(light.position_range, light.direction_outer, vec4<f32>(light.color_intensity.rgb * light.color_intensity.w, f32(light.light_type)), vec4<f32>(light.inner_angle, 0.0, 0.0, 0.0));
        count++;
    }
    local_lights.count = vec4<u32>(count, 0u, 0u, 0u);
}
