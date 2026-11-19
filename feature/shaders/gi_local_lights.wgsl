// How a probe ray's hit samples the scene's point and spot lights
// (`//!use gi_local_lights`, `helio_pass_gi::local_lights`): the live ones
// gathered once a frame (`gi_gather_lights.wgsl`), then one chosen at a hit
// by weighted reservoir over their unshadowed light there. GI's stand-in for
// the frame's light sampling, which the lights plugin will own (rendering.md
// r5): a reader binds `local_lights` (a `GiLocalLights`).

const GI_MAX_LOCAL_LIGHTS: u32 = 256u;

// A point or spot light: position, range; direction (spot), outer cosine;
// radiant colour (colour times intensity), type; inner cosine.
struct GiLocalLight {
    position_range: vec4<f32>,
    direction_outer: vec4<f32>,
    color_type: vec4<f32>,
    inner: vec4<f32>,
}

struct GiLocalLights {
    count: vec4<u32>,
    lights: array<GiLocalLight, GI_MAX_LOCAL_LIGHTS>,
}

// One light chosen at a point: toward it (unit), how far, and its
// irradiance there over its chance (unshadowed: the caller traces the
// shadow ray), 0 with none.
struct GiLightSample {
    dir: vec3<f32>,
    distance: f32,
    irradiance: vec3<f32>,
}

fn gi_light_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// One of the gathered lights at frame point `p` with normal `n`, chosen in
// proportion to its unshadowed irradiance there (as deferred lighting
// attenuates it), by uniform numbers from `seed` (`gi_hash`).
fn gi_local_light(p: vec3<f32>, n: vec3<f32>, seed: u32) -> GiLightSample {
    var total = 0.0;
    var chosen = vec4<f32>(0.0);
    var chosen_e = vec3<f32>(0.0);
    for (var i = 0u; i < local_lights.count.x; i++) {
        let light = local_lights.lights[i];
        let to_light = light.position_range.xyz - p;
        let d = length(to_light);
        if d >= light.position_range.w || d < 1e-4 { continue; }
        let l = to_light / d;
        let cosine = dot(n, l);
        if cosine <= 0.0 { continue; }
        let x = d / light.position_range.w;
        var atten = max(0.0, 1.0 - x * x * x * x) / (d * d + 0.0001);
        if light.color_type.w == 2.0 {
            atten *= smoothstep(light.direction_outer.w, light.inner.x, dot(-l, light.direction_outer.xyz));
        }
        let e = light.color_type.rgb * atten * cosine;
        let w = gi_light_luminance(e);
        if w <= 0.0 { continue; }
        total += w;
        if f32(gi_hash(seed ^ (i * 0x9e3779b9u)) >> 8u) / 16777216.0 * total < w {
            chosen = vec4<f32>(l, d);
            chosen_e = e;
        }
    }
    if total <= 0.0 { return GiLightSample(vec3<f32>(0.0, 1.0, 0.0), 0.0, vec3<f32>(0.0)); }
    return GiLightSample(chosen.xyz, chosen.w, chosen_e * (total / gi_light_luminance(chosen_e)));
}
