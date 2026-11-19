//!use frame_ray_stream
//!use frame_ray_surface
//!use frame_surface
// A ray tracer of analytic shapes (the GI tests' stand-in for a kind): a
// nearest-hit stream's rows get the nearest of what they hold and these
// shapes' hit, with its surface; an occlusion stream's what the shapes let
// through. Shapes are in the frame's space.

// `kind` in `a.w`: 0 the ground (solid below `a.y`, its top facing +y;
// a ray from inside it meets its inside at once), 1 a
// solid box (`a.xyz` to `b.xyz`), 2 a sheet (two-sided, at height `a.y`
// over `a.xz` to `b.xz`, letting `b.w` of the light through), 3 a room (a
// box seen from inside: its walls face in). Base colour and emission.
struct Shape {
    a: vec4<f32>,
    b: vec4<f32>,
    base: vec4<f32>,
    emissive: vec4<f32>,
}

// x: the tracer's id, y: 1 for an occlusion stream, z: shapes.
struct TracerInfo {
    info: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> header: RayStreamHeader;
@group(0) @binding(1) var<storage, read> rays: array<SceneRay>;
@group(0) @binding(2) var<storage, read_write> hits: array<SceneHit>;
@group(0) @binding(3) var<storage, read_write> surfaces: array<SceneSurface>;
@group(0) @binding(4) var<storage, read> shapes: array<Shape>;
@group(0) @binding(5) var<uniform> tracer: TracerInfo;

struct Meet {
    t: f32,
    n: vec3<f32>,
    inside: bool,
}

const NONE: f32 = -1.0;

fn none() -> Meet {
    return Meet(NONE, vec3<f32>(0.0, 1.0, 0.0), false);
}

// The slabs of the box `lo`..`hi` along the ray: entry, exit and the
// entry's and exit's outward normals.
fn slabs(o: vec3<f32>, d: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> array<vec4<f32>, 2> {
    let inv = 1.0 / d;
    let t0 = (lo - o) * inv;
    let t1 = (hi - o) * inv;
    let near = min(t0, t1);
    let far = max(t0, t1);
    let enter = max(near.x, max(near.y, near.z));
    let exit = min(far.x, min(far.y, far.z));
    var n_in = vec3<f32>(0.0);
    var n_out = vec3<f32>(0.0);
    if enter == near.x { n_in = vec3<f32>(-sign(d.x), 0.0, 0.0); } else if enter == near.y { n_in = vec3<f32>(0.0, -sign(d.y), 0.0); } else { n_in = vec3<f32>(0.0, 0.0, -sign(d.z)); }
    if exit == far.x { n_out = vec3<f32>(sign(d.x), 0.0, 0.0); } else if exit == far.y { n_out = vec3<f32>(0.0, sign(d.y), 0.0); } else { n_out = vec3<f32>(0.0, 0.0, sign(d.z)); }
    return array<vec4<f32>, 2>(vec4<f32>(n_in, enter), vec4<f32>(n_out, exit));
}

fn meet(s: Shape, o: vec3<f32>, d: vec3<f32>) -> Meet {
    let kind = u32(s.a.w);
    if kind == 0u || kind == 2u {
        // Inside the ground: its inside at once, whichever way.
        if kind == 0u && o.y < s.a.y { return Meet(1e-3, vec3<f32>(0.0, 1.0, 0.0), true); }
        if abs(d.y) < 1e-12 { return none(); }
        let t = (s.a.y - o.y) / d.y;
        if t <= 0.0 { return none(); }
        if kind == 0u {
            return Meet(t, vec3<f32>(0.0, 1.0, 0.0), false);
        }
        let at = o + d * t;
        if at.x < s.a.x || at.x > s.b.x || at.z < s.a.z || at.z > s.b.z { return none(); }
        return Meet(t, vec3<f32>(0.0, -sign(d.y), 0.0), false);
    }
    let sl = slabs(o, d, s.a.xyz, s.b.xyz);
    let enter = sl[0].w;
    let exit = sl[1].w;
    if enter > exit || exit <= 0.0 { return none(); }
    if enter > 0.0 {
        // From outside: a box's face; a room's back.
        return Meet(enter, sl[0].xyz, kind == 3u);
    }
    // From inside: a room's wall facing in; a box's back.
    return Meet(exit, -sl[1].xyz, kind == 1u);
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= header.count { return; }
    let ray = rays[id.x];
    if ray.t_max <= 0.0 { return; }
    if tracer.info.y == 1u {
        var through = scene_occlusion_through(hits[id.x]);
        for (var k = 0u; k < tracer.info.z; k++) {
            let s = shapes[k];
            let m = meet(s, ray.origin, ray.dir);
            if m.t < 0.0 || m.t > ray.t_max { continue; }
            if u32(s.a.w) == 2u {
                through *= s.b.w * s.base.rgb;
            } else {
                through = vec3<f32>(0.0);
            }
        }
        hits[id.x] = scene_occlusion(through);
        return;
    }
    let reach = scene_hit_reach(ray, hits[id.x]);
    var best = none();
    var at = 0u;
    for (var k = 0u; k < tracer.info.z; k++) {
        let m = meet(shapes[k], ray.origin, ray.dir);
        if m.t < 0.0 || m.t >= reach { continue; }
        if best.t < 0.0 || m.t < best.t {
            best = m;
            at = k;
        }
    }
    if best.t < 0.0 { return; }
    let s = shapes[at];
    hits[id.x] = scene_nearest(tracer.info.x, best.t, best.inside, at, 0u, 0u);
    let transmission = select(0.0, s.b.w, u32(s.a.w) == 2u);
    let surface = surface_data(s.base.rgb, best.n, 1.0, 0.0, s.emissive.rgb);
    surfaces[id.x] = scene_surface_pack(surface_with_transmission(surface, s.base.rgb * transmission), best.n);
}
