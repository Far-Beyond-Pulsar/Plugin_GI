//! GI knowing no kind: its node built over an analytic tracer offered
//! through the frame's ray streams (`common::analytic_tracer`, a compute
//! pass that intersects rays with planes, boxes and sheets), and the
//! probes' light against what it must be. The frame's origin is off the
//! world's, so the rays are checked in the frame's space.

mod common;

use common::{gpu, probe, updated, Harness, Setup, Shape};
use glam::{DVec3, Vec3};
use helio_pass_gi::MAX_UPDATES;

const PI: f32 = std::f32::consts::PI;
/// Base colour 128/255 (RGB8 in a surface).
const HALF: f32 = 128.0 / 255.0;

fn harness(setup: Setup, shapes: Vec<Shape>) -> Option<Harness> {
    Some(Harness::new(gpu()?, setup, shapes))
}

/// Renders frames until `done` holds for the probes' records, under a
/// wall-clock guard: a probe whose hits the radiance cache cannot light yet
/// holds its update (cells are lit from the frame after a hit first marks
/// them), so a test waits for the probes it checks, not for a count of
/// frames.
fn frames_until(h: &mut Harness, what: &str, done: impl Fn(&[u32]) -> bool) -> Vec<u32> {
    let start = std::time::Instant::now();
    loop {
        h.frames(1);
        let words = h.probes();
        if done(&words) {
            return words;
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(300),
            "{what}: not within the guard"
        );
    }
}

#[test]
fn with_nothing_to_trace_gi_builds_nothing() {
    let Some(gpu) = gpu() else { return };
    let setup = Setup {
        traced: false,
        ..Default::default()
    };
    let shapes = Default::default();
    let stats = helio_core::FrameStats::new(&gpu.device);
    let camera = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 512,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    for kind in ["helio.pass.gi", "helio.pass.gi_resolve"] {
        let (graph, _) = common::build(gpu, &setup, &shapes, &stats, &camera, kind, false);
        assert_eq!(common::pass_names(&graph), Vec::<String>::new(), "{kind}");
    }
}

/// The node is GI's stream's generators (the probes' rays, the radiance
/// cache's), its tracers, the hits' light (the cache's light batch's
/// generator), the light samplers (the lights': the grid built once, then
/// the batch), the shadow stream's generator, its tracers, the probes'
/// integration and the cache's fold, in that order.
#[test]
fn the_node_traces_both_streams_between_its_passes() {
    let Some(h) = harness(Setup::default(), Vec::new()) else {
        return;
    };
    assert_eq!(
        common::pass_names(&h.graph),
        [
            "GI",
            "RadianceCache",
            "AnalyticTracer",
            "GILight",
            "LightSampleGrid",
            "LightSamples",
            "GIShadow",
            "AnalyticTracer",
            "GIIntegrate",
            "RadianceCacheFold",
            "Capture",
            "CacheReader"
        ]
    );
}

/// A closed room whose walls glow with radiance 1 (and reflect nothing)
/// round the whole clipmap, a white furnace: every updated probe's
/// irradiance / pi is 1 in every direction.
#[test]
fn a_glowing_room_lights_probes_evenly() {
    let setup = Setup::default();
    let eye = setup.world_eye();
    let room = Shape::room(eye - 2000.0, eye + 2000.0, 0.0).glowing(1.0);
    let Some(mut h) = harness(setup, vec![room]) else {
        return;
    };
    h.frames(4);
    let words = h.probes();
    let probes = updated(&words, &h.clipmap(), |_| true);
    for &(index, _) in &probes {
        let (_, state) = probe(&words, index, Vec3::Y);
        assert_eq!(state & 1, 1, "probe {index} inactive");
        for d in [
            Vec3::X,
            Vec3::NEG_Y,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(0.6, -0.48, 0.64),
        ] {
            let e = probe(&words, index, d).0;
            assert!(
                (e - Vec3::ONE).abs().max_element() < 0.02,
                "probe {index} toward {d}: {e}"
            );
        }
    }
    assert!(
        probes.len() >= 3 * MAX_UPDATES as usize,
        "{} probes updated",
        probes.len()
    );
}

/// A scene's start: a grey room round the clipmap lit by a point light, so
/// every probe hit reflects light the radiance cache has to gather first. A
/// probe's first update holds (its hits made new cells), goes on the held
/// list and comes back to the schedule's front once the list is read back,
/// a few frames later, when the cache (new cells first) has lit those cells:
/// the probes the first three frames' updates hold are updated before their
/// turn would come round again, the finest probes' period.
#[test]
fn held_probes_come_back_before_their_turn() {
    let mut setup = Setup::default();
    let eye = setup.world_eye();
    let light = (eye + DVec3::new(0.0, 1.0, 0.0) - setup.origin).as_vec3();
    setup.lights = vec![common::point_light(light, 4000.0, 3000.0)];
    let room = Shape::room(eye - 300.0, eye + 300.0, HALF);
    let Some(mut h) = harness(setup, vec![room]) else {
        return;
    };
    let start = std::time::Instant::now();
    let mut frames = 0;
    loop {
        h.frames(1);
        frames += 1;
        let words = h.probes();
        if updated(&words, &h.clipmap(), |_| true).len() >= 3 * MAX_UPDATES as usize {
            break;
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(300),
            "probes not lit within the guard"
        );
    }
    let period = helio_pass_gi::cache::probe_period() as usize;
    println!("three frames' probes updated after {frames} frames (their turn: {period})");
    assert!(frames < period, "{frames} frames, period {period}");
}

/// A sky of radiance 1 over black ground: a probe over it sees the sky above
/// and nothing below (irradiance / pi 1 facing up, half of it sideways, none
/// facing down: L1 harmonics give a hemisphere's step exactly facing along
/// it). Probes under the ground are inside it: inactive, dark.
#[test]
fn the_sky_lights_probes_from_above() {
    let setup = Setup {
        sky: 1.0,
        ..Default::default()
    };
    let Some(mut h) = harness(setup, vec![Shape::ground(0.0, 0.0)]) else {
        return;
    };
    h.frames(4);
    let words = h.probes();
    let clipmap = h.clipmap();
    let above = updated(&words, &clipmap, |at| at.y >= 0.5);
    for &(index, at) in &above {
        let up = probe(&words, index, Vec3::Y).0;
        let side = probe(&words, index, Vec3::X).0;
        let down = probe(&words, index, Vec3::NEG_Y).0;
        assert!(
            (up - Vec3::ONE).abs().max_element() < 0.05,
            "probe at {at}: up {up}"
        );
        assert!(
            (side - Vec3::splat(0.5)).abs().max_element() < 0.08,
            "probe at {at}: side {side}"
        );
        assert!(down.max_element() < 0.05, "probe at {at}: down {down}");
    }
    let under = updated(&words, &clipmap, |at| at.y < 0.0);
    for &(index, at) in &under {
        let (e, state) = probe(&words, index, Vec3::Y);
        assert_eq!(state & 1, 0, "probe at {at} active");
        assert!(e.max_element() < 1e-3, "probe at {at} inside: {e}");
    }
    assert!(
        above.len() > 1000 && under.len() > 1000,
        "{} above, {} under",
        above.len(),
        under.len()
    );
}

/// With an atmosphere, a probe ray that meets nothing takes the
/// atmosphere's sky (here black) wherever the sky publishes it, not the
/// flat ambient (radiance 1 here): every probe over black ground stays dark.
#[test]
fn a_miss_takes_the_atmosphere_s_sky() {
    let setup = Setup {
        sky: 1.0,
        black_atmosphere: true,
        ..Default::default()
    };
    let Some(mut h) = harness(setup, vec![Shape::ground(0.0, 0.0)]) else {
        return;
    };
    h.frames(4);
    let words = h.probes();
    let above = updated(&words, &h.clipmap(), |at| at.y >= 0.5);
    for &(index, at) in &above {
        let up = probe(&words, index, Vec3::Y).0;
        assert!(up.max_element() < 0.01, "probe at {at}: up {up}");
    }
    assert!(above.len() > 1000, "{} above", above.len());
}

/// No sky; the sun (illuminance 2) straight down on flat ground of base
/// colour 128/255: a probe over it sees the ground's radiance a 2 / pi below
/// (its shadow ray to the sun clear) and nothing above.
#[test]
fn the_sun_on_the_ground_lights_probes_from_below() {
    let setup = Setup {
        sun: Some(2.0),
        ..Default::default()
    };
    let Some(mut h) = harness(setup, vec![Shape::ground(0.0, HALF)]) else {
        return;
    };
    let clipmap = h.clipmap();
    let words = frames_until(&mut h, "probes over the ground update", |words| {
        updated(words, &clipmap, |at| at.y >= 0.5).len() > 1000
    });
    let below = HALF * 2.0 / PI;
    let above = updated(&words, &h.clipmap(), |at| at.y >= 0.5);
    for &(index, at) in &above {
        let down = probe(&words, index, Vec3::NEG_Y).0;
        let side = probe(&words, index, Vec3::X).0;
        let up = probe(&words, index, Vec3::Y).0;
        assert!(
            (down - Vec3::splat(below)).abs().max_element() < 0.05 * below,
            "probe at {at}: down {down}, want {below}"
        );
        assert!(
            (side - Vec3::splat(below * 0.5)).abs().max_element() < 0.08 * below,
            "probe at {at}: side {side}"
        );
        assert!(up.max_element() < 0.05 * below, "probe at {at}: up {up}");
    }
    assert!(above.len() > 1000, "{} above", above.len());
}

/// The irradiance / pi facing down of the probe at world `at` (a probe of
/// the finest level), once it has taken `updates` updates: its first may
/// come while the cells its rays meet are new, answered from cells 8 times
/// wider (`radiance_cache::ANCESTOR`), which a shadow's edge smears; the
/// updates after read their own.
fn down_at(h: &mut Harness, at: DVec3, updates: u32) -> Vec3 {
    let clipmap = h.clipmap();
    let index = helio_pass_gi::Clipmap::index(0, (at / clipmap.finest).round().as_i64vec3());
    let words = frames_until(h, "the probe updates", |words| {
        probe(words, index, Vec3::NEG_Y).1 >> 8 >= updates
    });
    let (e, state) = probe(&words, index, Vec3::NEG_Y);
    assert!(state & 1 == 1, "probe at {at}: {state:x}");
    e
}

/// The sun straight down on lit ground, and a black roof (12 m square, 3 m
/// up) over a probe a metre up: the ground under the roof is in its shadow,
/// so the probe sees the lit ground only past it (2% of its view down
/// weighted by the cosine: a square of half-width 6 m a metre down holds
/// about 98%), where a probe in the open sees it all. The roof reflects
/// nothing and flat ground sees no ground, so nothing is bounced in.
#[test]
fn a_roof_shadows_the_sun() {
    let setup = Setup {
        sun: Some(2.0),
        ..Default::default()
    };
    let eye = setup.world_eye().floor();
    let roof = Shape::solid(
        DVec3::new(eye.x - 6.0, 3.0, eye.z - 6.0),
        DVec3::new(eye.x + 6.0, 3.5, eye.z + 6.0),
        0.0,
    );
    let probe_at = DVec3::new(eye.x, 1.0, eye.z);
    let Some(mut open) = harness(setup.clone(), vec![Shape::ground(0.0, HALF)]) else {
        return;
    };
    let lit = down_at(&mut open, probe_at, 3);
    let Some(mut covered) = harness(setup, vec![Shape::ground(0.0, HALF), roof]) else {
        return;
    };
    let shaded = down_at(&mut covered, probe_at, 3);
    let below = HALF * 2.0 / PI;
    assert!(
        (lit - Vec3::splat(below)).abs().max_element() < 0.05 * below,
        "{lit}"
    );
    assert!(
        shaded.max_element() < 0.1 * lit.min_element(),
        "{shaded} under the roof, {lit} open"
    );
}

/// A point light 3 m over lit ground and a table (6 m square, 1.5 to 1.7 m
/// up) between it and a probe a metre up: the table's shadow covers nearly
/// all of the ground the probe sees.
#[test]
fn a_table_shadows_a_point_light() {
    let base = Setup::default();
    let eye = base.world_eye().floor();
    let light_at = (DVec3::new(eye.x, 3.0, eye.z) - base.origin).as_vec3();
    let setup = Setup {
        lights: vec![common::point_light(light_at, 40.0, 30.0)],
        ..base.clone()
    };
    let table = Shape::solid(
        DVec3::new(eye.x - 3.0, 1.5, eye.z - 3.0),
        DVec3::new(eye.x + 3.0, 1.7, eye.z + 3.0),
        0.0,
    );
    let probe_at = DVec3::new(eye.x, 1.0, eye.z);
    let Some(mut open) = harness(setup.clone(), vec![Shape::ground(0.0, HALF)]) else {
        return;
    };
    let lit = down_at(&mut open, probe_at, 3);
    let Some(mut covered) = harness(setup, vec![Shape::ground(0.0, HALF), table]) else {
        return;
    };
    let shaded = down_at(&mut covered, probe_at, 3);
    assert!(
        lit.min_element() > 0.1,
        "the light lights the ground: {lit}"
    );
    assert!(
        shaded.max_element() < 0.15 * lit.min_element(),
        "{shaded} under the table, {lit} open"
    );
}

/// No sky, no sun; a floor of base colour 128/255 glowing with radiance 2:
/// a probe over it sees that radiance below and nothing above.
#[test]
fn glowing_ground_lights_probes() {
    let Some(mut h) = harness(
        Setup::default(),
        vec![Shape::ground(0.0, HALF).glowing(2.0)],
    ) else {
        return;
    };
    let clipmap = h.clipmap();
    let words = frames_until(&mut h, "probes over the floor update", |words| {
        updated(words, &clipmap, |at| at.y >= 0.5).len() > 1000
    });
    let over = updated(&words, &h.clipmap(), |at| at.y >= 0.5);
    for &(index, at) in &over {
        let down = probe(&words, index, Vec3::NEG_Y).0;
        let up = probe(&words, index, Vec3::Y).0;
        assert!(
            (down - Vec3::splat(2.0)).abs().max_element() < 0.05 * 2.0,
            "probe at {at}: down {down}"
        );
        assert!(up.max_element() < 0.1, "probe at {at}: up {up}");
    }
    assert!(over.len() > 1000, "{} probes over the floor", over.len());
}

/// No sky; the sun (illuminance 2) straight down on a canopy at height 4.5
/// that lets a leaf's share of light through (77/255; base colour 128/255).
/// A probe under it sees the canopy's underside give that share of the sun
/// on its top: radiance 77/255 * 128/255 * 2 / pi above, nothing below.
#[test]
fn light_through_a_canopy_lights_probes_under_it() {
    let canopy = Shape::sheet(4.5, HALF, 77.0 / 255.0);
    let setup = Setup {
        sun: Some(2.0),
        ..Default::default()
    };
    let Some(mut h) = harness(setup, vec![canopy]) else {
        return;
    };
    let clipmap = h.clipmap();
    let words = frames_until(&mut h, "probes under the canopy update", |words| {
        updated(words, &clipmap, |at| (0.0..4.0).contains(&at.y)).len() > 500
    });
    let through = 77.0 / 255.0;
    let above = through * HALF * 2.0 / PI;
    // The underside also passes on the light under it (the probes' own,
    // their multi-bounce term), a gain of its transmission times base
    // colour: up to the series' sum, as far as the probes have taken it in.
    let most = above / (1.0 - through * HALF);
    let under = updated(&words, &h.clipmap(), |at| (0.0..4.0).contains(&at.y));
    for &(index, at) in &under {
        let up = probe(&words, index, Vec3::Y).0;
        let down = probe(&words, index, Vec3::NEG_Y).0;
        // A few percent either way for those just under it (a spacing
        // away), which leak a little of their own through the visibility
        // test (DDGI's thin occluders).
        assert!(
            up.min_element() > 0.95 * above && up.max_element() < 1.05 * most,
            "probe at {at}: up {up}, want {above} to {most}"
        );
        assert!(
            down.max_element() < 0.06 * above,
            "probe at {at}: down {down}"
        );
    }
    assert!(under.len() > 500, "{} under the canopy", under.len());
}

/// A sky of radiance 1 and a solid block: the probes inside the block meet
/// its inside on every ray, so they are inactive and take no light (an
/// inside gives nothing), while those outside it see the sky.
#[test]
fn inside_hits_give_nothing() {
    let setup = Setup {
        sky: 1.0,
        ..Default::default()
    };
    let eye = setup.world_eye().floor();
    let block = Shape::solid(
        eye + DVec3::new(2.5, -3.5, -3.5),
        eye + DVec3::new(9.5, 3.5, 3.5),
        0.5,
    );
    let Some(mut h) = harness(setup, vec![block]) else {
        return;
    };
    // Every probe of the finest level updated: 13824 at 680 a frame.
    h.frames(22);
    let words = h.probes();
    let lo = block.lo;
    let hi = block.hi;
    let inside = updated(&words, &h.clipmap(), |at| {
        at.cmpgt(lo + 0.01).all() && at.cmplt(hi - 0.01).all()
    });
    for &(index, at) in &inside {
        let (e, state) = probe(&words, index, Vec3::Y);
        assert_eq!(state & 1, 0, "probe at {at} active");
        assert!(e.max_element() < 1e-3, "probe at {at}: {e}");
    }
    assert!(!inside.is_empty());
}

/// A scene loading: the tracer has nothing to answer with yet, so every ray
/// misses. The probes hold (none takes the sky everywhere, a room's probes
/// would then glow with it for seconds) until geometry answers; then they
/// converge; and with the geometry gone again they hold what they have.
#[test]
fn probes_hold_until_geometry_answers() {
    let setup = Setup {
        sky: 5.0,
        ..Default::default()
    };
    let eye = setup.world_eye();
    let Some(mut h) = harness(setup, Vec::new()) else {
        return;
    };
    h.frames(3);
    let words = h.probes();
    assert!(
        updated(&words, &h.clipmap(), |_| true).is_empty(),
        "a probe took the sky before anything answered"
    );
    *h.shapes.lock().unwrap() = vec![Shape::room(eye - 2000.0, eye + 2000.0, 0.0).glowing(1.0)];
    h.frames(3);
    let words = h.probes();
    let lit = updated(&words, &h.clipmap(), |_| true);
    assert!(lit.len() >= 3 * MAX_UPDATES as usize);
    for &(index, _) in &lit {
        let e = probe(&words, index, Vec3::X).0;
        assert!(
            (e - Vec3::ONE).abs().max_element() < 0.02,
            "probe {index}: {e}"
        );
    }
    h.shapes.lock().unwrap().clear();
    h.frames(3);
    assert_eq!(h.probes(), words, "the probes held while nothing answered");
}

/// A tracer that is late (its kernels still compiling) answers short: the
/// probes hold while it does, though another tracer answers (a room here),
/// so they do not learn a world without its geometry; once it answers in
/// full they converge.
#[test]
fn probes_hold_while_a_tracer_answers_short() {
    let late = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let setup = Setup {
        sky: 5.0,
        late: Some(late.clone()),
        ..Default::default()
    };
    let eye = setup.world_eye();
    let room = Shape::room(eye - 2000.0, eye + 2000.0, 0.0).glowing(1.0);
    let Some(mut h) = harness(setup, vec![room]) else {
        return;
    };
    h.frames(3);
    assert!(
        updated(&h.probes(), &h.clipmap(), |_| true).is_empty(),
        "a probe took rays a late tracer answered short"
    );
    late.store(false, std::sync::atomic::Ordering::Relaxed);
    h.frames(3);
    let words = h.probes();
    let lit = updated(&words, &h.clipmap(), |_| true);
    assert!(lit.len() >= 3 * MAX_UPDATES as usize);
    for &(index, _) in &lit {
        let e = probe(&words, index, Vec3::X).0;
        assert!(
            (e - Vec3::ONE).abs().max_element() < 0.02,
            "probe {index}: {e}"
        );
    }
}

/// Each stream counts its rays into the frame's statistics: GI's stream
/// every probe ray and the cache's (`rays.radiance_cache`, counted apart),
/// the shadow stream the cache rays' shadow rays (its cells' and their
/// hits', over sunlit ground: at most two a cache ray). The cache lights
/// cells from the frame after a reader first marked them.
#[test]
fn each_stream_counts_its_rays() {
    let setup = Setup {
        sun: Some(2.0),
        ..Default::default()
    };
    let Some(mut h) = harness(setup, vec![Shape::ground(0.0, HALF)]) else {
        return;
    };
    let start = std::time::Instant::now();
    let latest = loop {
        let stats = h.frame_stats();
        if stats.get("rays.radiance_cache").unwrap_or(0) > 0 {
            break stats;
        }
        assert!(start.elapsed().as_secs() < 120, "the cache traced no rays");
    };
    let rays = latest.get("rays.gi").expect("GI's stream counts");
    let cache = latest.get("rays.radiance_cache").unwrap();
    let shadows = latest
        .get("rays.radiance_cache_shadow")
        .expect("the shadow stream counts");
    assert_eq!(rays, u64::from(helio_pass_gi::MAX_RAYS) + cache);
    assert_eq!(cache % u64::from(helio_pass_gi::cache::RAYS_PER_CELL), 0);
    // The ground's cells see the sun (a shadow ray each), their rays up
    // miss (none), their rays' hits on the ground from below are insides.
    assert!(
        shadows >= cache / 2 && shadows <= 2 * cache,
        "{shadows} shadow rays of {cache} cache rays"
    );
}

/// The resolve at pixels of a furnace's probes: every point the clipmap
/// holds gets irradiance / pi 1, held fully.
#[test]
fn the_resolve_reads_the_probes_at_pixels() {
    let setup = Setup {
        eye: Vec3::ZERO,
        ..Default::default()
    };
    let eye = setup.world_eye();
    let Some(mut h) = harness(
        setup,
        vec![Shape::room(eye - 2000.0, eye + 2000.0, 0.0).glowing(1.0)],
    ) else {
        return;
    };
    // Every probe updated: 13824 a level at 340 a frame.
    h.frames(42);
    let gpu = h.gpu;
    let size = [16u32, 16];
    let texture = |format, label| {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    };
    let (normal, depth) = (
        texture(wgpu::TextureFormat::Rgba16Float, "normal"),
        texture(wgpu::TextureFormat::Depth32Float, "depth"),
    );
    let (normal_view, depth_view) = (
        normal.create_view(&Default::default()),
        depth.create_view(&Default::default()),
    );
    let view = glam::camera::rh::view::look_to_mat4(
        Vec3::ZERO,
        Vec3::new(0.2, -0.3, -1.0).normalize(),
        Vec3::Y,
    );
    let proj = helio_core::depth::perspective(1.0, 1.0, 0.1);
    let camera = helio_core::GpuCameraUniforms::new(
        view,
        proj,
        Vec3::ZERO,
        0.1,
        1000.0,
        0,
        [0.0; 2],
        proj * view,
    );
    use wgpu::util::DeviceExt;
    let camera = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::bytes_of(&camera),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let mut resolve = helio_pass_gi::GiResolvePass::new(&gpu.device);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    // Points about 4 m away (reversed depth near / distance), facing up.
    drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &normal_view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.0,
                    g: 1.0,
                    b: 0.0,
                    a: 0.0,
                }),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0.1 / 4.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    }));
    let mut cmds = helio_core::CommandRecorder::from_encoder(&mut encoder);
    let [frame_buffer, probes] = h.buffers();
    resolve.record(
        &gpu.device,
        &gpu.queue,
        &mut cmds,
        [&frame_buffer, &probes, &camera],
        &normal_view,
        &depth_view,
        size,
    );
    drop(cmds);
    gpu.queue.submit([encoder.finish()]);
    let words = common::read(&gpu.device, &gpu.queue, resolve.irradiance().unwrap());
    for (p, w) in words.chunks(2).enumerate() {
        let (rg, ba) = (common::half2(w[0]), common::half2(w[1]));
        let e = Vec3::new(rg[0], rg[1], ba[0]);
        assert!(
            (e - Vec3::ONE).abs().max_element() < 0.03 && (ba[1] - 1.0).abs() < 1e-3,
            "pixel {p}: {e} held {}",
            ba[1]
        );
    }
}
