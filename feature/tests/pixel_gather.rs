//! The pixels' gather (`helio_pass_gi::gather`) in a mesh scene's default
//! graph, traced by the meshes' tracer (ray queries): a glowing closed box
//! whose irradiance is known everywhere, the reservoirs against one plain
//! ray a pixel, the confidence a light switched off leaves, and the near
//! field's reach. Every wait is on a signal (the image settling, a
//! confidence dropping) under a wall-clock guard, never a count of frames.

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::Vec3;
use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    RendererBuilder, RendererConfig,
};
use helio_frame_api::denoise::TraceBlock;
use helio_mesh_host::ray_tracing::SceneDbRayTracing;
use helio_pass_gi::{GiGatherPass, GiGatherSignalPass, GiQuality, GiResolvePass, GiSettings};

mod mesh_scene;
use mesh_scene::{
    adapter, furnace, indirect, read, read_texture, room, room_light, scene_db, spawn_box,
    spawn_glowing_box,
};

const SIZE: u32 = 64;
/// How long any wait may take.
const GUARD: Duration = Duration::from_secs(240);

struct Run {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    scene: pulsar_scenedb::SceneDb,
    renderer: helio::Renderer,
    rays: SceneDbRayTracing,
    target: wgpu::TextureView,
    camera: Camera,
    frames: u64,
}

impl Run {
    /// A renderer of `scene` (built by `fill`) with GI at `quality`, or
    /// none without ray queries.
    fn new(
        quality: GiQuality,
        reuse: bool,
        camera: Camera,
        fill: impl FnOnce(&mut pulsar_scenedb::World),
    ) -> Option<Self> {
        let adapter = adapter()?;
        let features = required_wgpu_features(helio::usable_adapter_features(&adapter));
        if !features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
            eprintln!("no ray queries: skipped");
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(features),
            ..Default::default()
        }))
        .unwrap();
        let (device, queue) = (Arc::new(device), Arc::new(queue));
        let mut scene = scene_db(&device, &queue);
        fill(&mut scene.world);
        let config = RendererConfig::new(SIZE, SIZE, wgpu::TextureFormat::Rgba8Unorm).with_feature(
            GiSettings {
                quality,
                pixel_reuse: reuse,
                ..Default::default()
            },
        );
        let renderer =
            RendererBuilder::new(config.clone(), scene.world.gpu_mirror().cloned().unwrap())
                .with_graph(helio_mesh_host::presets::with(
                    helio_presets::Preset::Deferred,
                    &helio_feature_api::FeatureSet::new().with(helio_pass_gi::GiFeature),
                ))
                .with_external_device()
                .build(
                    device.clone(),
                    queue.clone(),
                    SIZE,
                    SIZE,
                    config.surface_format,
                );
        let target = device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: config.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let rays = SceneDbRayTracing::partial(device.clone(), queue.clone());
        Some(Self {
            device,
            queue,
            scene,
            renderer,
            rays,
            target,
            camera,
            frames: 0,
        })
    }

    fn frame(&mut self) {
        self.scene.world.flush_gpu_mirror(&self.queue);
        self.rays.prepare(&self.scene.world).unwrap();
        self.rays.publish(&mut self.renderer);
        self.renderer.render(&self.camera, &self.target).unwrap();
        self.frames += 1;
    }

    /// The frame's indirect diffuse, a pixel each (none while the renderer
    /// still compiles its graph's pipelines and shows its status screen).
    fn indirect(&self) -> Option<Vec<[f32; 4]>> {
        let pass = self.renderer.find_pass::<GiResolvePass>()?;
        Some(indirect(&read(
            &self.device,
            &self.queue,
            pass.irradiance()?,
        )))
    }

    fn near_field(&self) -> Option<Vec<f32>> {
        let pass = self.renderer.find_pass::<GiGatherPass>()?;
        Some(read_texture(&self.device, &self.queue, pass.near_field()?))
    }

    fn signal(&self) -> Option<[Vec<f32>; 2]> {
        let pass = self.renderer.find_pass::<GiGatherSignalPass>()?;
        let [a, b] = pass.signal_textures()?;
        Some([
            read_texture(&self.device, &self.queue, a),
            read_texture(&self.device, &self.queue, b),
        ])
    }

    /// Renders until `ready` holds, under the guard.
    fn until<T>(&mut self, what: &str, mut ready: impl FnMut(&mut Self) -> Option<T>) -> T {
        let start = Instant::now();
        loop {
            assert!(start.elapsed() < GUARD, "{what}: not within {GUARD:?}");
            self.frame();
            if let Some(value) = ready(self) {
                return value;
            }
        }
    }

    /// Renders until the mean luminance of the indirect diffuse moves by
    /// less than 0.5% over 16 frames (the cache and the gather settled).
    fn settle(&mut self, what: &str) -> f32 {
        let start = Instant::now();
        let mut history: Vec<f32> = Vec::new();
        loop {
            assert!(
                start.elapsed() < GUARD,
                "{what}: did not settle ({history:?})"
            );
            self.frame();
            let Some(image) = self.indirect() else {
                continue;
            };
            let lit: Vec<f32> = image
                .iter()
                .filter(|p| p[3] > 0.0)
                .map(|p| p[0] + p[1] + p[2])
                .collect();
            let mean = lit.iter().sum::<f32>() / lit.len().max(1) as f32;
            history.push(mean);
            if history.len() > 16 {
                let last = &history[history.len() - 17..];
                let (lo, hi) = last
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
                if mean > 0.0 && hi - lo <= 0.005 * mean {
                    eprintln!("{what}: settled after {} frames at {mean}", self.frames);
                    return mean;
                }
            }
        }
    }

    /// Renders until the mean luminance of the indirect diffuse over 64
    /// frames moves by less than 0.5% from the 64 before (a scene whose
    /// frames are noisy and whose cache warms slowly, one plain ray a block
    /// in a mostly dark room lit by bounces, settles on its means rather than
    /// on single frames).
    fn settle_means(&mut self, what: &str) -> f32 {
        let start = Instant::now();
        let mut history: Vec<f32> = Vec::new();
        loop {
            assert!(start.elapsed() < GUARD, "{what}: did not settle");
            self.frame();
            let Some(image) = self.indirect() else {
                continue;
            };
            let lit: Vec<f32> = image
                .iter()
                .filter(|p| p[3] > 0.0)
                .map(|p| p[0] + p[1] + p[2])
                .collect();
            history.push(lit.iter().sum::<f32>() / lit.len().max(1) as f32);
            if history.len() >= 128 {
                let n = history.len();
                let last = history[n - 64..].iter().sum::<f32>() / 64.0;
                let before = history[n - 128..n - 64].iter().sum::<f32>() / 64.0;
                if last > 0.0 && (last - before).abs() <= 0.005 * last {
                    eprintln!("{what}: settled after {} frames at {last}", self.frames);
                    return last;
                }
            }
        }
    }

    /// The per-pixel mean of the indirect diffuse over `frames` frames.
    fn mean(&mut self, frames: usize) -> Vec<[f32; 4]> {
        // The internal resolution's pixels (the renderer's render scale).
        let mut sum = Vec::new();
        for _ in 0..frames {
            self.frame();
            let image = self.indirect().unwrap();
            sum.resize(image.len(), [0.0f32; 4]);
            for (s, p) in sum.iter_mut().zip(image) {
                for c in 0..4 {
                    s[c] += p[c] / frames as f32;
                }
            }
        }
        sum
    }
}

fn furnace_camera() -> Camera {
    Camera::perspective_look_at(
        Vec3::new(-3.5, 1.2, 3.5),
        Vec3::new(4.0, 0.0, -4.0),
        Vec3::Y,
        1.2,
        1.0,
        0.1,
        100.0,
    )
}

fn room_camera() -> Camera {
    Camera::perspective_look_at(
        Vec3::new(-2.0, 1.7, 3.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::Y,
        1.0,
        1.0,
        0.1,
        100.0,
    )
}

/// A furnace (a closed box of albedo 1/2 glowing with radiance 1): the light
/// arriving anywhere is pi times the radiance everywhere, 1 / (1 - 1/2) = 2,
/// so the indirect diffuse (irradiance / pi) is 2 at every pixel, at every
/// quality. Checked where the near field leaves the block whole (a block
/// with an occluded pixel moves light between its pixels by the AO
/// approximation, reported).
#[test]
fn a_glowing_box_s_indirect_diffuse_is_its_radiance_at_every_quality() {
    for quality in [
        GiQuality::Probes,
        GiQuality::Pixels(TraceBlock::Sixteen),
        GiQuality::Pixels(TraceBlock::Four),
        GiQuality::Pixels(TraceBlock::One),
    ] {
        let Some(mut run) = Run::new(quality, true, furnace_camera(), |w| furnace(w, 0.5, 1.0))
        else {
            return;
        };
        run.settle(&format!("{quality:?}"));
        let mean = run.mean(32);
        let open: Vec<bool> = match quality {
            GiQuality::Probes => vec![true; mean.len()],
            _ => run
                .near_field()
                .unwrap()
                .chunks(4)
                .map(|v| v[0] >= 0.999)
                .collect(),
        };
        let (mut sum, mut error, mut count, mut corner, mut corners) = (0.0, 0.0, 0, 0.0, 0);
        for (p, open) in mean.iter().zip(&open) {
            if p[3] < 0.999 {
                continue;
            }
            let value = (p[0] + p[1] + p[2]) / 3.0;
            if *open {
                sum += value;
                error += (value / 2.0 - 1.0).abs();
                count += 1;
            } else {
                corner += value;
                corners += 1;
            }
        }
        let ratio = sum / count.max(1) as f32 / 2.0;
        let relative = error / count.max(1) as f32;
        eprintln!(
            "{quality:?}: {count} open pixels, mean / truth {ratio:.4}, mean |relative error| \
             {relative:.4}; {corners} near-field pixels at {:.4}",
            corner / corners.max(1) as f32 / 2.0
        );
        // Most of the image away from the box's edges (a block's footprint
        // from them at the coarsest block).
        assert!(count > mean.len() / 4, "{count} open pixels");
        // Every hit is lit from the radiance cache, whose closed box reads
        // within 10% of a radiosity solution (its own test's bound; 5-8% low
        // measured): the probes and the gather alike.
        assert!((ratio - 1.0).abs() < 0.10, "{quality:?}: {ratio}");
        assert!(relative < 0.12, "{quality:?}: {relative}");
    }
}

/// The reservoirs (temporal and spatial reuse, a visibility ray to a sample
/// another receiver took) against one plain cosine-distributed ray a pixel, both
/// accumulated over many frames at a ray a pixel, in a lit room with a
/// pillar: their means agree (the reuse's bias), tile by tile.
#[test]
fn the_reservoirs_agree_with_one_plain_ray_a_pixel() {
    let mut means = Vec::new();
    for reuse in [false, true] {
        let Some(mut run) = Run::new(
            GiQuality::Pixels(TraceBlock::One),
            reuse,
            room_camera(),
            |w| {
                room(w);
            },
        ) else {
            return;
        };
        run.settle(if reuse { "reuse" } else { "plain" });
        means.push(run.mean(128));
    }
    let tile = 8usize;
    let width = (means[0].len() as f64).sqrt() as usize;
    let side = width / tile;
    let mut worst = 0.0f32;
    let (mut a_all, mut b_all) = (0.0, 0.0);
    for ty in 0..side {
        for tx in 0..side {
            let (mut a, mut b) = (0.0, 0.0);
            for y in 0..tile {
                for x in 0..tile {
                    let i = (ty * tile + y) * width + tx * tile + x;
                    a += means[0][i][0] + means[0][i][1] + means[0][i][2];
                    b += means[1][i][0] + means[1][i][1] + means[1][i][2];
                }
            }
            a_all += a;
            b_all += b;
            if a > 1e-3 {
                worst = worst.max((b / a - 1.0).abs());
            }
        }
    }
    let ratio = b_all / a_all;
    eprintln!("reuse / plain: whole image {ratio:.4}, worst 8x8 tile {worst:.4}");
    assert!((ratio - 1.0).abs() < 0.05, "{ratio}");
    assert!(worst < 0.15, "{worst}");
}

/// Reuse across an occluder: a room halved by a wall to half its height, a
/// glowing panel on one side, the camera above the wall seeing the floor on
/// both sides of it. The floor's pixels on the far side neighbour pixels
/// that see the panel, on the same plane: reuse without visibility would
/// hand them samples of the panel. With reuse they take no more light than
/// one plain ray a pixel gives them (what reaches them over the wall, off
/// the ceiling), and the near side the same as it does.
#[test]
fn reuse_does_not_carry_light_through_an_occluder() {
    let mut means = Vec::new();
    for reuse in [false, true] {
        let Some(mut run) = Run::new(
            GiQuality::Pixels(TraceBlock::Four),
            reuse,
            Camera::perspective_look_at(
                Vec3::new(0.0, 4.5, 4.0),
                Vec3::new(0.0, 0.0, -1.5),
                Vec3::Y,
                1.2,
                1.0,
                0.1,
                100.0,
            ),
            |w| {
                let dark = [0.05; 3];
                for (centre, half, colour) in [
                    (
                        Vec3::new(0.0, -0.25, 0.0),
                        Vec3::new(5.0, 0.25, 5.0),
                        [0.7; 3],
                    ),
                    (Vec3::new(0.0, 5.25, 0.0), Vec3::new(5.0, 0.25, 5.0), dark),
                    (Vec3::new(-5.25, 2.5, 0.0), Vec3::new(0.25, 2.5, 5.0), dark),
                    (Vec3::new(5.25, 2.5, 0.0), Vec3::new(0.25, 2.5, 5.0), dark),
                    (Vec3::new(0.0, 2.5, -5.25), Vec3::new(5.0, 2.5, 0.25), dark),
                    (Vec3::new(0.0, 2.5, 5.25), Vec3::new(5.0, 2.5, 0.25), dark),
                    // The occluder: the wall at x = 0, half the room's height.
                    (
                        Vec3::new(0.0, 1.25, 0.0),
                        Vec3::new(0.05, 1.25, 5.0),
                        [0.7; 3],
                    ),
                ] {
                    spawn_box(w, centre, half, colour);
                }
                // A glowing panel low on the far wall of one side: below the
                // occluder's top seen from the other side's floor, and wide
                // (a small emitter's few hits a frame keep the image from
                // settling).
                spawn_glowing_box(
                    w,
                    Vec3::new(-4.9, 1.25, -1.0),
                    Vec3::new(0.05, 1.0, 3.0),
                    [0.5; 3],
                    5.0,
                );
            },
        ) else {
            return;
        };
        run.settle_means(if reuse { "reuse" } else { "plain" });
        means.push(run.mean(512));
    }
    // The camera looks along -z from above the wall: +x (the dark side) is
    // the image's right half, the wall its middle column. The dark side's
    // floor within the spatial reach of the wall (3 texels of 4 pixels: 12
    // pixels), past the wall's own pixels; and the lit side's alike.
    let width = (means[0].len() as f64).sqrt() as usize;
    let centre = width / 2;
    let band = |side: i32| -> [f32; 2] {
        let mut sums = [0.0f32; 2];
        for y in width / 3..width {
            for k in 2..(width / 4).min(12) {
                let x = (centre as i32 + side * k as i32) as usize;
                for (sum, mean) in sums.iter_mut().zip(&means) {
                    let p = mean[y * width + x];
                    *sum += p[0] + p[1] + p[2];
                }
            }
        }
        sums
    };
    let [dark_plain, dark_reuse] = band(1);
    let [lit_plain, lit_reuse] = band(-1);
    eprintln!(
        "dark side: plain {dark_plain:.3}, reuse {dark_reuse:.3} ({:.4}); lit side: plain          {lit_plain:.3}, reuse {lit_reuse:.3} ({:.4})",
        dark_reuse / dark_plain,
        lit_reuse / lit_plain
    );
    // The far side is darker than the near one by far (the scene does
    // occlude); reuse carries none of the near side's light to it (without
    // the visibility ray it read 1.59 of the plain rays there) and leaves the
    // near side as the plain rays do. The far side's band holds about 2% of
    // the near side's light and its plain mean moves by 10-15% between runs
    // over 128 frames (512 here): what reuse does there below the plain rays
    // is reported, not bounded.
    assert!(dark_plain < 0.25 * lit_plain, "{dark_plain} {lit_plain}");
    assert!(dark_reuse < 1.15 * dark_plain, "{dark_reuse} {dark_plain}");
    assert!(
        (lit_reuse / lit_plain - 1.0).abs() < 0.10,
        "{lit_reuse} {lit_plain}"
    );
}

/// A light switched off: the re-traced samples find their light gone and
/// the confidence the denoiser is given drops, so the indirect diffuse
/// follows the cache rather than the denoiser's history.
#[test]
fn a_light_switched_off_lowers_the_confidence() {
    let mut light = None;
    let Some(mut run) = Run::new(
        GiQuality::Pixels(TraceBlock::Four),
        true,
        room_camera(),
        |w| {
            light = Some(room(w));
        },
    ) else {
        return;
    };
    let lit = run.settle("lit");
    let light = light.unwrap();
    run.scene.world.insert(light, room_light(0.0));
    let start = Instant::now();
    let switched = run.frames;
    let mut dropped = None;
    loop {
        assert!(start.elapsed() < GUARD, "no confidence dropped");
        run.frame();
        let Some([s0, s1]) = run.signal() else {
            continue;
        };
        let low = s0
            .chunks(4)
            .zip(s1.chunks(4))
            .filter(|(a, b)| a[3] < 60000.0 && b[3] < 0.5)
            .count();
        if low > 0 && dropped.is_none() {
            dropped = Some(run.frames - switched);
        }
        let Some(image) = run.indirect() else {
            continue;
        };
        let mean = image
            .iter()
            .filter(|p| p[3] > 0.0)
            .map(|p| p[0] + p[1] + p[2])
            .sum::<f32>()
            / image.iter().filter(|p| p[3] > 0.0).count().max(1) as f32;
        if let Some(dropped) = dropped.filter(|_| mean < 0.1 * lit) {
            eprintln!(
                "confidence below 1/2 {dropped} frames after the switch; the indirect diffuse under 10% of its light after {}",
                run.frames - switched
            );
            return;
        }
    }
}

/// The near field reaches a block's footprint and no farther: the share of
/// pixels it darkens shrinks with the block, and traced at every pixel it
/// vanishes (nothing to move light between). It never changes a block's
/// mean: its factors average 1 over each block's pixels.
#[test]
fn the_near_field_shrinks_with_the_block() {
    let mut shares = Vec::new();
    for block in [TraceBlock::Sixteen, TraceBlock::Four, TraceBlock::One] {
        let Some(mut run) = Run::new(GiQuality::Pixels(block), true, room_camera(), |w| {
            room(w);
        }) else {
            return;
        };
        let near = run.until("the near field", |run| run.near_field());
        let width = (near.len() as f64 / 4.0).sqrt() as usize;
        let side = block.side() as usize;
        let mut worst = 0.0f32;
        for by in 0..width / side {
            for bx in 0..width / side {
                let (mut sum, mut count) = (0.0f32, 0.0f32);
                for y in 0..side {
                    for x in 0..side {
                        let i = ((by * side + y) * width + bx * side + x) * 4;
                        // The sky's pixels hold 1, as the block's mean.
                        sum += near[i + 1];
                        count += 1.0;
                    }
                }
                if count > 0.0 {
                    worst = worst.max((sum / count - 1.0).abs());
                }
            }
        }
        // Half floats: 2^-10 relative.
        assert!(
            worst < 4e-3,
            "{block:?}: a block's factors average 1 + {worst}"
        );
        let shaded = near.chunks(4).filter(|v| v[1] < 0.999).count();
        let share = shaded as f32 / (near.len() / 4) as f32;
        let least = near.chunks(4).map(|v| v[1]).fold(1.0f32, f32::min);
        eprintln!("{block:?}: {share:.4} of pixels darkened, least {least:.3}");
        shares.push(share);
    }
    assert!(shares[0] > shares[1] && shares[1] > shares[2], "{shares:?}");
    assert!(shares[2] < 0.05, "{shares:?}");
}
