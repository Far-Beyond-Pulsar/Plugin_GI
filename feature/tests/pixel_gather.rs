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
use helio_pass_gi::{GiGatherPass, GiGatherShadePass, GiQuality, GiResolvePass, GiSettings};

mod mesh_scene;
use mesh_scene::{adapter, furnace, indirect, read, read_texture, room, room_light, scene_db};

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
        let pass = self.renderer.find_pass::<GiGatherShadePass>()?;
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

    /// The per-pixel mean of the indirect diffuse over `frames` frames.
    fn mean(&mut self, frames: usize) -> Vec<[f32; 4]> {
        let mut sum = vec![[0.0f32; 4]; (SIZE * SIZE) as usize];
        for _ in 0..frames {
            self.frame();
            for (s, p) in sum.iter_mut().zip(self.indirect().unwrap()) {
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
/// quality. Checked where the near field is open (its corners' transfer is
/// the AO approximation's, reported).
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
                .map(|v| v[1] >= 0.999)
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
        assert!(count > (SIZE * SIZE / 4) as usize, "{count} open pixels");
        // Every hit is lit from the radiance cache, whose closed box reads
        // within 10% of a radiosity solution (its own test's bound; 5-8% low
        // measured): the probes and the gather alike.
        assert!((ratio - 1.0).abs() < 0.10, "{quality:?}: {ratio}");
        assert!(relative < 0.12, "{quality:?}: {relative}");
    }
}

/// The reservoirs (temporal and spatial reuse, no visibility ray between
/// neighbours) against one plain cosine-distributed ray a pixel, both
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
    let side = SIZE as usize / tile;
    let mut worst = 0.0f32;
    let (mut a_all, mut b_all) = (0.0, 0.0);
    for ty in 0..side {
        for tx in 0..side {
            let (mut a, mut b) = (0.0, 0.0);
            for y in 0..tile {
                for x in 0..tile {
                    let i = (ty * tile + y) * SIZE as usize + tx * tile + x;
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
        if dropped.is_some() && mean < 0.1 * lit {
            eprintln!(
                "confidence below 1/2 {} frames after the switch; the indirect diffuse under \
                 10% of its light after {}",
                dropped.unwrap(),
                run.frames - switched
            );
            return;
        }
    }
}

/// The near field reaches a block's footprint and no farther: the share of
/// pixels it darkens shrinks with the block, and traced at every pixel it
/// all but vanishes (a pixel's own neighbours only).
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
        let shaded = near.chunks(4).filter(|v| v[1] < 0.999).count();
        let share = shaded as f32 / (SIZE * SIZE) as f32;
        let least = near.chunks(4).map(|v| v[1]).fold(1.0f32, f32::min);
        eprintln!("{block:?}: {share:.4} of pixels darkened, least {least:.3}");
        shares.push(share);
    }
    assert!(shares[0] > shares[1] && shares[1] > shares[2], "{shares:?}");
    assert!(shares[2] < 0.05, "{shares:?}");
}
