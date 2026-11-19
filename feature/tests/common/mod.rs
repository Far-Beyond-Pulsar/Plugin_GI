//! What the GI tests share: GI's node built into a graph of its own with an
//! analytic tracer offered through the frame's ray streams (no kind) and the
//! lights' sampler through the frame's light samples, a scene of a camera
//! and light rows, buffer readback and the probes' records read back on the
//! CPU.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Vec3};
use helio_core::{
    BufferHandle, GpuCameraUniforms, PassContext, PrepareContext, RenderGraph, RenderPass,
    Result as HelioResult, SceneBufferProjection, SceneInput,
};
use helio_feature_api::{BuildEnv, FeatureSet, FrameDesc, NodeBuild, NodeRegistry};
use helio_frame_api::rays::light_samples;
use helio_frame_api::rays::stream::{self, RayClass, Stream, Tracer, TracerId};
use helio_pass_gi::{GiFeature, GiSettings, GI_FRAME, GI_PROBES, PROBE_WORDS};

pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

/// The tests' device (none without an adapter: the tests then pass empty).
pub fn gpu() -> Option<&'static Gpu> {
    static GPU: std::sync::OnceLock<Option<Gpu>> = std::sync::OnceLock::new();
    GPU.get_or_init(|| {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        let limits = adapter.limits();
        // GPU timestamps where the adapter has them (the cost test).
        let timestamps =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: adapter.features() & timestamps,
            required_limits: wgpu::Limits {
                max_buffer_size: limits.max_buffer_size.min(u32::MAX as u64),
                ..limits
            },
            ..Default::default()
        }))
        .ok()?;
        Some(Gpu {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    })
    .as_ref()
}

pub fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec()
}

pub fn half2(w: u32) -> [f32; 2] {
    let f = |h: u32| {
        let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
        let v = if e == 0 {
            m as f32 * 2f32.powi(-24)
        } else {
            (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15)
        };
        if s == 1 {
            -v
        } else {
            v
        }
    };
    [f(w & 0xffff), f(w >> 16)]
}

/// Probe `index`'s irradiance / pi toward unit `n` (its L1 harmonics), its
/// state.
pub fn probe(words: &[u32], index: u32, n: Vec3) -> (Vec3, u32) {
    let at = (index * PROBE_WORDS) as usize;
    let h: Vec<[f32; 2]> = (0..6).map(|w| half2(words[at + w])).collect();
    let (r, g, b) = (
        [h[0][0], h[1][1], h[3][0], h[4][1]],
        [h[0][1], h[2][0], h[3][1], h[5][0]],
        [h[1][0], h[2][1], h[4][0], h[5][1]],
    );
    let basis = [0.282095, 0.488603 * n.y, 0.488603 * n.z, 0.488603 * n.x];
    let dot = |c: [f32; 4]| (0..4).map(|k| c[k] * basis[k]).sum::<f32>().max(0.0);
    (Vec3::new(dot(r), dot(g), dot(b)), words[at + 6])
}

/// A point light at frame point `at` of `intensity` (white) and `range`
/// (`GpuLight`).
pub fn point_light(at: Vec3, intensity: f32, range: f32) -> helio_light_api::GpuLight {
    helio_light_api::GpuLight {
        position_range: [at.x, at.y, at.z, range],
        color_intensity: [1.0, 1.0, 1.0, intensity],
        shadow_index: u32::MAX,
        light_type: 1,
        ..Default::default()
    }
}

/// A shape of the analytic tracer (`analytic_tracer.wgsl`), in world space.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub kind: u32,
    pub lo: DVec3,
    pub hi: DVec3,
    pub transmission: f32,
    pub base: Vec3,
    pub emissive: Vec3,
}

const GROUND: u32 = 0;
const BOX: u32 = 1;
const SHEET: u32 = 2;
const ROOM: u32 = 3;

impl Shape {
    /// Solid ground below height `y`, of base colour `base`.
    pub fn ground(y: f64, base: f32) -> Self {
        Self {
            kind: GROUND,
            lo: DVec3::new(0.0, y, 0.0),
            hi: DVec3::ZERO,
            transmission: 0.0,
            base: Vec3::splat(base),
            emissive: Vec3::ZERO,
        }
    }

    /// A solid box from `lo` to `hi`.
    pub fn solid(lo: DVec3, hi: DVec3, base: f32) -> Self {
        Self {
            kind: BOX,
            lo,
            hi,
            transmission: 0.0,
            base: Vec3::splat(base),
            emissive: Vec3::ZERO,
        }
    }

    /// A two-sided horizontal sheet at height `y`, as wide as anything
    /// sees, letting `transmission` of the light through.
    pub fn sheet(y: f64, base: f32, transmission: f32) -> Self {
        Self {
            kind: SHEET,
            lo: DVec3::new(-1e7, y, -1e7),
            hi: DVec3::new(1e7, y, 1e7),
            transmission,
            base: Vec3::splat(base),
            emissive: Vec3::ZERO,
        }
    }

    /// A closed room from `lo` to `hi` (its walls face in).
    pub fn room(lo: DVec3, hi: DVec3, base: f32) -> Self {
        Self {
            kind: ROOM,
            lo,
            hi,
            transmission: 0.0,
            base: Vec3::splat(base),
            emissive: Vec3::ZERO,
        }
    }

    pub fn glowing(mut self, radiance: f32) -> Self {
        self.emissive = Vec3::splat(radiance);
        self
    }

    fn row(&self, origin: DVec3) -> GpuShape {
        let frame = |p: DVec3| (p - origin).as_vec3();
        let (lo, hi) = (frame(self.lo), frame(self.hi));
        GpuShape {
            a: [lo.x, lo.y, lo.z, self.kind as f32],
            b: [hi.x, hi.y, hi.z, self.transmission],
            base: self.base.extend(0.0).to_array(),
            emissive: self.emissive.extend(0.0).to_array(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuShape {
    a: [f32; 4],
    b: [f32; 4],
    base: [f32; 4],
    emissive: [f32; 4],
}

const MAX_SHAPES: u64 = 16;

/// The scene the tracer traces: shapes the test may change between frames.
pub type Shapes = Arc<Mutex<Vec<Shape>>>;

/// The analytic tracer's pass for one stream.
struct AnalyticPass {
    stream: Stream,
    id: TracerId,
    shapes: Shapes,
    origin: DVec3,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    rows: wgpu::Buffer,
    info: wgpu::Buffer,
    /// What an occlusion stream binds for its surfaces.
    no_surfaces: wgpu::Buffer,
    count: u32,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl AnalyticPass {
    fn new(
        device: &wgpu::Device,
        stream: Stream,
        id: TracerId,
        shapes: Shapes,
        origin: DVec3,
    ) -> Self {
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("analytic tracer"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                storage(3, false),
                storage(4, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let module = helio_core::shader::module_with(
            device,
            "analytic tracer",
            include_str!("analytic_tracer.wgsl"),
            &[
                stream::SNIPPET,
                stream::SURFACE_SNIPPET,
                helio_frame_api::gbuffer::SURFACE_SNIPPET,
            ],
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("analytic tracer"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("analytic tracer"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("analytic tracer"),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mut writes = vec![stream.hits];
        writes.extend(stream.surfaces);
        Self {
            reads: helio_frame_api::static_names(&[stream.header, stream.rays, stream.hits]),
            writes: helio_frame_api::static_names(&writes),
            stream,
            id,
            shapes,
            origin,
            layout,
            pipeline,
            rows: buffer(MAX_SHAPES * 64, wgpu::BufferUsages::STORAGE),
            info: buffer(16, wgpu::BufferUsages::UNIFORM),
            no_surfaces: buffer(64, wgpu::BufferUsages::STORAGE),
            count: 0,
        }
    }
}

impl RenderPass for AnalyticPass {
    fn name(&self) -> &'static str {
        "AnalyticTracer"
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let rows: Vec<GpuShape> = self
            .shapes
            .lock()
            .unwrap()
            .iter()
            .map(|s| s.row(self.origin))
            .collect();
        self.count = rows.len() as u32;
        if !rows.is_empty() {
            ctx.queue
                .write_buffer(&self.rows, 0, bytemuck::cast_slice(&rows));
        }
        let occlusion = u32::from(self.stream.class == RayClass::Occlusion);
        ctx.queue.write_buffer(
            &self.info,
            0,
            bytemuck::cast_slice(&[self.id.0, occlusion, self.count, 0]),
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let binding = |name: &str| ctx.registry.buffer_binding(name);
        let (Some(header), Some(rays), Some(hits)) = (
            binding(self.stream.header),
            binding(self.stream.rays),
            binding(self.stream.hits),
        ) else {
            return Ok(());
        };
        let surfaces = match self.stream.surfaces {
            Some(name) => binding(name),
            None => Some(&self.no_surfaces),
        };
        let Some(surfaces) = surfaces else {
            return Ok(());
        };
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        // A test: a bind group a frame is fine here.
        let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("analytic tracer"),
            layout: &self.layout,
            entries: &[
                entry(0, header),
                entry(1, rays),
                entry(2, hits),
                entry(3, surfaces),
                entry(4, &self.rows),
                entry(5, &self.info),
            ],
        });
        let header = header.clone();
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("analytic tracer"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&self.pipeline);
        pass.dispatch_workgroups_indirect(&header, 0);
        Ok(())
    }
}

/// The analytic tracer of `shapes` (world space, the frame at `origin`).
pub fn analytic_tracer(shapes: Shapes, origin: DVec3) -> Tracer {
    Tracer {
        name: "analytic",
        supported: Arc::new(|_| true),
        trace: Arc::new(move |build, stream, id| {
            let pass = AnalyticPass::new(build.env.device, *stream, id, shapes.clone(), origin);
            build.graph.add_pass(Box::new(pass));
            Ok(())
        }),
        // The trace pass writes its hits' surfaces.
        surface: Arc::new(|_, _, _| Ok(())),
    }
}

/// A late tracer's pass: it traces nothing, and answers its stream short
/// while `late` is set (a kind whose kernels still compile).
struct LatePass {
    late: Arc<std::sync::atomic::AtomicBool>,
    answers: Arc<stream::Answers>,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl RenderPass for LatePass {
    fn name(&self) -> &'static str {
        "LateTracer"
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        if self.late.load(std::sync::atomic::Ordering::Relaxed) {
            self.answers.short();
        }
        Ok(())
    }
}

/// A tracer that knows geometry it cannot trace while `late` is set.
pub fn late_tracer(late: Arc<std::sync::atomic::AtomicBool>) -> Tracer {
    Tracer {
        name: "late",
        supported: Arc::new(|_| true),
        trace: Arc::new(move |build, stream, _| {
            let pass = LatePass {
                late: late.clone(),
                answers: stream::answers(build.env, stream),
                reads: helio_frame_api::static_names(&[stream.header, stream.hits]),
                writes: helio_frame_api::static_names(&[stream.hits]),
            };
            build.graph.add_pass(Box::new(pass));
            Ok(())
        }),
        surface: Arc::new(|_, _, _| Ok(())),
    }
}

/// Keeps GI's frame and probes (`gi_frame`, `gi_probes`) as they are
/// published, for the test to read.
struct Capture {
    out: Arc<Mutex<Option<[wgpu::Buffer; 2]>>>,
}

impl RenderPass for Capture {
    fn name(&self) -> &'static str {
        "Capture"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[GI_FRAME, GI_PROBES]
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if let (Some(frame), Some(probes)) = (
            ctx.registry.buffer_binding(GI_FRAME),
            ctx.registry.buffer_binding(GI_PROBES),
        ) {
            *self.out.lock().unwrap() = Some([frame.clone(), probes.clone()]);
        }
        Ok(())
    }
}

/// What a test frame holds besides the shapes.
#[derive(Clone, Debug)]
pub struct Setup {
    /// The frame's origin (world) and the eye in it.
    pub origin: DVec3,
    pub eye: Vec3,
    /// The flat sky's radiance (no atmosphere).
    pub sky: f32,
    /// The sun's illuminance straight down, if any.
    pub sun: Option<f32>,
    /// Point lights (frame space).
    pub lights: Vec<helio_light_api::GpuLight>,
    /// Whether the analytic tracer is offered at all.
    pub traced: bool,
    /// A second tracer, short while this is set ([`late_tracer`]).
    pub late: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// An atmosphere whose sky is black (its frame and sky table published
    /// as the sky publishes them), over the flat sky.
    pub black_atmosphere: bool,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            origin: DVec3::new(1000.0, 2.0, -500.0),
            eye: Vec3::new(0.3, -0.3, -0.2),
            sky: 0.0,
            sun: None,
            lights: Vec::new(),
            traced: true,
            late: None,
            black_atmosphere: false,
        }
    }
}

impl Setup {
    /// The eye's world position.
    pub fn world_eye(&self) -> DVec3 {
        self.origin + self.eye.as_dvec3()
    }
}

struct Scene {
    gpu: &'static Gpu,
    frame: u64,
    origin: DVec3,
    camera: wgpu::Buffer,
    camera_data: GpuCameraUniforms,
    buffers: SceneBufferProjection,
}

impl SceneInput for Scene {
    fn world_origin(&self) -> DVec3 {
        self.origin
    }
    fn device(&self) -> &Arc<wgpu::Device> {
        &self.gpu.device
    }
    fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.gpu.queue
    }
    fn frame_count(&self) -> u64 {
        self.frame
    }
    fn camera(&self) -> &wgpu::Buffer {
        &self.camera
    }
    fn camera_data(&self) -> &GpuCameraUniforms {
        &self.camera_data
    }
    fn camera_generation(&self) -> u64 {
        0
    }
    fn scene_buffers(&self) -> &SceneBufferProjection {
        &self.buffers
    }
}

/// GI's node in a graph of its own, over the analytic tracer.
pub struct Harness {
    pub gpu: &'static Gpu,
    pub setup: Setup,
    pub shapes: Shapes,
    pub graph: RenderGraph,
    pub stats: helio_core::FrameStats,
    scene: Scene,
    target: wgpu::TextureView,
    depth: wgpu::TextureView,
    captured: Arc<Mutex<Option<[wgpu::Buffer; 2]>>>,
    /// The atmosphere's frame and sky table, when the setup has one.
    atmosphere: Option<[wgpu::Buffer; 2]>,
}

fn storage_init(gpu: &Gpu, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    gpu.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
}

/// The passes GI's node (`helio.pass.gi`) builds for `setup`, by name; the
/// graph and the environment it was built in.
pub fn build(
    gpu: &'static Gpu,
    setup: &Setup,
    shapes: &Shapes,
    stats: &helio_core::FrameStats,
    camera: &wgpu::Buffer,
    kind: &str,
    external: bool,
) -> RenderGraph {
    let frame = FrameDesc {
        width: 16,
        height: 16,
        internal_width: 16,
        internal_height: 16,
        surface_format: wgpu::TextureFormat::Rgba8Unorm,
        xr: false,
    };
    let mut env = BuildEnv::new(&gpu.device, &gpu.queue, frame, camera, camera, stats, true)
        .with_setting(GiSettings::default());
    if setup.traced {
        stream::offer(&mut env, analytic_tracer(shapes.clone(), setup.origin));
    }
    if let Some(late) = &setup.late {
        stream::offer(&mut env, late_tracer(late.clone()));
    }
    // The lights' answer to GI's light batch (its sun and point lights).
    light_samples::offer(&mut env, helio_pass_light_sampler::sampler());
    let mut registry = NodeRegistry::new();
    FeatureSet::new()
        .with(GiFeature)
        .register_into(&mut registry);
    for prepare in registry.preparers().to_vec() {
        prepare(&mut env);
    }
    let mut graph = if external {
        RenderGraph::new_with_external_device(&gpu.device, &gpu.queue)
    } else {
        RenderGraph::new(&gpu.device, &gpu.queue)
    };
    let node = registry.get(kind).expect("GI registers the node");
    let params = node.effective_params(&Default::default());
    node.run(&mut NodeBuild {
        graph: &mut graph,
        env: &mut env,
        params: &params,
        node_id: "gi",
    })
    .expect("GI's node builds");
    graph
}

/// The names of `graph`'s passes, in order.
pub fn pass_names(graph: &RenderGraph) -> Vec<String> {
    (0..graph.pass_count())
        .filter_map(|index| graph.pass_declaration(index))
        .map(|pass| pass.pass.to_string())
        .collect()
}

impl Harness {
    pub fn new(gpu: &'static Gpu, setup: Setup, shapes: Vec<Shape>) -> Self {
        Self::with(gpu, setup, shapes, false)
    }

    /// A harness whose graph neither waits for nor reads back a frame (the
    /// test polls the device), its GPU timestamps on: frames run pipelined.
    pub fn pipelined(gpu: &'static Gpu, setup: Setup, shapes: Vec<Shape>) -> Self {
        let mut harness = Self::with(gpu, setup, shapes, true);
        harness.graph.profiler_mut().set_enabled(true);
        harness
    }

    fn with(gpu: &'static Gpu, setup: Setup, shapes: Vec<Shape>, external: bool) -> Self {
        let shapes: Shapes = Arc::new(Mutex::new(shapes));
        let stats = helio_core::FrameStats::new(&gpu.device);
        let camera = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test camera"),
            size: std::mem::size_of::<GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut graph = build(
            gpu,
            &setup,
            &shapes,
            &stats,
            &camera,
            "helio.pass.gi",
            external,
        );
        let captured = Arc::new(Mutex::new(None));
        graph.add_pass(Box::new(Capture {
            out: captured.clone(),
        }));
        graph.lock(16, 16);
        let mut camera_data = GpuCameraUniforms::zeroed();
        camera_data.position_near = [setup.eye.x, setup.eye.y, setup.eye.z, 0.1];
        // The sun shines straight down: a directional light row among the
        // point lights.
        let mut lights = setup.lights.clone();
        if let Some(e) = setup.sun {
            lights.push(helio_light_api::GpuLight {
                direction_outer: [0.0, -1.0, 0.0, 0.0],
                color_intensity: [1.0, 1.0, 1.0, e],
                shadow_index: u32::MAX,
                light_type: 0,
                ..Default::default()
            });
        }
        let handle = |buffer: wgpu::Buffer, row_bytes: u64| BufferHandle {
            buffer,
            epoch: 0,
            row_bytes,
            content_generation: 1,
        };
        let mut handles = Vec::new();
        if !lights.is_empty() {
            handles.push((
                helio_light_api::SCENE_LIGHTS.key(),
                handle(
                    storage_init(gpu, "test lights", bytemuck::cast_slice(&lights)),
                    std::mem::size_of::<helio_light_api::GpuLight>() as u64,
                ),
            ));
        }
        let texture = |format, label| {
            gpu.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: 16,
                        height: 16,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        use helio_frame_api::lighting::sky::{SKY_FRAME_BYTES, SKY_TABLE_BYTES};
        let atmosphere = setup.black_atmosphere.then(|| {
            let uniform = |bytes: &[u8]| {
                use wgpu::util::DeviceExt;
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("test atmosphere"),
                        contents: bytes,
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            };
            // `SkyFrame.planet.w` 1: an atmosphere is active; nothing else.
            let mut frame = vec![0f32; (SKY_FRAME_BYTES / 4) as usize];
            frame[3] = 1.0;
            let table = vec![0u8; SKY_TABLE_BYTES as usize];
            [uniform(bytemuck::cast_slice(&frame)), uniform(&table)]
        });
        Self {
            gpu,
            atmosphere,
            scene: Scene {
                gpu,
                frame: 0,
                origin: setup.origin,
                camera,
                camera_data,
                buffers: SceneBufferProjection::from_handles(handles),
            },
            setup,
            shapes,
            graph,
            stats,
            target: texture(wgpu::TextureFormat::Rgba8Unorm, "test target"),
            depth: texture(wgpu::TextureFormat::Depth32Float, "test depth"),
            captured,
        }
    }

    /// Renders a frame without waiting for it.
    pub fn submit(&mut self) {
        let mut registry = helio_core::ResourceRegistry::empty();
        let sky = self.setup.sky;
        registry.write(
            helio_core::resource_keys::render_environment(),
            helio_core::RenderEnvironment {
                clear_color: [0.0; 4],
                ambient_color: [1.0; 3],
                ambient_intensity: sky,
                tlas: None,
            },
            "test",
        );
        if let Some([frame, table]) = &self.atmosphere {
            use helio_frame_api::lighting::sky::{SKY_FRAME, SKY_TABLE};
            registry.write(
                helio_core::ResourceKey::<&wgpu::Buffer>::new(SKY_FRAME),
                frame,
                "test",
            );
            registry.write(
                helio_core::ResourceKey::<&wgpu::Buffer>::new(SKY_TABLE),
                table,
                "test",
            );
        }
        self.graph
            .execute_with_registry(&self.scene, &self.target, &self.depth, &mut registry)
            .expect("the frame renders");
        self.scene.frame += 1;
    }

    /// Renders `n` frames, each finished before the next.
    pub fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.submit();
            self.gpu
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
        }
    }

    /// GI's frame and probe buffers (after a frame).
    pub fn buffers(&self) -> [wgpu::Buffer; 2] {
        self.captured
            .lock()
            .unwrap()
            .clone()
            .expect("GI published its probes")
    }

    /// The probes' records.
    pub fn probes(&self) -> Vec<u32> {
        let [_, probes] = self.buffers();
        read(&self.gpu.device, &self.gpu.queue, &probes)
    }

    /// The clipmap of the eye's place.
    pub fn clipmap(&self) -> helio_pass_gi::Clipmap {
        helio_pass_gi::Clipmap::around(
            self.setup.world_eye(),
            helio_pass_gi::DEFAULT_FINEST_SPACING,
        )
    }
}

/// The updated probes of `words` whose world position `keep` takes, with
/// that position.
pub fn updated(
    words: &[u32],
    clipmap: &helio_pass_gi::Clipmap,
    keep: impl Fn(DVec3) -> bool,
) -> Vec<(u32, DVec3)> {
    (0..helio_pass_gi::PROBES)
        .filter_map(|index| {
            let (_, state) = probe(words, index, Vec3::Y);
            let at = clipmap.position(index);
            (state >> 8 != 0 && keep(at)).then_some((index, at))
        })
        .collect()
}
