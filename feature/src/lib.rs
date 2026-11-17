//! World-space diffuse global illumination (README.md): a camera-centred
//! clipmap of irradiance probes, kept up to date by rays every kind's tracer
//! answers, read by deferred lighting at every pixel.
//!
//! GI is a consumer of the frame's ray streams
//! (`helio_frame_api::rays::stream`) and knows no kind. Each frame, after the
//! G-buffer phase (the frame's geometry as it was drawn):
//!
//! 1. [`GiPass`]: the sun, the probes that came into the clipmap reset, this
//!    frame's probe rays into the probe stream ([`probe_stream`]).
//! 2. The kinds' tracers answer it (`stream::trace`): each ray's nearest hit
//!    and its surface.
//! 3. [`GiLightPass`]: the light of every hit, the part of it that waits on
//!    a shadow ray (the sun's, one local light's) put aside, and those
//!    shadow rays into the shadow stream ([`shadow_stream`]).
//! 4. The tracers answer that (what gets through each).
//! 5. [`GiIntegratePass`]: the rays' radiance into their probes, this frame.
//!
//! [`GiResolvePass`] runs before deferred lighting: the probes' irradiance at
//! every pixel (`gi_irradiance`). The clipmap is a function of the eye's
//! world position and the finest spacing alone ([`Clipmap::around`],
//! [`eye`]).

use glam::{DVec3, Vec3};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_frame_api::lighting::sky::{
    SKY_FRAME, SKY_FRAME_BYTES, SKY_SNIPPET, SKY_TABLE, SKY_TABLE_BYTES, SKY_TABLE_SNIPPET,
};
use helio_frame_api::rays::stream::{self, RayClass, Stream, StreamBuffers};
use std::collections::VecDeque;

mod feature;
mod integrate;
mod light;
pub mod local_lights;
pub mod settings;
pub use feature::GiFeature;
pub use integrate::GiIntegratePass;
pub use light::GiLightPass;
pub use settings::{GiSettings, GI};

pub use helio_gi_api::{
    eye, Clipmap, GpuFrame, GpuLevel, DEFAULT_FINEST_SPACING, FRAME_BYTES, GI_COMMON_SNIPPET,
    GI_FRAME, GI_IRRADIANCE, GI_PROBES, GI_PROBES_VIEW, GI_SAMPLE_SNIPPET, GI_SUN, LEVELS,
    LEVEL_PROBES, LEVEL_UPDATES, MAX_RAYS, MAX_UPDATES, PROBES, PROBE_WORDS, RAYS, SIDE,
};

/// The probes' rays (`SceneRay`, a row a ray, probe-major: 64 a probe in
/// update-list order): the nearest hit and its surface.
pub fn probe_stream() -> Stream {
    Stream::new("gi_probe", RayClass::Nearest, MAX_RAYS)
}

/// The shadow rays of the probe rays' hits, toward the sun and one local
/// light: as many rows as there are such rays this frame, packed from 0
/// (the light pass writes the count on the GPU), at most two a probe ray.
pub fn shadow_stream() -> Stream {
    Stream::new("gi_shadow", RayClass::Occlusion, 2 * MAX_RAYS)
}

/// GI's own graph keys between its passes: this frame's update and reset
/// lists, the frame's rays (`GiTrace`), what each ray's light waits on
/// (`GiPending`) and whether anything answered.
const GI_LISTS: &str = "gi_lists";
const GI_TRACE: &str = "gi_trace";
const GI_PENDING: &str = "gi_pending";
const GI_ANSWERED: &str = "gi_answered";

/// `GiTrace` (`gi_rays.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTrace {
    eye: [f32; 4],
    info: [u32; 4],
}

const TRACE_BYTES: u64 = std::mem::size_of::<GpuTrace>() as u64;
/// Bytes of a `GiPending`.
const PENDING_BYTES: u64 = 24;

/// The probe rays' WGSL (`gi_rays.wgsl`, `//!use gi_rays`).
const GI_RAYS_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use gi_rays", "../shaders/gi_rays.wgsl");

/// Which probes update each frame: per level, those that came in (oldest
/// first), then the others in turn.
struct Schedule {
    entered: [VecDeque<u32>; LEVELS],
    queued: Vec<bool>,
    cursor: [u32; LEVELS],
}

impl Schedule {
    fn new() -> Self {
        Self {
            entered: Default::default(),
            queued: vec![false; PROBES as usize],
            cursor: [0; LEVELS],
        }
    }

    fn enter(&mut self, index: u32) {
        if !std::mem::replace(&mut self.queued[index as usize], true) {
            self.entered[(index / LEVEL_PROBES) as usize].push_back(index);
        }
    }

    fn next(&mut self, out: &mut Vec<u32>) {
        out.clear();
        for (level, &updates) in LEVEL_UPDATES.iter().enumerate() {
            let start = out.len();
            while out.len() - start < updates {
                let Some(index) = self.entered[level].pop_front() else {
                    break;
                };
                self.queued[index as usize] = false;
                out.push(index);
            }
            while out.len() - start < updates {
                out.push(level as u32 * LEVEL_PROBES + self.cursor[level]);
                self.cursor[level] = (self.cursor[level] + 1) % LEVEL_PROBES;
            }
        }
    }
}

/// A unit quaternion, uniformly random from `seed` (Shoemake).
fn rotation(seed: u64) -> [f32; 4] {
    let mut h = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xd1b5_4a32_d192_ed03;
    let mut next = || {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        (h >> 11) as f64 / (1u64 << 53) as f64
    };
    let (u1, u2, u3) = (next(), next(), next());
    let tau = std::f64::consts::TAU;
    let (a, b) = ((1.0 - u1).sqrt(), u1.sqrt());
    [
        (a * (tau * u2).sin()) as f32,
        (a * (tau * u2).cos()) as f32,
        (b * (tau * u3).sin()) as f32,
        (b * (tau * u3).cos()) as f32,
    ]
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

fn texture(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// A compute pipeline of `module`'s `entry` over `layouts`, with the
/// pipeline-override `constants`.
fn compute(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    entry: &str,
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    helio_core::shader::create_compute_pipeline(
        device,
        wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache: None,
        },
    )
}

fn pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    })
}

/// The probes' state across frames (kept over graph rebuilds).
struct Probes {
    frame: wgpu::Buffer,
    probes: wgpu::Buffer,
    /// This frame's update list (`MAX_UPDATES`), then its reset list.
    lists: wgpu::Buffer,
    sun: wgpu::Buffer,
    trace: wgpu::Buffer,
    answered: wgpu::Buffer,
    rays: StreamBuffers,
    clipmap: Option<Clipmap>,
    schedule: Schedule,
}

impl Probes {
    fn new(device: &wgpu::Device) -> Self {
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        Self {
            frame: buffer(device, "GI frame", FRAME_BYTES, uniform),
            probes: buffer(
                device,
                "GI probes",
                u64::from(PROBES * PROBE_WORDS) * 4,
                storage,
            ),
            lists: buffer(
                device,
                "GI lists",
                u64::from(MAX_UPDATES + PROBES) * 4,
                storage,
            ),
            sun: buffer(device, "GI sun", 32, storage),
            trace: buffer(device, "GI trace", TRACE_BYTES, uniform),
            answered: buffer(device, "GI answered", 16, storage),
            rays: StreamBuffers::new(device, &probe_stream()),
            clipmap: None,
            schedule: Schedule::new(),
        }
    }
}

/// How a [`GiPass`] runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiConfig {
    /// The finest level's probe spacing (metres; each coarser level's
    /// doubles): finer indoors, where rooms, pillars and furniture are a few
    /// metres apart, than over open ground.
    pub finest_spacing: f64,
}

impl Default for GiConfig {
    fn default() -> Self {
        Self {
            finest_spacing: DEFAULT_FINEST_SPACING,
        }
    }
}

/// The start of the probes' update (module docs): the probe stream's
/// generator.
pub struct GiPass {
    config: GiConfig,
    stream: Stream,
    state: Probes,
    layout: wgpu::BindGroupLayout,
    sun: wgpu::ComputePipeline,
    reset: wgpu::ComputePipeline,
    rays: wgpu::ComputePipeline,
    fallback_atmosphere: wgpu::Buffer,
    fallback_sky_table: wgpu::Buffer,
    fallback_directional: wgpu::Buffer,
    cache: helio_core::gpu_cache::GpuCache,
    /// The frame's statistics and the probe stream's count in them.
    stats: helio_core::FrameStats,
    stat: helio_core::StatSlot,
    writes: &'static [&'static str],
    /// This frame's probe counts: updated, reset; its probes to update and
    /// those that came in.
    counts: [u32; 2],
    list: Vec<u32>,
    entered: Vec<u32>,
}

impl GiPass {
    pub fn new(device: &wgpu::Device, config: GiConfig, stats: &helio_core::FrameStats) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI update"),
            entries: &[
                uniform(0, FRAME_BYTES),
                storage(1, false),
                storage(2, false),
                storage(3, false),
                storage(4, true),
                storage(5, false),
                storage(6, true),
                uniform(7, SKY_FRAME_BYTES),
                uniform(8, SKY_TABLE_BYTES),
                uniform(9, TRACE_BYTES),
            ],
        });
        let module = helio_core::shader::module_with(
            device,
            "GI update",
            helio_core::include_wgsl!("../shaders/gi_update.wgsl"),
            &[
                SKY_SNIPPET,
                SKY_TABLE_SNIPPET,
                stream::SNIPPET,
                GI_COMMON_SNIPPET,
                GI_RAYS_SNIPPET,
                helio_light_api::LIGHT_LIST_SNIPPET,
            ],
        );
        let pipeline_layout = pipeline_layout(device, "GI update", &layout);
        let pipeline = |entry: &str| compute(device, &module, &pipeline_layout, entry, &[]);
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let stream = probe_stream();
        Self {
            config,
            stream,
            state: Probes::new(device),
            sun: pipeline("gi_sun_main"),
            reset: pipeline("gi_reset"),
            rays: pipeline("gi_rays_main"),
            layout,
            fallback_atmosphere: buffer(device, "GI no atmosphere", SKY_FRAME_BYTES, uniform),
            fallback_sky_table: buffer(device, "GI no sky table", SKY_TABLE_BYTES, uniform),
            fallback_directional: buffer(
                device,
                "GI no directional lights",
                std::mem::size_of::<helio_light_api::DirectionalLights>() as u64,
                wgpu::BufferUsages::STORAGE,
            ),
            cache: helio_core::gpu_cache::GpuCache::new(),
            stats: stats.clone(),
            stat: stream.stat(stats),
            writes: helio_frame_api::static_names(&[
                GI_FRAME,
                GI_PROBES,
                GI_SUN,
                GI_LISTS,
                GI_TRACE,
                GI_ANSWERED,
                stream.header,
                stream.rays,
                stream.hits,
            ]),
            counts: [0; 2],
            list: Vec::with_capacity(MAX_UPDATES as usize),
            entered: Vec::new(),
        }
    }

    /// Starts frame `frame` with the eye at world point `eye`, at frame
    /// point `frame_eye` (the frame's space is the rays'); without an
    /// atmosphere the sky's radiance is `sky` above the horizon and `ground`
    /// below it.
    fn begin(
        &mut self,
        queue: &wgpu::Queue,
        eye: DVec3,
        frame_eye: Vec3,
        frame: u64,
        [sky, ground]: [Vec3; 2],
    ) {
        let state = &mut self.state;
        let clipmap = Clipmap::around(eye, self.config.finest_spacing);
        clipmap.entered_into(state.clipmap.as_ref(), &mut self.entered);
        for &index in &self.entered {
            state.schedule.enter(index);
        }
        state.clipmap = Some(clipmap);
        state.schedule.next(&mut self.list);
        let updated = self.list.len() as u32;
        let gpu = GpuFrame {
            levels: clipmap.levels(eye),
            sky: sky.extend(0.0).to_array(),
            ground: ground.extend(0.0).to_array(),
            counts: [updated, 0, self.entered.len() as u32, 0],
            rotation: rotation(frame),
            prev_rotation: [0.0, 0.0, 0.0, 1.0],
        };
        let trace = GpuTrace {
            eye: frame_eye.extend(0.0).to_array(),
            info: [(frame as u32).wrapping_mul(0x9e37_79b9), 0, 0, 0],
        };
        let write = |buffer: &wgpu::Buffer, offset: u64, bytes: &[u8]| {
            helio_core::upload::write_buffer(queue, buffer, offset, bytes);
        };
        write(&state.frame, 0, bytemuck::bytes_of(&gpu));
        write(&state.trace, 0, bytemuck::bytes_of(&trace));
        write(&state.lists, 0, bytemuck::cast_slice(&self.list));
        if !self.entered.is_empty() {
            write(
                &state.lists,
                u64::from(MAX_UPDATES) * 4,
                bytemuck::cast_slice(&self.entered),
            );
        }
        state.rays.set_count(queue, updated * RAYS);
        self.stats.add(self.stat, u64::from(updated * RAYS));
        self.counts = [updated, self.entered.len() as u32];
    }

    /// Records the frame `begin` started: the sun, the probes that came in
    /// reset, this frame's rays (their hits cleared). `atmosphere` and
    /// `sky_table` are the atmosphere's, `directional` the lights'
    /// directional lights (none: no atmosphere, no sun).
    fn record(
        &mut self,
        device: &wgpu::Device,
        cmds: &mut helio_core::CommandRecorder<'_>,
        atmosphere: Option<&wgpu::Buffer>,
        sky_table: Option<&wgpu::Buffer>,
        directional: Option<&wgpu::Buffer>,
    ) {
        let s = &self.state;
        self.cache.end_frame();
        let group = self.cache.bind_group(
            device,
            &wgpu::BindGroupDescriptor {
                label: Some("GI update"),
                layout: &self.layout,
                entries: &[
                    entry(0, &s.frame),
                    entry(1, &s.probes),
                    entry(2, &s.rays.rays),
                    entry(3, &s.rays.hits),
                    entry(4, &s.lists),
                    entry(5, &s.sun),
                    entry(6, directional.unwrap_or(&self.fallback_directional)),
                    entry(7, atmosphere.unwrap_or(&self.fallback_atmosphere)),
                    entry(8, sky_table.unwrap_or(&self.fallback_sky_table)),
                    entry(9, &s.trace),
                ],
            },
        );
        // Whether any ray of this frame meets anything (`gi_light`).
        cmds.clear_buffer(&s.answered, 0, None);
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI update"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&self.sun);
        pass.dispatch_workgroups(1, 1, 1);
        if self.counts[1] > 0 {
            pass.set_pipeline(&self.reset);
            pass.dispatch_workgroups(self.counts[1].div_ceil(64), 1, 1);
        }
        if self.counts[0] > 0 {
            pass.set_pipeline(&self.rays);
            pass.dispatch_workgroups((self.counts[0] * RAYS).div_ceil(64), 1, 1);
        }
    }

    /// The clipmap `begin` placed.
    pub fn clipmap(&self) -> Option<Clipmap> {
        self.state.clipmap
    }
}

impl RenderPass for GiPass {
    fn name(&self) -> &'static str {
        "GI"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            SKY_FRAME,
            SKY_TABLE,
            helio_core::resource_keys::RENDER_ENVIRONMENT,
        ]
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn scene_buffers(&self) -> &'static [&'static str] {
        &[helio_light_api::DIRECTIONAL_LIGHTS]
    }

    fn declare_resources(&self, _builder: &mut ResourceBuilder) {}

    fn inherit_persistent_state(&mut self, previous: &mut dyn RenderPass) -> bool {
        let Some(previous) = previous.as_any_mut().downcast_mut::<Self>() else {
            return false;
        };
        std::mem::swap(&mut self.state, &mut previous.state);
        true
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let environment = ctx
            .registry
            .get::<helio_core::RenderEnvironment>(helio_core::resource_keys::render_environment());
        let (ambient, intensity) = environment.map_or(([0.5, 0.5, 0.6], 1.0), |e| {
            (e.ambient_color, e.ambient_intensity)
        });
        let sky = Vec3::from(ambient) * intensity;
        let frame_eye = Vec3::from_slice(&ctx.camera_data.position_near[..3]);
        self.begin(
            ctx.queue,
            eye(ctx),
            frame_eye,
            ctx.frame_num,
            [sky, sky * 0.15],
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let atmosphere = ctx
            .registry
            .get::<&wgpu::Buffer>(ResourceKey::new(SKY_FRAME))
            .cloned();
        let sky_table = ctx
            .registry
            .get::<&wgpu::Buffer>(ResourceKey::new(SKY_TABLE))
            .cloned();
        let directional = ctx
            .scene_buffers
            .get(helio_light_api::DIRECTIONAL_LIGHTS_ROW.key())
            .map(|h| h.buffer.clone());
        let mut cmds = ctx.graphics_cmds();
        self.record(
            ctx.device,
            &mut cmds,
            atmosphere.as_ref(),
            sky_table.as_ref(),
            directional.as_ref(),
        );
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        let s = &self.state;
        frame.write_buffer_binding(GI_FRAME, &s.frame, "GI");
        frame.write_buffer_binding(GI_PROBES, &s.probes, "GI");
        frame.write_buffer_binding(GI_SUN, &s.sun, "GI");
        frame.write_buffer_binding(GI_LISTS, &s.lists, "GI");
        frame.write_buffer_binding(GI_TRACE, &s.trace, "GI");
        frame.write_buffer_binding(GI_ANSWERED, &s.answered, "GI");
        s.rays.publish(&self.stream, frame, "GI");
    }
}

/// The probes' irradiance at every pixel (module docs): after the
/// integration, before deferred lighting.
pub struct GiResolvePass {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    out: Option<(wgpu::Buffer, [u32; 2])>,
    /// The renderer's debug mode (`GI_PROBES_VIEW` shows the probes), and
    /// the one last written to `view`.
    view: wgpu::Buffer,
    debug_mode: u32,
    view_written: Option<u32>,
    /// Its bind group, kept while it binds the same targets.
    cache: helio_core::gpu_cache::GpuCache,
}

impl GiResolvePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let depth = wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI resolve"),
            entries: &[
                uniform(0, FRAME_BYTES),
                storage(1, true),
                storage(2, true),
                texture(3),
                depth,
                storage(5, false),
                uniform(6, 16),
            ],
        });
        let module = helio_core::shader::module_with(
            device,
            "GI resolve",
            helio_core::include_wgsl!("../shaders/gi_resolve.wgsl"),
            &[GI_COMMON_SNIPPET, GI_SAMPLE_SNIPPET],
        );
        let pipeline_layout = pipeline_layout(device, "GI resolve", &layout);
        let pipeline = compute(device, &module, &pipeline_layout, "gi_resolve", &[]);
        let view = buffer(
            device,
            "GI resolve view",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        Self {
            layout,
            pipeline,
            out: None,
            view,
            debug_mode: 0,
            view_written: None,
            cache: helio_core::gpu_cache::GpuCache::new(),
        }
    }

    /// Records the probes' irradiance at each pixel of a `size` frame
    /// (`gi_irradiance`) from `frame`, `probes` and `camera` (GiPass's
    /// frame and probes, Helio's camera) and the G-buffer's normals and
    /// depth.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cmds: &mut helio_core::CommandRecorder<'_>,
        [frame, probes, camera]: [&wgpu::Buffer; 3],
        normal: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        size: [u32; 2],
    ) {
        let size = [size[0].max(1), size[1].max(1)];
        if self.out.as_ref().is_none_or(|(_, s)| *s != size) {
            let bytes = u64::from(size[0]) * u64::from(size[1]) * 8;
            self.out = Some((
                buffer(
                    device,
                    "GI irradiance",
                    bytes,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                ),
                size,
            ));
        }
        if self.view_written != Some(self.debug_mode) {
            helio_core::upload::write_buffer(
                queue,
                &self.view,
                0,
                bytemuck::cast_slice(&[self.debug_mode, 0, 0, 0]),
            );
            self.view_written = Some(self.debug_mode);
        }
        let Some((out, _)) = &self.out else { return };
        self.cache.end_frame();
        let group = self.cache.bind_group(
            device,
            &wgpu::BindGroupDescriptor {
                label: Some("GI resolve"),
                layout: &self.layout,
                entries: &[
                    entry(0, frame),
                    entry(1, probes),
                    entry(2, camera),
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(depth),
                    },
                    entry(5, out),
                    entry(6, &self.view),
                ],
            },
        );
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI resolve"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&self.pipeline);
        pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
    }

    /// The last frame's `gi_irradiance` (none before one).
    pub fn irradiance(&self) -> Option<&wgpu::Buffer> {
        self.out.as_ref().map(|(b, _)| b)
    }
}

impl RenderPass for GiResolvePass {
    fn name(&self) -> &'static str {
        "GIResolve"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            helio_frame_api::gbuffer::GROUP,
            helio_frame_api::gbuffer::DEPTH,
            GI_FRAME,
            GI_PROBES,
        ]
    }

    fn writes(&self) -> &'static [&'static str] {
        &[GI_IRRADIANCE]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(helio_frame_api::gbuffer::GROUP);
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let (Some(frame), Some(probes)) = (
            ctx.registry.buffer_binding(GI_FRAME),
            ctx.registry.buffer_binding(GI_PROBES),
        ) else {
            self.out = None;
            return Ok(());
        };
        let Some(gbuffer) = ctx
            .registry
            .get::<helio_core::ViewGroup<'_, 4>>(ResourceKey::new(helio_frame_api::gbuffer::GROUP))
        else {
            return Err(helio_core::Error::ResourceNotFound(
                helio_frame_api::gbuffer::GROUP.into(),
            ));
        };
        let mut cmds = ctx.graphics_cmds();
        self.record(
            ctx.device,
            ctx.queue,
            &mut cmds,
            [frame, probes, ctx.camera],
            gbuffer.views[1],
            ctx.depth,
            [ctx.width, ctx.height],
        );
        Ok(())
    }

    fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode;
    }

    fn debug_views(&self) -> &'static [helio_core::DebugViewDescriptor] {
        static VIEWS: &[helio_core::DebugViewDescriptor] = &[helio_core::DebugViewDescriptor {
            name: "GI Probes",
            debug_mode: GI_PROBES_VIEW,
            description: "Each pixel's nearest GI probe (finest level holding it): its irradiance, dark red inside the ground, dark purple not yet updated; its cell's edges darker",
        }];
        VIEWS
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let Some((out, _)) = &self.out {
            frame.write_buffer_binding(GI_IRRADIANCE, out, "GIResolve");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_keep_their_storage_while_they_stay_in() {
        let a = Clipmap::around(DVec3::new(10.3, -4.0, 1e6 + 0.5), DEFAULT_FINEST_SPACING);
        let b = Clipmap::around(DVec3::new(13.9, -4.0, 1e6 + 0.5), DEFAULT_FINEST_SPACING);
        for index in 0..PROBES {
            let (level, world) = b.probe(index);
            assert_eq!(Clipmap::index(level, world), index);
            assert!(b.holds(level, world));
        }
        let entered = b.entered(Some(&a));
        // Level 0 moved by 3 probes in x, level 1 by 1 or 2, coarser by at
        // most 1: whole slabs came in.
        for &index in &entered {
            let (level, world) = b.probe(index);
            assert!(!a.holds(level, world));
        }
        let count = |level: usize| {
            entered
                .iter()
                .filter(|&&i| (i / LEVEL_PROBES) as usize == level)
                .count() as i64
        };
        let moved = |level: usize| (b.first[level].x - a.first[level].x).abs();
        for level in 0..LEVELS {
            assert_eq!(count(level), moved(level) * SIDE * SIDE);
        }
        assert_eq!(a.entered(None).len(), PROBES as usize);
        // Into a kept list, the same.
        let mut kept = vec![7, 7, 7];
        b.entered_into(Some(&a), &mut kept);
        assert_eq!(kept, entered);
        // Another spacing: every probe is new.
        let finer = Clipmap::around(DVec3::new(10.3, -4.0, 1e6 + 0.5), 0.5);
        assert_eq!(finer.entered(Some(&a)).len(), PROBES as usize);
    }

    #[test]
    fn the_eye_is_between_each_level_middle_probes() {
        let eye = DVec3::new(-123.45, 6.0, 789.0);
        for finest in [DEFAULT_FINEST_SPACING, 0.5] {
            let c = Clipmap::around(eye, finest);
            for level in 0..LEVELS {
                let local = eye / c.spacing(level) - c.first[level].as_dvec3();
                assert!(
                    local.cmpge(DVec3::splat(11.0)).all() && local.cmplt(DVec3::splat(12.0)).all(),
                    "{level} {local}"
                );
            }
        }
    }

    /// The shaders' copies of the layout constants (`gi_common.wgsl`,
    /// `gi_rays.wgsl`, `gi_local_lights.wgsl`).
    #[test]
    fn shader_constants_match() {
        let source = GI_COMMON_SNIPPET.source;
        for (name, value) in [
            ("GI_LEVELS", LEVELS as u32),
            ("GI_SIDE", SIDE as u32),
            ("GI_LEVEL_PROBES", LEVEL_PROBES),
            ("GI_PROBES", PROBES),
            ("GI_RAYS", RAYS),
            ("GI_MAX_UPDATES", MAX_UPDATES),
            ("GI_PROBE_WORDS", PROBE_WORDS),
        ] {
            assert!(
                source.contains(&format!("const {name}: u32 = {value}u;")),
                "{name} is not {value} in gi_common.wgsl"
            );
        }
        assert!(local_lights::SNIPPET.source.contains(&format!(
            "const GI_MAX_LOCAL_LIGHTS: u32 = {}u;",
            local_lights::MAX_LOCAL_LIGHTS
        )));
        // `GiTrace` and `GiPending`: their fields, as the shaders read them.
        let rays = GI_RAYS_SNIPPET.source;
        assert!(rays.contains("struct GiTrace {\n    eye: vec4<f32>,\n    info: vec4<u32>,\n}"));
        assert_eq!(rays.matches("    sun: u32,\n    local: u32,\n    rest: u32,\n    t: f32,\n    sun_row: u32,\n    local_row: u32,\n").count(), 1);
        assert_eq!(PENDING_BYTES, 6 * 4);
        assert_eq!(TRACE_BYTES, 32);
    }

    #[test]
    fn updates_take_entered_probes_first_then_all_in_turn() {
        let mut s = Schedule::new();
        s.enter(5);
        s.enter(5);
        s.enter(LEVEL_PROBES + 7);
        let mut out = Vec::new();
        s.next(&mut out);
        assert_eq!(out.len(), MAX_UPDATES as usize);
        assert_eq!(out[0], 5);
        assert_eq!(out[1], 0);
        assert_eq!(out[LEVEL_UPDATES[0]], LEVEL_PROBES + 7);
        // Every probe of every level comes round.
        let mut seen = vec![false; PROBES as usize];
        for _ in 0..(LEVEL_PROBES as usize).div_ceil(LEVEL_UPDATES[LEVELS - 1]) + 1 {
            s.next(&mut out);
            for &i in &out {
                seen[i as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn rotations_are_unit() {
        for seed in 0..64 {
            let q = rotation(seed);
            let n: f32 = q.iter().map(|c| c * c).sum();
            assert!((n - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn the_streams_hold_every_ray() {
        let probes = probe_stream();
        assert_eq!(probes.class, RayClass::Nearest);
        assert_eq!(probes.capacity, MAX_RAYS);
        let shadows = shadow_stream();
        assert_eq!(shadows.class, RayClass::Occlusion);
        // A sun ray and a local light's a probe ray at most.
        assert_eq!(shadows.capacity, 2 * MAX_RAYS);
        assert_ne!(probes.rays, shadows.rays);
    }
}

#[cfg(test)]
mod shader_tests {
    use super::*;

    fn validate(label: &str, source: &str, snippets: &[helio_core::shader::ShaderSnippet]) {
        let text = helio_core::shader::resolve_with(source, snippets);
        let module = wgpu::naga::front::wgsl::parse_str(&text)
            .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&text)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&text)));
        // Every probe ray or pixel runs these: no per-frame table (the
        // atmosphere's directional lights) indexed through a value, which
        // copies the whole table per invocation.
        let copies = helio_core::shader::value_array_copies(&text).unwrap();
        assert!(copies.is_empty(), "{label}: {copies:#?}");
    }

    #[test]
    fn shaders_validate() {
        validate(
            "gi_update",
            include_str!("../shaders/gi_update.wgsl"),
            &[
                SKY_SNIPPET,
                SKY_TABLE_SNIPPET,
                stream::SNIPPET,
                GI_COMMON_SNIPPET,
                GI_RAYS_SNIPPET,
                helio_light_api::LIGHT_LIST_SNIPPET,
            ],
        );
        validate(
            "gi_light",
            include_str!("../shaders/gi_light.wgsl"),
            &light::SNIPPETS,
        );
        validate(
            "gi_gather_lights",
            include_str!("../shaders/gi_gather_lights.wgsl"),
            &local_lights::GATHER_SNIPPETS,
        );
        validate(
            "gi_integrate",
            include_str!("../shaders/gi_integrate.wgsl"),
            &integrate::SNIPPETS,
        );
        validate(
            "gi_resolve",
            include_str!("../shaders/gi_resolve.wgsl"),
            &[GI_COMMON_SNIPPET, GI_SAMPLE_SNIPPET],
        );
    }
}
