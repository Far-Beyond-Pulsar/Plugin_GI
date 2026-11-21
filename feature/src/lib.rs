//! World-space diffuse global illumination (README.md): a camera-centred
//! clipmap of irradiance probes and the frame's radiance cache, kept up to
//! date by rays every kind's tracer answers, read by deferred lighting at
//! every pixel.
//!
//! GI is a consumer of the frame's ray streams
//! (`helio_frame_api::rays::stream`) and knows no kind; it publishes the
//! frame's radiance cache (`helio_frame_api::rays::radiance_cache`), where a
//! surface is lit for indirect light once for every reader. Each frame,
//! after the G-buffer phase (the frame's geometry as it was drawn):
//!
//! 1. [`GiPass`]: the probes that came into the clipmap reset, this frame's
//!    probe rays into GI's stream ([`gi_stream`]).
//! 2. [`RadianceCache`]: the cells no reader marked lately evicted, this
//!    frame's cells to light listed, their rays after the probes'.
//! 3. The kinds' tracers answer the stream (`stream::trace`): each ray's
//!    nearest hit and its surface.
//! 4. [`GiLightPass`]: every hit's light from the cache (a probe's hit all of
//!    it, a cache ray's hit its indirect part), marking its cell; a light
//!    query at each cache ray's hit and at its cell into the cache's light
//!    batch ([`light_batch`]).
//! 5. The frame's light samplers answer it (`light_samples::sample`): one
//!    light a query, from every kind of light that registered a sampler (the
//!    sun among them).
//! 6. [`GiShadowPass`]: each cache ray's light samples, waiting on shadow
//!    rays, into the shadow stream ([`shadow_stream`]).
//! 7. The tracers answer that (what gets through each).
//! 8. [`GiIntegratePass`]: the probes' rays' radiance into their probes.
//! 9. [`RadianceCacheFold`]: the cache rays' light into their cells.
//!
//! GI reads no light kind's rows: lights reach it as light samples.
//!
//! [`GiResolvePass`] runs before deferred lighting: the probes' irradiance at
//! every pixel (`gi_irradiance`). The clipmap is a function of the eye's
//! world position and the finest spacing alone ([`Clipmap::around`],
//! [`eye`]).

use glam::{DVec3, Vec3};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_frame_api::lighting::sky::{SKY_FRAME, SKY_TABLE};
use helio_frame_api::lighting::{
    VIEW_REMAPS, VIEW_REMAPS_BYTES, VIEW_REMAP_SPACES, VIEW_REMAP_SPACES_BYTES,
};
use helio_frame_api::rays::light_samples::LightBatch;
use helio_frame_api::rays::radiance_cache;
use helio_frame_api::rays::stream::{self, RayClass, Stream, StreamBuffers, SurfaceDetail};
use std::collections::VecDeque;

pub mod cache;
mod feature;
pub mod gather;
mod integrate;
mod light;
pub mod settings;
pub use cache::{CacheConfig, RadianceCache, RadianceCacheFold};
pub use feature::GiFeature;
pub use gather::{GatherConfig, GiGatherPass, GiGatherShadePass};
pub use integrate::GiIntegratePass;
pub use light::{GiLightPass, GiShadowPass};
pub use settings::{GiQuality, GiSettings, GI};

pub use helio_gi_api::{
    eye, Clipmap, GpuFrame, GpuLevel, DEFAULT_FINEST_SPACING, FRAME_BYTES, GI_COMMON_SNIPPET,
    GI_FRAME, GI_IRRADIANCE, GI_NEAR_FIELD_VIEW, GI_PROBES, GI_PROBES_VIEW, GI_SAMPLE_SNIPPET,
    LEVELS, LEVEL_PROBES, LEVEL_UPDATES, MAX_RAYS, MAX_UPDATES, PROBES, PROBE_WORDS,
    RADIANCE_CACHE_VIEW, RAYS, SIDE,
};

/// GI's rays (`SceneRay`, a row a ray): the probes' first, probe-major (64 a
/// probe in update-list order, `MAX_RAYS` at most), then the radiance
/// cache's (`CacheConfig::rays` at most, `cache::RAYS_PER_CELL` a cell): the
/// nearest hit and its surface, one stream so every tracer's fixed cost a
/// stream (a dispatch a volume, a binned material pass a program) is paid
/// once. Not runs: each probe ray starts behind its probe along its own
/// direction (`gi_update.wgsl`), and a stream's runs are the whole stream's.
pub fn gi_stream(config: &GiConfig) -> Stream {
    Stream::new("gi", RayClass::Nearest, MAX_RAYS + config.cache.rays)
        .with_detail(SurfaceDetail::Diffuse)
}

/// The shadow rays of the cache rays: toward each cache ray's hit's light
/// sample and its cell's, as many rows as there are such rays this frame,
/// packed from 0 (the shadow pass writes the count on the GPU), at most two
/// a cache ray.
pub fn shadow_stream(cache: &CacheConfig) -> Stream {
    Stream::new("radiance_cache_shadow", RayClass::Occlusion, 2 * cache.rays)
}

/// The cache rays asking the frame's light samplers for one light each at
/// their hit and at their cell: as many queries as that this frame, packed
/// from 0 (the light pass writes the count on the GPU).
pub fn light_batch(cache: &CacheConfig) -> LightBatch {
    LightBatch::new("radiance_cache", 2 * cache.rays)
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
const PENDING_BYTES: u64 = 16;

/// The snippets `gi_resolve.wgsl` is built with.
const RESOLVE_SNIPPETS: [helio_core::shader::ShaderSnippet; 5] = [
    stream::SNIPPET,
    radiance_cache::SNIPPET,
    GI_COMMON_SNIPPET,
    GI_SAMPLE_SNIPPET,
    helio_frame_api::denoise::IRRADIANCE_SNIPPET,
];

/// The snippets `gi_update.wgsl` is built with.
const UPDATE_SNIPPETS: [helio_core::shader::ShaderSnippet; 3] =
    [stream::SNIPPET, GI_COMMON_SNIPPET, GI_RAYS_SNIPPET];

/// The probe rays' WGSL (`gi_rays.wgsl`, `//!use gi_rays`).
const GI_RAYS_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use gi_rays", "../shaders/gi_rays.wgsl");

/// How GI lights a ray's hit or miss (`gi_hit.wgsl`, `//!use gi_hit`): the
/// sky along a direction, a hit's light from the radiance cache (the probes
/// standing in where it has none), what a hit sends back along its ray. The
/// one copy every reader of GI's streams (the probes' and the cache's rays,
/// the pixels' rays) lights its hits with.
pub(crate) const GI_HIT_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use gi_hit", "../shaders/gi_hit.wgsl");

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

/// The atmosphere's frame and sky table, as the sky publishes them (registry
/// values under `SKY_FRAME` and `SKY_TABLE`), if any: the one lookup GI's
/// passes share (the sky a probe ray meets).
fn sky<'a>(registry: &helio_core::ResourceRegistry<'a>) -> [Option<&'a wgpu::Buffer>; 2] {
    [SKY_FRAME, SKY_TABLE].map(|name| registry.get::<&'a wgpu::Buffer>(ResourceKey::new(name)))
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
    trace: wgpu::Buffer,
    answered: wgpu::Buffer,
    rays: StreamBuffers,
    clipmap: Option<Clipmap>,
    schedule: Schedule,
}

impl Probes {
    fn new(device: &wgpu::Device, stream: &Stream) -> Self {
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
            trace: buffer(device, "GI trace", TRACE_BYTES, uniform),
            answered: buffer(device, "GI answered", 16, storage),
            rays: StreamBuffers::new(device, stream),
            clipmap: None,
            schedule: Schedule::new(),
        }
    }
}

/// How GI's passes run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiConfig {
    /// The finest level's probe spacing (metres; each coarser level's
    /// doubles): finer indoors, where rooms, pillars and furniture are a few
    /// metres apart, than over open ground.
    pub finest_spacing: f64,
    /// The radiance cache's capacity and ray budget.
    pub cache: CacheConfig,
    /// The pixels' gather ([`gather`]), or none: the probes' irradiance at
    /// every pixel.
    pub pixels: Option<GatherConfig>,
}

impl Default for GiConfig {
    fn default() -> Self {
        GiSettings::default().config()
    }
}

/// The start of the probes' update (module docs): GI's stream's generator
/// (its probe rays; the cache's follow).
pub struct GiPass {
    config: GiConfig,
    stream: Stream,
    /// The probe stream's answers, started each frame.
    answers: std::sync::Arc<stream::Answers>,
    state: Probes,
    layout: wgpu::BindGroupLayout,
    reset: wgpu::ComputePipeline,
    rays: wgpu::ComputePipeline,
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
    /// The pass, GI's stream's answers `answers` ([`stream::answers`]).
    pub fn new(
        device: &wgpu::Device,
        config: GiConfig,
        stats: &helio_core::FrameStats,
        answers: std::sync::Arc<stream::Answers>,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI update"),
            entries: &[
                uniform(0, FRAME_BYTES),
                storage(1, false),
                storage(2, false),
                storage(3, false),
                storage(4, true),
                uniform(5, TRACE_BYTES),
            ],
        });
        let module = helio_core::shader::module_with(
            device,
            "GI update",
            helio_core::include_wgsl!("../shaders/gi_update.wgsl"),
            &UPDATE_SNIPPETS,
        );
        let pipeline_layout = pipeline_layout(device, "GI update", &layout);
        let pipeline = |entry: &str| compute(device, &module, &pipeline_layout, entry, &[]);
        let stream = gi_stream(&config);
        Self {
            config,
            stream,
            answers,
            state: Probes::new(device, &stream),
            reset: pipeline("gi_reset"),
            rays: pipeline("gi_rays_main"),
            layout,
            cache: helio_core::gpu_cache::GpuCache::new(),
            stats: stats.clone(),
            stat: stream.stat(stats),
            writes: helio_frame_api::static_names(&[
                GI_FRAME,
                GI_PROBES,
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
    /// point `frame_eye` (the frame's space is the rays'), the probes' rays
    /// in coordinate space `space` (`SceneRay::space`: the camera's, as the
    /// probes light what it sees); without an atmosphere the sky's radiance
    /// is `sky` above the horizon and `ground` below it.
    fn begin(
        &mut self,
        queue: &wgpu::Queue,
        eye: DVec3,
        frame_eye: Vec3,
        space: u32,
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
            rotation: rotation(frame),
            updated,
            reset: self.entered.len() as u32,
            _pad: [0; 2],
        };
        let trace = GpuTrace {
            eye: frame_eye.extend(0.0).to_array(),
            info: [(frame as u32).wrapping_mul(0x9e37_79b9), space, 0, 0],
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

    /// Records the frame `begin` started: the probes that came in reset,
    /// this frame's rays (their hits cleared).
    fn record(&mut self, device: &wgpu::Device, cmds: &mut helio_core::CommandRecorder<'_>) {
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
                    entry(5, &s.trace),
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

    /// The probes' records (`gi_probes`), for a reader outside the graph.
    pub fn probes(&self) -> &wgpu::Buffer {
        &self.state.probes
    }
}

impl RenderPass for GiPass {
    fn name(&self) -> &'static str {
        "GI"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[helio_core::resource_keys::RENDER_ENVIRONMENT]
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn declare_resources(&self, _builder: &mut ResourceBuilder) {}

    fn inherit_persistent_state(&mut self, previous: &mut dyn RenderPass) -> bool {
        let Some(previous) = previous.as_any_mut().downcast_mut::<Self>() else {
            return false;
        };
        // The stream's rows are sized by the cache's budget.
        if previous.stream.capacity != self.stream.capacity {
            return false;
        }
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
            ctx.camera_data.view_info[3],
            ctx.frame_num,
            [sky, sky * 0.15],
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.answers.begin();
        let mut cmds = ctx.graphics_cmds();
        self.record(ctx.device, &mut cmds);
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        let s = &self.state;
        frame.write_buffer_binding(GI_FRAME, &s.frame, "GI");
        frame.write_buffer_binding(GI_PROBES, &s.probes, "GI");
        frame.write_buffer_binding(GI_LISTS, &s.lists, "GI");
        frame.write_buffer_binding(GI_TRACE, &s.trace, "GI");
        frame.write_buffer_binding(GI_ANSWERED, &s.answered, "GI");
        s.rays.publish(&self.stream, frame, "GI");
    }
}

/// The frame's indirect diffuse at every pixel (module docs): after the
/// integration, before deferred lighting. The probes' irradiance, or the
/// pixels' gather as the frame's denoiser filtered it times the near field
/// ([`Self::with_gather`]; the probes' where it has no light yet). Its debug
/// views show the probes ([`GI_PROBES_VIEW`]) and the radiance cache's cells
/// ([`RADIANCE_CACHE_VIEW`]: what a pixel-wide ray's read of the frame's
/// cache finds at each pixel's surface).
pub struct GiResolvePass {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    /// The pixels' gather it resolves, if any.
    gather: Option<ResolveGather>,
    out: Option<(wgpu::Buffer, [u32; 2])>,
    /// The renderer's debug mode (`GI_PROBES_VIEW` shows the probes), and
    /// the one last written to `view`.
    view: wgpu::Buffer,
    no_remaps: wgpu::Buffer,
    debug_mode: u32,
    view_written: Option<(u32, bool)>,
    /// What it binds for the radiance cache where none is published.
    no_cache: radiance_cache::StandIn,
    /// Its bind group, kept while it binds the same targets.
    cache: helio_core::gpu_cache::GpuCache,
}

/// What the resolve reads of the pixels' gather.
struct ResolveGather {
    filtered: helio_frame_api::denoise::Filtered,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    /// `resolve_gather` (gi_resolve.wgsl) and the values last written.
    uniform: wgpu::Buffer,
    values: [u32; 4],
    written: Option<[u32; 4]>,
    reads: &'static [&'static str],
}

impl GiResolvePass {
    /// The resolve of the pixels' gather: `filtered` (what the frame's
    /// denoiser made of the gather's signal, traced at one pixel a block of
    /// `side` x `side`) times the near field ([`gather::NEAR_FIELD`]), the
    /// probes' irradiance where it has no light yet.
    pub fn with_gather(
        device: &wgpu::Device,
        filtered: helio_frame_api::denoise::Filtered,
        side: u32,
    ) -> Self {
        let mut pass = Self::new(device);
        let mut entries = Self::entries();
        entries.extend([uniform(13, 16), texture(14), texture(15), texture(16)]);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI resolve pixels"),
            entries: &entries,
        });
        let module = helio_core::shader::module_with(
            device,
            "GI resolve",
            helio_core::include_wgsl!("../shaders/gi_resolve.wgsl"),
            &RESOLVE_SNIPPETS,
        );
        let pipeline_layout = pipeline_layout(device, "GI resolve pixels", &layout);
        let pipeline = compute(device, &module, &pipeline_layout, "gi_resolve_pixels", &[]);
        let mut reads = vec![
            helio_frame_api::gbuffer::GROUP,
            helio_frame_api::gbuffer::DEPTH,
            GI_FRAME,
            GI_PROBES,
            gather::NEAR_FIELD,
        ];
        reads.extend(filtered.textures);
        pass.gather = Some(ResolveGather {
            filtered,
            layout,
            pipeline,
            uniform: buffer(
                device,
                "GI resolve gather",
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            values: [side, 0, u32::from(filtered.denoiser.is_some()), 0],
            written: None,
            reads: helio_frame_api::static_names(&reads),
        });
        pass
    }

    /// The probes' resolve's bindings (0..12 of `gi_resolve.wgsl`).
    fn entries() -> Vec<wgpu::BindGroupLayoutEntry> {
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
        vec![
            uniform(0, FRAME_BYTES),
            storage(1, true),
            storage(2, true),
            texture(3),
            depth,
            storage(5, false),
            uniform(6, 16),
            texture(7),
            uniform(8, VIEW_REMAPS_BYTES),
            uniform(9, VIEW_REMAP_SPACES_BYTES),
            uniform(10, radiance_cache::RadianceCacheFrame::BYTES),
            storage(11, false),
            storage(12, false),
        ]
    }

    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI resolve"),
            entries: &Self::entries(),
        });
        let module = helio_core::shader::module_with(
            device,
            "GI resolve",
            helio_core::include_wgsl!("../shaders/gi_resolve.wgsl"),
            &RESOLVE_SNIPPETS,
        );
        let pipeline_layout = pipeline_layout(device, "GI resolve", &layout);
        let pipeline = compute(device, &module, &pipeline_layout, "gi_resolve", &[]);
        let view = buffer(
            device,
            "GI resolve view",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let no_remaps = buffer(
            device,
            "GI no view remaps",
            VIEW_REMAPS_BYTES,
            wgpu::BufferUsages::UNIFORM,
        );
        Self {
            layout,
            pipeline,
            gather: None,
            out: None,
            view,
            no_remaps,
            debug_mode: 0,
            view_written: None,
            no_cache: radiance_cache::stand_in(device),
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
        self.record_with_remaps(
            device,
            queue,
            cmds,
            [frame, probes, camera],
            normal,
            depth,
            size,
            None,
            None,
            None,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn record_with_remaps(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cmds: &mut helio_core::CommandRecorder<'_>,
        [frame, probes, camera]: [&wgpu::Buffer; 3],
        normal: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        size: [u32; 2],
        remaps: Option<(&wgpu::TextureView, &wgpu::Buffer, &wgpu::Buffer)>,
        cache: Option<[&wgpu::Buffer; 3]>,
        gathered: Option<[&wgpu::TextureView; 3]>,
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
        let view = (self.debug_mode, remaps.is_some());
        if self.view_written != Some(view) {
            helio_core::upload::write_buffer(
                queue,
                &self.view,
                0,
                bytemuck::cast_slice(&[view.0, u32::from(view.1), 0, 0]),
            );
            self.view_written = Some(view);
        }
        let (surface, maps, spaces) = remaps.unwrap_or((normal, &self.no_remaps, &self.no_remaps));
        let [cache_frame, slots, entries] = cache.unwrap_or([
            &self.no_cache.frame,
            &self.no_cache.slots,
            &self.no_cache.entries,
        ]);
        let Some((out, _)) = &self.out else { return };
        // The gather's textures, where it ran this frame (the probes'
        // resolve where it did not).
        if let (Some(g), Some(_)) = (self.gather.as_mut(), gathered) {
            if g.written != Some(g.values) {
                helio_core::upload::write_buffer(
                    queue,
                    &g.uniform,
                    0,
                    bytemuck::cast_slice(&g.values),
                );
                g.written = Some(g.values);
            }
        }
        let gather = self.gather.as_ref().zip(gathered);
        self.cache.end_frame();
        let common = [
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
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(surface),
            },
            entry(8, maps),
            entry(9, spaces),
            entry(10, cache_frame),
            entry(11, slots),
            entry(12, entries),
        ];
        let texture_entry = |binding, view| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        };
        let (group, pipeline) = match gather {
            Some((g, [a, b, near])) => {
                let [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12] = common;
                let entries = [
                    e0,
                    e1,
                    e2,
                    e3,
                    e4,
                    e5,
                    e6,
                    e7,
                    e8,
                    e9,
                    e10,
                    e11,
                    e12,
                    entry(13, &g.uniform),
                    texture_entry(14, a),
                    texture_entry(15, b),
                    texture_entry(16, near),
                ];
                let group = self.cache.bind_group(
                    device,
                    &wgpu::BindGroupDescriptor {
                        label: Some("GI resolve pixels"),
                        layout: &g.layout,
                        entries: &entries,
                    },
                );
                (group, &g.pipeline)
            }
            None => {
                let group = self.cache.bind_group(
                    device,
                    &wgpu::BindGroupDescriptor {
                        label: Some("GI resolve"),
                        layout: &self.layout,
                        entries: &common,
                    },
                );
                (group, &self.pipeline)
            }
        };
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI resolve"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(pipeline);
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
        if let Some(g) = &self.gather {
            return g.reads;
        }
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

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // The traced pixel a pass-through signal's block holds this frame
        // (the denoiser's frame number).
        if let Some(g) = &mut self.gather {
            g.values[1] = ctx.frame_num as u32;
        }
        Ok(())
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(helio_frame_api::gbuffer::GROUP);
        builder.read_optional(helio_frame_api::gbuffer::SURFACE);
        builder.read_optional(VIEW_REMAPS);
        builder.read_optional(VIEW_REMAP_SPACES);
        for key in [
            radiance_cache::FRAME,
            radiance_cache::SLOTS,
            radiance_cache::ENTRIES,
        ] {
            builder.read_optional(key);
        }
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
        let remaps = ctx
            .registry
            .texture_view(ResourceKey::new(helio_frame_api::gbuffer::SURFACE))
            .zip(ctx.registry.buffer_binding(VIEW_REMAPS))
            .zip(ctx.registry.buffer_binding(VIEW_REMAP_SPACES))
            .map(|((surface, maps), spaces)| (surface, maps, spaces));
        let binding = |name: &str| ctx.registry.buffer_binding(name);
        let cache = match (
            binding(radiance_cache::FRAME),
            binding(radiance_cache::SLOTS),
            binding(radiance_cache::ENTRIES),
        ) {
            (Some(frame), Some(slots), Some(entries)) => Some([frame, slots, entries]),
            _ => None,
        };
        let gathered = self.gather.as_ref().and_then(|g| {
            let view = |name: &'static str| ctx.registry.texture_view(ResourceKey::new(name));
            Some([
                view(g.filtered.textures[0])?,
                view(g.filtered.textures[1])?,
                view(gather::NEAR_FIELD)?,
            ])
        });
        let mut cmds = ctx.graphics_cmds();
        self.record_with_remaps(
            ctx.device,
            ctx.queue,
            &mut cmds,
            [frame, probes, ctx.camera],
            gbuffer.views[1],
            ctx.depth,
            [ctx.width, ctx.height],
            remaps,
            cache,
            gathered,
        );
        Ok(())
    }

    fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode;
    }

    fn debug_views(&self) -> &'static [helio_core::DebugViewDescriptor] {
        static VIEWS: &[helio_core::DebugViewDescriptor] = &[
            helio_core::DebugViewDescriptor {
                name: "GI Probes",
                debug_mode: GI_PROBES_VIEW,
                description: "Each pixel's nearest GI probe (finest level holding it): its irradiance, dark red inside the ground, dark purple not yet updated; its cell's edges darker",
            },
            helio_core::DebugViewDescriptor {
                name: "GI Near Field",
                debug_mode: GI_NEAR_FIELD_VIEW,
                description: "With the pixels' gather: what each pixel's gathered irradiance is multiplied by, the occlusion within a traced block's footprint (white: none; a surface with its own occlusion shows its share over its material's)",
            },
            helio_core::DebugViewDescriptor {
                name: "Radiance Cache",
                debug_mode: RADIANCE_CACHE_VIEW,
                description: "The radiance cache's cell at each pixel's surface, as a pixel-wide ray would read it: its irradiance (direct and indirect) / pi, dark purple where no cell holds a value; its cell's edges darker",
            },
        ];
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
        let coarser = Clipmap::around(DVec3::new(10.3, -4.0, 1e6 + 0.5), 1.0);
        assert_eq!(coarser.entered(Some(&a)).len(), PROBES as usize);
    }

    #[test]
    fn the_eye_is_between_each_level_middle_probes() {
        let eye = DVec3::new(-123.45, 6.0, 789.0);
        for finest in [DEFAULT_FINEST_SPACING, 1.0] {
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
    /// `gi_rays.wgsl`).
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
        // `GiTrace` and `GiPending`: their fields, as the shaders read them.
        // As written, whatever line endings the checkout gave the file.
        let rays = GI_RAYS_SNIPPET.source.replace("\r\n", "\n");
        assert!(rays.contains("struct GiTrace {\n    eye: vec4<f32>,\n    info: vec4<u32>,\n}"));
        assert_eq!(
            rays.matches("    light: u32,\n    rest: u32,\n    t: f32,\n    row: u32,\n")
                .count(),
            1
        );
        assert_eq!(PENDING_BYTES, 4 * 4);
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
        let config = GiConfig::default();
        let rays = gi_stream(&config);
        assert_eq!(rays.class, RayClass::Nearest);
        // Every probe ray, then the cache's.
        assert_eq!(rays.capacity, MAX_RAYS + config.cache.rays);
        assert_eq!(rays.run, 1);
        let shadows = shadow_stream(&config.cache);
        assert_eq!(shadows.run, 1);
        assert_eq!(shadows.class, RayClass::Occlusion);
        // Two shadow rays and two light queries a cache ray at most (its
        // hit's and its cell's); none for a probe's.
        assert_eq!(shadows.capacity, 2 * config.cache.rays);
        assert_eq!(light_batch(&config.cache).capacity, 2 * config.cache.rays);
        assert_ne!(rays.rays, shadows.rays);
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
            "gi_cache",
            include_str!("../shaders/gi_cache.wgsl"),
            &cache::SNIPPETS,
        );
        validate(
            "gi_update",
            include_str!("../shaders/gi_update.wgsl"),
            &UPDATE_SNIPPETS,
        );
        validate(
            "gi_light",
            include_str!("../shaders/gi_light.wgsl"),
            &light::SNIPPETS,
        );
        validate(
            "gi_integrate",
            include_str!("../shaders/gi_integrate.wgsl"),
            &integrate::SNIPPETS,
        );
        validate(
            "gi_resolve",
            include_str!("../shaders/gi_resolve.wgsl"),
            &RESOLVE_SNIPPETS,
        );
        validate(
            "gi_gather",
            include_str!("../shaders/gi_gather.wgsl"),
            &gather::SNIPPETS,
        );
    }
}
