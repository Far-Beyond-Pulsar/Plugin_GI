//! Indirect diffuse at a pixel (`gi_gather.wgsl`, README.md "The pixels'
//! gather"; Helio's `docs/architecture/rendering.md`, section 5): one ray a
//! block of internal pixels from the G-buffer's surface, its hit lit from the
//! frame's radiance cache, kept in a reservoir a traced texel and reused
//! across frames and neighbours (ReSTIR GI), written as the frame's
//! denoiser's irradiance signal, which the resolve turns into the frame's
//! indirect diffuse. Frame contracts only: the G-buffer, a ray stream, the
//! radiance cache, the frame's history, the denoiser.
//!
//! [`GiGatherPass`]: the near field at every pixel (occlusion within a
//! block's footprint, the horizon over the depth buffer), each traced texel's
//! ray into the stream [`pixel_stream`], and one reservoir in
//! [`RETRACE_PERIOD`] re-traced along its kept sample. The tracers answer;
//! [`GiGatherShadePass`]: the re-traced samples' light now (a change lowers
//! the confidence the denoiser is given), this frame's sample combined with
//! the reservoir where the pixel was (through the frame's history) and with
//! neighbours', into the signal.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_frame_api::denoise::{self, Filtered, Signal, TraceBlock};
use helio_frame_api::history::{self, Need};
use helio_frame_api::lighting::sky::{
    SKY_FRAME_BYTES, SKY_SNIPPET, SKY_TABLE_BYTES, SKY_TABLE_SNIPPET,
};
use helio_frame_api::rays::radiance_cache::{self, RadianceCacheFrame};
use helio_frame_api::rays::stream::{self, RayClass, Stream, StreamBuffers, SurfaceDetail};

use crate::{
    buffer, compute, entry, storage, uniform, FRAME_BYTES, GI_COMMON_SNIPPET, GI_FRAME,
    GI_HIT_SNIPPET, GI_PROBES, GI_RAYS_SNIPPET, GI_SAMPLE_SNIPPET, GI_TRACE, TRACE_BYTES,
};

/// What the pixels' gather traces: the block of internal pixels one ray
/// stands for, and whether its samples are reused across frames and
/// neighbours (off: one cosine-distributed ray a traced pixel, the plain
/// estimate the reservoirs are checked against).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatherConfig {
    pub block: TraceBlock,
    pub reuse: bool,
}

impl GatherConfig {
    /// The denoiser's signal the gather writes.
    pub fn signal(&self) -> Signal {
        Signal::irradiance("gi_pixels", self.block).with_confidence()
    }
}

/// The most samples' worth (M) a temporal reservoir keeps. A kept sample
/// stays about M + 1 frames (a new one replaces it with odds 1 in M + 1),
/// so over the denoiser's 32 frames of history a traced pixel's estimates
/// rest on at least 32 / (M + 1) independent samples: 8 keeps about four,
/// while resampling still picks among 8 samples' worth.
pub const M_CAP: f32 = 8.0;
/// One reservoir in this many is re-traced a frame: a light switched on or
/// off reaches every reservoir's confidence within this many frames (67 ms
/// at 60 Hz), at a quarter more rays.
pub const RETRACE_PERIOD: u32 = 4;
/// The spatial neighbours' radius (traced texels): the denoiser's own reach
/// at full history (its two a-trous steps of 1 and 2 blocks), so the
/// correlation reuse leaves between texels is within what the denoiser
/// averages anyway (beyond it, blotches it cannot remove).
pub const SPATIAL_RADIUS: f32 = 3.0;
/// How near a surface the gather resolves occlusion (metres): its rays start
/// `gi_ray_offset` off their pixel's surface (2 mm and f32's rounding there);
/// nearer than a block's footprint, the near field resolves it at every
/// pixel. Offered as `indirect_occlusion`.
pub const OCCLUSION_START: f32 = 0.002;

/// A reservoir's bytes (`GatherReservoir`).
pub const RESERVOIR_BYTES: u64 = 48;

/// The stream of the pixels' rays: a traced texel's ray at its index, then
/// the re-traced samples' (one texel in [`RETRACE_PERIOD`]).
pub fn pixel_stream(traced: [u32; 2]) -> Stream {
    Stream::new("gi_pixels", RayClass::Nearest, rows(traced)).with_detail(SurfaceDetail::Diffuse)
}

/// Rows of the stream for `traced` texels.
fn rows(traced: [u32; 2]) -> u32 {
    let n = traced[0] * traced[1];
    n + n.div_ceil(RETRACE_PERIOD)
}

/// The re-traced texels of `n` on frame `frame`: those whose index is the
/// frame's phase modulo [`RETRACE_PERIOD`], and how many there are.
fn retrace_phase(n: u32, frame: u32) -> (u32, u32) {
    let phase = (RETRACE_PERIOD - frame % RETRACE_PERIOD) % RETRACE_PERIOD;
    let count = if phase < n {
        (n - phase).div_ceil(RETRACE_PERIOD)
    } else {
        0
    };
    (phase, count)
}

/// Graph keys between the gather's passes and the resolve.
const GATHER_FRAME: &str = "gi_gather_frame";
const GATHER_PREVIOUS: &str = "gi_gather_previous";
const GATHER_RESERVOIRS: &str = "gi_gather_reservoirs";
const GATHER_TEXELS: &str = "gi_gather_texels";
/// The near field (`Rgba16Float`, internal resolution, x and y; `gather_near`).
pub const NEAR_FIELD: &str = "gi_near_field";

/// `GatherFrame` (gi_gather.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
struct GpuGatherFrame {
    size: [u32; 4],
    info: [u32; 4],
    rebase: [f32; 4],
    reuse: [f32; 4],
    forward: [f32; 4],
}

const GATHER_FRAME_BYTES: u64 = std::mem::size_of::<GpuGatherFrame>() as u64;

const HAS_PREVIOUS: u32 = 1;
const REUSE: u32 = 2;
const TAGS: u32 = 4;

/// The snippets `gi_gather.wgsl` is built with.
pub(crate) const SNIPPETS: [helio_core::shader::ShaderSnippet; 13] = [
    SKY_SNIPPET,
    SKY_TABLE_SNIPPET,
    stream::SNIPPET,
    stream::SURFACE_SNIPPET,
    helio_frame_api::gbuffer::SURFACE_SNIPPET,
    helio_frame_api::gbuffer::BSDF_SNIPPET,
    radiance_cache::SNIPPET,
    history::SNIPPET,
    denoise::IRRADIANCE_SNIPPET,
    GI_COMMON_SNIPPET,
    GI_SAMPLE_SNIPPET,
    GI_RAYS_SNIPPET,
    GI_HIT_SNIPPET,
];

fn module(device: &wgpu::Device) -> wgpu::ShaderModule {
    helio_core::shader::module_with(
        device,
        "GI gather",
        helio_core::include_wgsl!("../shaders/gi_gather.wgsl"),
        &SNIPPETS,
    )
}

fn texture(binding: u32) -> wgpu::BindGroupLayoutEntry {
    crate::texture(binding)
}

fn depth_texture(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_texture(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn view_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

/// The G-buffer's bindings every gather pipeline reads (0..5): the frame,
/// the camera, depth, normal, surface tags.
fn surface_entries() -> [wgpu::BindGroupLayoutEntry; 5] {
    [
        uniform(0, GATHER_FRAME_BYTES),
        storage(1, true),
        depth_texture(2),
        texture(3),
        texture(4),
    ]
}

/// The G-buffer as the frame published it: depth, normal, ORM, and the
/// surface tags if any (the normal stands in, unread, where none).
struct GBuffer<'a> {
    depth: &'a wgpu::TextureView,
    normal: &'a wgpu::TextureView,
    orm: &'a wgpu::TextureView,
    tags: Option<&'a wgpu::TextureView>,
}

fn gbuffer<'a>(
    registry: &helio_core::ResourceRegistry<'a>,
    depth: &'a wgpu::TextureView,
) -> Option<GBuffer<'a>> {
    let group = registry
        .get::<helio_core::ViewGroup<'a, 4>>(ResourceKey::new(helio_frame_api::gbuffer::GROUP))?;
    Some(GBuffer {
        depth,
        normal: group.views[1],
        orm: group.views[2],
        tags: registry.texture_view(ResourceKey::new(helio_frame_api::gbuffer::SURFACE)),
    })
}

/// A view's reservoirs (kept across frames: by view id, as the denoiser and
/// the history keep theirs).
struct GatherView {
    id: u32,
    used: u64,
    size: [u32; 2],
    /// The two reservoir sets: this frame's is `latest`, the previous
    /// frame's the other.
    reservoirs: [wgpu::Buffer; 2],
    latest: usize,
    /// A texel's hit distance and confidences (`texels`).
    texels: wgpu::Buffer,
    /// The view rendered its previous frame at this size.
    primed: bool,
    /// The frame's origin when it last rendered.
    origin: DVec3,
}

/// Views a gather keeps reservoirs for at once.
const MAX_VIEWS: usize = 8;

/// The near field and the pixels' rays (module docs): the gather's stream's
/// generator.
pub struct GiGatherPass {
    config: GatherConfig,
    stream: Stream,
    answers: Arc<stream::Answers>,
    stats: helio_core::FrameStats,
    stat: helio_core::StatSlot,
    views: Vec<GatherView>,
    active: Option<usize>,
    rays: Option<StreamBuffers>,
    rows: u32,
    near: Option<(wgpu::Texture, wgpu::TextureView, [u32; 2])>,
    frame: wgpu::Buffer,
    gpu: GpuGatherFrame,
    near_layout: wgpu::BindGroupLayout,
    rays_layout: wgpu::BindGroupLayout,
    near_pipeline: wgpu::ComputePipeline,
    rays_pipeline: wgpu::ComputePipeline,
    cache: helio_core::gpu_cache::GpuCache,
    writes: &'static [&'static str],
}

impl GiGatherPass {
    pub fn new(
        device: &wgpu::Device,
        config: GatherConfig,
        stream: Stream,
        answers: Arc<stream::Answers>,
        stats: &helio_core::FrameStats,
    ) -> Self {
        let mut near = surface_entries().to_vec();
        near.extend([
            texture(5),
            storage_texture(6, wgpu::TextureFormat::Rgba16Float),
        ]);
        let near_layout = layout(device, "GI gather near", &near);
        let mut rays = surface_entries().to_vec();
        rays.extend([storage(7, false), storage(8, false)]);
        let rays_layout = layout(device, "GI gather rays", &rays);
        let module = module(device);
        // Its tile is written whole before it is read: no zeroing of the
        // workgroup's memory first (naga's is one invocation's loop).
        let near_pipeline = helio_core::shader::create_compute_pipeline(
            device,
            wgpu::ComputePipelineDescriptor {
                label: Some("gather_near"),
                layout: Some(&crate::pipeline_layout(
                    device,
                    "GI gather near",
                    &near_layout,
                )),
                module: &module,
                entry_point: Some("gather_near"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            },
        );
        let rays_pipeline = compute(
            device,
            &module,
            &crate::pipeline_layout(device, "GI gather rays", &rays_layout),
            "gather_rays",
            &[],
        );
        Self {
            config,
            answers,
            stats: stats.clone(),
            stat: stream.stat(stats),
            views: Vec::new(),
            active: None,
            rays: None,
            rows: 0,
            near: None,
            frame: buffer(
                device,
                "GI gather frame",
                GATHER_FRAME_BYTES,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            gpu: GpuGatherFrame::default(),
            near_layout,
            rays_layout,
            near_pipeline,
            rays_pipeline,
            cache: helio_core::gpu_cache::GpuCache::new(),
            writes: helio_frame_api::static_names(&[
                GATHER_FRAME,
                GATHER_PREVIOUS,
                GATHER_RESERVOIRS,
                GATHER_TEXELS,
                NEAR_FIELD,
                stream.header,
                stream.rays,
                stream.hits,
            ]),
            stream,
        }
    }

    fn view(&mut self, device: &wgpu::Device, id: u32, frame: u64, size: [u32; 2]) -> usize {
        let index = match self.views.iter().position(|v| v.id == id) {
            Some(i) => i,
            None => {
                if self.views.len() >= MAX_VIEWS {
                    let oldest = (0..self.views.len())
                        .min_by_key(|&i| self.views[i].used)
                        .unwrap();
                    self.views.swap_remove(oldest);
                }
                self.views.push(Self::new_view(device, id, [0, 0]));
                self.views.len() - 1
            }
        };
        if self.views[index].size != size {
            self.views[index] = Self::new_view(device, id, size);
        }
        self.views[index].used = frame;
        index
    }

    fn new_view(device: &wgpu::Device, id: u32, size: [u32; 2]) -> GatherView {
        let n = u64::from(size[0].max(1)) * u64::from(size[1].max(1));
        let usage = wgpu::BufferUsages::STORAGE;
        GatherView {
            id,
            used: 0,
            size,
            reservoirs: [0, 1]
                .map(|_| buffer(device, "GI gather reservoirs", n * RESERVOIR_BYTES, usage)),
            latest: 0,
            texels: buffer(device, "GI gather texels", n * 16, usage),
            primed: false,
            origin: DVec3::ZERO,
        }
    }
}

impl GiGatherPass {
    /// The near field as the last frame wrote it (`Rgba16Float`: x what the
    /// signal is divided by, y what a pixel's irradiance is multiplied by),
    /// for tests and tools that read it back.
    pub fn near_field(&self) -> Option<&wgpu::Texture> {
        self.near.as_ref().map(|(t, _, _)| t)
    }
}

impl RenderPass for GiGatherPass {
    fn name(&self) -> &'static str {
        "GIGather"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            helio_frame_api::gbuffer::GROUP,
            helio_frame_api::gbuffer::DEPTH,
        ]
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(helio_frame_api::gbuffer::GROUP);
        builder.read_optional(helio_frame_api::gbuffer::SURFACE);
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _size: helio_core::graph::RenderSize) {
        // Each view's reservoirs are remade at its next frame's size.
        for view in &mut self.views {
            view.primed = false;
        }
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let size = [ctx.width.max(1), ctx.height.max(1)];
        let signal = self.config.signal();
        let traced = signal.traced_size(size);
        let id = ctx.camera_data.jitter_frame[3].to_bits();
        let index = self.view(ctx.device, id, ctx.frame_num, traced);
        self.active = Some(index);
        let view = &mut self.views[index];
        let has_previous = view.primed;
        if has_previous {
            view.latest ^= 1;
        }
        // A point of the previous frame's space is this frame's plus the
        // origin's move back (whole 1024 m steps: exact in f32).
        let rebase = if has_previous {
            (view.origin - ctx.world_origin).as_vec3()
        } else {
            glam::Vec3::ZERO
        };
        view.origin = ctx.world_origin;
        let frame = ctx.frame_num as u32;
        let n = traced[0] * traced[1];
        let (phase, retraced) = retrace_phase(n, frame);
        let rows = rows(traced);
        if self.rows < rows || self.rays.is_none() {
            let stream = Stream {
                capacity: rows,
                ..self.stream
            };
            self.rays = Some(StreamBuffers::new(ctx.device, &stream));
            self.rows = rows;
        }
        let count = n + if self.config.reuse { retraced } else { 0 };
        if let Some(rays) = &self.rays {
            rays.set_count(ctx.queue, count);
        }
        self.stats.add(self.stat, u64::from(count));
        let p = ctx.camera_data.proj;
        let flags =
            (u32::from(has_previous) * HAS_PREVIOUS) | (u32::from(self.config.reuse) * REUSE);
        let forward = ctx.camera_data.forward_far;
        self.gpu = GpuGatherFrame {
            size: [size[0], size[1], traced[0], traced[1]],
            info: [
                self.config.block.side(),
                frame,
                flags,
                ctx.camera_data.view_info[3],
            ],
            rebase: [
                rebase.x,
                rebase.y,
                rebase.z,
                // A pixel's width a metre of view depth (the projection's
                // vertical scale: 2 / (P[1][1] h)).
                2.0 / (p[5].abs().max(1e-12) * size[1] as f32),
            ],
            reuse: [M_CAP, RETRACE_PERIOD as f32, phase as f32, SPATIAL_RADIUS],
            forward: [forward[0], forward[1], forward[2], 0.0],
        };
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.answers.begin();
        let Some(index) = self.active else {
            return Ok(());
        };
        let Some(g) = gbuffer(ctx.registry, ctx.depth) else {
            return Ok(());
        };
        let size = [ctx.width.max(1), ctx.height.max(1)];
        if self.near.as_ref().is_none_or(|(_, _, s)| *s != size) {
            let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("GI near field"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.near = Some((texture, view, size));
        }
        let (Some((_, near, _)), Some(rays)) = (&self.near, &self.rays) else {
            return Ok(());
        };
        let mut gpu = self.gpu;
        if g.tags.is_some() {
            gpu.info[2] |= TAGS;
        }
        helio_core::upload::write_buffer(ctx.queue, &self.frame, 0, bytemuck::bytes_of(&gpu));
        let view = &mut self.views[index];
        let tags = g.tags.unwrap_or(g.normal);
        let [s0, s1, s2, s3, s4] = [
            entry(0, &self.frame),
            entry(1, ctx.camera),
            view_entry(2, g.depth),
            view_entry(3, g.normal),
            view_entry(4, tags),
        ];
        self.cache.end_frame();
        let near_group = self.cache.bind_group(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("GI gather near"),
                layout: &self.near_layout,
                entries: &[
                    s0.clone(),
                    s1.clone(),
                    s2.clone(),
                    s3.clone(),
                    s4.clone(),
                    view_entry(5, g.orm),
                    view_entry(6, near),
                ],
            },
        );
        let rays_group = self.cache.bind_group(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("GI gather rays"),
                layout: &self.rays_layout,
                entries: &[
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    entry(7, &rays.rays),
                    entry(8, &view.reservoirs[view.latest ^ 1]),
                ],
            },
        );
        let traced = [self.gpu.size[2], self.gpu.size[3]];
        let mut cmds = ctx.graphics_cmds();
        rays.clear(&mut cmds);
        {
            let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GI gather"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, Some(&near_group), &[]);
            pass.set_pipeline(&self.near_pipeline);
            pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
            pass.set_bind_group(0, Some(&rays_group), &[]);
            pass.set_pipeline(&self.rays_pipeline);
            pass.dispatch_workgroups(traced[0].div_ceil(8), traced[1].div_ceil(8), 1);
        }
        view.primed = true;
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        let (Some(index), Some(rays), Some((_, near, _))) = (self.active, &self.rays, &self.near)
        else {
            return;
        };
        let view = &self.views[index];
        frame.write_buffer_binding(GATHER_FRAME, &self.frame, "GIGather");
        frame.write_buffer_binding(
            GATHER_PREVIOUS,
            &view.reservoirs[view.latest ^ 1],
            "GIGather",
        );
        frame.write_buffer_binding(GATHER_RESERVOIRS, &view.reservoirs[view.latest], "GIGather");
        frame.write_buffer_binding(GATHER_TEXELS, &view.texels, "GIGather");
        frame.write_texture_binding(NEAR_FIELD, near, "GIGather");
        rays.publish(&self.stream, frame, "GIGather");
    }
}

/// The re-traced samples, the temporal and spatial reuse, the signal
/// (module docs).
pub struct GiGatherShadePass {
    signal: Signal,
    stream: Stream,
    answers: Arc<stream::Answers>,
    outputs: Option<([wgpu::Texture; 2], [wgpu::TextureView; 2], [u32; 2])>,
    answer: wgpu::Buffer,
    answered: Option<bool>,
    layout: wgpu::BindGroupLayout,
    validate: wgpu::ComputePipeline,
    temporal: wgpu::ComputePipeline,
    spatial: wgpu::ComputePipeline,
    no_atmosphere: wgpu::Buffer,
    no_sky_table: wgpu::Buffer,
    no_cache: radiance_cache::StandIn,
    cache: helio_core::gpu_cache::GpuCache,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl GiGatherShadePass {
    pub fn new(
        device: &wgpu::Device,
        signal: Signal,
        stream: Stream,
        answers: Arc<stream::Answers>,
    ) -> Self {
        let mut entries = surface_entries().to_vec();
        entries.extend([
            storage(8, false),
            storage(9, false),
            storage(10, false),
            storage(11, true),
            storage(12, true),
            texture(13),
            texture(14),
            storage_texture(15, wgpu::TextureFormat::Rgba16Float),
            storage_texture(16, wgpu::TextureFormat::Rgba16Float),
            uniform(17, 16),
            uniform(18, FRAME_BYTES),
            storage(19, true),
            uniform(20, TRACE_BYTES),
            uniform(21, SKY_FRAME_BYTES),
            uniform(22, SKY_TABLE_BYTES),
            uniform(23, RadianceCacheFrame::BYTES),
            storage(24, false),
            storage(25, false),
            storage(26, true),
        ]);
        let layout = layout(device, "GI gather shade", &entries);
        let module = module(device);
        let pipeline_layout = crate::pipeline_layout(device, "GI gather shade", &layout);
        let pipeline = |entry: &str| compute(device, &module, &pipeline_layout, entry, &[]);
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let mut reads = vec![
            GATHER_FRAME,
            GATHER_PREVIOUS,
            GATHER_RESERVOIRS,
            GATHER_TEXELS,
            NEAR_FIELD,
            history::PIXELS,
            GI_FRAME,
            GI_PROBES,
            GI_TRACE,
            stream.rays,
            stream.hits,
        ];
        reads.extend(stream.surfaces);
        Self {
            validate: pipeline("gather_validate"),
            temporal: pipeline("gather_temporal"),
            spatial: pipeline("gather_spatial"),
            layout,
            outputs: None,
            answer: buffer(device, "GI gather answer", 16, uniform_usage),
            answered: None,
            no_atmosphere: buffer(
                device,
                "GI gather no atmosphere",
                SKY_FRAME_BYTES,
                uniform_usage,
            ),
            no_sky_table: buffer(
                device,
                "GI gather no sky table",
                SKY_TABLE_BYTES,
                uniform_usage,
            ),
            no_cache: radiance_cache::stand_in(device),
            cache: helio_core::gpu_cache::GpuCache::new(),
            reads: helio_frame_api::static_names(&reads),
            writes: helio_frame_api::static_names(&signal.inputs),
            signal,
            stream,
            answers,
        }
    }
}

impl GiGatherShadePass {
    /// The signal's two textures as the last frame wrote them (for tests and
    /// tools that read them back): `denoise`'s inputs, a texel a block.
    pub fn signal_textures(&self) -> Option<&[wgpu::Texture; 2]> {
        self.outputs.as_ref().map(|(t, _, _)| t)
    }
}

impl RenderPass for GiGatherShadePass {
    fn name(&self) -> &'static str {
        "GIGatherShade"
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(helio_frame_api::gbuffer::GROUP);
        builder.read_optional(helio_frame_api::gbuffer::SURFACE);
        for key in [
            radiance_cache::FRAME,
            radiance_cache::SLOTS,
            radiance_cache::ENTRIES,
        ] {
            builder.read_optional(key);
        }
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let binding = |name: &str| ctx.registry.buffer_binding(name);
        let view = |name: &'static str| ctx.registry.texture_view(ResourceKey::new(name));
        let (
            Some(frame),
            Some(previous),
            Some(reservoirs),
            Some(texels),
            Some(near),
            Some(pixels),
            Some(gi_frame),
            Some(probes),
            Some(trace),
            Some(rays),
            Some(hits),
            Some(surfaces),
            Some(g),
        ) = (
            binding(GATHER_FRAME),
            binding(GATHER_PREVIOUS),
            binding(GATHER_RESERVOIRS),
            binding(GATHER_TEXELS),
            view(NEAR_FIELD),
            view(history::PIXELS),
            binding(GI_FRAME),
            binding(GI_PROBES),
            binding(GI_TRACE),
            binding(self.stream.rays),
            binding(self.stream.hits),
            self.stream.surfaces.and_then(binding),
            gbuffer(ctx.registry, ctx.depth),
        )
        else {
            self.outputs = None;
            return Ok(());
        };
        let size = [ctx.width.max(1), ctx.height.max(1)];
        let traced = self.signal.traced_size(size);
        if self.outputs.as_ref().is_none_or(|(_, _, s)| *s != traced) {
            let make = || {
                ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("GI gather signal"),
                    size: wgpu::Extent3d {
                        width: traced[0],
                        height: traced[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
            };
            let textures = [make(), make()];
            let views = [0, 1].map(|i| textures[i].create_view(&Default::default()));
            self.outputs = Some((textures, views, traced));
        }
        // Whether every tracer answered in full: where one did not, the
        // reservoirs hold (this frame's rays miss what it did not trace).
        let answered = self.answers.complete();
        if self.answered != Some(answered) {
            helio_core::upload::write_buffer(
                ctx.queue,
                &self.answer,
                0,
                bytemuck::cast_slice(&[u32::from(answered), 0, 0, 0]),
            );
            self.answered = Some(answered);
        }
        let [atmosphere, sky_table] = crate::sky(ctx.registry);
        let atmosphere = atmosphere.unwrap_or(&self.no_atmosphere);
        let sky_table = sky_table.unwrap_or(&self.no_sky_table);
        let cache = match (
            binding(radiance_cache::FRAME),
            binding(radiance_cache::SLOTS),
            binding(radiance_cache::ENTRIES),
        ) {
            (Some(f), Some(s), Some(e)) => [f, s, e],
            _ => [
                &self.no_cache.frame,
                &self.no_cache.slots,
                &self.no_cache.entries,
            ],
        };
        let Some((_, [signal0, signal1], _)) = &self.outputs else {
            return Ok(());
        };
        self.cache.end_frame();
        let group = self.cache.bind_group(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("GI gather shade"),
                layout: &self.layout,
                entries: &[
                    entry(0, frame),
                    entry(1, ctx.camera),
                    view_entry(2, g.depth),
                    view_entry(3, g.normal),
                    view_entry(4, g.tags.unwrap_or(g.normal)),
                    entry(8, previous),
                    entry(9, reservoirs),
                    entry(10, texels),
                    entry(11, hits),
                    entry(12, surfaces),
                    view_entry(13, near),
                    view_entry(14, pixels),
                    view_entry(15, signal0),
                    view_entry(16, signal1),
                    entry(17, &self.answer),
                    entry(18, gi_frame),
                    entry(19, probes),
                    entry(20, trace),
                    entry(21, atmosphere),
                    entry(22, sky_table),
                    entry(23, cache[0]),
                    entry(24, cache[1]),
                    entry(25, cache[2]),
                    entry(26, rays),
                ],
            },
        );
        let groups = [traced[0].div_ceil(8), traced[1].div_ceil(8)];
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI gather shade"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        for pipeline in [&self.validate, &self.temporal, &self.spatial] {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(groups[0], groups[1], 1);
        }
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let Some((_, views, _)) = &self.outputs {
            for (name, view) in self.signal.inputs.iter().zip(views) {
                frame.write_texture_binding(*name, view, "GIGatherShade");
            }
        }
    }
}

/// Adds the gather's passes to the graph being built (after the G-buffer and
/// GI's update, before the resolve) and the frame's denoiser's after them;
/// what the resolve reads. `None` where the frame has no history to reuse
/// samples through (a host without the spine): the probes resolve alone.
pub(crate) fn build(
    build: &mut helio_feature_api::NodeBuild<'_, '_>,
    config: GatherConfig,
) -> Result<Option<Filtered>, String> {
    if !history::request(build, Need::Pixels)? {
        log::warn!(
            "GI's pixels' gather needs the frame's history, which nothing publishes: probes only"
        );
        return Ok(None);
    }
    let signal = config.signal();
    let frame = build.env.frame;
    let traced = signal.traced_size([frame.internal_width.max(1), frame.internal_height.max(1)]);
    let stream = pixel_stream(traced);
    let answers = stream::answers(build.env, &stream);
    let device = build.env.device.clone();
    let stats = build.env.stats.clone();
    build.graph.add_pass(Box::new(GiGatherPass::new(
        &device,
        config,
        stream,
        answers.clone(),
        &stats,
    )));
    stream::trace(build, &stream)?;
    build.graph.add_pass(Box::new(GiGatherShadePass::new(
        &device, signal, stream, answers,
    )));
    denoise::filter(build, &signal).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_retraced_texels_fill_their_rows_once() {
        for n in [1u32, 5, 16, 17, 129_600] {
            let mut seen = 0;
            for frame in 0..RETRACE_PERIOD {
                let (phase, count) = retrace_phase(n, frame);
                let selected = (0..n).filter(|i| i % RETRACE_PERIOD == phase).count() as u32;
                assert_eq!(selected, count, "n {n} frame {frame}");
                // Rows n + i / period are distinct and within the stream.
                for i in (0..n).filter(|i| i % RETRACE_PERIOD == phase) {
                    assert!(n + i / RETRACE_PERIOD < n + n.div_ceil(RETRACE_PERIOD));
                }
                seen += count;
            }
            // Every texel once in a period.
            assert_eq!(seen, n);
        }
        assert_eq!(rows([480, 270]), 129_600 + 32_400);
    }

    #[test]
    fn the_frame_and_reservoir_rows_have_their_shader_layouts() {
        let source =
            helio_core::shader::resolve_with(include_str!("../shaders/gi_gather.wgsl"), &SNIPPETS);
        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
        let mut layouter = wgpu::naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        for (name, bytes) in [
            ("GatherFrame", GATHER_FRAME_BYTES),
            ("GatherReservoir", RESERVOIR_BYTES),
        ] {
            let (handle, _) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(name))
                .unwrap();
            assert_eq!(u64::from(layouter[handle].size), bytes, "{name}");
        }
        // The flags and constants the shader shares with this file.
        for line in [
            "const GATHER_HAS_PREVIOUS: u32 = 1u;",
            "const GATHER_REUSE: u32 = 2u;",
            "const GATHER_TAGS: u32 = 4u;",
        ] {
            assert!(source.contains(line), "{line}");
        }
        assert_eq!((HAS_PREVIOUS, REUSE, TAGS), (1, 2, 4));
    }

    /// The constants' derivations (module docs and rendering.md).
    #[test]
    fn the_constants_follow_their_derivations() {
        // Four independent kept samples over the denoiser's 32 frames.
        assert!(32.0 / (M_CAP + 1.0) >= 3.5);
        // The spatial radius: the denoiser's two a-trous steps of 1 and 2
        // blocks.
        assert_eq!(SPATIAL_RADIUS, 1.0 + 2.0);
        // Spatial neighbours times the M cap cover the disc's texels.
        let disc = (std::f32::consts::PI * SPATIAL_RADIUS * SPATIAL_RADIUS).round();
        assert!(4.0 * M_CAP >= disc, "{disc}");
        // The rays' start offset is `gi_ray_offset`'s constant part.
        assert!(include_str!("../shaders/gi_rays.wgsl").contains("return 0.002 + "));
        assert_eq!(OCCLUSION_START, 0.002);
        // The cache noise a re-traced light may show: 1 / sqrt(32).
        assert!(include_str!("../shaders/gi_gather.wgsl").contains(&format!(
            "const CACHE_NOISE: f32 = {:.8};",
            1.0f64 / 32f64.sqrt()
        )));
    }
}
