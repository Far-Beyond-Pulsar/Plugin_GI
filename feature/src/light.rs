//! The light of the probe rays' hits (`gi_light.wgsl`, README.md "The
//! frame"): the light batch's and the shadow stream's generators. They run
//! once every tracer answered the probe stream, and know no kind: a hit is
//! its `SceneSurface` and its place, a light is a sample of the frame's
//! light samplers (`helio_frame_api::rays::light_samples`), whichever kinds
//! of light registered one.
//!
//! [`GiLightPass`]: for each probe ray that met an outside, the light that
//! waits on no shadow ray (the probes' irradiance at the hit, through it
//! from its far side, what it emits; a miss takes the sky) into
//! `gi_pending`, and a light query at the hit into GI's light batch
//! ([`crate::light_batch`]), compacted from row 0. The samplers then choose
//! one light a hit (the sun is one more of them).
//!
//! [`GiShadowPass`]: each hit's sample off its surface (Lambertian on the side
//! the light is on; through it by its transmission share from behind), into
//! `gi_pending`, waiting on a shadow ray that starts off the surface along
//! its face normal (on the side it leaves from); the shadow rays packed into
//! the shadow stream from row 0, a workgroup's rows together. A last
//! one-invocation dispatch writes the stream's dispatch from its count and
//! counts it into the frame's statistics.

use helio_core::{PassContext, RenderPass, Result as HelioResult};
use helio_frame_api::lighting::sky::{
    SKY_FRAME, SKY_FRAME_BYTES, SKY_SNIPPET, SKY_TABLE, SKY_TABLE_BYTES, SKY_TABLE_SNIPPET,
};
use helio_frame_api::rays::light_samples::{self, LightBatch, LightBatchBuffers};
use helio_frame_api::rays::stream::{self, Stream, StreamBuffers};

use crate::{
    buffer, compute, entry, storage, uniform, FRAME_BYTES, GI_ANSWERED, GI_COMMON_SNIPPET,
    GI_FRAME, GI_LISTS, GI_PENDING, GI_PROBES, GI_RAYS_SNIPPET, GI_SAMPLE_SNIPPET, GI_TRACE,
    MAX_RAYS, PENDING_BYTES, TRACE_BYTES,
};

/// The snippets `gi_light.wgsl` is built with.
pub(crate) const SNIPPETS: [helio_core::shader::ShaderSnippet; 9] = [
    SKY_SNIPPET,
    SKY_TABLE_SNIPPET,
    stream::SNIPPET,
    stream::SURFACE_SNIPPET,
    helio_frame_api::gbuffer::SURFACE_SNIPPET,
    light_samples::SNIPPET,
    GI_COMMON_SNIPPET,
    GI_SAMPLE_SNIPPET,
    GI_RAYS_SNIPPET,
];

fn module(device: &wgpu::Device) -> wgpu::ShaderModule {
    helio_core::shader::module_with(
        device,
        "GI light",
        helio_core::include_wgsl!("../shaders/gi_light.wgsl"),
        &SNIPPETS,
    )
}

/// The hits' light that needs no shadow ray and their light queries (module
/// docs).
pub struct GiLightPass {
    probes: Stream,
    batch: LightBatch,
    lights: LightBatchBuffers,
    pending: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    light: wgpu::ComputePipeline,
    header: wgpu::ComputePipeline,
    no_atmosphere: wgpu::Buffer,
    no_sky_table: wgpu::Buffer,
    cache: helio_core::gpu_cache::GpuCache,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl GiLightPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI light"),
            entries: &[
                uniform(0, FRAME_BYTES),
                storage(1, true),
                storage(2, true),
                uniform(3, SKY_FRAME_BYTES),
                uniform(4, SKY_TABLE_BYTES),
                uniform(5, TRACE_BYTES),
                storage(6, true),
                storage(7, true),
                storage(11, false),
                storage(12, false),
                storage(13, false),
                storage(14, false),
            ],
        });
        let module = module(device);
        let pipeline_layout = crate::pipeline_layout(device, "GI light", &layout);
        let probes = crate::probe_stream();
        let batch = crate::light_batch();
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let mut reads = vec![
            GI_FRAME,
            GI_PROBES,
            GI_LISTS,
            GI_TRACE,
            GI_ANSWERED,
            probes.hits,
            SKY_FRAME,
            SKY_TABLE,
        ];
        reads.extend(probes.surfaces);
        Self {
            light: compute(device, &module, &pipeline_layout, "gi_light", &[]),
            header: compute(device, &module, &pipeline_layout, "gi_light_header", &[]),
            layout,
            lights: LightBatchBuffers::new(device, &batch),
            pending: buffer(
                device,
                "GI pending light",
                u64::from(MAX_RAYS) * PENDING_BYTES,
                wgpu::BufferUsages::STORAGE,
            ),
            no_atmosphere: buffer(
                device,
                "GI light no atmosphere",
                SKY_FRAME_BYTES,
                uniform_usage,
            ),
            no_sky_table: buffer(
                device,
                "GI light no sky table",
                SKY_TABLE_BYTES,
                uniform_usage,
            ),
            cache: helio_core::gpu_cache::GpuCache::new(),
            reads: helio_frame_api::static_names(&reads),
            writes: helio_frame_api::static_names(&[
                GI_PENDING,
                GI_ANSWERED,
                batch.header,
                batch.queries,
                batch.samples,
            ]),
            probes,
            batch,
        }
    }
}

impl RenderPass for GiLightPass {
    fn name(&self) -> &'static str {
        "GILight"
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let binding = |name: &str| ctx.registry.buffer_binding(name);
        let (
            Some(frame),
            Some(probes),
            Some(lists),
            Some(trace),
            Some(answered),
            Some(hits),
            Some(surfaces),
        ) = (
            binding(GI_FRAME),
            binding(GI_PROBES),
            binding(GI_LISTS),
            binding(GI_TRACE),
            binding(GI_ANSWERED),
            binding(self.probes.hits),
            self.probes.surfaces.and_then(binding),
        )
        else {
            return Ok(());
        };
        let [atmosphere, sky_table] = crate::sky(ctx.registry);
        let atmosphere = atmosphere.unwrap_or(&self.no_atmosphere);
        let sky_table = sky_table.unwrap_or(&self.no_sky_table);
        self.cache.end_frame();
        let group = self.cache.bind_group(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("GI light"),
                layout: &self.layout,
                entries: &[
                    entry(0, frame),
                    entry(1, probes),
                    entry(2, lists),
                    entry(3, atmosphere),
                    entry(4, sky_table),
                    entry(5, trace),
                    entry(6, hits),
                    entry(7, surfaces),
                    entry(11, &self.pending),
                    entry(12, answered),
                    entry(13, &self.lights.header),
                    entry(14, &self.lights.queries),
                ],
            },
        );
        let mut cmds = ctx.graphics_cmds();
        // The batch's count, from 0, and no samples yet.
        cmds.clear_buffer(&self.lights.header, 0, None);
        self.lights.clear(&mut cmds);
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI light"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&self.light);
        // A workgroup a probe; those past this frame's count (`GiFrame`)
        // return at once.
        pass.dispatch_workgroups(crate::MAX_UPDATES, 1, 1);
        pass.set_pipeline(&self.header);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        frame.write_buffer_binding(GI_PENDING, &self.pending, "GILight");
        self.lights.publish(&self.batch, frame, "GILight");
    }
}

/// The hits' light samples into shadow rays (module docs).
pub struct GiShadowPass {
    probes: Stream,
    shadows: Stream,
    batch: LightBatch,
    /// The shadow stream's answers, started each frame.
    answers: std::sync::Arc<stream::Answers>,
    rays: StreamBuffers,
    layout: wgpu::BindGroupLayout,
    shadow: wgpu::ComputePipeline,
    header: wgpu::ComputePipeline,
    stats: wgpu::Buffer,
    cache: helio_core::gpu_cache::GpuCache,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl GiShadowPass {
    /// The pass, counting its shadow rays into `stats`, the shadow stream's
    /// answers `answers`.
    pub fn new(
        device: &wgpu::Device,
        stats: &helio_core::FrameStats,
        answers: std::sync::Arc<stream::Answers>,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI shadow"),
            entries: &[
                uniform(0, FRAME_BYTES),
                storage(2, true),
                uniform(5, TRACE_BYTES),
                storage(6, true),
                storage(7, true),
                storage(8, false),
                storage(9, false),
                storage(10, false),
                storage(11, false),
                storage(15, true),
                storage(16, false),
            ],
        });
        let module = module(device);
        let pipeline_layout = crate::pipeline_layout(device, "GI shadow", &layout);
        let shadows = crate::shadow_stream();
        let constants =
            stats.slot_constants(&[("GI_SHADOW_STAT", &format!("rays.{}", shadows.name))]);
        let probes = crate::probe_stream();
        let batch = crate::light_batch();
        let mut reads = vec![
            GI_FRAME,
            GI_LISTS,
            GI_TRACE,
            GI_PENDING,
            probes.hits,
            batch.samples,
        ];
        reads.extend(probes.surfaces);
        Self {
            shadow: compute(device, &module, &pipeline_layout, "gi_shadow", &constants),
            header: compute(
                device,
                &module,
                &pipeline_layout,
                "gi_shadow_header",
                &constants,
            ),
            layout,
            rays: StreamBuffers::new(device, &shadows),
            stats: stats.buffer().clone(),
            cache: helio_core::gpu_cache::GpuCache::new(),
            reads: helio_frame_api::static_names(&reads),
            writes: helio_frame_api::static_names(&[
                GI_PENDING,
                shadows.header,
                shadows.rays,
                shadows.hits,
            ]),
            probes,
            shadows,
            batch,
            answers,
        }
    }
}

impl RenderPass for GiShadowPass {
    fn name(&self) -> &'static str {
        "GIShadow"
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.answers.begin();
        let binding = |name: &str| ctx.registry.buffer_binding(name);
        let (
            Some(frame),
            Some(lists),
            Some(trace),
            Some(pending),
            Some(hits),
            Some(surfaces),
            Some(samples),
        ) = (
            binding(GI_FRAME),
            binding(GI_LISTS),
            binding(GI_TRACE),
            binding(GI_PENDING),
            binding(self.probes.hits),
            self.probes.surfaces.and_then(binding),
            binding(self.batch.samples),
        )
        else {
            return Ok(());
        };
        self.cache.end_frame();
        let group = self.cache.bind_group(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("GI shadow"),
                layout: &self.layout,
                entries: &[
                    entry(0, frame),
                    entry(2, lists),
                    entry(5, trace),
                    entry(6, hits),
                    entry(7, surfaces),
                    entry(8, &self.rays.rays),
                    entry(9, &self.rays.hits),
                    entry(10, &self.rays.header),
                    entry(11, pending),
                    entry(15, samples),
                    entry(16, &self.stats),
                ],
            },
        );
        let mut cmds = ctx.graphics_cmds();
        // The shadow stream's count, from 0.
        cmds.clear_buffer(&self.rays.header, 0, None);
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI shadow"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&self.shadow);
        pass.dispatch_workgroups(crate::MAX_UPDATES, 1, 1);
        pass.set_pipeline(&self.header);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        self.rays.publish(&self.shadows, frame, "GIShadow");
    }
}
