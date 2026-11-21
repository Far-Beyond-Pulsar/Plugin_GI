//! The light of GI's rays' hits (`gi_light.wgsl`, README.md "The frame"):
//! the light batch's and the shadow stream's generators. They run once
//! every tracer answered GI's stream, and know no kind: a hit is its
//! `SceneSurface` and its place, a light is a sample of the frame's light
//! samplers (`helio_frame_api::rays::light_samples`), whichever kinds of
//! light registered one, and what reaches a surface for indirect purposes is
//! the frame's radiance cache's (`helio_frame_api::rays::radiance_cache`,
//! published by [`crate::RadianceCache`]).
//!
//! [`GiLightPass`]: for each ray of GI's stream that met an outside, its
//! hit's light from the radiance cache (a probe's hit: all of it; a cache
//! ray's hit: its indirect part), through it from its far side and what it
//! emits, or the sky on a miss, into `gi_pending`, every hit marking its
//! cell in use; and for each cache ray a light query at its hit and one at
//! its cell's point into the cache's light batch ([`crate::light_batch`]),
//! compacted from row 0. The samplers then choose one light a query (the sun
//! is one more of them).
//!
//! [`GiShadowPass`]: each cache ray's samples, its hit's off the hit's
//! surface (Lambertian on the side the light is on; through it by its
//! transmission share from behind) into `gi_pending` and its cell's as the
//! irradiance at the cell into `gi_cell_pending`, each waiting on a shadow
//! ray that starts off its surface (on the side it leaves from); the shadow
//! rays packed into the shadow stream from row 0, a workgroup's rows
//! together. A last one-invocation dispatch writes the stream's dispatch
//! from its count and counts it into the frame's statistics.

use helio_core::{PassContext, RenderPass, Result as HelioResult};
use helio_frame_api::lighting::sky::{
    SKY_FRAME, SKY_FRAME_BYTES, SKY_SNIPPET, SKY_TABLE, SKY_TABLE_BYTES, SKY_TABLE_SNIPPET,
};
use helio_frame_api::rays::light_samples::{self, LightBatch, LightBatchBuffers};
use helio_frame_api::rays::radiance_cache::{self, RadianceCacheFrame};
use helio_frame_api::rays::stream::{self, Stream, StreamBuffers};

use crate::cache::{CACHE_LIST, CACHE_PLAN, CELL_PENDING};
use crate::{
    buffer, compute, entry, storage, uniform, GiConfig, FRAME_BYTES, GI_ANSWERED,
    GI_COMMON_SNIPPET, GI_FRAME, GI_LISTS, GI_PENDING, GI_PROBES, GI_RAYS_SNIPPET,
    GI_SAMPLE_SNIPPET, GI_TRACE, MAX_UPDATES, PENDING_BYTES, TRACE_BYTES,
};

/// The snippets `gi_light.wgsl` is built with.
pub(crate) const SNIPPETS: [helio_core::shader::ShaderSnippet; 11] = [
    SKY_SNIPPET,
    SKY_TABLE_SNIPPET,
    stream::SNIPPET,
    stream::SURFACE_SNIPPET,
    helio_frame_api::gbuffer::SURFACE_SNIPPET,
    light_samples::SNIPPET,
    radiance_cache::SNIPPET,
    GI_COMMON_SNIPPET,
    GI_SAMPLE_SNIPPET,
    GI_RAYS_SNIPPET,
    crate::GI_HIT_SNIPPET,
];

fn module(device: &wgpu::Device) -> wgpu::ShaderModule {
    helio_core::shader::module_with(
        device,
        "GI light",
        helio_core::include_wgsl!("../shaders/gi_light.wgsl"),
        &SNIPPETS,
    )
}

/// The cache's bindings both passes declare (17..23 of `gi_light.wgsl`).
fn cache_entries() -> [wgpu::BindGroupLayoutEntry; 7] {
    [
        uniform(17, RadianceCacheFrame::BYTES),
        storage(18, false),
        storage(19, false),
        storage(20, true),
        storage(21, true),
        storage(22, false),
        storage(23, true),
    ]
}

/// The hits' light from the cache, and the cache rays' light queries (module
/// docs).
pub struct GiLightPass {
    rays: Stream,
    batch: LightBatch,
    lights: LightBatchBuffers,
    pending: wgpu::Buffer,
    cells: wgpu::Buffer,
    cache_cells: u32,
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
    pub fn new(device: &wgpu::Device, config: &GiConfig) -> Self {
        let mut entries = vec![
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
        ];
        entries.extend(cache_entries());
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI light"),
            entries: &entries,
        });
        let module = module(device);
        let pipeline_layout = crate::pipeline_layout(device, "GI light", &layout);
        let rays = crate::gi_stream(config);
        let batch = crate::light_batch(&config.cache);
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let constants = [("CACHE_CELLS", f64::from(config.cache.cells()))];
        let mut reads = vec![
            GI_FRAME,
            GI_PROBES,
            GI_LISTS,
            GI_TRACE,
            GI_ANSWERED,
            rays.rays,
            rays.hits,
            SKY_FRAME,
            SKY_TABLE,
            radiance_cache::FRAME,
            radiance_cache::SLOTS,
            radiance_cache::ENTRIES,
            CACHE_PLAN,
            CACHE_LIST,
        ];
        reads.extend(rays.surfaces);
        Self {
            light: compute(device, &module, &pipeline_layout, "gi_light", &constants),
            header: compute(
                device,
                &module,
                &pipeline_layout,
                "gi_light_header",
                &constants,
            ),
            layout,
            lights: LightBatchBuffers::new(device, &batch),
            pending: buffer(
                device,
                "GI pending light",
                u64::from(rays.capacity) * PENDING_BYTES,
                wgpu::BufferUsages::STORAGE,
            ),
            cells: buffer(
                device,
                "GI cell pending light",
                u64::from(config.cache.rays) * 8,
                wgpu::BufferUsages::STORAGE,
            ),
            cache_cells: config.cache.cells(),
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
                CELL_PENDING,
                GI_ANSWERED,
                radiance_cache::SLOTS,
                radiance_cache::ENTRIES,
                batch.header,
                batch.queries,
                batch.samples,
            ]),
            rays,
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
            Some(rays),
            Some(hits),
            Some(surfaces),
            Some(cache_frame),
            Some(slots),
            Some(entries),
            Some(plan),
            Some(list),
        ) = (
            binding(GI_FRAME),
            binding(GI_PROBES),
            binding(GI_LISTS),
            binding(GI_TRACE),
            binding(GI_ANSWERED),
            binding(self.rays.rays),
            binding(self.rays.hits),
            self.rays.surfaces.and_then(binding),
            binding(radiance_cache::FRAME),
            binding(radiance_cache::SLOTS),
            binding(radiance_cache::ENTRIES),
            binding(CACHE_PLAN),
            binding(CACHE_LIST),
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
                    entry(17, cache_frame),
                    entry(18, slots),
                    entry(19, entries),
                    entry(20, plan),
                    entry(21, list),
                    entry(22, &self.cells),
                    entry(23, rays),
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
        // A workgroup a probe, then one for 64 cache rays; those past this
        // frame's counts return at once.
        let cache_groups = (self.cache_cells * crate::cache::RAYS_PER_CELL).div_ceil(64);
        pass.dispatch_workgroups(MAX_UPDATES + cache_groups, 1, 1);
        pass.set_pipeline(&self.header);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        frame.write_buffer_binding(GI_PENDING, &self.pending, "GILight");
        frame.write_buffer_binding(CELL_PENDING, &self.cells, "GILight");
        self.lights.publish(&self.batch, frame, "GILight");
    }
}

/// The cache rays' light samples into shadow rays (module docs).
pub struct GiShadowPass {
    rays: Stream,
    shadows: Stream,
    batch: LightBatch,
    cache_cells: u32,
    /// The shadow stream's answers, started each frame.
    answers: std::sync::Arc<stream::Answers>,
    buffers: StreamBuffers,
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
        config: &GiConfig,
        stats: &helio_core::FrameStats,
        answers: std::sync::Arc<stream::Answers>,
    ) -> Self {
        let mut entries = vec![
            uniform(0, FRAME_BYTES),
            uniform(5, TRACE_BYTES),
            storage(6, true),
            storage(7, true),
            storage(8, false),
            storage(9, false),
            storage(10, false),
            storage(11, false),
            storage(15, true),
            storage(16, false),
        ];
        entries.extend(cache_entries());
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI shadow"),
            entries: &entries,
        });
        let module = module(device);
        let pipeline_layout = crate::pipeline_layout(device, "GI shadow", &layout);
        let shadows = crate::shadow_stream(&config.cache);
        let mut constants =
            stats.slot_constants(&[("GI_SHADOW_STAT", &format!("rays.{}", shadows.name))]);
        constants.push(("CACHE_CELLS", f64::from(config.cache.cells())));
        let rays = crate::gi_stream(config);
        let batch = crate::light_batch(&config.cache);
        let mut reads = vec![
            GI_FRAME,
            GI_TRACE,
            GI_PENDING,
            CELL_PENDING,
            rays.rays,
            rays.hits,
            batch.samples,
            radiance_cache::FRAME,
            radiance_cache::ENTRIES,
            CACHE_PLAN,
            CACHE_LIST,
        ];
        reads.extend(rays.surfaces);
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
            buffers: StreamBuffers::new(device, &shadows),
            stats: stats.buffer().clone(),
            cache: helio_core::gpu_cache::GpuCache::new(),
            reads: helio_frame_api::static_names(&reads),
            writes: helio_frame_api::static_names(&[
                GI_PENDING,
                CELL_PENDING,
                shadows.header,
                shadows.rays,
                shadows.hits,
            ]),
            cache_cells: config.cache.cells(),
            rays,
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
            Some(trace),
            Some(pending),
            Some(cells),
            Some(rays),
            Some(hits),
            Some(surfaces),
            Some(samples),
            Some(cache_frame),
            Some(slots),
            Some(entries),
            Some(plan),
            Some(list),
        ) = (
            binding(GI_FRAME),
            binding(GI_TRACE),
            binding(GI_PENDING),
            binding(CELL_PENDING),
            binding(self.rays.rays),
            binding(self.rays.hits),
            self.rays.surfaces.and_then(binding),
            binding(self.batch.samples),
            binding(radiance_cache::FRAME),
            binding(radiance_cache::SLOTS),
            binding(radiance_cache::ENTRIES),
            binding(CACHE_PLAN),
            binding(CACHE_LIST),
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
                    entry(5, trace),
                    entry(6, hits),
                    entry(7, surfaces),
                    entry(8, &self.buffers.rays),
                    entry(9, &self.buffers.hits),
                    entry(10, &self.buffers.header),
                    entry(11, pending),
                    entry(15, samples),
                    entry(16, &self.stats),
                    entry(17, cache_frame),
                    entry(18, slots),
                    entry(19, entries),
                    entry(20, plan),
                    entry(21, list),
                    entry(22, cells),
                    entry(23, rays),
                ],
            },
        );
        let mut cmds = ctx.graphics_cmds();
        // The shadow stream's count, from 0.
        cmds.clear_buffer(&self.buffers.header, 0, None);
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI shadow"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&self.shadow);
        // A workgroup for 64 cache rays; those past this frame's count
        // return at once.
        pass.dispatch_workgroups(
            (self.cache_cells * crate::cache::RAYS_PER_CELL).div_ceil(64),
            1,
            1,
        );
        pass.set_pipeline(&self.header);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        self.buffers.publish(&self.shadows, frame, "GIShadow");
    }
}
