//! The light of the probe rays' hits (`gi_light.wgsl`, README.md "The
//! frame"): the shadow stream's generator. It runs once every tracer
//! answered the probe stream, and knows no kind: a hit is its
//! `SceneSurface` and its place.
//!
//! For each probe ray that met an outside, it writes what its integration
//! needs (`gi_pending`): the sun's light off the surface's sunlit side
//! (cosine, base colour, the clouds' transmittance at the hit; through the
//! surface, by its transmission share, when the sunlit side is its far one)
//! and one local light's ([`crate::local_lights`]), each pending on a shadow
//! ray; and the rest, unshadowed: the probes' irradiance at the hit (every
//! further bounce), through it from its far side, and what it emits. A miss
//! takes the sky. The shadow rays start off the surface along its face
//! normal (on the side they leave from) and are packed into the shadow
//! stream from row 0, a workgroup's rows together (its sun rays, then its
//! local lights'): a tracer walks only rays there are, sun rays side by side.
//! A last one-invocation dispatch writes the stream's dispatch from its count
//! and counts it into the frame's statistics.

use helio_core::{PassContext, RenderPass, Result as HelioResult};
use helio_frame_api::lighting::sky::{
    SKY_FRAME, SKY_FRAME_BYTES, SKY_SNIPPET, SKY_TABLE, SKY_TABLE_BYTES, SKY_TABLE_SNIPPET,
};
use helio_frame_api::lighting::sun_transmittance::{self, SunTransmittance};
use helio_frame_api::rays::stream::{self, Stream, StreamBuffers};

use crate::local_lights::LocalLights;
use crate::{
    buffer, compute, entry, pipeline_layout, storage, texture, uniform, FRAME_BYTES, GI_ANSWERED,
    GI_COMMON_SNIPPET, GI_FRAME, GI_LISTS, GI_PENDING, GI_PROBES, GI_RAYS_SNIPPET,
    GI_SAMPLE_SNIPPET, GI_SUN, GI_TRACE, MAX_RAYS, PENDING_BYTES, TRACE_BYTES,
};

/// The snippets `gi_light.wgsl` is built with.
pub(crate) const SNIPPETS: [helio_core::shader::ShaderSnippet; 8] = [
    SKY_SNIPPET,
    SKY_TABLE_SNIPPET,
    stream::SNIPPET,
    GI_COMMON_SNIPPET,
    GI_SAMPLE_SNIPPET,
    GI_RAYS_SNIPPET,
    crate::local_lights::SNIPPET,
    sun_transmittance::SNIPPET,
];

/// The hits' light and the shadow stream's generator (module docs).
pub struct GiLightPass {
    probes: Stream,
    shadows: Stream,
    /// The shadow stream's answers, started each frame.
    answers: std::sync::Arc<stream::Answers>,
    rays: StreamBuffers,
    pending: wgpu::Buffer,
    lights: LocalLights,
    layout: wgpu::BindGroupLayout,
    light: wgpu::ComputePipeline,
    header: wgpu::ComputePipeline,
    /// The sun's transmittance a medium published when this pass was built
    /// (`helio_frame_api::lighting::sun_transmittance`), if any; and what
    /// stands in for none (a 1x1 map, a zeroed frame).
    sun_transmittance: Option<SunTransmittance>,
    no_sun_transmittance: (wgpu::TextureView, wgpu::Buffer),
    no_atmosphere: wgpu::Buffer,
    no_sky_table: wgpu::Buffer,
    stats: wgpu::Buffer,
    cache: helio_core::gpu_cache::GpuCache,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl GiLightPass {
    /// The pass, lighting the hits through the sun's transmittance `sun`
    /// published (none: the sun reaches every hit), counting its shadow rays
    /// into `stats`, the shadow stream's answers `answers`.
    pub fn new(
        device: &wgpu::Device,
        sun: Option<SunTransmittance>,
        stats: &helio_core::FrameStats,
        answers: std::sync::Arc<stream::Answers>,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI light"),
            entries: &[
                uniform(0, FRAME_BYTES),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                uniform(4, SKY_FRAME_BYTES),
                uniform(5, SKY_TABLE_BYTES),
                uniform(6, TRACE_BYTES),
                storage(7, true),
                storage(8, true),
                storage(9, false),
                storage(10, false),
                storage(11, false),
                storage(12, false),
                storage(13, false),
                storage(14, true),
                texture(15),
                uniform(16, sun_transmittance::FRAME_BYTES),
                storage(17, false),
            ],
        });
        let module = helio_core::shader::module_with(
            device,
            "GI light",
            helio_core::include_wgsl!("../shaders/gi_light.wgsl"),
            &SNIPPETS,
        );
        let pipeline_layout = pipeline_layout(device, "GI light", &layout);
        let shadows = crate::shadow_stream();
        let constants =
            stats.slot_constants(&[("GI_SHADOW_STAT", &format!("rays.{}", shadows.name))]);
        let probes = crate::probe_stream();
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let mut reads = vec![
            GI_FRAME,
            GI_PROBES,
            GI_SUN,
            GI_LISTS,
            GI_TRACE,
            GI_ANSWERED,
            probes.hits,
            SKY_FRAME,
            SKY_TABLE,
        ];
        reads.extend(probes.surfaces);
        if let Some(sun) = sun {
            reads.extend([sun.map, sun.frame]);
        }
        Self {
            light: compute(device, &module, &pipeline_layout, "gi_light", &constants),
            header: compute(
                device,
                &module,
                &pipeline_layout,
                "gi_shadow_header",
                &constants,
            ),
            layout,
            rays: StreamBuffers::new(device, &shadows),
            pending: buffer(
                device,
                "GI pending light",
                u64::from(MAX_RAYS) * PENDING_BYTES,
                wgpu::BufferUsages::STORAGE,
            ),
            lights: LocalLights::new(device),
            sun_transmittance: sun,
            no_sun_transmittance: sun_transmittance::stand_in(device),
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
            stats: stats.buffer().clone(),
            cache: helio_core::gpu_cache::GpuCache::new(),
            reads: helio_frame_api::static_names(&reads),
            writes: helio_frame_api::static_names(&[
                GI_PENDING,
                GI_ANSWERED,
                shadows.header,
                shadows.rays,
                shadows.hits,
            ]),
            probes,
            shadows,
            answers,
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

    fn scene_buffers(&self) -> &'static [&'static str] {
        &[helio_light_api::LIGHT_LIST]
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.answers.begin();
        let binding = |name: &str| ctx.registry.buffer_binding(name);
        let (
            Some(frame),
            Some(probes),
            Some(lists),
            Some(sun),
            Some(trace),
            Some(answered),
            Some(hits),
            Some(surfaces),
        ) = (
            binding(GI_FRAME),
            binding(GI_PROBES),
            binding(GI_LISTS),
            binding(GI_SUN),
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
        let (cloud_map, cloud_frame) = self
            .sun_transmittance
            .and_then(|published| {
                ctx.registry
                    .texture_binding(published.map)
                    .zip(binding(published.frame))
            })
            .unwrap_or((&self.no_sun_transmittance.0, &self.no_sun_transmittance.1));
        let scene_lights = ctx
            .scene_buffers
            .get(helio_light_api::SCENE_LIGHTS.key())
            .map(|h| h.buffer.clone());
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
                    entry(3, sun),
                    entry(4, atmosphere),
                    entry(5, sky_table),
                    entry(6, trace),
                    entry(7, hits),
                    entry(8, surfaces),
                    entry(9, &self.rays.rays),
                    entry(10, &self.rays.hits),
                    entry(11, &self.rays.header),
                    entry(12, &self.pending),
                    entry(13, answered),
                    entry(14, self.lights.buffer()),
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: wgpu::BindingResource::TextureView(cloud_map),
                    },
                    entry(16, cloud_frame),
                    entry(17, &self.stats),
                ],
            },
        );
        let mut cmds = ctx.graphics_cmds();
        // The shadow stream's count, from 0.
        cmds.clear_buffer(&self.rays.header, 0, None);
        self.lights
            .record(ctx.device, &mut cmds, scene_lights.as_ref());
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
        self.rays.publish(&self.shadows, frame, "GILight");
    }
}
