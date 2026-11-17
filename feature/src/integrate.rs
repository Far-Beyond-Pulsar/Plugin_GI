//! This frame's probe rays into their probes (`gi_integrate.wgsl`, README.md
//! "The frame"), once the shadow stream's tracers answered: the probes
//! update in the frame their rays were traced in.

use helio_core::{PassContext, RenderPass, Result as HelioResult};
use helio_frame_api::rays::stream;

use crate::{
    compute, entry, pipeline_layout, storage, uniform, FRAME_BYTES, GI_ANSWERED, GI_COMMON_SNIPPET,
    GI_FRAME, GI_LISTS, GI_PENDING, GI_PROBES, GI_RAYS_SNIPPET, MAX_UPDATES,
};

/// The snippets `gi_integrate.wgsl` is built with.
pub(crate) const SNIPPETS: [helio_core::shader::ShaderSnippet; 3] =
    [stream::SNIPPET, GI_COMMON_SNIPPET, GI_RAYS_SNIPPET];

/// The probes' integration (module docs).
pub struct GiIntegratePass {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    shadow_hits: &'static str,
    cache: helio_core::gpu_cache::GpuCache,
    reads: &'static [&'static str],
}

impl GiIntegratePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI integrate"),
            entries: &[
                uniform(0, FRAME_BYTES),
                storage(1, false),
                storage(2, true),
                storage(3, true),
                storage(4, true),
                storage(5, true),
            ],
        });
        let module = helio_core::shader::module_with(
            device,
            "GI integrate",
            helio_core::include_wgsl!("../shaders/gi_integrate.wgsl"),
            &SNIPPETS,
        );
        let pipeline_layout = pipeline_layout(device, "GI integrate", &layout);
        let shadow_hits = crate::shadow_stream().hits;
        Self {
            pipeline: compute(device, &module, &pipeline_layout, "gi_integrate", &[]),
            layout,
            shadow_hits,
            cache: helio_core::gpu_cache::GpuCache::new(),
            reads: helio_frame_api::static_names(&[
                GI_FRAME,
                GI_PROBES,
                GI_LISTS,
                GI_PENDING,
                GI_ANSWERED,
                shadow_hits,
            ]),
        }
    }
}

impl RenderPass for GiIntegratePass {
    fn name(&self) -> &'static str {
        "GIIntegrate"
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        &[GI_PROBES]
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let binding = |name: &str| ctx.registry.buffer_binding(name);
        let (Some(frame), Some(probes), Some(lists), Some(pending), Some(shadows), Some(answered)) = (
            binding(GI_FRAME),
            binding(GI_PROBES),
            binding(GI_LISTS),
            binding(GI_PENDING),
            binding(self.shadow_hits),
            binding(GI_ANSWERED),
        ) else {
            return Ok(());
        };
        self.cache.end_frame();
        let group = self.cache.bind_group(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("GI integrate"),
                layout: &self.layout,
                entries: &[
                    entry(0, frame),
                    entry(1, probes),
                    entry(2, lists),
                    entry(3, pending),
                    entry(4, shadows),
                    entry(5, answered),
                ],
            },
        );
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI integrate"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&self.pipeline);
        // A workgroup a probe; those past this frame's count (`GiFrame`)
        // return at once.
        pass.dispatch_workgroups(MAX_UPDATES, 1, 1);
        Ok(())
    }
}
