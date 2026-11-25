//! This frame's probe rays into their probes (`gi_integrate.wgsl`, README.md
//! "The frame"), once their hits were lit (from the radiance cache, or on
//! the spot where it has no light for them, through the shadow stream): the
//! probes update in the frame their rays were traced in. On a frame where a
//! tracer answered GI's stream or the shadow stream short (`stream::Answers`:
//! a kind's kernels still compiling, what it traces not drawn yet) the
//! probes hold: rays that miss geometry a tracer knows of would teach them a
//! world without it, and their history keeps what it is taught for many
//! updates.
//!
//! A probe never updated whose hits met cells still new holds too, and goes
//! on the held list, read back without waiting ([`Held`]): GI's schedule
//! takes those probes first again once the list comes back, a few frames
//! later, when the cache has lit the cells their hits made (the cache lights
//! new cells first). A probe's first update so waits on the cache's light,
//! not on its turn in the schedule; every update after takes every ray.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use helio_core::{PassContext, RenderPass, Result as HelioResult};
use helio_frame_api::rays::stream;

use crate::{
    compute, entry, pipeline_layout, storage, uniform, FRAME_BYTES, GI_ANSWERED, GI_COMMON_SNIPPET,
    GI_FRAME, GI_LISTS, GI_PENDING, GI_PROBES, GI_RAYS_SNIPPET, MAX_UPDATES,
};

/// The snippets `gi_integrate.wgsl` is built with.
pub(crate) const SNIPPETS: [helio_core::shader::ShaderSnippet; 3] =
    [stream::SNIPPET, GI_COMMON_SNIPPET, GI_RAYS_SNIPPET];

/// Places of the held list: a few frames' updates (the list's way back).
pub(crate) const HELD: u32 = 4 * MAX_UPDATES;

/// The probes held for want of the cache's light, as their list comes back
/// (module docs): GI's schedule takes them first.
#[derive(Clone, Default)]
pub struct Held(Arc<Mutex<Vec<u32>>>);

impl Held {
    /// The probes held since the last take.
    pub fn take(&self) -> Vec<u32> {
        std::mem::take(&mut *self.0.lock().unwrap_or_else(|p| p.into_inner()))
    }

    fn add(&self, probes: &[u32]) {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .extend_from_slice(probes);
    }
}

/// The probes' integration (module docs).
pub struct GiIntegratePass {
    /// GI's stream's answers and the shadow stream's.
    answers: [Arc<stream::Answers>; 2],
    /// The shadow stream's hits (what a probe's own light sample waits on).
    shadow_hits: &'static str,
    /// The held list on the GPU, its way back and where it goes.
    held: wgpu::Buffer,
    readbacks: helio_core::gpu_cache::ReadbackPool,
    readback: Option<(helio_core::gpu_cache::PooledBuffer, Arc<AtomicBool>, bool)>,
    retries: Held,
    layout: wgpu::BindGroupLayout,
    pipeline: helio_core::shader::DeferredCompute,
    cache: helio_core::gpu_cache::GpuCache,
    reads: &'static [&'static str],
}

impl GiIntegratePass {
    /// The pass, holding the probes on frames where `answers` (GI's
    /// stream's and the shadow stream's, `shadows`) were short, its held
    /// probes back to `retries`.
    pub fn new(
        device: &wgpu::Device,
        answers: [Arc<stream::Answers>; 2],
        shadows: &stream::Stream,
        retries: Held,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI integrate"),
            entries: &[
                uniform(0, FRAME_BYTES),
                storage(1, false),
                storage(2, true),
                storage(3, true),
                storage(4, true),
                storage(5, true),
                storage(6, false),
            ],
        });
        let module = helio_core::shader::module_with(
            device,
            "GI integrate",
            helio_core::include_wgsl!("../shaders/gi_integrate.wgsl"),
            &SNIPPETS,
        );
        let pipeline_layout = pipeline_layout(device, "GI integrate", &layout);
        Self {
            answers,
            shadow_hits: shadows.hits,
            held: crate::buffer(
                device,
                "GI held probes",
                u64::from(1 + HELD) * 4,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            readbacks: helio_core::gpu_cache::ReadbackPool::new("GI held probes"),
            readback: None,
            retries,
            pipeline: compute(
                device,
                &module,
                &pipeline_layout,
                "gi_integrate",
                &[("GI_HELD", f64::from(HELD))],
            ),
            layout,
            cache: helio_core::gpu_cache::GpuCache::new(),
            reads: helio_frame_api::static_names(&[
                GI_FRAME,
                GI_PROBES,
                GI_LISTS,
                GI_PENDING,
                GI_ANSWERED,
                shadows.hits,
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

    fn prepare(&mut self, _ctx: &helio_core::PrepareContext) -> HelioResult<()> {
        // The held list copied out last frame: map it once that copy was
        // submitted, hand its probes to the schedule once mapped.
        if let Some((buffer, ready, mapping)) = self.readback.as_mut() {
            if !*mapping {
                buffer.map_read(Arc::clone(ready));
                *mapping = true;
            } else if ready.load(Ordering::Acquire) {
                let words: Vec<u32> = {
                    let view = buffer.slice(..).get_mapped_range();
                    view.as_ref()
                        .map(|bytes| bytemuck::cast_slice::<u8, u32>(bytes).to_vec())
                        .unwrap_or_default()
                };
                if let Some((&count, probes)) = words.split_first() {
                    let count = (count.min(HELD) as usize).min(probes.len());
                    self.retries.add(&probes[..count]);
                }
                buffer.unmap();
                self.readback = None;
            }
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.answers.iter().all(|a| a.complete()) {
            return Ok(());
        }
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
                    entry(6, &self.held),
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
        drop(pass);
        // The held list on its way back, emptied for the frames after,
        // unless the last copy is still on its way.
        if self.readback.is_none() {
            let bytes = u64::from(1 + HELD) * 4;
            let buffer = self.readbacks.get(ctx.device, bytes);
            cmds.copy_buffer_to_buffer(&self.held, 0, &buffer, 0, bytes);
            cmds.clear_buffer(&self.held, 0, Some(4));
            self.readback = Some((buffer, Arc::new(AtomicBool::new(false)), false));
        }
        Ok(())
    }
}
