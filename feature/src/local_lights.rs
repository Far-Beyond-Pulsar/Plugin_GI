//! How a probe ray's hit samples the scene's point and spot lights
//! (`gi_local_lights.wgsl`): GI's stand-in for the frame's light sampling.
//!
//! Once a frame, `gi_gather_lights` lists the scene's live point and spot
//! lights ([`helio_light_api::SCENE_LIGHTS`]), the first
//! [`MAX_LOCAL_LIGHTS`] of them, in row order; at a hit, `gi_local_light`
//! picks one by weighted reservoir over their unshadowed irradiance there
//! and returns it over its chance, for the caller to trace its shadow ray.
//! Directional lights are the sun's (`gi_sun`).
//!
//! This is the one piece of GI that knows how lights are sampled. The
//! frame's light sampling belongs to the lights plugin (rendering.md, r5:
//! many lights sampled with traced visibility); when it lands, this module
//! and its two WGSL files are replaced by the lights' sampler, and nothing
//! else in GI changes (`gi_light.wgsl` asks for one sample at a point).

use crate::{buffer, compute, entry, pipeline_layout, storage};

/// Point and spot lights a hit chooses from.
pub const MAX_LOCAL_LIGHTS: u32 = 256;
/// Bytes of the gathered lights: a count, then 64 bytes a light.
const LOCAL_LIGHTS_BYTES: u64 = 16 + 64 * MAX_LOCAL_LIGHTS as u64;

/// The sampling's WGSL (`//!use gi_local_lights`): a reader binds
/// `local_lights` (a `GiLocalLights`).
pub const SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use gi_local_lights", "../shaders/gi_local_lights.wgsl");

/// The snippets `gi_gather_lights.wgsl` is built with.
pub(crate) const GATHER_SNIPPETS: [helio_core::shader::ShaderSnippet; 3] = [
    helio_light_api::LIGHT_LIST_SNIPPET,
    SNIPPET,
    crate::GI_RAYS_SNIPPET,
];

/// The gathered lights and the pass that gathers them (module docs).
pub struct LocalLights {
    lights: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    gather: wgpu::ComputePipeline,
    /// No light rows: a binding with none.
    no_lights: wgpu::Buffer,
    cache: helio_core::gpu_cache::GpuCache,
}

impl LocalLights {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI local lights"),
            entries: &[storage(0, true), storage(1, false)],
        });
        let module = helio_core::shader::module_with(
            device,
            "GI local lights",
            helio_core::include_wgsl!("../shaders/gi_gather_lights.wgsl"),
            &GATHER_SNIPPETS,
        );
        let pipeline_layout = pipeline_layout(device, "GI local lights", &layout);
        Self {
            gather: compute(device, &module, &pipeline_layout, "gi_gather_lights", &[]),
            layout,
            lights: buffer(
                device,
                "GI local lights",
                LOCAL_LIGHTS_BYTES,
                wgpu::BufferUsages::STORAGE,
            ),
            no_lights: buffer(
                device,
                "GI no lights",
                std::mem::size_of::<helio_light_api::GpuLight>() as u64,
                wgpu::BufferUsages::STORAGE,
            ),
            cache: helio_core::gpu_cache::GpuCache::new(),
        }
    }

    /// The gathered lights (`GiLocalLights`), for the sampling's readers.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.lights
    }

    /// Records this frame's gathering from the scene's light rows
    /// `scene_lights` (none: no lights).
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        cmds: &mut helio_core::CommandRecorder<'_>,
        scene_lights: Option<&wgpu::Buffer>,
    ) {
        self.cache.end_frame();
        let group = self.cache.bind_group(
            device,
            &wgpu::BindGroupDescriptor {
                label: Some("GI local lights"),
                layout: &self.layout,
                entries: &[
                    entry(0, scene_lights.unwrap_or(&self.no_lights)),
                    entry(1, &self.lights),
                ],
            },
        );
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI local lights"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&self.gather);
        pass.dispatch_workgroups(1, 1, 1);
    }
}
