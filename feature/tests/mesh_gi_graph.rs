//! World-space GI in a mesh scene's default graph (no voxel terrain): with
//! ray queries the probes' streams are traced by the meshes' tracer
//! (`helio-pass-mesh-rays`, in the meshes' features), and the probes hold
//! until a frame's rays meet something (before the renderer has a TLAS:
//! `SceneDbRayTracing::partial` published); without ray queries nothing
//! traces rays, so there is no GI, the sky's ambient as before.

use std::sync::Arc;

use glam::Vec3;
use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    RendererBuilder, RendererConfig,
};
use helio_mesh_host::ray_tracing::SceneDbRayTracing;
use helio_pass_gi::{GiPass, PROBES, PROBE_WORDS};
use helio_pass_mesh_rays::MeshRayPass;

mod mesh_scene;
use mesh_scene::{adapter, read, room, scene_db};

/// The probes updated at least once, and of those the ones with light.
fn probes_lit(words: &[u32]) -> (usize, usize) {
    let mut updated = 0;
    let mut lit = 0;
    for index in 0..PROBES as usize {
        let at = index * PROBE_WORDS as usize;
        if words[at + 6] >> 8 == 0 {
            continue;
        }
        updated += 1;
        // The DC term's red and green (half floats): any light at all.
        if words[at] & 0x7fff_7fff != 0 {
            lit += 1;
        }
    }
    (updated, lit)
}

#[test]
fn a_mesh_scene_s_probes_trace_its_meshes_on_frames_with_a_tlas() {
    let Some(adapter) = adapter() else { return };
    let features = required_wgpu_features(helio::usable_adapter_features(&adapter));
    if !features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
        eprintln!("no ray queries: skipped");
        return;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        required_limits: required_wgpu_limits(adapter.limits()),
        experimental_features: required_experimental_features(features),
        ..Default::default()
    }))
    .unwrap();
    let (device, queue) = (Arc::new(device), Arc::new(queue));
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut scene = scene_db(&device, &queue);
    room(&mut scene.world);
    let config = RendererConfig::new(64, 64, wgpu::TextureFormat::Rgba8Unorm);
    let mut renderer =
        RendererBuilder::new(config.clone(), scene.world.gpu_mirror().cloned().unwrap())
            .with_graph(helio_mesh_host::presets::with(
                helio_presets::Preset::Deferred,
                &helio_feature_api::FeatureSet::new().with(helio_pass_gi::GiFeature),
            ))
            .with_external_device()
            .build(
                device.clone(),
                queue.clone(),
                config.width,
                config.height,
                config.surface_format,
            );
    assert!(
        renderer.find_pass::<MeshRayPass>().is_some(),
        "no mesh tracer in the graph"
    );
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.surface_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let camera = Camera::perspective_look_at(
        Vec3::new(-2.0, 1.7, 3.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::Y,
        1.0,
        1.0,
        0.1,
        100.0,
    );
    let probes = |renderer: &helio::Renderer| {
        read(
            &device,
            &queue,
            renderer.find_pass::<GiPass>().unwrap().probes(),
        )
    };
    // No TLAS published: every ray misses, and the probes hold.
    for _ in 0..3 {
        scene.world.flush_gpu_mirror(&queue);
        renderer.render(&camera, &view).unwrap();
    }
    assert_eq!(
        probes_lit(&probes(&renderer)),
        (0, 0),
        "probes updated without a TLAS"
    );
    let mut rays = SceneDbRayTracing::partial(device.clone(), queue.clone());
    for _ in 0..40 {
        scene.world.flush_gpu_mirror(&queue);
        rays.prepare(&scene.world).unwrap();
        rays.publish(&mut renderer);
        renderer.render(&camera, &view).unwrap();
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let (updated, lit) = probes_lit(&probes(&renderer));
    // 40 frames of 2720 updates, the room's light bounced into most of the
    // finest probes inside it.
    assert!(updated > 40_000, "{updated} probes updated");
    assert!(lit > 1000, "{lit} probes lit of {updated}");
    let error = pollster::block_on(errors.pop());
    assert!(error.is_none(), "{error:?}");
}

#[test]
fn without_ray_queries_a_mesh_scene_has_no_gi() {
    let Some(adapter) = adapter() else { return };
    let features =
        required_wgpu_features(adapter.features()) - wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        required_limits: required_wgpu_limits(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        ..Default::default()
    }))
    .unwrap();
    let (device, queue) = (Arc::new(device), Arc::new(queue));
    assert!(!helio_pass_mesh_rays::mesh_rays_supported(&device));
    let mut scene = scene_db(&device, &queue);
    room(&mut scene.world);
    let config = RendererConfig::new(32, 32, wgpu::TextureFormat::Rgba8Unorm);
    let mut renderer =
        RendererBuilder::new(config.clone(), scene.world.gpu_mirror().cloned().unwrap())
            .with_graph(helio_mesh_host::presets::with(
                helio_presets::Preset::Deferred,
                &helio_feature_api::FeatureSet::new().with(helio_pass_gi::GiFeature),
            ))
            .with_external_device()
            .build(
                device.clone(),
                queue.clone(),
                config.width,
                config.height,
                config.surface_format,
            );
    assert!(
        renderer.find_pass::<GiPass>().is_none(),
        "GI without a tracer"
    );
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.surface_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    scene.world.flush_gpu_mirror(&queue);
    let camera = Camera::perspective_look_at(
        Vec3::new(-2.0, 1.7, 3.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::Y,
        1.0,
        1.0,
        0.1,
        100.0,
    );
    renderer
        .render(&camera, &target.create_view(&Default::default()))
        .unwrap();
}
