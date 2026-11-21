//! A mesh scene for GI's graph tests (`mesh_gi_graph`, `pixel_gather`):
//! SceneDB with its GPU mirror and the mesh, material, object and light
//! rows, boxes of a colour (and a glow), a room, buffer readback, an adapter.
#![allow(dead_code)]

use std::sync::Arc;

use glam::{Mat4, Vec3};
use helio_light_host::LightComponent;
use helio_mats::host_rows::MaterialComponent;
use helio_mesh_host::{MeshComponent, StaticObjectComponent};

pub fn scene_db(device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>) -> pulsar_scenedb::SceneDb {
    let mut scene_db = pulsar_scenedb::SceneDb::new();
    let ctx = pulsar_scenedb::gpu::EngineGpuContext::new(device.clone(), queue.clone());
    let store = pulsar_scenedb::gpu::SceneGpuStore::new(
        &ctx,
        pulsar_scenedb::gpu::SceneGpuConfig {
            classes: Vec::new(),
            tombstone_headroom: 0,
            max_cells_metadata: 0,
        },
    );
    MeshComponent::register_gpu_columns_growable(&store, 64, device);
    MaterialComponent::register_gpu_columns_growable(&store, 64, device);
    StaticObjectComponent::register_gpu_columns_growable(&store, 64, device);
    helio_light_host::LightComponent::register_gpu_columns_growable(
        &store,
        helio_light_api::MAX_LIGHTS,
        device,
    );
    scene_db
        .world
        .attach_gpu_mirror(pulsar_scenedb::gpu::GpuMirrorHandle::new(
            Arc::new(store),
            queue.clone(),
        ));
    scene_db
}

/// An axis-aligned box of half extent `half` at `centre` in `colour`.
pub fn spawn_box(world: &mut pulsar_scenedb::World, centre: Vec3, half: Vec3, colour: [f32; 3]) {
    spawn_glowing_box(world, centre, half, colour, 0.0);
}

/// [`spawn_box`] glowing with radiance `glow` (white) from every face.
pub fn spawn_glowing_box(
    world: &mut pulsar_scenedb::World,
    centre: Vec3,
    half: Vec3,
    colour: [f32; 3],
    glow: f32,
) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for axis in 0..3 {
        for sign in [-1.0f32, 1.0] {
            let n = Vec3::from_array(std::array::from_fn(|k| if k == axis { sign } else { 0.0 }));
            let (u, v) = (
                Vec3::from_array(std::array::from_fn(|k| f32::from(k == (axis + 1) % 3))),
                Vec3::from_array(std::array::from_fn(|k| f32::from(k == (axis + 2) % 3))),
            );
            let (u, v) = if sign > 0.0 { (u, v) } else { (v, u) };
            let base = vertices.len() as u32;
            for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let p = (n + u * a + v * b) * half;
                vertices.push(helio_mesh_api::PackedVertex::from_components(
                    p.to_array(),
                    n.to_array(),
                    [0.0; 2],
                    u.to_array(),
                    1.0,
                ));
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    let mesh = world.spawn();
    world.insert(mesh, MeshComponent { vertices, indices });
    let material = world.spawn();
    let row = MaterialComponent::from(helio_mats::MaterialRow::new(
        [colour[0], colour[1], colour[2], 1.0],
        0.8,
        0.0,
        [1.0; 3],
        glow,
    ));
    world.insert(material, row);
    let store = world.gpu_mirror().cloned().unwrap();
    let vertex_range = MeshComponent::vertices_gpu_handle(store.store(), mesh.index()).unwrap();
    let index_range = MeshComponent::indices_gpu_handle(store.store(), mesh.index()).unwrap();
    let transform = Mat4::from_translation(centre);
    let object = StaticObjectComponent::from(helio_mesh_api::StaticObjectRow::new(
        mesh.index(),
        mesh.generation().wrapping_add(1),
        material.index(),
        material.generation().wrapping_add(1),
        transform,
        [centre.x, centre.y, centre.z, half.length()],
        index_range.count,
        index_range.offset,
        vertex_range.offset as i32,
        row.material_class,
        0,
        0,
    ));
    let entity = world.spawn();
    world.insert(entity, object);
}

/// The room's point light at `intensity`.
pub fn room_light(intensity: f32) -> LightComponent {
    LightComponent::from(helio_light_api::LightRow::from(helio_light_api::GpuLight {
        position_range: [0.0, 2.5, 0.0, 12.0],
        color_intensity: [1.0, 0.9, 0.8, intensity],
        shadow_index: u32::MAX,
        light_type: 1,
        ..Default::default()
    }))
}

/// A room (floor, walls, ceiling: boxes) with a pillar and a point light.
pub fn room(world: &mut pulsar_scenedb::World) -> pulsar_scenedb::Entity {
    room_at(world, Vec3::ZERO)
}

/// [`room`] moved by `offset`.
pub fn room_at(world: &mut pulsar_scenedb::World, offset: Vec3) -> pulsar_scenedb::Entity {
    let light = world.spawn();
    let mut row = room_light(20.0);
    row.position_range[0] += offset.x;
    row.position_range[1] += offset.y;
    row.position_range[2] += offset.z;
    world.insert(light, row);
    let white = [0.7; 3];
    spawn_box(
        world,
        offset + Vec3::new(0.0, -0.25, 0.0),
        Vec3::new(5.0, 0.25, 5.0),
        white,
    );
    spawn_box(
        world,
        offset + Vec3::new(0.0, 4.25, 0.0),
        Vec3::new(5.0, 0.25, 5.0),
        white,
    );
    spawn_box(
        world,
        offset + Vec3::new(-5.25, 2.0, 0.0),
        Vec3::new(0.25, 2.5, 5.0),
        [0.7, 0.1, 0.1],
    );
    spawn_box(
        world,
        offset + Vec3::new(5.25, 2.0, 0.0),
        Vec3::new(0.25, 2.5, 5.0),
        [0.1, 0.7, 0.1],
    );
    spawn_box(
        world,
        offset + Vec3::new(0.0, 2.0, -5.25),
        Vec3::new(5.0, 2.5, 0.25),
        white,
    );
    spawn_box(
        world,
        offset + Vec3::new(0.0, 2.0, 5.25),
        Vec3::new(5.0, 2.5, 0.25),
        white,
    );
    spawn_box(
        world,
        offset + Vec3::new(1.5, 1.0, -1.0),
        Vec3::new(0.4, 1.0, 0.4),
        white,
    );
    light
}

/// A closed box room 10 x 5 x 10 m (floor, ceiling, four walls) of albedo
/// `albedo` glowing with radiance `glow` from every face, and nothing else:
/// a furnace, whose radiance everywhere is `glow / (1 - albedo)`.
pub fn furnace(world: &mut pulsar_scenedb::World, albedo: f32, glow: f32) {
    let c = [albedo; 3];
    for (centre, half) in [
        (Vec3::new(0.0, -0.25, 0.0), Vec3::new(5.5, 0.25, 5.5)),
        (Vec3::new(0.0, 5.25, 0.0), Vec3::new(5.5, 0.25, 5.5)),
        (Vec3::new(-5.25, 2.5, 0.0), Vec3::new(0.25, 2.5, 5.0)),
        (Vec3::new(5.25, 2.5, 0.0), Vec3::new(0.25, 2.5, 5.0)),
        (Vec3::new(0.0, 2.5, -5.25), Vec3::new(5.0, 2.5, 0.25)),
        (Vec3::new(0.0, 2.5, 5.25), Vec3::new(5.0, 2.5, 0.25)),
    ] {
        spawn_glowing_box(world, centre, half, c, glow);
    }
}

/// A 2D texture's texels read back as f32 (half-float formats unpacked):
/// `channels` a texel.
pub fn read_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Vec<f32> {
    let (width, height) = (texture.width(), texture.height());
    let channels = match texture.format() {
        wgpu::TextureFormat::Rg16Float => 2,
        wgpu::TextureFormat::Rgba16Float => 4,
        other => panic!("{other:?}"),
    };
    let row = (width * channels * 2).div_ceil(256) * 256;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
    let mut out = Vec::with_capacity((width * height * channels) as usize);
    for y in 0..height {
        let line = &bytes[(y * row) as usize..];
        for x in 0..width * channels {
            let at = (x * 2) as usize;
            out.push(half(u16::from_le_bytes([line[at], line[at + 1]])));
        }
    }
    out
}

/// A half float's value.
pub fn half(bits: u16) -> f32 {
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    let value = match exponent {
        0 => mantissa * (-24.0f32).exp2(),
        31 => f32::INFINITY,
        e => (1.0 + mantissa / 1024.0) * ((e - 15) as f32).exp2(),
    };
    if bits & 0x8000 != 0 {
        -value
    } else {
        value
    }
}

/// The frame's indirect diffuse (`INDIRECT_DIFFUSE`, two words a pixel) as
/// rgb irradiance / pi and the weight it holds a pixel with.
pub fn indirect(words: &[u32]) -> Vec<[f32; 4]> {
    words
        .chunks(2)
        .map(|w| {
            [
                w[0] as u16,
                (w[0] >> 16) as u16,
                w[1] as u16,
                (w[1] >> 16) as u16,
            ]
            .map(half)
        })
        .collect()
}

pub fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec()
}

pub fn adapter() -> Option<wgpu::Adapter> {
    let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok()
}
