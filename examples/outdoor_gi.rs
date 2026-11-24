//! An open scene lit by GI: a forest at the edge of a city of tall blocks,
//! on a 2.4 km ground, under the sky plugin's atmosphere and a low sun.
//!
//! The mesh scene GI's radiance cache is measured on outdoors (the cathedral
//! is its interior): a few hundred metres of visible surface around the eye,
//! large smooth occluders (the blocks, 6 to 60 m tall, out to 1.1 km), and
//! volume-filling clutter (trees whose crowns are leaf cards, two-sided
//! foliage, around the eye). Built from the GI examples' helpers alone.
//!
//! Example switches (as the cathedral's, `common.rs`): `HELIO_GI_QUALITY`,
//! `HELIO_GI_CELL_ANGLE`, `HELIO_CAPTURE`, `HELIO_CAPTURE_FRAME`,
//! `HELIO_CAPTURE_FRAMES`, `HELIO_CAPTURE_TIMINGS`, `HELIO_DEBUG_MODE`; and
//! `HELIO_LOOK_AROUND=1`: the camera turns by one radian about up over the
//! 30 frames before the capture, so a burst is shot just after a look around
//! (as voxel_craft's pose 12). The camera holds still otherwise.
//!
//! Controls: mouse drag to look, WASD / Space / Shift to move, Escape to exit.

mod common;

use common::{
    box_mesh, directional_light, make_material, new_scene_db_with_gpu_mirror, plane_mesh,
    scene_db_handle, spawn_light, spawn_material, spawn_mesh, spawn_object, spawn_sky, SceneRays,
    SurfaceCapture,
};
use glam::{Mat4, Quat, Vec3};
use helio::{
    Camera, HelioGraphicsConfig, HelioGraphicsContext, Renderer, RendererBuilder, RendererConfig,
};
use helio_mesh_api::{MeshUpload, PackedVertex};
use pulsar_scenedb::SceneDb;
use std::collections::HashSet;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

/// The ground's half-extent (m).
const GROUND: f32 = 1200.0;
/// The city: a block every `BLOCK` metres, from `CITY_NEAR` to `CITY_FAR`
/// from the eye's start, on the half of the ground the eye looks across.
const BLOCK: f32 = 45.0;
const CITY_NEAR: f32 = 160.0;
const CITY_FAR: f32 = 1100.0;
/// The forest: trees on a jittered grid `TREE_STEP` apart, within
/// `FOREST` metres of the eye's start.
const TREE_STEP: f32 = 7.0;
const FOREST: f32 = 150.0;
/// Leaf cards a crown, each `LEAF` metres square.
const LEAVES: usize = 220;
const LEAF: f32 = 0.45;
/// Frames the look around takes.
const TURN_FRAMES: u32 = 30;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App { state: None };
    event_loop.run_app(&mut app).expect("run");
}

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    scene_db: SceneDb,
    rays: SceneRays,
    capture: SurfaceCapture,
    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    look_around: bool,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
    last_frame: std::time::Instant,
}

/// A seeded generator for the scene's layout (the same scene every run).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }

    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next()
    }
}

/// One tree's crown: `LEAVES` leaf cards (two triangles each, turned every
/// way) in an ellipsoid about `centre`, as one mesh every tree instances.
fn crown_mesh(rng: &mut Rng, centre: Vec3, radius: Vec3) -> MeshUpload {
    let mut vertices = Vec::with_capacity(LEAVES * 4);
    let mut indices = Vec::with_capacity(LEAVES * 6);
    for i in 0..LEAVES {
        let p = loop {
            let q = Vec3::new(
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            );
            if q.length_squared() <= 1.0 {
                break centre + q * radius;
            }
        };
        let turn = Quat::from_euler(
            glam::EulerRot::YXZ,
            rng.range(0.0, std::f32::consts::TAU),
            rng.range(-1.2, 1.2),
            rng.range(-0.6, 0.6),
        );
        let (u, v, n) = (
            turn * Vec3::X * LEAF * 0.5,
            turn * Vec3::Z * LEAF * 0.5,
            turn * Vec3::Y,
        );
        let base = (i * 4) as u32;
        for (corner, uv) in [
            (-u - v, [0.0, 0.0]),
            (u - v, [1.0, 0.0]),
            (u + v, [1.0, 1.0]),
            (-u + v, [0.0, 1.0]),
        ] {
            vertices.push(PackedVertex::from_components(
                (p + corner).to_array(),
                n.to_array(),
                uv,
                u.normalize().to_array(),
                1.0,
            ));
        }
        indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
    MeshUpload { vertices, indices }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let capture = SurfaceCapture::from_env();
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio – Outdoor GI")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1920u32, 1080u32)),
                )
                .expect("window"),
        );
        let size = window.inner_size();
        let mut graphics = pollster::block_on(HelioGraphicsContext::new(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            HelioGraphicsConfig {
                application_name: "Outdoor GI".into(),
                surface_usage: capture.usage(),
                ..Default::default()
            },
        ))
        .expect("Helio graphics context");
        let device = graphics.device.clone();
        let queue = graphics.queue.clone();
        let format = graphics.surface_config.format;
        device.on_uncaptured_error(Arc::new(|e: wgpu::Error| {
            panic!("[GPU UNCAPTURED ERROR] {e:?}");
        }));
        // Open ground: GI's finest probes a metre apart, as the voxel
        // examples set it; every other GI setting the default.
        let config = RendererConfig::new(size.width, size.height, format)
            .with_feature(helio_light_api::ShadowSettings {
                quality: helio_light_api::ShadowQuality::Ultra,
                ..Default::default()
            })
            .with_feature(common::gi_settings(helio_pass_gi::GiSettings {
                probe_spacing: Some(1.0),
                ..Default::default()
            }));
        let mut scene_db = new_scene_db_with_gpu_mirror(&device, &queue);
        let mut renderer = RendererBuilder::new(config, scene_db_handle(&scene_db))
            .with_graph(common::preset())
            .build_with_graphics_context(&mut graphics)
            .expect("Helio renderer");
        let surface = graphics.surface;
        let world = &mut scene_db.world;

        // The sun before any other entity (the lights' rows are scanned
        // from index 0: `indoor_cathedral_gi`'s note).
        spawn_light(
            world,
            directional_light([-0.55, -0.45, -0.7], [1.0, 0.95, 0.85], 3.0),
        );
        spawn_sky(world, [0.3, 0.55, 1.0]);

        let ground_mat = spawn_material(
            world,
            make_material([0.32, 0.36, 0.25, 1.0], 0.9, 0.0, [0.0; 3], 0.0),
        );
        let block_mat = spawn_material(
            world,
            make_material([0.62, 0.6, 0.56, 1.0], 0.8, 0.0, [0.0; 3], 0.0),
        );
        let bark_mat = spawn_material(
            world,
            make_material([0.25, 0.18, 0.12, 1.0], 0.9, 0.0, [0.0; 3], 0.0),
        );
        let leaf_mat = spawn_material(
            world,
            helio_mats::host_rows::MaterialComponent::from(
                helio_mats::MaterialRow::new([0.16, 0.3, 0.08, 1.0], 0.7, 0.0, [0.0; 3], 0.0)
                    .foliage(0.3),
            ),
        );
        let mut triangles = 0usize;

        let ground = spawn_mesh(world, plane_mesh([0.0; 3], GROUND));
        triangles += 2;
        let _ = spawn_object(world, ground, ground_mat, Mat4::IDENTITY, GROUND * 1.42);

        // The city, ahead of the eye (toward -z) past the forest.
        let mut rng = Rng(0x5eed_c17e);
        let mut blocks = 0usize;
        let reach = (CITY_FAR / BLOCK) as i32;
        for i in -reach..=reach {
            for j in -reach..=0 {
                let c = Vec3::new(i as f32 * BLOCK, 0.0, j as f32 * BLOCK);
                let d = c.length();
                if !(CITY_NEAR..=CITY_FAR).contains(&d) {
                    continue;
                }
                let (hw, hd, hh) = (
                    rng.range(8.0, 16.0),
                    rng.range(8.0, 16.0),
                    rng.range(3.0, 30.0),
                );
                let mesh = spawn_mesh(world, box_mesh([0.0; 3], [hw, hh, hd]));
                let _ = spawn_object(
                    world,
                    mesh,
                    block_mat,
                    Mat4::from_translation(c + Vec3::Y * hh),
                    hw.max(hd).max(hh) * 1.74,
                );
                blocks += 1;
                triangles += 12;
            }
        }

        // The forest about the eye: a trunk and a crown, one mesh each,
        // instanced, each tree turned and scaled a little.
        let trunk = spawn_mesh(world, box_mesh([0.0, 2.5, 0.0], [0.18, 2.5, 0.18]));
        let crown = spawn_mesh(
            world,
            crown_mesh(&mut rng, Vec3::new(0.0, 6.0, 0.0), Vec3::new(2.6, 2.2, 2.6)),
        );
        let mut trees = 0usize;
        let n = (FOREST / TREE_STEP) as i32;
        for i in -n..=n {
            for j in -n..=n {
                let p = Vec3::new(
                    i as f32 * TREE_STEP + rng.range(-2.5, 2.5),
                    0.0,
                    j as f32 * TREE_STEP + rng.range(-2.5, 2.5),
                );
                if p.length() > FOREST || p.length() < 4.0 {
                    continue;
                }
                let t = Mat4::from_scale_rotation_translation(
                    Vec3::splat(rng.range(0.8, 1.25)),
                    Quat::from_rotation_y(rng.range(0.0, std::f32::consts::TAU)),
                    p,
                );
                let _ = spawn_object(world, trunk, bark_mat, t, 3.0);
                let _ = spawn_object(world, crown, leaf_mat, t, 9.0);
                trees += 1;
                triangles += 12 + LEAVES * 2;
            }
        }
        println!(
            "[scene] ground {:.0} m square, {blocks} blocks to {CITY_FAR} m, {trees} trees within {FOREST} m ({LEAVES} leaf cards each), {triangles} triangles drawn",
            2.0 * GROUND
        );

        renderer.set_feature_live(helio::AmbientLight {
            color: [0.45, 0.58, 0.8],
            intensity: 0.35,
        });
        if let Some(mode) = std::env::var("HELIO_DEBUG_MODE")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            renderer.set_debug_mode(mode);
        }

        let rays = SceneRays::new(&device, &queue);
        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            renderer,
            scene_db,
            rays,
            cam_pos: Vec3::new(0.0, 1.7, 0.0),
            cam_yaw: -0.5,
            cam_pitch: 0.05,
            look_around: std::env::var("HELIO_LOOK_AROUND").as_deref() == Ok("1"),
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            last_frame: std::time::Instant::now(),
            capture,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ks,
                        physical_key: PhysicalKey::Code(key),
                        ..
                    },
                ..
            } => {
                if key == KeyCode::Escape && ks == ElementState::Pressed {
                    if state.cursor_grabbed {
                        state.cursor_grabbed = false;
                        let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                        state.window.set_cursor_visible(true);
                    } else {
                        event_loop.exit();
                    }
                } else if ks == ElementState::Pressed {
                    state.keys.insert(key);
                } else {
                    state.keys.remove(&key);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !state.cursor_grabbed => {
                if state
                    .window
                    .set_cursor_grab(CursorGrabMode::Confined)
                    .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                    .is_ok()
                {
                    state.window.set_cursor_visible(false);
                    state.cursor_grabbed = true;
                }
            }
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: state.capture.usage(),
                        format: state.surface_format,
                        width: s.width,
                        height: s.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                        color_space: wgpu::SurfaceColorSpace::Auto,
                    },
                );
                state.renderer.set_render_size(s.width, s.height);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.render(dt);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

impl AppState {
    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 8.0;
        const SENS: f32 = 0.002;
        self.cam_yaw += self.mouse_delta.0 * SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.4, 1.4);
        self.mouse_delta = (0.0, 0.0);
        // The look around: a radian over the frames before the capture.
        let frame = self.capture.upcoming();
        let at = self.capture.at();
        if self.look_around && frame + TURN_FRAMES >= at && frame < at {
            self.cam_yaw += 1.0 / TURN_FRAMES as f32;
        }
        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = Vec3::new(sy * cp, sp, -cy * cp);
        let right = Vec3::new(cy, 0.0, sy);
        for (key, step) in [
            (KeyCode::KeyW, forward),
            (KeyCode::KeyS, -forward),
            (KeyCode::KeyA, -right),
            (KeyCode::KeyD, right),
            (KeyCode::Space, Vec3::Y),
            (KeyCode::ShiftLeft, -Vec3::Y),
        ] {
            if self.keys.contains(&key) {
                self.cam_pos += step * SPEED * dt;
            }
        }
        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            Vec3::Y,
            70f32.to_radians(),
            aspect,
            0.1,
            5000.0,
        );
        common::flush_scene_db(&self.scene_db, &self.queue);
        self.rays.publish(&self.scene_db.world, &mut self.renderer);
        if let Err(e) = self
            .capture
            .render_window_frame(&mut self.renderer, &camera, &self.surface)
        {
            log::error!("Render error: {e:?}");
        }
    }
}
