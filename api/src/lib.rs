//! The GI feature's public contract.
//!
//! What a reader of the probes needs: the clipmap's layout ([`Clipmap`],
//! [`eye`]), the graph keys of the probes and the irradiance the GI passes
//! publish, and the WGSL that reads them (`gi_common`, `gi_sample`). Holds no
//! state; a feature that reads GI's irradiance depends on this crate, never on
//! `helio-pass-gi`.
//!
//! GI traces its rays through the frame's ray streams
//! (`helio_frame_api::rays::stream`): no kind traces for GI by name, so this
//! crate holds no tracer interface. What is left of the old one
//! ([`TerrainTracer`], [`GI_RAYS`], [`GI_HITS`], [`GI_CLIPMAP`], `GiRay` and
//! `GiHit` in `gi_common.wgsl`, [`GpuFrame`]'s `prev_rotation`) is there only
//! while the voxel plugin's old probe tracer still compiles against it; GI
//! itself reads and publishes none of it, and it goes with that tracer.

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, I64Vec3, Vec3};
use helio_core::PrepareContext;

/// The voxel plugin's old marker that it traced the probes. GI no longer
/// reads it (whether anything traces is `stream::traced`, the spacing is
/// `GiSettings::probe_spacing`); deleted with the voxel's old tracer.
#[derive(Clone, Copy, Debug, Default)]
pub struct TerrainTracer;

/// Levels of the clipmap, each twice as far apart as the one inside it.
pub const LEVELS: usize = 6;
/// Probes along a level's side.
pub const SIDE: i64 = 24;
pub const LEVEL_PROBES: u32 = (SIDE * SIDE * SIDE) as u32;
pub const PROBES: u32 = LEVEL_PROBES * LEVELS as u32;
/// The finest level's probe spacing by default (metres): a clipmap reaching
/// 384 m around the eye. Scenes of rooms set a finer one
/// (`GiSettings::probe_spacing`).
pub const DEFAULT_FINEST_SPACING: f64 = 1.0;
/// Rays a probe update traces.
pub const RAYS: u32 = 64;
/// Probes updated a frame, by level: the two finest every 20 frames or so,
/// the coarser (the far surroundings, slow to change, their rays the
/// longest walks) half as often.
pub const LEVEL_UPDATES: [usize; LEVELS] = [680, 680, 340, 340, 340, 340];
pub const MAX_UPDATES: u32 = {
    let mut sum = 0;
    let mut level = 0;
    while level < LEVELS {
        sum += LEVEL_UPDATES[level];
        level += 1;
    }
    sum as u32
};
pub const MAX_RAYS: u32 = MAX_UPDATES * RAYS;
/// Words of a probe's record (`gi_common.wgsl`).
pub const PROBE_WORDS: u32 = 72;

/// Graph keys of what the passes publish: the frame's [`GpuFrame`] and the
/// probes' records (what `gi_sample` reads), and the sun the probes' hits
/// are lit by (`array<vec4<f32>, 2>`: towards it, its illuminance).
pub const GI_FRAME: &str = "gi_frame";
pub const GI_PROBES: &str = "gi_probes";
pub const GI_SUN: &str = "gi_sun";
/// The probes' irradiance at each pixel: the frame's indirect diffuse.
pub const GI_IRRADIANCE: &str = helio_frame_api::lighting::INDIRECT_DIFFUSE;
/// Debug mode of the probes' view (`GiResolvePass`; deferred lighting shows
/// it as it shows `gi_irradiance`).
pub const GI_PROBES_VIEW: u32 = 36;

/// The voxel plugin's old probe tracer's keys (crate docs): GI publishes
/// none of them.
pub const GI_RAYS: &str = "gi_rays";
pub const GI_HITS: &str = "gi_hits";
/// The frame's [`Clipmap`] (a registry value), as that tracer read it.
pub const GI_CLIPMAP: &str = "gi_clipmap";

/// The probes' shared WGSL (`gi_common.wgsl`: the clipmap, a probe's record,
/// its rays' directions): a reader opts in with `//!use gi_common`.
pub const GI_COMMON_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use gi_common", "shaders/gi_common.wgsl");
/// Irradiance from the probes at a point (`gi_sample.wgsl`, with
/// `gi_common`): `//!use gi_sample`.
pub const GI_SAMPLE_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use gi_sample", "shaders/gi_sample.wgsl");

/// The eye's world position this frame: the frame's origin and the
/// camera's position in it.
pub fn eye(ctx: &PrepareContext) -> DVec3 {
    ctx.world_origin + Vec3::from_slice(&ctx.camera_data.position_near[..3]).as_dvec3()
}

/// Where each level's probes are: the world coordinates (in its spacings)
/// of its logical probe 0, and the finest level's spacing. Probes sit at
/// whole multiples of their spacing and are stored toroidally (storage
/// coordinates are world coordinates modulo [`SIDE`]), so a probe keeps its
/// storage while it stays in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clipmap {
    pub first: [I64Vec3; LEVELS],
    pub finest: f64,
}

impl Clipmap {
    /// The clipmap around world point `eye` with the finest level's probes
    /// `finest` apart: each level's grid with the eye between its middle
    /// probes.
    pub fn around(eye: DVec3, finest: f64) -> Self {
        let mut first = [I64Vec3::ZERO; LEVELS];
        for (level, first) in first.iter_mut().enumerate() {
            let cell = (eye / (finest * f64::from(1u32 << level))).floor();
            *first = I64Vec3::new(cell.x as i64, cell.y as i64, cell.z as i64) - (SIDE / 2 - 1);
        }
        Self { first, finest }
    }

    /// The probe spacing of `level`.
    pub fn spacing(&self, level: usize) -> f64 {
        self.finest * f64::from(1u32 << level)
    }

    /// The storage index of world probe `world` (in spacings) of `level`.
    pub fn index(level: usize, world: I64Vec3) -> u32 {
        let s = world.rem_euclid(I64Vec3::splat(SIDE));
        level as u32 * LEVEL_PROBES + (s.x + SIDE * (s.y + SIDE * s.z)) as u32
    }

    /// Probe `index`'s level and world coordinates (in its spacings).
    pub fn probe(&self, index: u32) -> (usize, I64Vec3) {
        let level = (index / LEVEL_PROBES) as usize;
        let at = i64::from(index % LEVEL_PROBES);
        let s = I64Vec3::new(at % SIDE, (at / SIDE) % SIDE, at / (SIDE * SIDE));
        let logical = (s - self.first[level]).rem_euclid(I64Vec3::splat(SIDE));
        (level, self.first[level] + logical)
    }

    /// Probe `index`'s world position.
    pub fn position(&self, index: u32) -> DVec3 {
        let (level, world) = self.probe(index);
        world.as_dvec3() * self.spacing(level)
    }

    /// Whether world probe `world` of `level` is in the clipmap.
    pub fn holds(&self, level: usize, world: I64Vec3) -> bool {
        let d = world - self.first[level];
        d.cmpge(I64Vec3::ZERO).all() && d.cmplt(I64Vec3::splat(SIDE)).all()
    }

    /// The storage indices of the probes in this clipmap that were not in
    /// `before` (all of them without one).
    pub fn entered(&self, before: Option<&Clipmap>) -> Vec<u32> {
        let mut out = Vec::new();
        self.entered_into(before, &mut out);
        out
    }

    /// [`Self::entered`] into `out` (cleared first), so a caller each frame
    /// reuses its list.
    pub fn entered_into(&self, before: Option<&Clipmap>, out: &mut Vec<u32>) {
        out.clear();
        for level in 0..LEVELS {
            if before
                .is_some_and(|b| b.first[level] == self.first[level] && b.finest == self.finest)
            {
                continue;
            }
            for z in 0..SIDE {
                for y in 0..SIDE {
                    for x in 0..SIDE {
                        let world = self.first[level] + I64Vec3::new(x, y, z);
                        if !before.is_some_and(|b| b.finest == self.finest && b.holds(level, world))
                        {
                            out.push(Self::index(level, world));
                        }
                    }
                }
            }
        }
    }

    /// The frame's levels for the GPU, relative to world point `eye`.
    pub fn levels(&self, eye: DVec3) -> [GpuLevel; LEVELS] {
        std::array::from_fn(|level| {
            let s = self.spacing(level);
            let origin = (self.first[level].as_dvec3() * s - eye).as_vec3();
            let base = self.first[level].rem_euclid(I64Vec3::splat(SIDE));
            GpuLevel {
                origin: [origin.x, origin.y, origin.z, s as f32],
                base: [base.x as u32, base.y as u32, base.z as u32, 0],
            }
        })
    }
}

/// `GiLevel` (`gi_common.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuLevel {
    pub origin: [f32; 4],
    pub base: [u32; 4],
}

/// `GiFrame` (`gi_common.wgsl`). `counts`: probes updated this frame, 0,
/// probes reset this frame, 0. `prev_rotation` is unused (crate docs).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuFrame {
    pub levels: [GpuLevel; LEVELS],
    pub sky: [f32; 4],
    pub ground: [f32; 4],
    pub counts: [u32; 4],
    pub rotation: [f32; 4],
    pub prev_rotation: [f32; 4],
}

pub const FRAME_BYTES: u64 = std::mem::size_of::<GpuFrame>() as u64;
const _: () = assert!(FRAME_BYTES == 272);
