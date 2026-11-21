//! GI's radiance cache (`gi_cache.wgsl`, README.md "The radiance cache"):
//! the publisher of the frame's `helio_frame_api::rays::radiance_cache`, and
//! the one place a surface is lit for indirect light.
//!
//! [`RadianceCache`], at the start of GI's update after the probes' rays,
//! walks the cells in use (a list on the GPU: the cells the last frame kept,
//! then the new cells the readers' marks queued; never the hash's slots):
//! evicts the cells no reader marked lately, lists the cells to light this
//! frame by need ([`CacheConfig::cells`] at most) and writes
//! [`RAYS_PER_CELL`] rays a lit cell into GI's stream after the probes'
//! rays, cosine distributed about the cell's normal from its point. GI's
//! light and shadow passes then light those rays' hits (a light sample and
//! a shadow ray each, the sky on a miss, apart what the hit reflects of the
//! cache's own light) and the cell's own direct light (a light sample and a
//! shadow ray a ray), and [`RadianceCacheFold`] folds each cell's rays into
//! it.
//!
//! The constants, and where each comes from:
//!
//! - [`RAYS_PER_CELL`] (4): a cell's update is a stratified set of the
//!   hemisphere (two low rays in two opposite ways of its surface that turn
//!   each frame, two high in an azimuth half each), the fewest strata whose
//!   mean a change test can judge and that explore every way of its surface
//!   once in four updates.
//! - A cell's parts: direct (its light samples), indirect (what its rays'
//!   hits send as the rays meet them: emission, the light samples' light
//!   off the hits, the sky) and bounced (what the hits reflect of the
//!   cache's own light, the later bounces). The direct and indirect parts
//!   are stationary in a static scene; each is the plain mean of its
//!   updates since it last restarted (weights that depend on the samples
//!   bias a mean: a first version, restarting at the bright update that
//!   tripped its test, doubled the cathedral's indirect light), over as
//!   many updates as bring its relative standard error to [`ERROR`] (3%:
//!   `(1.2533 * mad / mean / ERROR)^2`, the mean absolute deviation the
//!   change test measures), between [`HISTORY_MIN`] (16, twice what the
//!   change test needs before it judges) and [`HISTORY_MAX`] (64, twice
//!   the first version's 32: a noisier part's error stays above the target,
//!   and the work a cell takes to settle, which a moving view's new cells
//!   multiply, stays a few dozen updates). The bounced part follows the cells its hits read, which
//!   converge themselves: [`BOUNCED`] updates (16), so each bounce settles
//!   in a few dozen updates whatever the noise of the others. One history
//!   for both, as before, made each bounce wait on the last: the CPU
//!   radiosity box read 5-8% low once its histories filled.
//! - Need: never lit, a part restarting, the refresh (every
//!   [`refresh_period`] frames, a cell's turn spread by its slot, settled or
//!   not), then the cells still converging, the least share of their
//!   histories first; the budget is a cap.
//! - [`SPREAD`]: a cell's ray stands for its share of the hemisphere over
//!   twice the least history, `sqrt(2 pi / 128)`, so its hit reads the
//!   cache at the detail the cell's estimate resolves.
//! - The change test (direct and indirect parts; the bounced part restarts
//!   with the indirect) only decides that the part restarts at its next
//!   update, whose sample it has not seen:
//!   - a fast mean ([`FAST`] updates) and a slow one (the history's) of each
//!     update's luminance clamped from above at the fast mean plus [`CLAMP`]
//!     (3) mean absolute deviations (one bright update of a heavy tail
//!     cannot move it), apart by more than [`LAG`] deviations: the
//!     difference of the two means has a standard deviation of
//!     `sqrt(1/7 + b/(2-b) - 2c)` of an update's (`c` the two exponential
//!     means' covariance, `b = 1/64`), so 1.74 is 4 standard deviations, a
//!     false alarm once in 16,000 updates (a part fills its longest history
//!     unrestarted 996 times in 1000); and by [`FLOOR`] (1%) of the slow mean,
//!     for parts with no noise at all;
//!   - after [`DETECT`] updates (twice the fast mean's length).
//!
//!   A light switched off restarts the parts it lit within a few updates
//!   (their luminance falls and is not clamped from below).
//! - Clear distances ([`CLEAR_GROWTH`]): each cell keeps, along 8 ways over
//!   its surface, how far its low rays (below 45 degrees) found nothing in
//!   the way; a way not bounded again grows by `2^(1/8)` an update, twice in
//!   8 updates (about four rays a way), so an opened way is found again.
//! - [`EVICT_AGE`] (128 frames): twice the longest a reader of the cache
//!   leaves between two reads of a cell it keeps reading (a probe of GI's
//!   coarser levels is updated every 41 frames), so no cell a reader still
//!   uses goes; past half the slots the least recently used go too, keeping
//!   buckets' room (a bucket of 16 at half load is full for 0.4% of keys).
//! - The eye angle ([`CacheConfig::eye_angle`]): cells at least `a * d` wide
//!   at a distance `d` from the eye hold at most `4 pi / a^2` cells a level
//!   for surfaces all around the eye; over the levels that is half the
//!   slots: `a = sqrt(4 pi LEVELS / (slots / 2))`, 0.040 rad at the default
//!   2^18 slots. A larger cache resolves finer cells.

use std::sync::Arc;

use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_frame_api::rays::radiance_cache::{
    self, RadianceCacheEntry, RadianceCacheFrame, RadianceCacheSlot, ENTRIES, FRAME, LEVELS, SLOTS,
};
use helio_frame_api::rays::stream::{self, Stream};

use crate::{
    buffer, compute, entry, pipeline_layout, storage, uniform, FRAME_BYTES, GI_ANSWERED,
    GI_COMMON_SNIPPET, GI_FRAME, GI_PENDING, GI_RAYS_SNIPPET,
};

/// Rays a lit cell traces a frame (module docs).
pub const RAYS_PER_CELL: u32 = 4;
/// Updates a part's history holds at least (module docs).
pub const HISTORY_MIN: u32 = 16;
/// Updates a part's history holds at most (module docs).
pub const HISTORY_MAX: u32 = 64;
/// The relative standard error a part's history brings its value to,
/// within those bounds (module docs).
pub const ERROR: f32 = 0.03;
/// Updates the bounced part's history holds (module docs).
pub const BOUNCED: u32 = 16;
// A part's updates fit their 8 bits.
const _: () = assert!(HISTORY_MAX <= 255 && BOUNCED <= HISTORY_MAX);
/// The spread of a cell's ray: `sqrt(2 pi / (RAYS_PER_CELL * 2 * HISTORY_MIN))`.
pub const SPREAD: f32 = 0.2216;
/// How much a cell's clear distance along a way grows an update its rays
/// meet nothing nearer that way: `2^(1 / 8)`, twice in eight updates, about
/// four rays a way (module docs).
pub const CLEAR_GROWTH: f32 = 1.0905;
/// Updates the change test's fast mean holds (module docs).
pub const FAST: u32 = 4;
/// The change test clamps an update's luminance at the fast mean plus this
/// many mean absolute deviations (module docs).
pub const CLAMP: f32 = 3.0;
/// The change test's fast and slow means this many deviations apart restart
/// the part at its next update (module docs).
pub const LAG: f32 = 1.74;
/// ... and at least this share of the slow mean apart.
pub const FLOOR: f32 = 0.01;
/// Updates of history before the change test runs.
pub const DETECT: u32 = 8;
/// Frames unused before a cell goes.
pub const EVICT_AGE: u32 = 128;
/// The default slots (log2) and ray budget (rays a frame).
pub const DEFAULT_SLOTS_LOG2: u32 = 18;
/// Rays a frame by default: at the load the eviction keeps (half the
/// default slots, 2^17 cells) every cell lit at least once in the time the
/// probes that read it take to come round (the finest levels' 21 frames,
/// [`probe_period`]): `2^17 * RAYS_PER_CELL / 21` = 24,966 rays, the power
/// of two above.
pub const DEFAULT_RAYS: u32 = 1 << 15;

/// Frames between two updates of a probe of GI's finest levels.
pub fn probe_period() -> u32 {
    helio_gi_api::LEVEL_PROBES.div_ceil(helio_gi_api::LEVEL_UPDATES[0] as u32)
}

/// Frames between two updates of a settled cell (module docs): the finest
/// probes' period, so a change of light no reader's frame shows reaches the
/// probes that read the cell within about one of their own cycles.
pub fn refresh_period() -> u32 {
    probe_period()
}

/// GI's graph keys of the cache's own buffers (between its passes).
const CACHE_LIFE: &str = "gi_cache_life";
const CACHE_CELLS: &str = "gi_cache_cells";
pub(crate) const CACHE_PLAN: &str = "gi_cache_plan";
pub(crate) const CACHE_LIST: &str = "gi_cache_list";
/// What GI's light pass leaves a cache ray for its cell's own direct light.
pub(crate) const CELL_PENDING: &str = "gi_cell_pending";
/// Words of `cache_plan`.
const PLAN_WORDS: u64 = 80;
/// Bytes of a `CacheLife`.
const LIFE_BYTES: u64 = 48;

/// The cache's capacity and budget (`GiSettings`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheConfig {
    /// Slots of the hash (a power of two, at least 2^12).
    pub slots: u32,
    /// Rays a frame the cache spends lighting its cells (a multiple of 64).
    pub rays: u32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            slots: 1 << DEFAULT_SLOTS_LOG2,
            rays: DEFAULT_RAYS,
        }
    }
}

impl CacheConfig {
    /// A cache of `2^slots_log2` slots (12 to 24) spending `rays` rays a
    /// frame (1024 up, rounded to 64).
    pub fn new(slots_log2: u32, rays: u32) -> Self {
        Self {
            slots: 1 << slots_log2.clamp(12, 24),
            rays: rays.clamp(1024, 1 << 20).next_multiple_of(64),
        }
    }

    /// Cells lit a frame at most.
    pub fn cells(&self) -> u32 {
        self.rays / RAYS_PER_CELL
    }

    /// The eye angle (module docs).
    pub fn eye_angle(&self) -> f32 {
        (4.0 * std::f32::consts::PI * LEVELS as f32 / (self.slots as f32 / 2.0)).sqrt()
    }

    /// Bytes the cache holds: its slots and the new cells' queue, entries
    /// and lives, the cells in use (two halves), the cells lit.
    pub fn bytes(&self) -> u64 {
        radiance_cache::slot_rows(self.slots) * RadianceCacheSlot::BYTES
            + u64::from(self.slots) * (RadianceCacheEntry::BYTES + LIFE_BYTES + 8)
            + 8
            + u64::from(self.cells()) * 4
    }
}

/// The snippets `gi_cache.wgsl` is built with.
pub(crate) const SNIPPETS: [helio_core::shader::ShaderSnippet; 6] = [
    stream::SNIPPET,
    radiance_cache::SNIPPET,
    helio_frame_api::gbuffer::SURFACE_SNIPPET,
    helio_frame_api::gbuffer::BSDF_SNIPPET,
    GI_COMMON_SNIPPET,
    GI_RAYS_SNIPPET,
];

fn module(device: &wgpu::Device) -> wgpu::ShaderModule {
    helio_core::shader::module_with(
        device,
        "GI radiance cache",
        helio_core::include_wgsl!("../shaders/gi_cache.wgsl"),
        &SNIPPETS,
    )
}

/// The cache's constants for its pipelines: its cells a frame and its
/// statistics' slots.
fn constants(
    config: &CacheConfig,
    stats: &helio_core::FrameStats,
    rays: &Stream,
) -> Vec<(&'static str, f64)> {
    let mut constants = stats.slot_constants(&[
        ("CACHE_STAT_CELLS", "radiance_cache.cells"),
        ("CACHE_STAT_NEW", "radiance_cache.new"),
        ("CACHE_STAT_EVICTED", "radiance_cache.evicted"),
        ("CACHE_STAT_LIT", "radiance_cache.lit"),
        ("CACHE_STAT_SETTLED", "radiance_cache.settled"),
        ("CACHE_STAT_RAYS", "rays.radiance_cache"),
        ("CACHE_STAT_STREAM", &format!("rays.{}", rays.name)),
    ]);
    constants.push(("CACHE_CELLS", f64::from(config.cells())));
    constants.push(("CACHE_REFRESH", f64::from(refresh_period())));
    constants
}

/// The cache across frames (kept over graph rebuilds of the same config).
struct State {
    config: CacheConfig,
    frame: wgpu::Buffer,
    slots: wgpu::Buffer,
    entries: wgpu::Buffer,
    life: wgpu::Buffer,
    plan: wgpu::Buffer,
    list: wgpu::Buffer,
    /// The cells in use: the current half and its length, then two halves.
    cells: wgpu::Buffer,
    /// The walks' and the rays' indirect dispatches.
    args: wgpu::Buffer,
}

impl State {
    fn new(device: &wgpu::Device, config: CacheConfig) -> Self {
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let slots = u64::from(config.slots);
        // Created zeroed: every slot empty.
        Self {
            config,
            frame: buffer(
                device,
                "radiance cache frame",
                RadianceCacheFrame::BYTES,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            slots: buffer(
                device,
                "radiance cache slots",
                radiance_cache::slot_rows(config.slots) * RadianceCacheSlot::BYTES,
                storage,
            ),
            entries: buffer(
                device,
                "radiance cache entries",
                slots * RadianceCacheEntry::BYTES,
                storage,
            ),
            life: buffer(device, "radiance cache lives", slots * LIFE_BYTES, storage),
            plan: buffer(device, "radiance cache plan", PLAN_WORDS * 4, storage),
            list: buffer(
                device,
                "radiance cache list",
                u64::from(config.cells()) * 4,
                storage,
            ),
            cells: buffer(device, "radiance cache cells", (2 + 2 * slots) * 4, storage),
            args: buffer(
                device,
                "radiance cache dispatches",
                8 * 4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            ),
        }
    }
}

/// The cache's eviction, its list of cells to light and their rays (module
/// docs): after the probes' rays, before the stream's tracers.
pub struct RadianceCache {
    state: State,
    rays: Stream,
    layout: wgpu::BindGroupLayout,
    /// The indirect dispatches' group (`cache_begin`, `cache_end`).
    args_layout: wgpu::BindGroupLayout,
    begin: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    plan: wgpu::ComputePipeline,
    select: wgpu::ComputePipeline,
    end: wgpu::ComputePipeline,
    generate: wgpu::ComputePipeline,
    stats: wgpu::Buffer,
    cache: helio_core::gpu_cache::GpuCache,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl RadianceCache {
    /// The cache of `config`, its rays in GI's stream `rays`.
    pub fn new(
        device: &wgpu::Device,
        config: CacheConfig,
        stats: &helio_core::FrameStats,
        rays: Stream,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI radiance cache"),
            entries: &[
                uniform(0, RadianceCacheFrame::BYTES),
                storage(1, false),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, false),
                uniform(6, FRAME_BYTES),
                storage(7, false),
                storage(8, false),
                storage(9, false),
                storage(10, false),
                storage(11, false),
            ],
        });
        let args_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI radiance cache dispatches"),
            entries: &[storage(0, false)],
        });
        let module = module(device);
        let pipeline_layout = pipeline_layout(device, "GI radiance cache", &layout);
        let args_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GI radiance cache dispatches"),
            bind_group_layouts: &[Some(&layout), Some(&args_layout)],
            immediate_size: 0,
        });
        let constants = constants(&config, stats, &rays);
        let pipeline = |entry: &str| compute(device, &module, &pipeline_layout, entry, &constants);
        let writer =
            |entry: &str| compute(device, &module, &args_pipeline_layout, entry, &constants);
        Self {
            begin: writer("cache_begin"),
            scan: pipeline("cache_scan"),
            plan: pipeline("cache_plan_main"),
            select: pipeline("cache_select"),
            end: writer("cache_end"),
            generate: pipeline("cache_rays"),
            args_layout,
            layout,
            state: State::new(device, config),
            stats: stats.buffer().clone(),
            cache: helio_core::gpu_cache::GpuCache::new(),
            reads: helio_frame_api::static_names(&[GI_FRAME, rays.header, rays.rays, rays.hits]),
            writes: helio_frame_api::static_names(&[
                FRAME,
                SLOTS,
                ENTRIES,
                CACHE_LIFE,
                CACHE_PLAN,
                CACHE_LIST,
                CACHE_CELLS,
                rays.header,
                rays.rays,
                rays.hits,
            ]),
            rays,
        }
    }

    /// The cache's frame, slots and entries (the frame's contract), for a
    /// reader outside the graph.
    pub fn buffers(&self) -> [&wgpu::Buffer; 3] {
        [&self.state.frame, &self.state.slots, &self.state.entries]
    }
}

impl RenderPass for RadianceCache {
    fn name(&self) -> &'static str {
        "RadianceCache"
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn inherit_persistent_state(&mut self, previous: &mut dyn RenderPass) -> bool {
        let Some(previous) = previous.as_any_mut().downcast_mut::<Self>() else {
            return false;
        };
        if previous.state.config != self.state.config {
            return false;
        }
        std::mem::swap(&mut self.state, &mut previous.state);
        true
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let s = &self.state;
        let eye = [0, 1, 2].map(|k| ctx.camera_data.position_near[k]);
        let frame = RadianceCacheFrame::new(
            ctx.world_origin.to_array(),
            eye,
            s.config.eye_angle(),
            s.config.slots,
            ctx.frame_num as u32,
        );
        helio_core::upload::write_buffer(ctx.queue, &s.frame, 0, bytemuck::bytes_of(&frame));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let binding = |name: &str| ctx.registry.buffer_binding(name);
        let (Some(gi), Some(header), Some(rays), Some(hits)) = (
            binding(GI_FRAME),
            binding(self.rays.header),
            binding(self.rays.rays),
            binding(self.rays.hits),
        ) else {
            return Ok(());
        };
        let s = &self.state;
        self.cache.end_frame();
        let group = self.cache.bind_group(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("GI radiance cache"),
                layout: &self.layout,
                entries: &[
                    entry(0, &s.frame),
                    entry(1, &s.slots),
                    entry(2, &s.entries),
                    entry(3, &s.life),
                    entry(4, &s.plan),
                    entry(5, &s.list),
                    entry(6, gi),
                    entry(7, rays),
                    entry(8, hits),
                    entry(9, header),
                    entry(10, &self.stats),
                    entry(11, &s.cells),
                ],
            },
        );
        let args = self.cache.bind_group(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("GI radiance cache dispatches"),
                layout: &self.args_layout,
                entries: &[entry(0, &s.args)],
            },
        );
        let mut cmds = ctx.graphics_cmds();
        cmds.clear_buffer(&s.plan, 0, None);
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI radiance cache"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_bind_group(1, Some(&args), &[]);
        // The walks over the cells in use and the new ones (their length on
        // the GPU), then the rays of the cells lit.
        pass.set_pipeline(&self.begin);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.scan);
        pass.dispatch_workgroups_indirect(&s.args, 0);
        pass.set_pipeline(&self.plan);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.select);
        pass.dispatch_workgroups_indirect(&s.args, 0);
        pass.set_pipeline(&self.end);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.generate);
        pass.dispatch_workgroups_indirect(&s.args, 16);
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        let s = &self.state;
        frame.write_buffer_binding(FRAME, &s.frame, "RadianceCache");
        frame.write_buffer_binding(SLOTS, &s.slots, "RadianceCache");
        frame.write_buffer_binding(ENTRIES, &s.entries, "RadianceCache");
        frame.write_buffer_binding(CACHE_LIFE, &s.life, "RadianceCache");
        frame.write_buffer_binding(CACHE_PLAN, &s.plan, "RadianceCache");
        frame.write_buffer_binding(CACHE_LIST, &s.list, "RadianceCache");
        frame.write_buffer_binding(CACHE_CELLS, &s.cells, "RadianceCache");
    }
}

/// The lit cells' rays folded into them (module docs), once the shadow
/// stream's tracers answered; on a frame either stream was answered short
/// the cells hold, as the probes do.
pub struct RadianceCacheFold {
    config: CacheConfig,
    /// GI's stream's answers and the shadow stream's.
    answers: [Arc<stream::Answers>; 2],
    shadow_hits: &'static str,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    stats: wgpu::Buffer,
    cache: helio_core::gpu_cache::GpuCache,
    reads: &'static [&'static str],
}

impl RadianceCacheFold {
    pub fn new(
        device: &wgpu::Device,
        config: CacheConfig,
        stats: &helio_core::FrameStats,
        rays: Stream,
        shadows: Stream,
        answers: [Arc<stream::Answers>; 2],
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI radiance cache fold"),
            entries: &[
                uniform(0, RadianceCacheFrame::BYTES),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, false),
                uniform(6, FRAME_BYTES),
                storage(10, false),
                storage(12, true),
                storage(13, true),
                storage(14, true),
                storage(15, true),
            ],
        });
        let module = module(device);
        let pipeline_layout = pipeline_layout(device, "GI radiance cache fold", &layout);
        let constants = constants(&config, stats, &rays);
        Self {
            pipeline: compute(device, &module, &pipeline_layout, "cache_fold", &constants),
            layout,
            config,
            answers,
            shadow_hits: shadows.hits,
            stats: stats.buffer().clone(),
            cache: helio_core::gpu_cache::GpuCache::new(),
            reads: helio_frame_api::static_names(&[
                FRAME,
                ENTRIES,
                CACHE_LIFE,
                CACHE_PLAN,
                CACHE_LIST,
                GI_FRAME,
                GI_PENDING,
                CELL_PENDING,
                GI_ANSWERED,
                shadows.hits,
            ]),
        }
    }
}

impl RenderPass for RadianceCacheFold {
    fn name(&self) -> &'static str {
        "RadianceCacheFold"
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        &[ENTRIES, CACHE_LIFE]
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.answers.iter().all(|a| a.complete()) {
            return Ok(());
        }
        let binding = |name: &str| ctx.registry.buffer_binding(name);
        let (
            Some(frame),
            Some(entries),
            Some(life),
            Some(plan),
            Some(list),
            Some(gi),
            Some(pending),
            Some(cells),
            Some(shadows),
            Some(answered),
        ) = (
            binding(FRAME),
            binding(ENTRIES),
            binding(CACHE_LIFE),
            binding(CACHE_PLAN),
            binding(CACHE_LIST),
            binding(GI_FRAME),
            binding(GI_PENDING),
            binding(CELL_PENDING),
            binding(self.shadow_hits),
            binding(GI_ANSWERED),
        )
        else {
            return Ok(());
        };
        self.cache.end_frame();
        let group = self.cache.bind_group(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("GI radiance cache fold"),
                layout: &self.layout,
                entries: &[
                    entry(0, frame),
                    entry(2, entries),
                    entry(3, life),
                    entry(4, plan),
                    entry(5, list),
                    entry(6, gi),
                    entry(10, &self.stats),
                    entry(12, pending),
                    entry(13, cells),
                    entry(14, shadows),
                    entry(15, answered),
                ],
            },
        );
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI radiance cache fold"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&self.pipeline);
        pass.dispatch_workgroups(self.config.cells().div_ceil(64), 1, 1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helio_frame_api::rays::radiance_cache::BUCKET;

    /// The shader's copies of the constants (`gi_cache.wgsl`, `gi_rays.wgsl`).
    #[test]
    fn shader_constants_match() {
        let source = include_str!("../shaders/gi_cache.wgsl");
        for (name, value) in [
            ("CACHE_HISTORY_MIN", format!("{HISTORY_MIN}u")),
            ("CACHE_HISTORY_MAX", format!("{HISTORY_MAX}u")),
            ("CACHE_ERROR", format!("{ERROR:.2}")),
            ("CACHE_BOUNCED", format!("{BOUNCED}u")),
            (
                "CACHE_EVICT_BIN",
                format!("{}u", EVICT_AGE.trailing_zeros() + 1),
            ),
            ("CACHE_CLAMP", format!("{CLAMP:.1}")),
            ("CACHE_LAG", format!("{LAG:.2}")),
            ("CACHE_FLOOR", format!("{FLOOR:.2}")),
            ("CACHE_FAST", format!("{FAST}u")),
            ("CACHE_DETECT", format!("{DETECT}u")),
            ("CACHE_SPREAD", format!("{SPREAD}")),
            ("CACHE_CLEAR_GROWTH", format!("{CLEAR_GROWTH}")),
        ] {
            assert!(
                source.contains(&format!("const {name}: "))
                    && source.contains(&format!(" = {value};")),
                "{name} is not {value} in gi_cache.wgsl"
            );
        }
        let rays = GI_RAYS_SNIPPET.source;
        assert!(rays.contains(&format!("const CACHE_RAYS: u32 = {RAYS_PER_CELL}u;")));
        assert!(rays.contains(&format!("const CACHE_PLAN_WORDS: u32 = {PLAN_WORDS}u;")));
    }

    #[test]
    fn the_constants_follow_from_their_derivations() {
        // The change test: 4 standard deviations of the difference of the
        // fast and the slow mean (the slow one over the longest history), in
        // mean absolute deviations (0.798 of a standard deviation): a false
        // alarm one update in 16,000, so a part fills its longest history
        // (64 updates) unrestarted 996 times in 1000.
        let (a, b) = (1.0 / FAST as f32, 1.0 / HISTORY_MAX as f32);
        let variance = a / (2.0 - a) + b / (2.0 - b) - 2.0 * a * b / (a + b - a * b);
        assert!(
            (4.0 * variance.sqrt() / 0.798 - LAG).abs() < 0.03,
            "{variance}"
        );
        assert_eq!(DETECT, 2 * FAST);
        // A ray's share of the hemisphere over twice the least history.
        let spread = (2.0 * std::f32::consts::PI / (RAYS_PER_CELL * 2 * HISTORY_MIN) as f32).sqrt();
        // Twice in as many updates as there are ways (four rays a way).
        let ways = helio_frame_api::rays::radiance_cache::WAYS;
        assert!((CLEAR_GROWTH.powi(ways as i32) - 2.0).abs() < 1e-3);
        // The least history measures the deviation the change test needs.
        assert_eq!(HISTORY_MIN, 2 * DETECT);
        assert!((spread - SPREAD).abs() < 1e-4, "{spread}");
        // Twice the longest a probe goes between updates.
        let period = helio_gi_api::LEVEL_UPDATES
            .iter()
            .map(|&u| helio_gi_api::LEVEL_PROBES.div_ceil(u as u32))
            .max()
            .unwrap();
        assert!(
            EVICT_AGE >= 2 * period && EVICT_AGE.is_power_of_two(),
            "{period}"
        );
        assert!(EVICT_AGE / 2 < 2 * period);
        // The default eye angle.
        let a = CacheConfig::default().eye_angle();
        assert!((a - 0.0404).abs() < 1e-3, "{a}");
        // The default budget lights half the default slots' cells, 4 rays
        // each, within a finest probe's period.
        let at_load = (1u32 << DEFAULT_SLOTS_LOG2) / 2 * RAYS_PER_CELL / probe_period();
        assert_eq!(DEFAULT_RAYS, at_load.next_power_of_two(), "{at_load}");
        // Default memory: about 162 bytes a slot, 40.5 MiB.
        let slots = 1u64 << 18;
        assert_eq!(
            CacheConfig::default().bytes(),
            (slots + 1 + slots / 4) * 8 + slots * 152 + 8 + 8192 * 4
        );
        assert_eq!(CacheConfig::default().slots % BUCKET, 0);
    }

    #[test]
    fn a_config_is_clamped_to_what_the_hash_and_the_stream_take() {
        let c = CacheConfig::new(4, 100);
        assert_eq!((c.slots, c.rays), (1 << 12, 1024));
        let c = CacheConfig::new(30, 5000);
        assert_eq!((c.slots, c.rays), (1 << 24, 5056));
        assert_eq!(CacheConfig::new(18, 1 << 15), CacheConfig::default());
    }
}
