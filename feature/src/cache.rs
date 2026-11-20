//! GI's radiance cache (`gi_cache.wgsl`, README.md "The radiance cache"):
//! the publisher of the frame's `helio_frame_api::rays::radiance_cache`, and
//! the one place a surface is lit for indirect light.
//!
//! [`RadianceCache`], at the start of GI's update after the probes' rays:
//! evicts the cells no reader marked lately, lists the cells to light this
//! frame (new ones first, then the longest unlit, [`CacheConfig::cells`] at
//! most) and writes [`RAYS_PER_CELL`] rays a listed cell into GI's stream
//! after the probes' rays, cosine distributed about the cell's normal from
//! its point. GI's light and shadow passes then light those rays' hits (a
//! light sample and a shadow ray each, the sky on a miss, the cache's own
//! indirect irradiance at the hit for the bounces after) and the cell's own
//! direct light (a light sample and a shadow ray a ray), and
//! [`RadianceCacheFold`] folds each cell's rays into it.
//!
//! The constants, and where each comes from:
//!
//! - [`RAYS_PER_CELL`] (4): a cell's update is a 2x2 stratified set of the
//!   hemisphere, the fewest strata whose mean a change test can judge; more
//!   rays a cell light fewer cells a frame for the same budget, and a cell's
//!   latency (cells in use over cells lit a frame) grows with it.
//! - [`HISTORY`] (32 updates): a part's value is the mean of its last
//!   `RAYS_PER_CELL * HISTORY` = 128 rays, a relative error of 1/sqrt(128)
//!   (9%) of one ray's spread, which the readers (a probe's 64 rays and its
//!   own history, a pixel's denoiser) average further.
//! - [`SPREAD`]: a cell's ray stands for its share of the hemisphere over the
//!   whole history, `sqrt(2 pi / 128)`, so its hit reads the cache at the
//!   detail the cell's estimate resolves.
//! - The fold keeps, beside a part's value (the mean of its last
//!   [`HISTORY`] updates), a fast mean of its luminance over [`FAST`]
//!   updates and the mean absolute deviation of its updates from it, and
//!   shortens the history in two ways (ReBLUR's anti-lag, Zhdan 2021):
//!   - **A jump**: an update [`CHANGE`] (5) deviations off the fast mean,
//!     once [`DETECT`] updates have measured the deviation, restarts the
//!     part at that update. 5 deviations are about 4 standard deviations of
//!     a Gaussian part (a mean absolute deviation is 0.8 of one): a false
//!     restart once in about 16,000 updates. A light switched off moves a
//!     part it alone lit by its whole value.
//!   - **A lag**: the fast mean [`LAG`] (1.5) deviations off the value
//!     brings the value to the fast mean and the history to [`FAST`]. The
//!     fast mean's noise is `sqrt(1 / (2 FAST - 1))` = 0.38 of an update's,
//!     the value's `sqrt(1 / (2 HISTORY - 1))` = 0.13, together 0.40
//!     standard deviations or 0.5 deviations: 1.5 deviations are 3 standard
//!     deviations of their difference. It catches what moves slower than a
//!     jump: a cell's indirect light falling a bounce at a time after a
//!     light goes off, which the value alone would follow over its whole
//!     history.
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
/// Updates a part's history holds at most (module docs).
pub const HISTORY: u32 = 32;
/// The spread of a cell's ray: `sqrt(2 pi / (RAYS_PER_CELL * HISTORY))`.
pub const SPREAD: f32 = 0.2216;
/// Updates the fast mean of a part's luminance holds (module docs).
pub const FAST: u32 = 4;
/// An update this many mean absolute deviations off the fast mean restarts
/// its part (module docs).
pub const CHANGE: f32 = 5.0;
/// The fast mean this many deviations off the value brings it there (module
/// docs).
pub const LAG: f32 = 1.5;
/// Updates of history before the change tests run.
pub const DETECT: u32 = 4;
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

/// GI's graph keys of the cache's own buffers (between its passes).
const CACHE_LIFE: &str = "gi_cache_life";
pub(crate) const CACHE_PLAN: &str = "gi_cache_plan";
pub(crate) const CACHE_LIST: &str = "gi_cache_list";
/// What GI's light pass leaves a cache ray for its cell's own direct light.
pub(crate) const CELL_PENDING: &str = "gi_cell_pending";
/// Words of `cache_plan`.
const PLAN_WORDS: u64 = 80;
/// Bytes of a `CacheLife`.
const LIFE_BYTES: u64 = 20;

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

    /// Bytes the cache holds: its slots, entries and lives, the list.
    pub fn bytes(&self) -> u64 {
        u64::from(self.slots) * (RadianceCacheSlot::BYTES + RadianceCacheEntry::BYTES + LIFE_BYTES)
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
        ("CACHE_STAT_RAYS", "rays.radiance_cache"),
        ("CACHE_STAT_STREAM", &format!("rays.{}", rays.name)),
    ]);
    constants.push(("CACHE_CELLS", f64::from(config.cells())));
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
                slots * RadianceCacheSlot::BYTES,
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
        }
    }
}

/// The cache's eviction, its list of cells to light and their rays (module
/// docs): after the probes' rays, before the stream's tracers.
pub struct RadianceCache {
    state: State,
    rays: Stream,
    layout: wgpu::BindGroupLayout,
    scan: wgpu::ComputePipeline,
    plan: wgpu::ComputePipeline,
    select: wgpu::ComputePipeline,
    generate: wgpu::ComputePipeline,
    header: wgpu::ComputePipeline,
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
            ],
        });
        let module = module(device);
        let pipeline_layout = pipeline_layout(device, "GI radiance cache", &layout);
        let constants = constants(&config, stats, &rays);
        let pipeline = |entry: &str| compute(device, &module, &pipeline_layout, entry, &constants);
        Self {
            scan: pipeline("cache_scan"),
            plan: pipeline("cache_plan_main"),
            select: pipeline("cache_select"),
            generate: pipeline("cache_rays"),
            header: pipeline("cache_header"),
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
                ],
            },
        );
        let slots = s.config.slots;
        let rays = s.config.cells() * RAYS_PER_CELL;
        let mut cmds = ctx.graphics_cmds();
        cmds.clear_buffer(&s.plan, 0, None);
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI radiance cache"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&self.scan);
        pass.dispatch_workgroups(slots.div_ceil(64), 1, 1);
        pass.set_pipeline(&self.plan);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.select);
        pass.dispatch_workgroups(slots.div_ceil(64), 1, 1);
        pass.set_pipeline(&self.generate);
        pass.dispatch_workgroups(rays.div_ceil(64), 1, 1);
        pass.set_pipeline(&self.header);
        pass.dispatch_workgroups(1, 1, 1);
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
            ("CACHE_HISTORY", format!("{HISTORY}u")),
            (
                "CACHE_EVICT_BIN",
                format!("{}u", EVICT_AGE.trailing_zeros() + 1),
            ),
            ("CACHE_CHANGE", format!("{CHANGE:.1}")),
            ("CACHE_LAG", format!("{LAG:.1}")),
            ("CACHE_FAST", format!("{FAST}u")),
            ("CACHE_DETECT", format!("{DETECT}u")),
            ("CACHE_SPREAD", format!("{SPREAD}")),
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
        // The lag test's 3 standard deviations of the fast mean against the
        // value, in mean absolute deviations (0.798 of a standard deviation).
        let noise = |n: u32| (1.0 / (2.0 * n as f32 - 1.0)).sqrt();
        let sigma = (noise(FAST).powi(2) + noise(HISTORY).powi(2)).sqrt();
        assert!((3.0 * sigma / 0.798 - LAG).abs() < 0.01, "{sigma}");
        // A ray's share of the hemisphere over a history.
        let spread = (2.0 * std::f32::consts::PI / (RAYS_PER_CELL * HISTORY) as f32).sqrt();
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
        // Default memory: 92 bytes a slot, 23 MiB.
        assert_eq!(CacheConfig::default().bytes(), (1 << 18) * 92 + 8192 * 4);
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
