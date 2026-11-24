# helio-pass-gi

World-space diffuse global illumination: a camera-centred clipmap of
irradiance probes kept up to date by rays every frame, read by deferred
lighting at every pixel (meshes and voxel terrain alike), and the frame's
radiance cache, where a surface is lit for indirect light once for every ray
that meets it. It gives bounce light from sunlit ground into shadow, sky
light that does not leak into caves or under overhangs, and light that
changes as the world is dug and built.

## Probes

- **Clipmap.** `LEVELS` (6) cubic grids of `SIDE`^3 (24^3) probes, each
  twice as far apart as the one inside it. The finest spacing is the
  pass's (`GiConfig::finest_spacing`, the `gi` settings group's
  `probe_spacing`): 0.5 m by default, for rooms, pillars and furniture a
  few metres apart (0.5, 1, 2 .. 16 m apart: half-extents of 6 to 192 m
  around the eye); a host over open ground sets 1 m (12 to 384 m), as the
  voxel examples do. What kinds the scene holds does not choose it: a host
  or project sets it. Probes sit at whole
  multiples of their spacing in world coordinates, and are stored
  toroidally: storage coordinates are world coordinates modulo `SIDE`. A
  probe keeps its record while it stays in the clipmap, and a grid that
  scrolls resets only the slabs that came in. The clipmap is a function of
  the eye's world position and the finest spacing alone
  (`Clipmap::around`), published each frame as `gi_clipmap`, so a tracer
  keeping state per probe follows it.
- **A probe** (`gi_probes`, 72 words) holds:
  - its irradiance / pi as L1 spherical harmonics (RGB, half floats);
  - its state (active, updates taken);
  - its visibility as an 8x8 octahedral map of ray distance moments (mean,
    mean^2; half floats), as in DDGI.

  A probe whose rays mostly meet insides (a quarter or more: the tracer's
  `SceneHit` says the ray met a surface from its inside) is inside the
  ground, a wall or a solid: inactive, so it takes no weight. A probe on a
  surface (a hit within millimetres of it: probes on a floor at a whole
  multiple of the spacing) counts as inside, so all of them are dropped
  alike rather than some, as f32 rounding would pick: its rays start that
  far behind it.
  Probes are not moved out of geometry (DDGI's relocation): a probe
  inside is dropped, and the visibility test keeps its neighbours from
  leaking through.
- **Updates.** Each frame updates `LEVEL_UPDATES` probes per level (680
  on the two finest, 340 on the coarser, whose far surroundings change
  slowly and whose rays walk farthest): those that came in first, oldest
  first, then the others in turn. Each
  update traces 64 rays along a spherical Fibonacci set turned by a random
  rotation each frame. Results blend into the probe with hysteresis (1/n
  for its first updates, then 3% for irradiance and 5% for moments).

## The frame

GI is a consumer of the frame's ray streams (`helio_frame_api::rays::stream`)
and knows no kind: whatever kinds the frame has answer its rays with their
own tracers (a voxel terrain's walk, the meshes' TLAS queries), and an
absent kind is a tracer that is not there. It publishes the frame's radiance
cache (`helio_frame_api::rays::radiance_cache`, below). Its node
(`helio.pass.gi`) runs once a frame at `Slot::Geometry`, after the G-buffer
phase, and holds the whole update:

1. **`GiPass`** (GI's stream's generator):
   - `gi_reset`: the probes that came into the clipmap.
   - `gi_rays_main`: this frame's probe rays into GI's stream (`gi`,
     `RayClass::Nearest`, `MAX_RAYS` probe rows, 64 a probe in update order,
     then the cache's), in the frame's space (the probes are eye-relative;
     the eye's frame position is added), each with no width at the probe and
     the spread of one of its probe's 64 rays (`GI_RAY_SPREAD`, sqrt(4 pi /
     64)), its hit cleared. A ray starts `gi_ray_offset` (a few millimetres
     and f32's rounding there) behind its probe along its own direction, and
     `gi_light` takes a hit within that of the probe as an inside, so every
     ray of a probe on a floor crosses the floor and the probe is dropped
     whichever side rounding put it. The CPU writes the probes' count into
     the header; the cache adds its rays' on the GPU.
2. **`RadianceCache`** (below): evicts, lists this frame's cells to light,
   writes their rays after the probes' and the stream's count.
3. **The tracers** (`stream::trace`), once for both kinds of ray: each ray's
   nearest hit (`SceneHit`: distance, tracer, inside) and its
   `SceneSurface` (shading and face normals, base colour, transmission,
   emission). One stream, so a tracer's fixed cost a stream (a dispatch a
   volume on the terrain, a binned material pass a program on meshes) is
   paid once; not in runs, since a probe's rays share no origin.
4. **`GiLightPass`** (`gi_light`; a workgroup a probe, a lane a ray, then a
   lane a cache ray; GI's light batch's generator). For a ray that met an
   outside, its hit's light as `gi_hit.wgsl` lights a hit (the one copy, the
   gather's too): the radiance cache at the hit (on the side the ray met, in
   its space, any reach: a probe is in the air, and a cache ray's hit is its
   cell's bounced part), marking the hit's cell in use (the side the ray met
   only: a far side lit through is read, never marked, since behind a voxel
   of leaves it is the voxel's inside, where a cell seated there would start
   every ray in a solid):
   - a probe's ray: `base * cache irradiance / pi + emission`, the cache's
     direct and indirect parts both. A side the cache has no light for is
     lit on the spot: a light query of its own for its direct light, and for
     its indirect light, by why the read found nothing (the contract's
     `miss`): the cell's own where its rays have not explored the way to the
     hit yet; the probes' irradiance (DDGI's recursion) where the cell is
     new or not claimed yet; none where every cell that could answer holds
     its light past an occluder (the probes see a thin wall's, a sheet's, a
     contact's point from both sides, so they cannot stand in there). A two-sided surface's far side:
     the cache's indirect part where it has one (nothing of the probes',
     which at a point on a thin surface hold the near side's light too), and
     its direct light always from the probe's own light sample through the
     frame BSDF's thin lobe, its shadow ray from the lit side (a far side's
     cell exists only where rays from that side met it);
   - a cache ray: its hit's emission into `gi_pending`, and apart
     (`gi_cell_pending`) what it reflects of the cache's indirect light there
     (the bounced part's update; none from a hit the cache has no light for),
     and a light query at the hit (`LightQuery`: its place, the ray's
     footprint there, its normal, lit from both sides when it lets light
     through, its space) and one at its cell's point.

   The queries are packed into GI's light batch (`radiance_cache`,
   `light_batch()`: a row a probe ray, two a cache ray) from row 0, a
   workgroup's together.

   A miss takes the sky (`gi_sky_radiance`, one call for every reader): the
   atmosphere's radiance table at the ray's origin (`sky_radiance_eval`:
   the sky's radiance by direction, which from the eye is the sky the sky
   pass draws); without an atmosphere, the flat ambient above the horizon
   and 0.15 of it below. What it has
   into `gi_pending` (RGB9E5, 16 bytes a ray), the cell's query's row into
   `gi_cell_pending`.
5. **The light samplers** (`light_samples::sample`): every kind of light
   that registered a sampler offers one of its lights at each query (the
   lights' sampler: its point, spot and directional lights, the sun through
   the air and the clouds' transmittance, `helio-pass-light-sampler`), and
   the row keeps one, chosen in proportion to what each brings. GI reads no
   light kind's rows: the sun is one more sample. The sampler's world-space
   light grid is the sampler's own; nothing in the cache depends on it.
6. **`GiShadowPass`** (`gi_shadow`, a workgroup a probe, then a lane a cache
   ray): a hit's sample off its surface on the sides that take it
   (`gi_hit_direct`: `base * radiance * cos / pi` on the side its ray met;
   from behind, the frame BSDF's thin lobe, `bsdf_eval`) and a cell's as
   irradiance at the cell (`radiance * cos`), each waiting on a shadow ray
   from its side. Shadow
   rays start off the surface by `0.002 + 4e-6 * |p|` (a few ulps of f32 at
   the point), a hit's as wide as the ray's footprint there and a cell's as
   wide as the cell, with no spread (cut-outs on the way are tested at it),
   and end short of a local light (a directional light's reach a planet
   across). They are packed into the shadow stream (`radiance_cache_shadow`,
   `RayClass::Occlusion`, two a cache ray and one a probe ray at most) from
   row 0, a
   workgroup's together; a one-invocation dispatch writes the stream's
   dispatch from the count.
7. **The tracers** of the shadow stream: what gets through each, per
   channel.
8. **`GiIntegratePass`** (`gi_integrate.wgsl`, a workgroup a probe, a lane
   a ray; after both streams' tracers, holding on a frame either was
   short): a ray's radiance is its `rest` plus, for a hit lit on the spot,
   its light sample's light through its shadow ray; an inside gives
   nothing. Every ray is projected over the whole sphere (no ray left out,
   no rescale). The one hold is readiness: a probe never updated more than
   1/32 of whose hits met cells still new holds and goes on the held list:
   read back without waiting, it puts those probes first in the schedule a
   few frames later, once the cache (new cells first) has lit their cells,
   so its first update is not taught a world lit by the probes' empty
   state. Once a probe holds light, a
   ray's radiance is capped at 16 times the probe's mean irradiance / pi
   plus 2: a ray meeting a surface centimetres from a point light (its
   fitting) would otherwise dominate its probe for many updates
   (fireflies). Projected on L1 harmonics and blended into the probe; its
   distances into the moments.
9. **`RadianceCacheFold`** (below): the cache rays' light into their cells.

The probes take this frame's rays in this frame: every tracer runs where
the node calls `trace`, so nothing waits for a later pass (before streams
the meshes traced later in the frame and the hits were integrated in the
next one). The node sits after the G-buffer phase because the frame's
geometry is this frame's there: a terrain updates and draws its cut for
this eye in that phase, and a frame's uploads land at the start of its
submission, so a tracer run before the drawer would see this frame's
uploads over last frame's GPU-built state. The probes' rays see the cut the
camera drew.

**Readiness.** GI builds where `stream::traced(env) > 0` (a kind registered
a tracer this device runs); otherwise neither the update nor the resolve
builds, and deferred lighting keeps the sky's ambient. On a frame where
tracers ran but no probe ray met anything (a scene loading, a TLAS still
building), the probes hold: `gi_light` marks the frame answered when any ray
met anything, and `gi_integrate` writes nothing otherwise. Taking the sky
there would seed a room's probes with it (alpha 1 for a probe's first
update, then a slow blend), so the room would glow for seconds; holding
leaves them as they were (not yet updated: the resolve's weight 0, the
sky's ambient) until geometry answers.

A tracer that knows geometry it cannot trace yet (the voxel terrain while
its ray kernels compile, or before its first draw) answers its stream short
(`stream::Answers`), and `GiIntegratePass` holds the probes that frame, though
other tracers answered: rays that miss the late kind's geometry would teach
the probes a world without it, and their history keeps what it is taught for
many updates (a slab over terrain whose meshes answered first: the open
ground's probes held 0.18 of light from below for hundreds of frames, where
their rays found 0.26).

**Known limit.** A scene emptied of all geometry keeps what its probes held
(every ray misses, so they hold): a tracer that cannot tell its geometry is
still loading (the meshes before their first TLAS) is covered by the
answered flag alone.

**Occlusion distance.** GI offers the frame's
`lighting::indirect_occlusion`: twice its finest spacing, the distance at
which the probes resolve occlusion, wherever both its nodes build. A kind
that lights its own surfaces (the voxel terrain's sky rays) adds the
occlusion nearer than that and leaves the rest to the probes.

`GiPass` publishes `gi_frame`, `gi_probes` and GI's stream; `RadianceCache`
the frame's radiance cache; `GiLightPass` the light batch, `GiShadowPass` the
shadow stream; the integration writes `gi_probes`, the fold the cache's
entries. The frame statistics count each stream's rays (`rays.gi`: the
probes' on the CPU, the cache's on the GPU; `rays.radiance_cache_shadow` on
the GPU) and the cache's own (`rays.radiance_cache`, `radiance_cache.cells`
in use, `.new`, `.evicted`, `.lit` a frame, `.settled`).

`GiResolvePass` runs after the integration and before deferred lighting. For
each pixel it writes `gi_irradiance` (`gi_sample.wgsl`): irradiance / pi
and how much the probes hold it. It reads the finest level whose grid holds
the point. The 8 probes around the point are weighted by:

- trilinear position;
- facing;
- a Chebyshev test of the point's distance against each probe's moments
  toward it.

Near its grid's edge, the level blends into the next one. Its debug views
show each pixel's nearest probe (36) and the radiance cache's cell a
pixel-wide read finds at each pixel's surface (39, `RADIANCE_CACHE_VIEW`:
its irradiance / pi, dark purple where none holds a value, cell edges
darker); deferred lighting shows both as it shows `gi_irradiance`.

## The radiance cache

The frame's contract (`helio_frame_api::rays::radiance_cache`, its WGSL
`//!use frame_radiance_cache`) is the frame's: a world-space hash of
irradiance arriving at surface cells, read and marked by every ray that
needs a lit hit (GI's probes and its own rays here; a pixel's gather,
reflections). GI publishes it (`cache.rs`, `gi_cache.wgsl`):

- **Keys** (the contract's): the world cell of the hit (64-bit counts an
  axis), its level (cells 2^-6 to 2^10 m: one a surface point, the eye
  angle's at its distance, whoever reads it), the dominant side of the
  hit's face turned toward the ray, the ray's coordinate space. Hashed into
  buckets of 16 slots (a key in its bucket or, where that was full, the
  next); a slot is its key's checksum and the frame it was last marked. A cell first marked takes the marking hit's point and normal
  and a place in the new cells' queue (after the slots: a quarter of them
  a frame); the point stays while the cell's rays find it on a free
  surface (its clear distances are that point's). A cell all of whose rays
  of an update started in a solid (the surface it was taken on is gone:
  terrain streamed in finer, an edit) is unseated: no value, its life
  cleared, not lit until the next mark seats it at that mark's hit.
- **Where a cell's light holds** (`radiance_cache_holds`): each cell keeps
  how far from its point along its surface its rays found nothing in the
  way, in 8 directions (from the rays that went low, below 45 degrees over
  the surface: a wall standing there bounds a way, at the line where the
  plane of the face met crosses the cell's surface; a ceiling far above
  does not; a way no ray explored yet holds a 32nd of the cell; a way an
  update's ray finds clear grows by 2^(1/2), one no ray explored keeps its
  bound). A reader off the cell's surface (the frame's surface stops,
  `frame_surface_stops`: another surface facing the same way within the
  cell, a curve turned away), or further along it than that, or than its
  reader's reach (a pixel's ray its length: the surface it left is in the
  hit's way there), reads the finer cell whose light holds there (at most
  3 levels finer), and a mark claims those finer cells: a cell wider than a
  thin wall, a slab or the gap to a parallel wall beside it lights its own
  side only. A read that finds nothing says why (`miss`).
- **The cells in use are a list** on the GPU (two halves, one current):
  what the last frame kept, then the queue the marks filled. Every pass
  walks it with indirect dispatches, so a frame's work follows the cells in
  use and those lit, never the slots.
- **Each frame** (`RadianceCache`: `cache_begin`, `cache_scan`,
  `cache_plan`, `cache_select`, `cache_end`, `cache_rays`): the cells' ages
  and needs into histograms; evicted, the cells no reader marked for 128
  frames and, past half the slots, the least recently used but never one
  marked within 64 frames (past the probes' longest period); lit by need,
  `rays / 4` cells at most: never lit, a part restarting, the refresh (each
  cell once in 21 frames, the finest probes' period), then those still
  converging, the least of their history first. A settled static scene
  spends the refresh alone. 4 rays a lit cell, cosine-distributed about its
  normal in a 2x2 set of jittered strata, from no width spreading by
  `sqrt(2 pi / 128)`.
- **Their light** (`GiLightPass`, `GiShadowPass`): a hit's emission (the
  sky's radiance table on a miss) and light sample and shadow ray; apart,
  what the hit reflects of the cache's own light (none from a hit the cache
  has no light for yet: the probes' irradiance there would carry light into
  the cache across what its cells keep apart); the cell's own light sample
  and shadow ray a ray.
- **The fold** (`RadianceCacheFold`, a lane a lit cell): three parts. The
  direct (the cell's light samples) and indirect (what the hits send as
  the rays meet them) are stationary: the plain mean of their updates
  since they last restarted, over as many as bring the relative standard
  error to 3%, 16 to 64. The bounced (what the hits reflect of the
  cache's light: the later bounces) follows the cells it reads: 16
  updates, so bounces settle in a few dozen updates. A change test (fast
  mean of 4 updates against the history's, of clamped luminance, 4
  standard deviations and 1% apart; the bounced part has its own) restarts a part at its next update. A
  ray that started inside a solid (the cell's point behind a corner's other
  surface) is left out of the update.
- **Statistics**: `radiance_cache.cells` (in use), `.new`, `.evicted`,
  `.lit`, `.settled`; and every reader's hits by how the cache answered the
  side its ray met, `gi.hits.<reader>.<state>` (readers `probes`, `cache`,
  `pixels`; states `cache`, `absent`, `new`, `unexplored`, `occluded`,
  `reach`, and `far_own`: a far side lit by the reader's own light sample).
- **Settings** (`GiSettings`, host group `gi`): `radiance_cache_slots`
  (2^12 to 2^24, default 2^18: about 162 bytes a slot, 40.5 MiB) and
  `radiance_cache_rays` (the most a frame, default 32,768). Where each
  constant comes from: `cache.rs`'s module docs.

## The pixels' gather

With the `quality` setting at `pixels_16`, `pixels_4` or `pixels_1` (the
`gi` host group; default `pixels_16`), the resolve node builds the pixels'
gather before the resolve, in the primary view (captures and other views
resolve the probes). Frame contracts only (`gather.rs`, `gi_gather.wgsl`):

1. **`GiGatherPass`** (`gather_near`, a pixel): the near field. One ray a
   block misses occlusion nearer than the block's footprint; a horizon
   search over the depth buffer, every pixel out to the block's side along
   4 slices (both sides), counts occluders within that footprint and no
   farther (the share of the cosine-weighted hemisphere left open, `V`). A
   surface whose occlusion is its own (`gbuffer_own_occlusion`: a traced
   terrain, a meadow) brings its `V` as its material occlusion, which the
   lighting multiplies in itself. Each pixel's factor is its `V` over the
   block's mean `V`: the near field moves light between a block's pixels
   and never changes their mean (no division by the traced pixel's `V`,
   which counted again the light a ray brings back from a near occluder; no
   floor). (`gather_rays`, a traced texel): its traced pixel
   (`denoise_traced_pixel`, the denoiser's rotation) sends one ray,
   cosine-distributed about the G-buffer normal (Roberts' R2 over frames,
   shifted by a hash of the texel), into the stream `gi_pixels`; one texel
   in 4 also re-traces its previous reservoir's sample.
2. The tracers answer the stream (`stream::trace`): terrain and meshes alike,
   with or without ray queries (whichever tracers the device runs).
3. **`GiGatherHitsPass`** (`gather_hits`): each hit's light as the probes'
   hits take it (`gi_hit`, the cell claimed), with the ray's length as its
   reach: a cell whose point lies farther from the hit than that averages
   light the hit does not see (the floor past a contact, the lit side of a
   thin wall), and the cache answers from a finer cell or not at all. A
   side the cache has no light for is lit on the spot (the probes'
   irradiance and a light query of its own into the light batch
   `gi_pixels`), as is a two-sided hit's far side's direct light. The
   frame's light samplers answer (`light_samples::sample`).
4. **`GiGatherShadowPass`** (`gather_shadow`): each sample's light at its
   hit (`gi_hit_direct`), waiting on a shadow ray from the light's side, the
   rays compacted into the stream `gi_pixels_shadow`. The tracers answer.
5. **`GiGatherShadePass`**: `gather_validate` (a re-traced sample's cache
   light now against the luminance kept with it: a change beyond the
   cache's own noise, 1/sqrt(32), lowers the confidence written for the
   denoiser and the reservoir's M; a ray that now meets something else
   drops the reservoir), `gather_temporal` (this frame's sample combined by
   generalised RIS with the reservoir where the pixel was, from the frame's
   history's taps, M capped at 8), `gather_spatial` (combined with 4
   neighbours' within 3 texels, balance-heuristic MIS over their receivers;
   not kept; a neighbour or a previous receiver is taken only where the
   frame's surface stops pass it, `frame_surface_stops`: the denoiser's
   normal and plane stops and its walk over the depth between the two) and, where the chosen sample was taken by another receiver (the
   previous frame's, a neighbour's), a visibility ray to it into the stream
   `gi_pixels_visibility`. The tracers answer. Reservoirs are 32 bytes:
   the receiver as a distance and direction from the eye, normals in 8:8
   octahedral, M and the kept cache luminance as half floats.
6. **`GiGatherSignalPass`** (`gather_signal`): the chosen sample's estimate
   times what its visibility ray lets through (an occluded reuse gives
   none), the probes' irradiance where the texel has no sample; written
   as the denoiser's irradiance signal (`denoise_irradiance_add`), the hit
   distance no nearer than a block's footprint.
7. The frame's denoiser (`denoise::filter`).
8. **`GiResolvePass`** (`gi_resolve_pixels`): the filtered irradiance at each
   pixel's normal times its near-field factor, into `INDIRECT_DIFFUSE` in
   its encoding; the probes' where the gather has no light yet or a remap
   put the pixel in another space. View 37 (`GI Near Field`) shows the
   factor.

GI offers `indirect_occlusion` as its rays' start off the surface (2 mm): a
kind's own short-range term (the terrain's sky rays) keeps its minimum
reach, and never darkens twice (the near field keeps each block's mean).
Constants' derivations: `gather.rs`'s docs and Helio's `rendering.md`,
section 5, "Indirect diffuse at a pixel" (with the measured cost and error
of each setting).

## Costs

2720 probe updates a frame (174k probe rays) and the cache's light by need
(32,768 rays a frame at most; light queries and shadow rays at most two a
cache ray and one a probe ray, 239,616).
GI's own passes are the generation, the cache's walks and rays, the hits'
light (cache reads and marks, the light queries), the shadow rays, the
integration and the fold; the samplers' cost is
theirs, the tracers' theirs. What each scales with: the probes' work with
the probes updated; the cache's walks with the cells in use; its rays,
queries, shadow rays and fold with the cells lit (the cap a setting), the
probes' own light queries with their hits the cache has no light for; a
read one level (one bucket of 16 slots, the next where that is full), more
next to an occluder that splits a cell (at most 3). CPU work a
frame: the probes that came in and the update list, four small uploads,
the held probes' list read back without waiting; bind groups are kept while
their inputs stay. Memory: 6 x 24^3 probes x 288 bytes (24 MB); the
cache's slots and queue, entries, lives and lists (about 162 bytes a slot,
40.5 MiB at 2^18); GI's stream's rows (80 bytes a ray, 16.5 MB), the light
batch's (64 bytes a query, 4.2 MB), the shadow stream's (48 bytes a row,
3.1 MB) and the pending light (16 bytes a ray, and 16 more a cache ray,
3.3 MB). Measured: `docs/architecture/rendering.md` (Helio), section 5,
"The radiance cache".
