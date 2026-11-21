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
   lane a cache ray; the cache's light batch's generator). For a ray that met
   an outside, its hit's light from the radiance cache at the hit (on the
   side the ray met, at the ray's footprint there, in its space), marking
   the hit's cell in use:
   - a probe's ray: `base * cache irradiance / pi + share * the far side's
     cache irradiance / pi + emission`, the cache's direct and indirect parts
     both. Where the cache has no value at any level yet (a cell new this
     frame or the last), the probes' own irradiance at the hit stands in (a
     bounce as it was before the cache), and the ray is marked unlit;
   - a cache ray: the same with the cache's indirect part alone, and a
     light query at the hit (`LightQuery`: its place, the ray's footprint
     there, its normal, lit from both sides when it lets light through, its
     space) and one at its cell's point, packed into the cache's light batch
     (`radiance_cache`, `light_batch()`) from row 0, a workgroup's together.

   A miss takes the sky (`sky_radiance_at`, one call for both kinds): the
   atmosphere's radiance table at the ray's origin (`sky_radiance_eval`:
   the sky's radiance by direction, which from the eye is the sky the sky
   pass draws); without an atmosphere, the flat ambient above the horizon
   and 0.15 of it below. What it has
   into `gi_pending` (RGB9E5, 16 bytes a ray), the cell's query's row into
   `gi_cell_pending`. GI lights no probe hit itself: lighting a surface for
   indirect purposes is the cache's alone.
5. **The light samplers** (`light_samples::sample`): every kind of light
   that registered a sampler offers one of its lights at each query (the
   lights' sampler: its point, spot and directional lights, the sun through
   the air and the clouds' transmittance, `helio-pass-light-sampler`), and
   the row keeps one, chosen in proportion to what each brings. GI reads no
   light kind's rows: the sun is one more sample. The sampler's world-space
   light grid is the sampler's own; nothing in the cache depends on it.
6. **`GiShadowPass`** (`gi_shadow`, a lane a cache ray): its hit's sample
   off the hit's surface (`base * radiance * cos / pi` on the side the light
   is on; a surface with a transmission share whose far side faces the
   light lets that share through) and its cell's as irradiance at the cell
   (`radiance * cos`), each waiting on a shadow ray from its side. Shadow
   rays start off the surface by `0.002 + 4e-6 * |p|` (a few ulps of f32 at
   the point), a hit's as wide as the ray's footprint there and a cell's as
   wide as the cell, with no spread (cut-outs on the way are tested at it),
   and end short of a local light (a directional light's reach a planet
   across). They are packed into the shadow stream (`radiance_cache_shadow`,
   `RayClass::Occlusion`, two a cache ray at most) from row 0, a
   workgroup's together; a one-invocation dispatch writes the stream's
   dispatch from the count.
7. **The tracers** of the shadow stream: what gets through each, per
   channel.
8. **`GiIntegratePass`** (`gi_integrate.wgsl`, a workgroup a probe, a lane
   a ray): a ray's radiance is its `rest`; an inside gives nothing. A probe
   more than 1/32 of whose hits the cache could not light yet (so its
   irradiance is off by at most that share) holds this update (a scene loading, the eye somewhere new), so it is not taught
   a world lit by its own previous state alone. Once a probe holds light, a
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
in use, `.new`, `.evicted`, `.lit` a frame).

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
irradiance arriving at surface cells, direct and indirect apart, read and
marked by every ray that needs a lit hit (GI's probes and its own rays here;
a pixel's gather, reflections). GI publishes it (`cache.rs`,
`gi_cache.wgsl`):

- **Keys** (the contract's): the world cell of the hit (64-bit counts an
  axis), its level (cells 2^-6 to 2^10 m, the coarser of the ray's
  footprint and the eye angle at its distance), the dominant side of the
  hit's face turned toward the ray, the ray's coordinate space. Hashed into
  buckets of 16 slots; a slot is its key's checksum and the frame it was
  last marked. A cell first marked takes the marking hit's point (15 bits
  an axis in the cell) and normal; one mark in 8 of a cell in use moves its
  point to the marking hit, so a cell's light is gathered over the places
  readers meet it. A mark also marks the cell three levels coarser: a new
  fine cell's read answers from it.
- **Each frame** (`RadianceCache`, five dispatches: two over the slots, two
  of one invocation, one over the rays): the cells' ages into histograms of
  powers of two of frames; evicted, the cells no reader marked for 128
  frames and, past half the slots, the least recently used (never those
  marked this frame or the last); listed, `rays / 4` cells at most, the
  never lit first and then the longest unlit; 4 rays a listed cell,
  cosine-distributed about its normal in a 2x2 stratified set from its
  point, from no width spreading by its share of the cell's history
  (`sqrt(2 pi / 128)`), so their hits read cells at the detail the cell's
  estimate resolves (and at least the eye angle's), not as coarse as the
  cell itself, which would mix a surface's parallel layers in them.
- **Their light** (`GiLightPass`, `GiShadowPass`): a hit's light sample and
  shadow ray, the cache's indirect irradiance at the hit for the bounces
  after, the sky on a miss; the cell's own light sample and shadow ray a ray.
- **The fold** (`RadianceCacheFold`, a lane a listed cell): the direct
  part's mean of its rays' light samples, the indirect part's pi times
  their mean radiance (cosine-distributed rays), each folded into its value,
  the plain mean of its updates since it last restarted (32 at most). A
  change test (a fast mean over 4 updates and a slow one of each update's
  luminance clamped from above at the fast mean plus 3 deviations, 1.2
  deviations apart: 3 standard deviations of their difference) restarts a
  part at its next update, whose sample it has not seen: weights chosen by
  the samples themselves bias the mean (a first version, restarting at the
  bright update that tripped it, doubled the cathedral's indirect light). On
  a frame either stream was answered short, or nothing answered, the cells
  hold.
- **Settings** (`GiSettings`, host group `gi`): `radiance_cache_slots`
  (2^12 to 2^24, default 2^18: 104 bytes a slot, 26 MiB) and
  `radiance_cache_rays` (default 32,768 a frame: at half the default slots'
  cells, each lit within a finest probe's period of 21 frames). The eye angle follows
  from the slots: `sqrt(4 pi * 17 / (slots / 2))`, 0.040 rad by default.
  Where each constant comes from: `cache.rs`'s module docs.

## Costs

2720 probe updates a frame (174k probe rays) and the cache's budget (32,768
rays, 65,536 light queries and shadow rays at most). GI's own passes are the
generation, the cache's selection and rays, the hits' light (cache reads and
marks, the cache rays' light queries), the cache rays' shadow rays, the
integration and the fold; the samplers' cost is theirs, the tracers' theirs.
What each scales with: the probes' work with the probes updated; the
cache's selection with its slots (two passes over them a frame: 2^18 slots,
2 MiB of slots and 5 MiB of lives read); its rays, queries, shadow rays and
fold with its ray budget; a read with the levels it walks (one bucket of 16
slots a level, usually one level). CPU work a frame: the probes that came in
and the update list, four small uploads; bind groups are kept while their
inputs stay. Memory: 6 x 24^3 probes x 288 bytes (24 MB); the cache's
slots, entries and lives (104 bytes a slot, 26 MiB at 2^18); GI's stream's
rows (80 bytes a ray, 16.5 MB), the light batch's (64 bytes a query, 4.2
MB), the shadow stream's (48 bytes a row, 3.1 MB) and the pending light (16
bytes a ray and 8 a cache ray, 3.6 MB). Measured: `docs/architecture/
rendering.md` (Helio), section 5, "The radiance cache".
