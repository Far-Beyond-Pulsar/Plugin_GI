# helio-pass-gi

World-space diffuse global illumination: a camera-centred clipmap of
irradiance probes kept up to date by rays every frame, read by deferred
lighting at every pixel (meshes and voxel terrain alike). It gives bounce
light from sunlit ground into shadow, sky light that does not leak into caves
or under overhangs, and light that changes as the world is dug and built.

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
absent kind is a tracer that is not there. Its node (`helio.pass.gi`) runs
once a frame at `Slot::Geometry`, after the G-buffer phase, and holds the
whole update:

1. **`GiPass`** (the probe stream's generator):
   - `gi_reset`: the probes that came into the clipmap.
   - `gi_rays_main`: this frame's rays into the probe stream (`gi_probe`,
     `RayClass::Nearest`, `MAX_RAYS` rows: 64 a probe, in update order), in
     the frame's space (the probes are eye-relative; the eye's frame
     position is added), each with no width at the probe and the spread of
     one of its probe's 64 rays (`GI_RAY_SPREAD`, sqrt(4 pi / 64)), its hit
     cleared. A ray starts `gi_ray_offset` (a few millimetres and f32's
     rounding there) behind its probe along its own direction, and
     `gi_light` takes a hit within that of the probe as an inside, so every
     ray of a probe on a floor crosses the floor and the probe is dropped
     whichever side rounding put it. The header's count is the CPU's
     (updated probes times 64).
2. **The tracers** (`stream::trace`): each ray's nearest hit (`SceneHit`:
   distance, tracer, inside) and its `SceneSurface` (shading and face
   normals on the side the ray met, base colour, transmission, emission).
3. **`GiLightPass`** (`gi_light`, a workgroup a probe, a lane a ray; the
   light batch's generator). For a ray that met an outside:
   - the light that waits on no shadow ray: `base * (probe irradiance at
     the hit + share * probe irradiance from its far side) + emission` (the
     probes' own irradiance gives every further bounce), read before any
     probe of the frame is written;
   - a light query at the hit (`LightQuery`: its place, the probe ray's
     footprint there, its normal, lit from both sides when it lets light
     through, the probes' coordinate space), packed into GI's light batch
     (`gi_probe`, `light_batch()`) from row 0, a workgroup's together.

   A miss takes the sky: the atmosphere's radiance table at the probe
   (`sky_radiance_eval`: the sky's radiance by direction, which from the
   eye is the sky the sky pass draws); without an atmosphere, the flat
   ambient above the horizon and 0.15 of it below. What it has into `gi_pending` (RGB9E5, 16 bytes a ray).
4. **The light samplers** (`light_samples::sample`): every kind of light
   that registered a sampler offers one of its lights at each query (the
   lights' sampler: its point, spot and directional lights, the sun through
   the air and the clouds' transmittance, `helio-pass-light-sampler`), and
   the row keeps one, chosen in proportion to what each brings. GI reads no
   light kind's rows: the sun is one more sample. Why the sun is not GI's
   own any more: it took a second shadow ray at every hit whether or not a
   local light lit it; one sample a hit chooses the sun where it dominates
   (outdoors, every time but where a lamp outshines it) and a lamp where it
   does, so outdoors costs the same rays and indoors half.
5. **`GiShadowPass`** (`gi_shadow`): each hit's sample off its surface
   (`base * radiance * cos / pi` on the side the light is on; a surface
   with a transmission share whose far side faces the light lets that share
   through), waiting on a shadow ray from that side. The shadow rays start
   off the surface along its face normal by `0.002 + 4e-6 * |p|` (a few
   ulps of f32 at the point), as wide as the probe ray's footprint at the
   hit (`GI_RAY_SPREAD * t`), keeping that width (no spread), as cut-outs on
   the way are tested at it, and end short of a local light (a directional
   light's reach a planet across). They are packed into the shadow stream
   (`gi_shadow`, `RayClass::Occlusion`, one a probe ray at most) from row 0,
   a workgroup's together; a one-invocation dispatch writes the stream's
   dispatch from the count. Packed rather than a fixed row a ray (an empty
   row `t_max = 0`): misses, insides and unlit hits would be empty rows,
   and a tracer's wave costs its slowest lane.
6. **The tracers** of the shadow stream: what gets through each, per
   channel (the terrain's shadow of the sun at a hit is its tracer's
   answer; a lamp's shadow ray sees every kind).
7. **`GiIntegratePass`** (`gi_integrate.wgsl`, a workgroup a probe, a lane
   a ray): a ray's radiance is `light * through + rest`; an inside gives
   nothing. Once a probe holds light, a ray's radiance is capped at 16 times
   the probe's mean irradiance / pi plus 2: a ray meeting a surface
   centimetres from a point light (its fitting) would otherwise dominate its
   probe for many updates (fireflies). Projected on L1 harmonics and
   blended into the probe; its distances into the moments.

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

`GiPass` publishes `gi_frame`, `gi_probes` and the probe stream;
`GiLightPass` the light batch, `GiShadowPass` the shadow stream; the integration writes `gi_probes`. The
frame statistics count each stream's rays (`rays.gi_probe` on the CPU,
`rays.gi_shadow` on the GPU).

`GiResolvePass` runs after the integration and before deferred lighting. For
each pixel it writes `gi_irradiance` (`gi_sample.wgsl`): irradiance / pi
and how much the probes hold it. It reads the finest level whose grid holds
the point. The 8 probes around the point are weighted by:

- trilinear position;
- facing;
- a Chebyshev test of the point's distance against each probe's moments
  toward it.

Near its grid's edge, the level blends into the next one.

## Costs (targets, 1080p, RTX 3060 class)

2720 probe updates a frame (174k probe rays and light queries, up to 174k shadow rays): about
2 ms with the terrain tracer. GI's own passes are the generation, the hits'
light (probe samples at each hit, a light query) and its samples' shadow
rays, and the integration; the samplers' cost is theirs; the tracers' cost is theirs. CPU work a frame: the
probes that came in and the update list (it scales with the probes
updated), three small uploads; bind groups are kept while their inputs
stay. Memory: 6 x 24^3 probes x 288 bytes (24 MB); the probe stream's rows
(80 bytes a ray, 13.9 MB), the light batch's (64 bytes a query, 11.1 MB),
the shadow stream's (48 bytes a row, 8.4 MB) and the pending light (16
bytes a ray, 2.8 MB).
