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
  `probe_spacing`): 1 m by default (1, 2, 4 .. 32 m apart: half-extents of
  12 to 384 m around the eye); a scene of rooms, pillars and furniture a
  few metres apart sets 0.5 m (6 to 192 m), as the cathedral example does.
  What kinds the scene holds does not choose it: a host or project sets it. Probes sit at whole
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
   - `gi_sun_main`: the sun. With an atmosphere, its direction and its
     illuminance through the air to the eye; without one, the lights'
     primary directional light.
   - `gi_reset`: the probes that came into the clipmap.
   - `gi_rays_main`: this frame's rays into the probe stream (`gi_probe`,
     `RayClass::Nearest`, `MAX_RAYS` rows: 64 a probe, in update order), in
     the frame's space (the probes are eye-relative; the eye's frame
     position is added), each with the cone of one of its probe's 64 rays
     (`GI_RAY_SPREAD`, sqrt(4 pi / 64)), its hit cleared. The header's count
     is the CPU's (updated probes times 64).
2. **The tracers** (`stream::trace`): each ray's nearest hit (`SceneHit`:
   distance, tracer, inside) and its `SceneSurface` (shading and face
   normals on the side the ray met, base colour, transmission, emission).
3. **`GiLightPass`** (`gi_light.wgsl`, a workgroup a probe, a lane a ray;
   the shadow stream's generator). For a ray that met an outside:
   - the sun's light off the side of the surface that faces it (`base *
     sun * clouds * cos / pi`), or, for a surface with a transmission share
     whose far side faces it, that share of it through the surface (a leaf
     lit from behind), waiting on a shadow ray from that side;
   - one local light (`local_lights`: the scene's point and spot lights
     gathered once a frame, one chosen at the hit by weighted reservoir over
     their unshadowed light, over its chance), waiting on a shadow ray to
     it. This is GI's stand-in for the frame's light sampling, which the
     lights plugin will own (rendering.md, r5);
   - the rest, unshadowed: `base * (probe irradiance at the hit + share *
     probe irradiance from its far side) + emission` (the probes' own
     irradiance gives every further bounce), read before any probe of the
     frame is written.

   A miss takes the sky: the atmosphere's sky table at the probe,
   cosine-convolved L2 harmonics turned back into radiance band by band;
   without an atmosphere, the flat ambient above the horizon and 0.15 of it
   below. `clouds` is the share of the sun the media let through to the hit
   (the frame's `sun_transmittance`, the sky's clouds publish it): without
   it, ground under a cloud was lit by bounce from ground the cloud shades.
   What waits on a shadow ray, and the rest, go into `gi_pending` (RGB9E5,
   24 bytes a ray). The shadow rays start off the surface along its face
   normal by `0.002 + 4e-6 * |p|` (a few ulps of f32 at the point). They are
   packed into the shadow stream (`gi_shadow`, `RayClass::Occlusion`, up to
   two a probe ray) from row 0, a workgroup's together (its sun rays, then
   its local lights'), counted by a workgroup atomic and one global atomic
   a workgroup; a one-invocation dispatch writes the stream's dispatch from
   the count. Packed rather than two fixed rows a ray (an empty row
   `t_max = 0`): half or more of the rows would be empty (misses, insides,
   surfaces facing away from the sun, scenes without local lights), and a
   tracer's wave costs its slowest lane, so empty lanes beside a terrain
   walk cost a whole walk; packing costs a few atomics a workgroup and
   8 bytes a ray (the rows in `gi_pending`).
4. **The tracers** of the shadow stream: what gets through each, per
   channel. The terrain's shadow of the sun at a hit is its tracer's answer,
   and a local light's shadow ray sees every kind (terrain too).
5. **`GiIntegratePass`** (`gi_integrate.wgsl`, a workgroup a probe, a lane
   a ray): a ray's radiance is `sun * through + local * through + rest`,
   the model before streams (`base * (sun * through * clouds * cos / pi +
   probe irradiance) + local * through + emission`); an inside gives
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
sky's ambient) until geometry answers. The cost: a scene emptied of all
geometry keeps what its probes held.

`GiPass` publishes `gi_frame`, `gi_probes`, `gi_sun` and the probe stream;
`GiLightPass` the shadow stream; the integration writes `gi_probes`. The
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

2720 probe updates a frame (174k probe rays, up to 348k shadow rays): about
2 ms with the terrain tracer. GI's own passes are the generation, the hits'
light (probe samples at each hit, one loop over at most 256 local lights)
and the integration; the tracers' cost is theirs. CPU work a frame: the
probes that came in and the update list (it scales with the probes
updated), three small uploads; bind groups are kept while their inputs
stay. Memory: 6 x 24^3 probes x 288 bytes (24 MB); the probe stream's rows
(68 bytes a ray, 11.8 MB), the shadow stream's (48 bytes a row, 16.7 MB)
and the pending light (24 bytes a ray, 4.2 MB).
