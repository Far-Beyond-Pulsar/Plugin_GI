// How GI lights a ray's hit and a ray's miss (`//!use gi_hit`, with
// `frame_sky`, `sky_radiance`, `frame_ray_stream`, `frame_ray_surface`,
// `frame_surface`, `frame_bsdf`, `frame_light_samples`, `frame_radiance_cache`,
// `gi_common`, `gi_sample` and `gi_rays`): the one place GI's readers of a
// stream (the probes' and the cache's rays in `gi_light`, the pixels' rays
// in `gi_gather`) turn a hit into the light it sends back along its ray. A
// shader that opts in binds `frame_gi`, `gi_probes`, `gi_trace`,
// `sky_frame`, `sky_radiance`, `sky_radiance_sampler` and the radiance cache
// under their names.

const PI: f32 = 3.14159265;

// The sky's radiance along unit direction `d` from eye-relative point
// `from_camera` (km): the atmosphere's radiance table there
// (`sky_radiance_eval`, from the eye the sky the sky pass draws); without an
// atmosphere, the flat sky and ground.
fn gi_sky_radiance(from_camera: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    if sky_frame.planet.w < 0.5 { return select(frame_gi.ground.rgb, frame_gi.sky.rgb, d.y >= 0.0); }
    return sky_radiance_eval(from_camera, d);
}

// A side of a hit as the cache holds it (`gi_hit_cell`): its irradiance / pi
// from the lights (`direct`) and from everything else (`indirect`); `found`
// where the cache has light for the hit. Elsewhere the hit is lit on the
// spot (its direct light a light sample of the reader's own), and `indirect`
// is, by why the cache had none (`miss`, RADIANCE_CACHE_MISS_*):
// - the cell's own where its rays have not explored the way to the hit yet
//   (it holds a value; whether something along its surface stands between
//   its point and the hit is known within a few of its updates);
// - the probes' irradiance at the hit where the cell is new or not claimed
//   yet (the probes take in what their rays meet, never the lights' own
//   light: DDGI's recursion), `waiting` where it is held without a value (a
//   scene starting, the eye somewhere new);
// - none where every cell that could answer holds its light past an
//   occluder (`radiance_cache_miss_stays`): the probes
//   cannot tell an occluder's two sides apart (they see the point from both
//   sides of a thin wall, a sheet, a contact), and that state lasts as long
//   as the cells do. What is lost is the indirect light of a hit beside an
//   occluder thinner than an eighth of its surface's cell; the direct light
//   stays exact.
struct GiHitCell {
    direct: vec3<f32>,
    indirect: vec3<f32>,
    found: bool,
    waiting: bool,
    miss: u32,
    // The read went to a cell finer than the surface's level (past an
    // occluder the surface's cell met, a surface turned away from it).
    finer: bool,
    // The read's mark claimed a new cell.
    claimed: bool,
}

// The cache at eye-relative hit `at` on the side of unit `side` (normal `n`)
// in space `space`, for a ray `footprint` wide there: read and marked with
// one find a level (`radiance_cache_read_mark`) when `mark`, read alone
// otherwise.
fn gi_hit_cell(at: vec3<f32>, side: vec3<f32>, n: vec3<f32>, space: u32, footprint: f32, mark: bool) -> GiHitCell {
    let p = at + gi_trace.eye.xyz;
    var c: RadianceCacheIrradiance;
    if mark { c = radiance_cache_read_mark(p, side, space, footprint); } else { c = radiance_cache_read(p, side, space, footprint); }
    let finer = radiance_cache_present() && c.level < radiance_cache_level_at(p, footprint);
    if c.found { return GiHitCell(c.direct / PI, c.indirect / PI, true, false, RADIANCE_CACHE_MISS_NONE, finer, false); }
    if c.miss == RADIANCE_CACHE_MISS_UNEXPLORED { return GiHitCell(vec3<f32>(0.0), c.indirect / PI, false, false, c.miss, finer, false); }
    if radiance_cache_miss_stays(c.miss) { return GiHitCell(vec3<f32>(0.0), vec3<f32>(0.0), false, false, c.miss, finer, false); }
    // Waiting: the cell the read went to is in the cache (this mark's claim,
    // or an earlier one) and has no value yet. A key the hash turned away
    // (both buckets full, the new cells' queue full: ABSENT) waits on
    // nothing.
    let waiting = c.miss == RADIANCE_CACHE_MISS_NEW;
    return GiHitCell(vec3<f32>(0.0), gi_irradiance(at, n).rgb, false, waiting, c.miss, finer, c.claimed);
}

// The sides of a hit (`GiHitLight::own`, `gi_hit_direct`): the one its ray
// met (the surface's normal turned toward the ray) and the far one.
const GI_SIDE_NEAR: u32 = 1u;
const GI_SIDE_FAR: u32 = 2u;

// What a hit sends back along its ray of the cache's light (`gi_hit_light`):
// `reflected`, what it reflects of the irradiance on the side its ray met and
// lets through of its far side's; `emitted`, what it emits; `answered`, the
// cache lit every side that reflects or lets light through; `waiting`, a
// side that does took the probes' while its cell is new (`GiHitCell`);
// `own`, the sides whose direct light the reader takes from a light sample
// of its own (`gi_hit_direct`; GI_SIDE_*).
struct GiHitLight {
    reflected: vec3<f32>,
    emitted: vec3<f32>,
    answered: bool,
    waiting: bool,
    own: u32,
    // The near side's `GiHitCell::miss`, `finer` and `claimed`.
    miss: u32,
    finer: bool,
    claimed: bool,
}

// The light a hit at eye-relative `at` sends back along its ray's unit
// direction `dir` (`footprint` wide there: the cache's level it reads,
// `radiance_cache_level_at`): a Lambertian surface `surface` (a stream's
// row) lit by the cache on the side the ray met, plus its transmission share
// of the far side's cell (two-sided leaves), and what it emits
// (`gi_hit_cell`). Where
// the cache has no light for a side, the hit is lit on the spot: its
// indirect light by why (`GiHitCell`) and, for
// a reader that lights such a side itself (`own_light`), its own light
// sample for the direct (the side is in `own`). A side's direct light is
// none for `indirect_only` (the reader lights every side itself). The near
// side's cell is claimed and marked where `mark` (the screen's reader: the
// pixels' gather, or the probes where there is none); the far side's is
// read, never marked:
// the hit is on the side its ray met, and behind it may be the inside of
// what lets light through (a voxel of leaves: a cell seated there would
// start its rays in it). The far side's cells are the ones rays from that
// side make.
fn gi_hit_light(at: vec3<f32>, dir: vec3<f32>, footprint: f32, surface: SceneSurface, space: u32, indirect_only: bool, own_light: bool, mark: bool) -> GiHitLight {
    let s = scene_surface_unpack(surface);
    let face = scene_surface_face(surface);
    let side = select(face, -face, dot(face, dir) > 0.0);
    let through = surface_transmission(s);
    let front = gi_hit_cell(at, side, s.normal, space, footprint, mark);
    let dark = all(s.albedo.rgb == vec3<f32>(0.0));
    var lit = GiHitLight(vec3<f32>(0.0), s.emissive, front.found || dark, front.waiting && !dark, 0u, front.miss, front.finer, front.claimed);
    if !dark {
        lit.reflected = s.albedo.rgb * (front.indirect + select(vec3<f32>(0.0), front.direct, !indirect_only));
        if own_light && !front.found { lit.own |= GI_SIDE_NEAR; }
    }
    if any(through > vec3<f32>(0.0)) {
        // What passes through from the far side: the far side's cell's
        // light where the cache has it (its direct and indirect parts, the
        // frame BSDF's thin lobe: transmission times irradiance / pi), and
        // nothing of the probes' where it has none (at a point on a thin
        // surface the probes hold the near side's light too, both sides'
        // probes seeing the point, so the surface's own light would come
        // back through it). A far side's cell exists only where rays from
        // that side met it (never inside a voxel of leaves, under a
        // canopy's top): where there is none, a reader that lights a side
        // itself takes the far side's direct light from a light sample of
        // its own through the thin lobe, the shadow ray from the lit side
        // (`gi_hit_direct`).
        let back = gi_hit_cell(at, -side, -s.normal, space, footprint, false);
        if back.found {
            lit.reflected += through * (back.indirect + select(vec3<f32>(0.0), back.direct, !indirect_only));
        } else if own_light {
            lit.own |= GI_SIDE_FAR;
        }
        lit.answered = lit.answered && back.found;
        lit.waiting = lit.waiting || back.waiting;
    }
    return lit;
}

// A hit's direct light from one light sample (`frame_light_samples`) on its
// sides `sides` (GI_SIDE_*): Lambertian on the side its ray met (its normal's,
// turned toward the ray); from behind, the surface's share of it through
// (two-sided leaves). `light` before its shadow ray, which leaves from `side`
// (the face turned toward the light's side); `lit` false where the sample
// gives those sides nothing.
struct GiHitDirect {
    light: vec3<f32>,
    side: vec3<f32>,
    lit: bool,
}

fn gi_hit_direct(surface: SceneSurface, sample: LightSample, sides: u32) -> GiHitDirect {
    var d = GiHitDirect(vec3<f32>(0.0), vec3<f32>(0.0), false);
    if !light_sample_found(sample) { return d; }
    let s = scene_surface_unpack(surface);
    let face = scene_surface_face(surface);
    let towards = light_sample_dir(sample);
    let radiance = light_sample_radiance(sample);
    let through = surface_transmission(s);
    let cosine = dot(s.normal, towards);
    if cosine > 0.0 && (sides & GI_SIDE_NEAR) != 0u {
        d.light = s.albedo.rgb * radiance * cosine / PI;
        d.side = face;
    } else if any(through > vec3<f32>(0.0)) && cosine < 0.0 && (sides & GI_SIDE_FAR) != 0u {
        // The frame BSDF's lobe through a thin two-sided surface.
        d.light = bsdf_eval(s, s.normal, towards) * radiance;
        d.side = -face;
    }
    d.lit = any(d.light > vec3<f32>(0.0));
    return d;
}

// The frame statistics of a reader's hits (`gi.hits.<reader>.<state>`,
// `crate::hit_stats`): the near side's RADIANCE_CACHE_MISS_* (0 lit by the
// cache), then GI_HIT_FAR_OWN (a far side lit by the reader's own light
// sample), GI_HIT_FINER (the near side's read went finer than its
// surface's level), GI_HIT_CLAIMED (its mark claimed a new cell: the
// readers' share of the cache's new cells) and GI_HIT_CLAIMED_FINER (a new
// cell finer than the surface's level).
const GI_HIT_STATES: u32 = 9u;
const GI_HIT_FAR_OWN: u32 = 5u;
const GI_HIT_FINER: u32 = 6u;
const GI_HIT_CLAIMED: u32 = 7u;
const GI_HIT_CLAIMED_FINER: u32 = 8u;

// A shadow ray from frame point `p` (already off its surface) toward a light
// sample: `distance` away (a directional light: as far as GI's rays reach),
// as wide as `width` with no spread (it stands for that patch of surface).
fn gi_shadow_ray(p: vec3<f32>, sample: LightSample, width: f32, space: u32) -> SceneRay {
    let off = gi_ray_offset(p);
    let distance = light_sample_distance(sample);
    let reach = select(max(distance - 2.0 * off, 1e-4), GI_RAY_REACH, distance >= LIGHT_SAMPLE_FAR);
    let cone = scene_cone_in_space(scene_ray_cone(width, 0.0, RAY_MASK_ALL), space);
    return SceneRay(p, reach, light_sample_dir(sample), cone);
}
