//! GI's radiance cache against what it must be: GI's node over the analytic
//! tracer (`common`), the lights' sampler, and a reader outside GI that
//! reads and marks the frame's cache as any pass does
//! (`frame_radiance_cache`). Every test waits on a signal (a cell's history
//! filled, a value within its bound) under a wall-clock guard, never a
//! count of frames.

mod common;

use std::time::{Duration, Instant};

use common::{gpu, CacheQuery, CacheRead, Harness, Setup, Shape};
use glam::{DVec3, Vec3};
use helio_frame_api::rays::stream::space_of_slot;
use helio_pass_gi::cache::{refresh_period, EVICT_AGE, HISTORY_MAX, HISTORY_MIN};
use helio_pass_gi::{CacheConfig, GiSettings};

const PI: f64 = std::f64::consts::PI;
/// Base colour 128/255 (RGB8 in a surface).
const HALF: f32 = 128.0 / 255.0;
/// The longest a test waits for its signal.
const GUARD: Duration = Duration::from_secs(300);

fn harness(setup: Setup, shapes: Vec<Shape>) -> Option<Harness> {
    Some(Harness::new(gpu()?, setup, shapes))
}

/// Renders frames until `done` holds for the reads (at least one frame),
/// under the wall-clock guard; the frames it took and the last reads.
fn until(
    h: &mut Harness,
    what: &str,
    mut done: impl FnMut(&[CacheRead]) -> bool,
) -> (usize, Vec<CacheRead>) {
    let start = Instant::now();
    let mut frames = 0;
    loop {
        h.frames(1);
        frames += 1;
        let reads = h.cache_reads();
        if done(&reads) {
            return (frames, reads);
        }
        assert!(
            start.elapsed() < GUARD,
            "{what}: not after {frames} frames; reads {reads:?}"
        );
    }
}

/// Every read found with at least the least history and its histories
/// unchanged over two refresh periods: its cell is settled (its parts hold
/// the histories their noise asks for, its bounced part's change test sees
/// the cells it reads no longer move; only refreshes reach it, which keep
/// them). Renders frames until then, under the wall-clock guard; the frames
/// it took and the last reads.
fn settle(h: &mut Harness, what: &str) -> (usize, Vec<CacheRead>) {
    let start = Instant::now();
    let hold = 2 * refresh_period() as usize;
    let mut frames = 0;
    let mut last: Vec<[u32; 2]> = Vec::new();
    let mut still = 0;
    loop {
        let stats = h.frame_stats();
        frames += 1;
        let reads = h.cache_reads();
        let histories: Vec<[u32; 2]> = reads.iter().map(|r| r.history).collect();
        still = if histories == last { still + 1 } else { 0 };
        last = histories;
        let cells = stats.get("radiance_cache.cells").unwrap_or(0);
        let settled = stats.get("radiance_cache.settled").unwrap_or(0);
        let found = reads.iter().all(|r| {
            r.level.is_some() && r.history[0] >= HISTORY_MIN && r.history[1] >= HISTORY_MIN
        });
        if found && still >= hold {
            println!("{what} after {frames} frames: {settled} of {cells} cells settled");
            return (frames, reads);
        }
        assert!(
            start.elapsed() < GUARD,
            "{what}: not after {frames} frames ({settled} of {cells} cells settled); reads {reads:?}"
        );
    }
}

/// The mean of each query's reads over `frames` more frames.
fn mean_reads(h: &mut Harness, frames: usize) -> Vec<(Vec3, Vec3)> {
    let n = h.queries.lock().unwrap().0.len();
    let mut sum = vec![(DVec3::ZERO, DVec3::ZERO); n];
    for _ in 0..frames {
        h.frames(1);
        for (s, r) in sum.iter_mut().zip(h.cache_reads()) {
            s.0 += r.direct.as_dvec3();
            s.1 += r.indirect.as_dvec3();
        }
    }
    sum.iter()
        .map(|(d, i)| {
            (
                (*d / frames as f64).as_vec3(),
                (*i / frames as f64).as_vec3(),
            )
        })
        .collect()
}

/// A point light's irradiance at `p` on a surface facing `n` (the lights'
/// one falloff, `light_arrival`: inverse square windowed by
/// `1 - (d / range)^4`).
fn point_irradiance(light: DVec3, intensity: f64, range: f64, p: DVec3, n: DVec3) -> f64 {
    let to = light - p;
    let d = to.length();
    if d > range {
        return 0.0;
    }
    let window = (1.0 - (d / range).powi(4)).max(0.0);
    intensity * window / (d * d + 1e-4) * (to / d).dot(n).max(0.0)
}

/// A closed box's walls as patches about `side` metres across: centre,
/// inward normal, the two tangent half-extents.
struct Patch {
    centre: DVec3,
    normal: DVec3,
    u: DVec3,
    v: DVec3,
}

fn patches(lo: DVec3, hi: DVec3, side: f64) -> Vec<Patch> {
    let mut out = Vec::new();
    let size = hi - lo;
    for axis in 0..3 {
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
        let (na, nb) = (
            (size[a] / side).round().max(1.0) as usize,
            (size[b] / side).round().max(1.0) as usize,
        );
        for wall in [lo[axis], hi[axis]] {
            let mut normal = DVec3::ZERO;
            normal[axis] = if wall == lo[axis] { 1.0 } else { -1.0 };
            for i in 0..na {
                for j in 0..nb {
                    let mut centre = DVec3::ZERO;
                    centre[axis] = wall;
                    centre[a] = lo[a] + (i as f64 + 0.5) * size[a] / na as f64;
                    centre[b] = lo[b] + (j as f64 + 0.5) * size[b] / nb as f64;
                    let mut u = DVec3::ZERO;
                    u[a] = 0.5 * size[a] / na as f64;
                    let mut v = DVec3::ZERO;
                    v[b] = 0.5 * size[b] / nb as f64;
                    out.push(Patch {
                        centre,
                        normal,
                        u,
                        v,
                    });
                }
            }
        }
    }
    out
}

/// The irradiance at `p` (facing `n`) from patch `q` of radiance 1: the
/// patch's area integral of `cos cos / r^2` by `k`^2 points.
fn from_patch(p: DVec3, n: DVec3, q: &Patch, k: usize) -> f64 {
    let area = 4.0 * q.u.length() * q.v.length() / (k * k) as f64;
    let mut sum = 0.0;
    for i in 0..k {
        for j in 0..k {
            let s = ((i as f64 + 0.5) / k as f64) * 2.0 - 1.0;
            let t = ((j as f64 + 0.5) / k as f64) * 2.0 - 1.0;
            let at = q.centre + q.u * s + q.v * t;
            let to = at - p;
            let r2 = to.length_squared();
            let d = to / r2.sqrt();
            let c = n.dot(d).max(0.0) * q.normal.dot(-d).max(0.0);
            sum += c / r2 * area;
        }
    }
    sum
}

/// Brute force on the CPU: the indirect irradiance at `queries` in a closed
/// Lambertian box of albedo `albedo` lit by one point light (radiosity over
/// patches, Jacobi to convergence).
fn box_indirect(
    lo: DVec3,
    hi: DVec3,
    albedo: f64,
    light: (DVec3, f64, f64),
    queries: &[(DVec3, DVec3)],
    query_samples: usize,
) -> Vec<f64> {
    let ps = patches(lo, hi, 1.0 / 3.0);
    let direct: Vec<f64> = ps
        .iter()
        .map(|p| point_irradiance(light.0, light.1, light.2, p.centre, p.normal))
        .collect();
    let n = ps.len();
    let mut g = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j && ps[i].normal != ps[j].normal {
                g[i * n + j] = from_patch(ps[i].centre, ps[i].normal, &ps[j], 2);
            }
        }
    }
    // Radiance L = albedo / pi (E_direct + sum_j L_j G_ij).
    let mut radiance: Vec<f64> = direct.iter().map(|e| albedo / PI * e).collect();
    for _ in 0..40 {
        radiance = (0..n)
            .map(|i| {
                let indirect: f64 = (0..n).map(|j| radiance[j] * g[i * n + j]).sum();
                albedo / PI * (direct[i] + indirect)
            })
            .collect();
    }
    queries
        .iter()
        .map(|&(p, normal)| {
            ps.iter()
                .zip(&radiance)
                .filter(|(q, _)| q.normal != normal)
                .map(|(q, l)| l * from_patch(p, normal, q, query_samples))
                .sum()
        })
        .collect()
}

/// The brute force itself, checked by the energy it must conserve: in a
/// closed box every watt the light sends is absorbed by the walls in the
/// end, so the walls' area times their direct irradiance is the light's
/// power (`4 pi I`), and times all their irradiance and `1 - albedo` too.
#[test]
fn the_cpu_radiosity_conserves_energy() {
    let (lo, hi) = (DVec3::new(-3.0, -1.5, -3.0), DVec3::new(3.0, 2.5, 3.0));
    let light = (DVec3::new(0.0, 1.0, 0.0), 40.0, 100.0);
    let albedo = 0.5;
    let ps = patches(lo, hi, 1.0 / 3.0);
    let points: Vec<_> = ps.iter().map(|p| (p.centre, p.normal)).collect();
    let indirect = box_indirect(lo, hi, albedo, light, &points, 2);
    let (mut direct_flux, mut absorbed) = (0.0, 0.0);
    for (p, i) in ps.iter().zip(&indirect) {
        let area = 4.0 * p.u.length() * p.v.length();
        let d = point_irradiance(light.0, light.1, light.2, p.centre, p.normal);
        direct_flux += area * d;
        absorbed += area * (d + i) * (1.0 - albedo);
    }
    let power = 4.0 * PI * light.1;
    println!("direct {direct_flux:.2}, absorbed {absorbed:.2}, power {power:.2}");
    assert!(
        (direct_flux / power - 1.0).abs() < 0.01,
        "{direct_flux} of {power}"
    );
    assert!(
        (absorbed / power - 1.0).abs() < 0.02,
        "{absorbed} of {power}"
    );
}

/// The lit box: a room 6 x 4 x 6 m of albedo 128/255 about the eye, a point
/// light a metre over it; four wall points, read as a pixel's ray would read
/// them, in the world's coordinate space (the probes' rays are in every
/// space's, so the points' cells are the reader's alone, their points the
/// reader's).
struct LitBox {
    lo: DVec3,
    hi: DVec3,
    light: DVec3,
    queries: Vec<CacheQuery>,
}

const INTENSITY: f32 = 40.0;
const RANGE: f32 = 100.0;

fn lit_box(setup: &Setup) -> LitBox {
    let c = setup.world_eye();
    let (lo, hi) = (
        c + DVec3::new(-3.0, -1.5, -3.0),
        c + DVec3::new(3.0, 2.5, 3.0),
    );
    let query = |offset: DVec3, normal: Vec3| CacheQuery {
        world: c + offset,
        normal,
        space: space_of_slot(0),
    };
    LitBox {
        lo,
        hi,
        light: c + DVec3::new(0.0, 1.0, 0.0),
        queries: vec![
            query(DVec3::new(0.4, -1.5, 0.3), Vec3::Y),
            query(DVec3::new(3.0, 0.2, -0.7), Vec3::NEG_X),
            query(DVec3::new(-1.2, 2.5, 1.1), Vec3::NEG_Y),
            query(DVec3::new(1.5, -0.5, -3.0), Vec3::Z),
        ],
    }
}

fn lit_box_setup(origin: DVec3) -> (Setup, LitBox) {
    let mut setup = Setup {
        origin,
        ..Default::default()
    };
    let b = lit_box(&setup);
    setup.lights = vec![common::point_light(
        (b.light - origin).as_vec3(),
        INTENSITY,
        RANGE,
    )];
    (setup, b)
}

/// A closed box with one point light: once a cell's history is full, the
/// irradiance the cache holds at four wall points matches the brute-force
/// radiosity solution on the CPU: the direct part (one delta light, no
/// occlusion: every sample exact) within 1% (RGB9E5's 9 bits of mantissa
/// in the samples and the cell's point in a 1024th of its cell); the
/// indirect part within 10%, which covers the estimate's noise (128 rays a
/// value, averaged over 128 frames: about 2% of an indirect ray's spread)
/// and the coarser cells its rays read at their hits (half a metre to a
/// metre, each holding one point's light).
#[test]
fn a_lit_box_converges_to_its_radiosity_solution() {
    let (setup, b) = lit_box_setup(DVec3::new(1000.0, 2.0, -500.0));
    let room = Shape::room(b.lo, b.hi, HALF);
    let Some(mut h) = harness(setup, vec![room]) else {
        return;
    };
    h.read_cache(b.queries.clone(), true);
    let (frames, _) = settle(&mut h, "the box's cells settle");
    let reads = mean_reads(&mut h, 128);
    let truth = box_indirect(
        b.lo,
        b.hi,
        f64::from(HALF),
        (b.light, f64::from(INTENSITY), f64::from(RANGE)),
        &b.queries
            .iter()
            .map(|q| (q.world, q.normal.as_dvec3()))
            .collect::<Vec<_>>(),
        4,
    );
    for ((q, (direct, indirect)), want) in b.queries.iter().zip(&reads).zip(&truth) {
        let d = point_irradiance(
            b.light,
            f64::from(INTENSITY),
            f64::from(RANGE),
            q.world,
            q.normal.as_dvec3(),
        );
        println!(
            "{:?}: direct {} (want {d:.4}), indirect {} (want {want:.4}), settled after {frames} frames",
            q.normal, direct.x, indirect.x
        );
        assert!(
            (f64::from(direct.x) / d - 1.0).abs() < 0.01,
            "{:?}: direct {direct} against {d}",
            q.normal
        );
        assert!(
            (f64::from(indirect.x) / want - 1.0).abs() < 0.10,
            "{:?}: indirect {indirect} against {want}",
            q.normal
        );
        // Grey light stays grey.
        assert!((indirect.x - indirect.z).abs() < 1e-3 * indirect.x.max(1e-3));
    }
}

/// A plate 5 cm thick under the sun, black ground below and no sky: the
/// plate's two sides at one place, read from 25 m away (the eye angle's cell
/// there a metre wide, far wider than the plate), stay apart: the top holds
/// the sun's 2, the underside, which sees nothing lit, nothing at all.
#[test]
fn a_thin_wall_s_two_sides_stay_apart() {
    let mut setup = Setup {
        sun: Some(2.0),
        ..Default::default()
    };
    let c = setup.world_eye();
    let y = c.y + 1.0;
    // The eye beside the plate, a metre's cell away.
    setup.eye += Vec3::new(1.0 / CacheConfig::default().eye_angle(), 0.0, 0.0);
    let plate = Shape::solid(
        DVec3::new(c.x - 5.0, y, c.z - 5.0),
        DVec3::new(c.x + 5.0, y + 0.05, c.z + 5.0),
        HALF,
    );
    let ground = Shape::ground(c.y - 3.0, 0.0);
    let Some(mut h) = harness(setup, vec![plate, ground]) else {
        return;
    };
    let side = |at_y: f64, normal: Vec3| CacheQuery {
        world: DVec3::new(c.x + 0.7, at_y, c.z - 0.4),
        normal,
        space: space_of_slot(0),
    };
    h.read_cache(vec![side(y + 0.05, Vec3::Y), side(y, Vec3::NEG_Y)], true);
    let (_, reads) = settle(&mut h, "both sides settle");
    assert_eq!(reads[0].level, reads[1].level, "one level for both");
    let means = mean_reads(&mut h, 64);
    let (lit, dark) = (means[0].0 + means[0].1, means[1].0 + means[1].1);
    println!("lit side {lit}, dark side {dark}");
    assert!((lit.x - 2.0).abs() < 0.02, "the lit side: {lit}");
    assert!(dark.max_element() < 1e-4, "dark {dark}, lit {lit}");
}

/// The lit box split by a wall 5 cm thick, the light on one side: the far
/// side gets no light at all. Read on the dark side next to the wall with
/// the eye as far from the reads as makes their surfaces' cells 1/8 m to
/// 4 m wide (the eye angle's), so each read's cell is wider than the wall
/// and holds both sides (the lit side's cells marked by the probes' and the
/// cache's rays, read on the lit side too): the dark reads find their own
/// side's cells, which hold under 1% of the lit side's light (without the
/// cells' clear distances the dark side reads the lit side's whole light
/// from 1/4 m up). Twice: a wall standing on the floor (the floor's cells
/// across it) and a slab across the box (the walls' cells across it).
#[test]
fn a_wall_thinner_than_its_cells_keeps_its_light_on_its_side() {
    let angle = f64::from(CacheConfig::default().eye_angle());
    for (vertical, size) in [true, false]
        .into_iter()
        .flat_map(|v| [0.125f64, 0.25, 0.5, 1.0, 2.0, 4.0].map(|s| (v, s)))
    {
        let (mut setup, b) = lit_box_setup(DVec3::new(1000.0, 2.0, -500.0));
        let c = setup.world_eye();
        // World 1001.3 to 1001.35 (or 2.3 to 2.35 up): inside a cell of
        // every level from 1/8 m to 4 m. Into the room's walls by 10 cm, so
        // no face of it lies on one of theirs (a seam a ray could slip
        // through between the two shapes).
        let into = DVec3::splat(0.1);
        let (wall, lit, dark, normal) = if vertical {
            (
                Shape::solid(
                    DVec3::new(c.x + 1.3, b.lo.y - into.y, b.lo.z - into.z),
                    DVec3::new(c.x + 1.35, b.hi.y + into.y, b.hi.z + into.z),
                    HALF,
                ),
                DVec3::new(c.x + 1.28, b.lo.y, c.z + 0.4),
                DVec3::new(c.x + 1.36, b.lo.y, c.z + 0.4),
                Vec3::Y,
            )
        } else {
            (
                Shape::solid(
                    DVec3::new(b.lo.x - into.x, c.y + 0.3, b.lo.z - into.z),
                    DVec3::new(b.hi.x + into.x, c.y + 0.35, b.hi.z + into.z),
                    HALF,
                ),
                DVec3::new(b.hi.x, c.y + 0.37, c.z + 0.4),
                DVec3::new(b.hi.x, c.y + 0.29, c.z + 0.4),
                Vec3::NEG_X,
            )
        };
        // The eye `size / angle` from the reads, along the reads' normal.
        let between = (lit + dark) / 2.0;
        let eye = between + normal.as_dvec3() * (size / angle);
        setup.eye = (eye - setup.origin).as_vec3();
        let Some(mut h) = harness(setup, vec![Shape::room(b.lo, b.hi, HALF), wall]) else {
            return;
        };
        let read = |at: DVec3| CacheQuery {
            world: at,
            normal,
            space: space_of_slot(0),
        };
        h.read_cache(vec![read(lit), read(dark)], true);
        settle(&mut h, "the split box settles");
        let means = mean_reads(&mut h, 32);
        let reads = h.cache_reads();
        let (lit_light, dark_light) = (means[0].0 + means[0].1, means[1].0 + means[1].1);
        println!(
            "{} wall, {size} m cells: lit {} (level {:?}), dark {} (level {:?})",
            if vertical { "standing" } else { "lying" },
            lit_light.x,
            reads[0].level,
            dark_light.x,
            reads[1].level
        );
        assert!(lit_light.x > 0.1, "the lit side at {size} m: {lit_light}");
        assert!(
            dark_light.max_element() < 0.01 * lit_light.min_element(),
            "the dark side at {size} m: {dark_light} (lit {lit_light})"
        );
    }
}

/// The lit box, the light switched off once the cache settled: the time
/// until the four points' irradiance is under 5% of what it was. A settled
/// cell is updated once a refresh period; the first updates without the
/// light move its change tests' fast means past their bound, so its parts
/// restart and it is lit as converging until it settles again; its bounced
/// part follows as the cells it reads restart at their own refreshes. The
/// bound: what a part whose history never restarted would take to fall to
/// 5% of a step at the refresh period (`ln 0.05 / ln(1 - 1 / HISTORY_MAX)`
/// updates): the cache must forget a light faster than its history alone
/// would.
#[test]
fn a_light_switched_off_leaves_the_cache_within_a_bound() {
    let (setup, b) = lit_box_setup(DVec3::new(1000.0, 2.0, -500.0));
    let room = Shape::room(b.lo, b.hi, HALF);
    let lights = setup.lights.clone();
    let Some(mut h) = harness(setup, vec![room]) else {
        return;
    };
    h.read_cache(b.queries.clone(), true);
    settle(&mut h, "the box's cells settle");
    let lit: Vec<Vec3> = mean_reads(&mut h, 16)
        .iter()
        .map(|(d, i)| *d + *i)
        .collect();
    let mut off = lights.clone();
    off[0].color_intensity[3] = 0.0;
    h.set_lights(&off);
    let (frames, reads) = until(&mut h, "the light leaves the cache", |reads| {
        reads
            .iter()
            .zip(&lit)
            .all(|(r, l)| r.irradiance().x < 0.05 * l.x)
    });
    let updates = (0.05f64.ln() / (1.0 - 1.0 / f64::from(HISTORY_MAX)).ln()).ceil() as usize;
    let bound = updates * refresh_period() as usize;
    println!("the light left the cache in {frames} frames (bound {bound}); {reads:?}");
    assert!(frames <= bound, "{frames} frames, bound {bound}");
}

/// A furnace: a room 100 m across glowing with radiance 1 and reflecting
/// nothing, so every update of every cell is exactly pi; reads at four of
/// its walls.
fn furnace(setup: &Setup, albedo: f32) -> (Shape, Vec<CacheQuery>) {
    let c = setup.world_eye();
    let room = Shape::room(c - 50.0, c + 50.0, albedo).glowing(1.0);
    let query = |offset: DVec3, normal: Vec3| CacheQuery {
        world: c + offset,
        normal,
        space: 0,
    };
    (
        room,
        vec![
            query(DVec3::new(3.0, -50.0, 2.0), Vec3::Y),
            query(DVec3::new(50.0, 1.0, -4.0), Vec3::NEG_X),
            query(DVec3::new(-7.0, 50.0, 5.0), Vec3::NEG_Y),
            query(DVec3::new(2.0, 3.0, -50.0), Vec3::Z),
        ],
    )
}

/// The eye crossing a 1024 m boundary moves the frame's origin by 1024 m:
/// no cell moves to another slot, no key and no value changes, and the
/// reads at the same world points find the same values.
#[test]
fn an_origin_change_moves_no_entry_and_changes_no_value() {
    let origin = DVec3::new(1024.0 * 977.0, 0.0, -1024.0 * 489.0);
    let setup = Setup {
        origin,
        eye: Vec3::new(1023.8, 1.3, 0.4),
        ..Default::default()
    };
    let (room, queries) = furnace(&setup, 0.0);
    let Some(mut h) = harness(setup, vec![room]) else {
        return;
    };
    h.read_cache(queries, true);
    settle(&mut h, "the furnace's cells settle");
    // The cells that held still over a frame (a coarse cell's point near a
    // corner sends a few rays out of the room and keeps moving).
    let (_, held) = h.cache_rows();
    h.frames(1);
    let before = h.cache_reads();
    let (slots, entries) = h.cache_rows();
    // The eye at the boundary: the origin snaps 1024 m on, the eye where it
    // was in the world.
    h.set_origin(origin + DVec3::new(1024.0, 0.0, 0.0));
    h.frames(1);
    let after = h.cache_reads();
    let (slots_after, entries_after) = h.cache_rows();
    let now = (0..entries.len()).map(|s| slots[s].used).max().unwrap_or(0);
    let mut kept = 0;
    for s in 0..entries.len() {
        let still = (
            held[s].key,
            held[s].history & 0xffff,
            held[s].indirect,
            held[s].bounced,
        ) == (
            entries[s].key,
            entries[s].history & 0xffff,
            entries[s].indirect,
            entries[s].bounced,
        );
        let full = |h: u32| (h & 0xff) >= HISTORY_MIN && ((h >> 8) & 0xff) >= HISTORY_MIN;
        // In use (marked in the last half of the eviction's age: a cell no
        // reader marked for that long may go at the change's frame).
        let recent = slots[s].used + EVICT_AGE / 2 > now;
        if slots[s].state == 0 || !full(entries[s].history) || !still || !recent {
            continue;
        }
        assert_eq!(slots_after[s].state, slots[s].state, "slot {s}");
        let (a, b) = (entries[s], entries_after[s]);
        assert_eq!(
            (a.key, a.history & 0xffff),
            (b.key, b.history & 0xffff),
            "slot {s}"
        );
        assert_eq!(
            (a.direct, a.indirect, a.bounced),
            (b.direct, b.indirect, b.bounced),
            "slot {s}"
        );
        kept += 1;
    }
    assert!(kept > 4, "{kept} settled cells");
    for (a, b) in before.iter().zip(&after) {
        assert_eq!(a.level, b.level);
        assert_eq!((a.direct, a.indirect), (b.direct, b.indirect));
        assert!((a.indirect.x - std::f32::consts::PI).abs() < 1e-5, "{a:?}");
    }
}

/// A furnace of albedo one half (radiance 1 glowing): every cell's
/// irradiance converges to `pi / (1 - 1/2)` = 2 pi. Near the world's zero
/// and an Earth's radius away on either side (the high words of a cell's
/// coordinates all ones there), it converges to the same value in about the
/// same frames.
#[test]
fn a_cell_at_earth_scale_behaves_as_one_near_zero() {
    let earth = 6_371_000.0 - 6_371_000.0 % 1024.0;
    let mut results = Vec::new();
    for origin in [
        DVec3::ZERO,
        DVec3::new(earth, 0.0, 0.0),
        DVec3::new(0.0, -earth, -earth),
    ] {
        let setup = Setup {
            origin,
            ..Default::default()
        };
        let (room, queries) = furnace(&setup, 0.5);
        let Some(mut h) = harness(setup, vec![room]) else {
            return;
        };
        h.read_cache(queries, true);
        let want = 2.0 * std::f32::consts::PI;
        let (frames, reads) = until(&mut h, "the furnace converges", |reads| {
            reads
                .iter()
                .all(|r| r.level.is_some() && (r.indirect.x / want - 1.0).abs() < 0.01)
        });
        println!("origin {origin}: within 1% of 2 pi after {frames} frames: {reads:?}");
        results.push((frames, reads));
    }
    let (near, _) = &results[0];
    for (frames, reads) in &results[1..] {
        assert!(
            frames.abs_diff(*near) * 5 <= *near,
            "{frames} frames against {near} near zero"
        );
        for (r, n) in reads.iter().zip(&results[0].1) {
            assert_eq!(r.level, n.level);
        }
    }
}

/// A small cache (2^13 slots): one set of cells read and marked, sized to
/// fit beside the probes' cells under half the slots, then another as large,
/// which does not fit beside it. The first set's cells go (unused, and past
/// half the slots the least recently used), the second's take their slots;
/// read again, the first set comes back.
#[test]
fn a_small_cache_evicts_and_reuses_its_slots() {
    let setup = Setup {
        gi: GiSettings {
            cache_slots_log2: Some(13),
            ..Default::default()
        },
        ..Default::default()
    };
    let (room, _) = furnace(&setup, 0.0);
    let c = setup.world_eye();
    let Some(mut h) = harness(setup, vec![room]) else {
        return;
    };
    let slots = 1usize << 13;
    // The probes' cells alone, once their rays have met the room for a few
    // frames.
    for _ in 0..8 {
        h.frame_stats();
    }
    let probes = h.frame_stats().get("radiance_cache.cells").unwrap_or(0) as usize;
    // 30 floor points a metre apart in each of `spaces` coordinate spaces a
    // set (rays of those spaces see nothing: a miss, an empty sky, lights
    // them black: found, holding 0); a space's points key 30 cells (one
    // level a surface).
    let spaces = ((slots * 45 / 100).saturating_sub(probes) / 30).max(8) as u32;
    println!("{probes} probe cells; sets of {spaces} spaces");
    let set = |first: u32| -> Vec<CacheQuery> {
        (0..spaces)
            .flat_map(|k| {
                (0..30).map(move |i| CacheQuery {
                    world: c + DVec3::new(f64::from(i % 6) - 2.5, -1.5, f64::from(i / 6) - 2.0),
                    normal: Vec3::Y,
                    space: space_of_slot(first + k),
                })
            })
            .collect()
    };
    let (a, b) = (set(1), set(1 + spaces));
    let all_found = |reads: &[CacheRead]| reads.iter().all(|r| r.level.is_some());
    h.read_cache(a.clone(), true);
    until(&mut h, "the first set is lit", all_found);
    let occupied_a = h.frame_stats().get("radiance_cache.cells").unwrap_or(0);
    h.read_cache(b.clone(), true);
    let start = Instant::now();
    let (mut frames, mut evicted) = (0, 0);
    loop {
        let stats = h.frame_stats();
        frames += 1;
        evicted += stats.get("radiance_cache.evicted").unwrap_or(0);
        let reads = h.cache_reads();
        if all_found(&reads) {
            break;
        }
        if frames % 200 == 0 {
            let found = reads.iter().filter(|r| r.level.is_some()).count();
            println!(
                "frame {frames}: {found} of {} found; cells {:?} new {:?} evicted {:?} lit {:?}",
                reads.len(),
                stats.get("radiance_cache.cells"),
                stats.get("radiance_cache.new"),
                stats.get("radiance_cache.evicted"),
                stats.get("radiance_cache.lit")
            );
        }
        assert!(
            start.elapsed() < GUARD,
            "the second set is not lit after {frames} frames"
        );
    }
    println!(
        "first set: {occupied_a} cells in use; second set lit after {frames} frames, {evicted} cells evicted meanwhile"
    );
    assert!(evicted as usize >= a.len() / 2, "{evicted} evicted");
    // The first set's cells went: unmarked reads find nothing.
    h.read_cache(a.clone(), false);
    h.frames(1);
    let reads = h.cache_reads();
    let found = reads.iter().filter(|r| r.level.is_some()).count();
    assert_eq!(
        found,
        0,
        "{found} of the first set's {} cells kept",
        a.len()
    );
    // And come back once read again.
    h.read_cache(a, true);
    until(&mut h, "the first set is lit again", all_found);
}

/// Two coordinate spaces at one place (two rooms in one box, one glowing 1,
/// the other 3, each seen only by its own space's rays): a wall point read
/// in each space finds its own room's light, pi and 3 pi.
#[test]
fn two_spaces_at_one_place_do_not_mix() {
    let setup = Setup::default();
    let c = setup.world_eye();
    let rooms = vec![
        Shape::room(c - 20.0, c + 20.0, 0.0)
            .glowing(1.0)
            .in_space(1),
        Shape::room(c - 20.0, c + 20.0, 0.0)
            .glowing(3.0)
            .in_space(2),
    ];
    let Some(mut h) = harness(setup, rooms) else {
        return;
    };
    let query = |slot: u32| CacheQuery {
        world: c + DVec3::new(1.0, -20.0, 2.0),
        normal: Vec3::Y,
        space: space_of_slot(slot),
    };
    h.read_cache(vec![query(1), query(2)], true);
    let (_, reads) = until(&mut h, "both spaces' cells are lit", |reads| {
        reads.iter().all(|r| r.level.is_some())
    });
    let pi = std::f32::consts::PI;
    assert!((reads[0].indirect.x - pi).abs() < 1e-4, "{:?}", reads[0]);
    assert!(
        (reads[1].indirect.x - 3.0 * pi).abs() < 1e-3,
        "{:?}",
        reads[1]
    );
}
