//! The GPU cost of GI's own passes (generation, the radiance cache's
//! selection and rays, the hits' light, the cache rays' shadow rays, the
//! integration, the cache's fold) over the analytic tracer, from GPU timestamps of pipelined
//! frames: the graph never waits for a frame, the test only polls. Ignored
//! by default (a measurement, not a check):
//! `cargo test -p helio-pass-gi --test cost -- --ignored --nocapture`.

mod common;

use common::{gpu, Harness, Setup, Shape};
use glam::{DVec3, Vec3};

/// Median GPU milliseconds of each of `names` over `frames` pipelined
/// frames after a warm-up.
fn measure(h: &mut Harness, names: &[&str], frames: usize) -> Vec<(String, f64)> {
    let device = h.gpu.device.clone();
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for frame in 0..frames + 60 {
        h.submit();
        device.poll(wgpu::PollType::Poll).unwrap();
        if frame < 60 {
            continue;
        }
        let timings = h.graph.profiler().get_gpu_timings();
        for (k, name) in names.iter().enumerate() {
            let ms: f64 = timings
                .iter()
                .filter(|t| t.name == *name)
                .map(|t| t.duration_ns as f64 / 1e6)
                .sum();
            if ms > 0.0 {
                samples[k].push(ms);
            }
        }
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    names
        .iter()
        .zip(samples)
        .map(|(name, mut s)| {
            s.sort_by(f64::total_cmp);
            (
                name.to_string(),
                s.get(s.len() / 2).copied().unwrap_or(f64::NAN),
            )
        })
        .collect()
}

#[test]
#[ignore = "a GPU measurement: run it on its own"]
fn gpu_cost_of_gi_passes() {
    let Some(gpu) = gpu() else { return };
    if !gpu
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
    {
        eprintln!("no GPU timestamps on this adapter");
        return;
    }
    let names = [
        "GI",
        "RadianceCache",
        "GILight",
        "GIShadow",
        "GIIntegrate",
        "RadianceCacheFold",
        "AnalyticTracer",
    ];
    let base = Setup {
        sky: 1.0,
        sun: Some(3.0),
        ..Default::default()
    };
    let eye = base.world_eye().floor();
    // Ground, a roof and a block: about half the rays hit, most hits face
    // the sun.
    let shapes = vec![
        Shape::ground(0.0, 0.5),
        Shape::solid(
            DVec3::new(eye.x - 6.0, 3.0, eye.z - 6.0),
            DVec3::new(eye.x + 6.0, 3.5, eye.z + 6.0),
            0.5,
        ),
        Shape::solid(
            eye + DVec3::new(4.0, -2.0, 4.0),
            eye + DVec3::new(9.0, 6.0, 9.0),
            0.4,
        ),
    ];
    let lights: Vec<_> = (0..64)
        .map(|k| {
            let a = k as f32 * 0.7;
            let at = (DVec3::new(eye.x, 2.5, eye.z) - base.origin).as_vec3()
                + Vec3::new(a.cos(), 0.0, a.sin()) * (2.0 + k as f32 * 0.3);
            common::point_light(at, 20.0, 15.0)
        })
        .collect();
    let setups = [
        ("sun, sky", base.clone()),
        (
            "sun, sky, 64 point lights",
            Setup {
                lights,
                ..base.clone()
            },
        ),
    ];
    // Two runs of each, interleaved, so a clock change shows.
    for run in 0..2 {
        for (label, setup) in &setups {
            let mut h = Harness::pipelined(gpu, setup.clone(), shapes.clone());
            let result = measure(&mut h, &names, 300);
            let line: Vec<String> = result
                .iter()
                .map(|(name, ms)| format!("{name} {ms:.3} ms"))
                .collect();
            eprintln!("{label} (run {run}): {}", line.join(", "));
        }
    }
}
