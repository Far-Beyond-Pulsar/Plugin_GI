//! GI as a feature: the nodes it adds to preset graphs.
//!
//! The probes' update runs at [`Slot::Geometry`], after the G-buffer phase:
//! every kind has drawn this frame's geometry there (a voxel terrain its cut
//! for this eye, uploads and all), so the tracers its rays meet answer for
//! the frame being drawn, and the probes take this frame's rays before the
//! resolve reads them. Its node holds the whole update: the probe stream's
//! generator, the stream's tracers (`stream::trace`), the hits' light and
//! the shadow stream's generator, its tracers, the integration. The resolve
//! joins at [`Slot::LightingInputs`], publishing the frame's indirect
//! diffuse. Where nothing traces rays (no kind registered a tracer this
//! device runs), neither builds.

use helio_feature_api::{Feature, FeatureRegistrar, NodeBuild, NodeDescriptor, ParamDesc};
use helio_frame_api::lighting::{inputs_order, sun_transmittance, InputsOrder};
use helio_frame_api::rays::stream;
use helio_frame_api::{FrameNode, Slot};

use crate::{GiIntegratePass, GiLightPass, GiPass, GiResolvePass, GiSettings};

/// World-space diffuse GI ([`crate`]).
pub struct GiFeature;

const CATEGORY: &str = "Passes: Lighting";

fn settings(build: &NodeBuild<'_, '_>) -> GiSettings {
    build
        .env
        .setting::<GiSettings>()
        .copied()
        .unwrap_or_default()
}

fn node(
    name: &str,
    title: &str,
    description: &str,
    build: impl Fn(&mut NodeBuild<'_, '_>) -> Result<(), String> + Send + Sync + 'static,
) -> NodeDescriptor {
    NodeDescriptor::new(
        &format!("helio.pass.{name}"),
        title,
        CATEGORY,
        description,
        build,
    )
    .with_param(ParamDesc::boolean("enabled", "Enabled", true))
    .with_generated_pins()
}

/// The probes' update (module docs): its passes and the tracers' between
/// them.
fn build_update(build: &mut NodeBuild<'_, '_>) -> Result<(), String> {
    if stream::traced(build.env) == 0 {
        log::warn!(
            "GI is off: nothing in this graph traces rays (no kind registered a ray tracer this \
             device runs; meshes need hardware ray queries, which the host opts into)"
        );
        return Ok(());
    }
    let config = settings(build).config();
    let sun = sun_transmittance::current(build.env);
    let device = build.env.device.clone();
    let stats = build.env.stats.clone();
    build
        .graph
        .add_pass(Box::new(GiPass::new(&device, config, &stats)));
    stream::trace(build, &crate::probe_stream())?;
    build
        .graph
        .add_pass(Box::new(GiLightPass::new(&device, sun, &stats)));
    stream::trace(build, &crate::shadow_stream())?;
    build
        .graph
        .add_pass(Box::new(GiIntegratePass::new(&device)));
    Ok(())
}

impl Feature for GiFeature {
    fn id(&self) -> &'static str {
        "helio.gi"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.consumes(helio_light_api::DIRECTIONAL_LIGHTS_ROW);
        registrar.consumes(helio_light_api::SCENE_LIGHTS);
        // What shader hot reload may swap without rebuilding the graph.
        registrar.swappable::<GiPass>();
        registrar.swappable::<GiLightPass>();
        registrar.swappable::<GiIntegratePass>();
        registrar.swappable::<GiResolvePass>();
        registrar.node(
            node(
                "gi",
                "GI: Probe Update",
                "Traces this frame's probe rays and their shadow rays through every kind's \
                 tracer, lights the hits and integrates them into the probe clipmap. Builds \
                 nothing where nothing traces rays.",
                build_update,
            )
            .once_per_frame()
            .in_slot(Slot::Geometry, 20),
        );
        registrar.node(
            node(
                "gi_resolve",
                "GI: Resolve",
                "The probes' irradiance at every pixel (the frame's indirect diffuse) for the lighting. Builds nothing where the lighting ran before it (forward) or nothing traces rays.",
                |build| {
                    let read = inputs_order(build.env) == InputsOrder::BeforeLighting;
                    if read && stream::traced(build.env) > 0 {
                        build
                            .graph
                            .add_pass(Box::new(GiResolvePass::new(build.env.device)));
                    }
                    Ok(())
                },
            )
            .in_slot(Slot::LightingInputs, 0),
        );
    }
}
