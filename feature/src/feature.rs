//! GI as a feature: the nodes it adds to preset graphs.
//!
//! The update runs at [`Slot::Geometry`], after the G-buffer phase: every
//! kind has drawn this frame's geometry there (a voxel terrain its cut for
//! this eye, uploads and all), so the tracers its rays meet answer for the
//! frame being drawn, and the probes take this frame's rays before the
//! resolve reads them. Its node holds the whole update: GI's stream's
//! generators (the probes' rays, the radiance cache's), the stream's tracers
//! (`stream::trace`), the hits' light from the cache and the cache rays'
//! light batch, the frame's light samplers (`light_samples::sample`), the
//! shadow stream's generator, its tracers, the probes' integration and the
//! cache's fold. GI reads no light kind: what lights its cells comes as
//! light samples. The resolve
//! joins at [`Slot::LightingInputs`], publishing the frame's indirect
//! diffuse. Where nothing traces rays (no kind registered a tracer this
//! device runs), neither builds.
//!
//! GI offers the distance at which its indirect diffuse resolves occlusion
//! (`helio_frame_api::lighting::indirect_occlusion`): twice the finest
//! probes' spacing, wherever both nodes build.

use std::sync::Arc;

use helio_feature_api::{
    BuildEnv, Feature, FeatureRegistrar, NodeBuild, NodeDescriptor, ParamDesc,
};
use helio_frame_api::lighting::indirect_occlusion::OfferIndirectOcclusion;
use helio_frame_api::lighting::{inputs_order, InputsOrder};
use helio_frame_api::rays::{light_samples, stream};
use helio_frame_api::{FrameNode, Slot};

use crate::{
    GiIntegratePass, GiLightPass, GiPass, GiResolvePass, GiSettings, GiShadowPass, RadianceCache,
    RadianceCacheFold,
};

/// World-space diffuse GI ([`crate`]).
pub struct GiFeature;

const CATEGORY: &str = "Passes: Lighting";

fn settings(env: &BuildEnv<'_>) -> GiSettings {
    env.setting::<GiSettings>().copied().unwrap_or_default()
}

/// The node names (`helio.pass.<name>`).
const UPDATE: &str = "gi";
const RESOLVE: &str = "gi_resolve";

/// Probe spacings within which the probes do not hold an occluder: their
/// irradiance at a point blends the probes of its cell, and a probe a
/// spacing off an occluder sees past it; at two spacings every probe of the
/// cell is on the occluder's near side.
const OCCLUSION_SPACINGS: f32 = 2.0;

/// Whether the probes' irradiance reaches the lighting in a graph built in
/// `env`: something traces their rays, the lighting reads it after the
/// resolve (not forward), and the graph builds both nodes.
fn resolves(env: &mut BuildEnv<'_>) -> bool {
    inputs_order(env) == InputsOrder::BeforeLighting
        && [UPDATE, RESOLVE]
            .iter()
            .all(|name| env.plans(&helio_feature_api::pass_kind(name)))
        && stream::traced(env) > 0
}

/// The distance GI's indirect diffuse resolves occlusion at in `env`'s
/// graph (module docs), if it builds there.
fn occlusion_distance(env: &mut BuildEnv<'_>) -> Option<f32> {
    if !resolves(env) {
        return None;
    }
    Some(OCCLUSION_SPACINGS * settings(env).config().finest_spacing as f32)
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
    let config = settings(build.env).config();
    let device = build.env.device.clone();
    let stats = build.env.stats.clone();
    let (rays, shadows) = (
        crate::gi_stream(&config),
        crate::shadow_stream(&config.cache),
    );
    let answers = [rays, shadows].map(|s| stream::answers(build.env, &s));
    build.graph.add_pass(Box::new(GiPass::new(
        &device,
        config,
        &stats,
        answers[0].clone(),
    )));
    build.graph.add_pass(Box::new(RadianceCache::new(
        &device,
        config.cache,
        &stats,
        rays,
    )));
    stream::trace(build, &rays)?;
    build
        .graph
        .add_pass(Box::new(GiLightPass::new(&device, &config)));
    light_samples::sample(build, &crate::light_batch(&config.cache))?;
    build.graph.add_pass(Box::new(GiShadowPass::new(
        &device,
        &config,
        &stats,
        answers[1].clone(),
    )));
    stream::trace(build, &shadows)?;
    build
        .graph
        .add_pass(Box::new(GiIntegratePass::new(&device, answers[0].clone())));
    build.graph.add_pass(Box::new(RadianceCacheFold::new(
        &device,
        config.cache,
        &stats,
        rays,
        shadows,
        answers,
    )));
    Ok(())
}

impl Feature for GiFeature {
    fn id(&self) -> &'static str {
        "helio.gi"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        // What shader hot reload may swap without rebuilding the graph.
        registrar.swappable::<GiPass>();
        registrar.swappable::<GiLightPass>();
        registrar.swappable::<GiShadowPass>();
        registrar.swappable::<GiIntegratePass>();
        registrar.swappable::<GiResolvePass>();
        registrar.swappable::<RadianceCache>();
        registrar.swappable::<RadianceCacheFold>();
        registrar.indirect_occlusion(Arc::new(occlusion_distance));
        registrar.node(
            node(
                UPDATE,
                "GI: Probe Update",
                "Traces this frame's probe rays and the radiance cache's rays and shadow rays \
                 through every kind's tracer, lights the hits from the cache, integrates the \
                 probes and folds the cache's cells. Builds nothing where nothing traces rays.",
                build_update,
            )
            .once_per_frame()
            .in_slot(Slot::Geometry, 20),
        );
        registrar.node(
            node(
                RESOLVE,
                "GI: Resolve",
                "The probes' irradiance at every pixel (the frame's indirect diffuse) for the lighting. Builds nothing where the lighting ran before it (forward) or nothing traces rays.",
                |build| {
                    if resolves(build.env) {
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
