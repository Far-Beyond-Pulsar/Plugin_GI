//! What the GI plugin registers for a game (and the editor): every runtime
//! contribution, grouped by the engine seam it plugs into.

use engine_backend::subsystems::render::RenderFeatureRegistration;
use pulsar_settings::project::rendering::SettingsGroupRegistration;

// ── Rendering ───────────────────────────────────────────────────────────────

/// GI's feature: the probe clipmap, its tracers (the meshes' ray queries; a
/// voxel terrain through `helio-gi-api`'s `TerrainTracer`) and the resolve
/// that publishes the frame's indirect diffuse.
fn render_features() -> helio_feature_api::FeatureSet {
    helio_feature_api::FeatureSet::new().with(helio_pass_gi::GiFeature)
}

/// The feature's settings from the project's GI settings.
fn configure_renderer(config: helio::RendererConfig) -> helio::RendererConfig {
    use engine_backend::subsystems::render::helio_renderer::project_settings;
    config.with_feature(helio_pass_gi::GiSettings::from_settings(
        &project_settings::resolved(&helio_pass_gi::GI),
    ))
}

/// Everything the plugin registers for a game (and the editor).
pub(crate) fn build(p: &mut pulsar_plugins::PluginBuilder) {
    // The probe clipmap, configured from the project's GI settings.
    p.add(RenderFeatureRegistration {
        origin: p.origin(),
        features: render_features,
        configure: configure_renderer,
    });

    // The project's `gi` settings group.
    p.add(SettingsGroupRegistration {
        origin: p.origin(),
        group: &helio_pass_gi::GI,
    });
}

#[cfg(test)]
mod tests {
    use engine_backend::subsystems::render::helio_renderer::render_features;

    #[test]
    fn registers_its_feature() {
        let features = render_features::plugin_features();
        assert!(
            features.ids().contains(&"helio.gi"),
            "features: {:?}",
            features.ids()
        );
    }

    #[test]
    fn configures_its_settings() {
        let config = render_features::configure(helio::RendererConfig::new(
            8,
            8,
            wgpu::TextureFormat::Rgba8Unorm,
        ));
        assert!(config.features.get::<helio_pass_gi::GiSettings>().is_some());
    }

    /// Every registration of the plugin names this root, so it resolves to
    /// the plugin's id.
    #[test]
    fn every_registration_resolves_to_the_plugin() {
        let ours: Vec<_> = pulsar_plugins::contributions()
            .into_iter()
            .filter(|summary| summary.origin == pulsar_plugins::origin!())
            .collect();
        assert!(ours
            .iter()
            .any(|summary| summary.category == "render.features"));
        assert!(ours
            .iter()
            .any(|summary| summary.category == "settings.group"));
        for summary in &ours {
            assert_eq!(
                pulsar_plugins::plugin_of(summary.origin).map(|plugin| plugin.id.as_str()),
                Some("pulsar.gi"),
                "{summary:?}"
            );
        }
    }
}
