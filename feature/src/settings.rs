//! GI's settings: what a renderer config carries for it
//! (`RendererConfig::features`) and the node builds read, and their host
//! description ([`GI`], as `helio_settings` describes the renderer's):
//! engines build their settings UI from it and read the values back
//! through it.

use helio_settings::{number, Group, Resolved};

use crate::GiConfig;

/// GI's settings (a renderer config's `features`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GiSettings {
    /// The finest probes' spacing (metres), or `None` for the default
    /// ([`crate::DEFAULT_FINEST_SPACING`], a clipmap reaching 384 m around
    /// the eye). A scene of rooms, pillars and furniture a few metres apart
    /// sets a finer one (half a metre: 192 m).
    pub probe_spacing: Option<f32>,
}

impl GiSettings {
    /// The probes' configuration these settings ask for.
    pub fn config(&self) -> GiConfig {
        GiConfig {
            finest_spacing: self
                .probe_spacing
                .map_or(crate::DEFAULT_FINEST_SPACING, f64::from),
        }
    }
}

/// GI's host settings: a host lists it with its other settings groups.
pub static GI: Group = Group {
    key: "gi",
    title: "Global Illumination",
    description: "World-space diffuse GI: the probe clipmap, traced by every kind's ray tracer (a \
                  voxel terrain's walk; meshes with hardware ray queries, which the host must \
                  enable on its device: its hardware ray query opt-in)",
    settings: &[number(
        "probe_spacing",
        "GI Probe Spacing",
        "Global Illumination",
        (0.0, 8.0, 0.25),
        0.0,
        "Metres between the finest GI probes (each coarser level doubles it); 0: the default, 1 m \
         (rooms and furniture want 0.5 m)",
    )],
};

impl GiSettings {
    /// The settings `settings` (of [`GI`]) hold.
    pub fn from_settings(settings: &Resolved) -> Self {
        debug_assert!(std::ptr::eq(settings.group(), &GI));
        Self {
            probe_spacing: settings.optional("probe_spacing").map(|m| m as f32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_defaults_are_the_default_settings() {
        assert_eq!(
            GiSettings::from_settings(&GI.defaults()),
            GiSettings::default()
        );
    }

    #[test]
    fn the_default_spacing_is_the_api_default() {
        assert_eq!(
            GiSettings::default().config().finest_spacing,
            crate::DEFAULT_FINEST_SPACING
        );
        let rooms = GiSettings {
            probe_spacing: Some(0.5),
        };
        assert_eq!(rooms.config().finest_spacing, 0.5);
    }

    #[test]
    fn a_spacing_maps() {
        let settings = GI
            .resolve(|key| (key == "probe_spacing").then_some(helio_settings::Value::Number(2.0)));
        assert_eq!(
            GiSettings::from_settings(&settings).probe_spacing,
            Some(2.0)
        );
    }
}
