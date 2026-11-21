//! GI's settings: what a renderer config carries for it
//! (`RendererConfig::features`) and the node builds read, and their host
//! description ([`GI`], as `helio_settings` describes the renderer's):
//! engines build their settings UI from it and read the values back
//! through it.

use helio_settings::{integer, number, Group, Resolved};

use crate::cache::{CacheConfig, DEFAULT_RAYS, DEFAULT_SLOTS_LOG2};
use crate::GiConfig;

/// GI's settings (a renderer config's `features`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GiSettings {
    /// The finest probes' spacing (metres), or `None` for the default
    /// ([`crate::DEFAULT_FINEST_SPACING`], half a metre: rooms, pillars and
    /// furniture, a clipmap reaching 192 m around the eye). A host over open
    /// ground sets a coarser one (1 m: 384 m).
    pub probe_spacing: Option<f32>,
    /// The radiance cache's slots, as a power of two (12 to 24), or `None`
    /// for the default ([`DEFAULT_SLOTS_LOG2`]: 2^18, 26 MiB). A larger cache
    /// holds finer cells (`CacheConfig::eye_angle`).
    pub cache_slots_log2: Option<u32>,
    /// Rays a frame the radiance cache spends lighting its cells, or `None`
    /// for the default ([`DEFAULT_RAYS`]). More rays light each cell more
    /// often: a change of light reaches the cache sooner.
    pub cache_rays: Option<u32>,
}

impl GiSettings {
    /// The configuration these settings ask for.
    pub fn config(&self) -> GiConfig {
        GiConfig {
            finest_spacing: self
                .probe_spacing
                .map_or(crate::DEFAULT_FINEST_SPACING, f64::from),
            cache: CacheConfig::new(
                self.cache_slots_log2.unwrap_or(DEFAULT_SLOTS_LOG2),
                self.cache_rays.unwrap_or(DEFAULT_RAYS),
            ),
        }
    }
}

/// GI's host settings: a host lists it with its other settings groups.
pub static GI: Group = Group {
    key: "gi",
    title: "Global Illumination",
    description: "World-space diffuse GI: the probe clipmap and the radiance cache, traced by \
                  every kind's ray tracer (a voxel terrain's walk; meshes with hardware ray \
                  queries, which the host must enable on its device: its hardware ray query \
                  opt-in)",
    settings: &[
        number(
            "probe_spacing",
            "GI Probe Spacing",
            "Global Illumination",
            (0.0, 8.0, 0.25),
            0.0,
            "Metres between the finest GI probes (each coarser level doubles it); 0: the \
             default, 0.5 m (open ground wants 1 m)",
        ),
        integer(
            "radiance_cache_slots",
            "Radiance Cache Size",
            "Global Illumination",
            (0, 24),
            0,
            "The radiance cache's slots as a power of two (12 to 24, 104 bytes a slot; a larger \
             cache holds finer cells); 0: the default, 2^18 (26 MiB)",
        ),
        integer(
            "radiance_cache_rays",
            "Radiance Cache Rays",
            "Global Illumination",
            (0, 1 << 20),
            0,
            "Rays a frame the radiance cache spends lighting surfaces (1024 up; more light each \
             cell more often); 0: the default, 32768",
        ),
    ],
};

impl GiSettings {
    /// The settings `settings` (of [`GI`]) hold.
    pub fn from_settings(settings: &Resolved) -> Self {
        debug_assert!(std::ptr::eq(settings.group(), &GI));
        Self {
            probe_spacing: settings.optional("probe_spacing").map(|m| m as f32),
            cache_slots_log2: settings.optional("radiance_cache_slots").map(|n| n as u32),
            cache_rays: settings.optional("radiance_cache_rays").map(|n| n as u32),
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
        let open_ground = GiSettings {
            probe_spacing: Some(1.0),
            ..Default::default()
        };
        assert_eq!(open_ground.config().finest_spacing, 1.0);
    }

    #[test]
    fn a_spacing_and_a_cache_map() {
        let settings = GI.resolve(|key| match key {
            "probe_spacing" => Some(helio_settings::Value::Number(2.0)),
            "radiance_cache_slots" => Some(helio_settings::Value::Number(20.0)),
            "radiance_cache_rays" => Some(helio_settings::Value::Number(65536.0)),
            _ => None,
        });
        let gi = GiSettings::from_settings(&settings);
        assert_eq!(gi.probe_spacing, Some(2.0));
        assert_eq!(
            gi.config().cache,
            CacheConfig {
                slots: 1 << 20,
                rays: 65536
            }
        );
        assert_eq!(GiSettings::default().config().cache, CacheConfig::default());
    }
}
