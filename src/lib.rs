//! Global Illumination (`pulsar.gi`): the plugin's root crate, the one place
//! the plugin registers everything it contributes.
//!
//! The plugin's other crates hold the code: `helio-gi-api` (the tracer
//! interface other kinds implement), the feature crate (`helio-pass-gi`: the
//! probe clipmap, its tracers and the resolve that publishes the frame's
//! indirect diffuse). GI has no SceneDB component (its probe clipmap follows
//! the camera; the scene's tracers feed it) and no editor UI of its own.
//! Neither crate registers a contribution; this crate imports them and
//! registers them
//! through its [`build`], so this file and the module below are the
//! whole list of what the plugin brings:
//!
//! - [`runtime`]: what a game links. The Helio feature, configured from the
//!   project's GI settings, and that settings group (the project settings
//!   window edits it like any other).
//!
//! Every registration names this crate as its origin
//! (`p.origin()`), so the plugin gate skips all of them while
//! a project disables the plugin, and the plugin catalog lists them under
//! `pulsar.gi`. Other plugins never depend on this crate (only on
//! `helio-gi-api`); hosts reach it through the installed set or a generated
//! game, a game without the `editor` feature.

// The plugin's descriptor (`pulsar-plugin.toml`): linking this crate
// installs the plugin; a project enables it or not (`pulsar_plugins`).
pulsar_plugins::installed!("pulsar-plugin.toml", build);

/// Everything the plugin registers.
pub fn build(p: &mut pulsar_plugins::PluginBuilder) {
    runtime::build(p);
}


pub mod runtime;
