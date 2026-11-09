# Plugin_GI

World-space diffuse GI: the probe clipmap the scene's tracers feed, and its settings.

`pulsar.gi` ("Global Illumination") is a plugin of
[Pulsar-Native](https://github.com/Far-Beyond-Pulsar/Pulsar-Native), the engine and editor that renders with
[Helio](https://github.com/Far-Beyond-Pulsar/Helio). Pulsar-Native vendors this repository as a git submodule
at `plugins/vendor/gi`.

## What it owns

Every object kind is a plugin. A plugin owns its SceneDB components (their
data and scriptable methods; the editor calls the same methods scripts do),
the contracts other kinds read (its `api` crate: graph keys, row ABIs, WGSL
snippets), the Helio feature that renders it (passes, scene derivations,
settings), its file formats and its editor UI. Helio knows no object kind,
and other kinds meet this one only through frame contracts, extension points,
draw phases and api crates.

| Crate | Path | What |
|---|---|---|
| `helio-gi-api` | `api/` | The GI plugin's public contract: the probe clipmap's layout, the rays, hits and irradiance it publishes, and the WGSL that reads them (a terrain that traces the probes reads it). |
| `gi_plugin` | `./` (`src/`) | The plugin's root crate: the one place it registers everything it contributes (`src/runtime.rs`: the feature set, its configuration from the project's GI settings, and the settings group). GI has no SceneDB component and no editor UI, so it has no component or editor crate. |
| `gi-examples` | `examples/` | The GI plugin's standalone hosts on Helio: the indoor cathedral lit by the probes |
| `helio-pass-gi` | `feature/` | Helio world-space diffuse global illumination: a probe clipmap traced by the scene's tracers |

- **Root crate** `gi_plugin`: the only crate that registers anything (the
  Helio feature and its settings group; GI registers nothing for the
  editor). A game links it without `editor`; the editor links it with. Other
  plugins never depend on it, only on `helio-gi-api`.
- **Requires**: nothing. A project enables or disables the plugin in its
  `Pulsar.toml` (`[plugins]`); disabled, the kind is absent and the frame
  draws without it.
- **Reads other kinds through**: `helio-light-api` (lights); `helio-mats` (materials); `helio-mesh-api` (meshes).
- **Its standalone hosts also compose**: `helio-light-api, helio-light-host` (lights); `helio-mats` (materials); `helio-mesh-api, helio-mesh-host, helio-pass-forward-lit, helio-vg-stream` (meshes); `helio-pass-sky` (sky) (demos and GPU tests
  draw a whole scene; the plugin itself does not depend on these).

## Building and testing

Like [Plugin_Physics](https://github.com/Far-Beyond-Pulsar/Plugin_Physics), this plugin's crates are members
of Pulsar-Native's workspace: their dependencies are `workspace = true`, and
Helio comes from Pulsar-Native's `crates/renderer/helio` submodule. It builds
only inside a Pulsar-Native checkout:

```bash
git clone --recursive https://github.com/Far-Beyond-Pulsar/Pulsar-Native
cd Pulsar-Native
cargo check -p helio-gi-api -p gi_plugin --features gi_plugin/editor -p gi-examples -p helio-pass-gi --tests
cargo test -p helio-gi-api -p gi_plugin -p helio-pass-gi
```

Suites that render real frames need a GPU; CI leaves them out by name
(Pulsar-Native's `.config/nextest.toml`, `ci` profile). This repository's CI
checks the plugin inside a Pulsar-Native checkout: rustfmt, clippy and the
`ci` profile's tests.

A change that spans this plugin and Pulsar-Native (or Helio) merges
dependency-first: merge here, re-pin `plugins/vendor/gi` in Pulsar-Native to
this repository's `main`, then merge the Pulsar-Native PR (Pulsar-Native's
`.agents/CRATES.md`, "Submodule pins").

## Architecture

- [Every object kind is a plugin](https://github.com/Far-Beyond-Pulsar/Helio/blob/main/docs/architecture/plugin-migration.md): the
  rule, what Helio is, the plugin table and the "As built" notes of each
  phase.
- [The layered rewrite](https://github.com/Far-Beyond-Pulsar/Helio/blob/main/docs/architecture/layered-rewrite.md): layers, buffer
  contracts, extension points and why the renderer has this shape.
- [The `helio-architecture` skill](https://github.com/Far-Beyond-Pulsar/Helio/blob/main/.agents/skills/helio-architecture/SKILL.md): required reading for any change
  here (see `AGENTS.md`).
- Pulsar-Native's `.agents/PLUGIN_SYSTEM.md` and `.agents/CRATES.md`: plugin
  descriptors, the installed set and the submodule pins.

History before 2026-11-02 (when the plugin moved into Pulsar-Native) is in
Helio's repository; this repository's history starts with that move
(`git subtree split` of `plugins/vendor/gi`).

## License

MIT, as the rest of Pulsar (`LICENSE.md`).
