# Contributing

Thanks for helping. This is the short version of how the project works.

## Build and test

```sh
cargo build --workspace
cargo test --workspace
```

GPU tests (`crates/render/tests/overlay.rs`) render headlessly and skip themselves when
there is no adapter. Set `OPENMOLS_SKIP_GPU_TESTS=1` to skip them explicitly (CI does).
Tests that need Docker or the network are `#[ignore]`d; run them with `cargo test -- --ignored`.

If the app panics, the backtrace is written to `crash.log` in the config folder
(`%APPDATA%\openmols` on Windows, `~/.config/openmols` elsewhere).

## Conventions

- Every parser has a unit test against a **real captured output file** in
  `crates/<crate>/tests/fixtures/`. Please do not hand-write expected output.
- `crates/render` must not depend on a UI toolkit.
- External tools run in Docker; they are never linked in.
- UI code must not read an entity that is already being updated: dialogs get their data
  as arguments when opened (see `open_settings`, `open_job_dialog`).
- Keep the claims in docs and the UI to what was checked. Say what is not covered.

## Adding a tool (a worked recipe)

AutoDock was added this way; `crates/jobs/src/autodock.rs` and `crates/app/src/autodock_ui.rs`
are the template.

1. **Image.** `docker/<tool>/Dockerfile`, built from packages that can be redistributed
   (conda-forge, bioconda, apt). If the software cannot be redistributed, build the image
   from a folder the user supplies, as FRODOCK does (`needs_context`). Register it in
   `crates/jobs/src/lib.rs` (`ToolImage`, `TOOLS`), give redistributable images a `registry`
   name, add the tool to the matrix in `.github/workflows/containers.yml` (a test keeps the
   names and `IMAGE_VERSION` in step), and put OCI `source` and `licenses` labels in the
   Dockerfile. Bump the `:N` tag everywhere when a Dockerfile changes.
2. **Command builder and parser** in `crates/jobs/src/<tool>.rs`.
   `command(settings) -> Result<String>` returns the shell script run in the job folder;
   refuse bad input there (it ends up in a shell). Write `parse_*` against a real output
   file and commit that file as a fixture.
3. **Check it in the container first.** Run the generated script on a case with a known
   answer (a crystal complex, say) before building any UI. Report what you compared and how.
4. **App side.** A `start_<tool>` that exports inputs and calls `run_job`, a `finish_<tool>`
   that parses results and adds structures to the scene, and a dialog in `job_dialogs.rs`.
   Saved defaults live in `Config.job_defaults`.
5. **Assistant tool.** Add a spec and a dispatch arm in `crates/app/src/tools.rs`
   (jobs that take minutes are dispatched in `chat.rs`).
6. Update `ROADMAP.md` and `docs/LICENSES.md`.

## Releases

Pushing a tag like `v0.1.0` runs `.github/workflows/release.yml`, which builds the app for
Windows, macOS (Apple silicon) and Linux, packages it, and attaches the archives and a
`SHA256SUMS.txt` to a GitHub Release. Before tagging, bump `version` in the root `Cargo.toml`
and the version string in the macOS `Info.plist` step of the workflow. "Run workflow" on the
Release workflow builds the same archives without publishing, which is a good dry run.

## Pull requests

Small, focused changes with a line on how you checked them. Screenshots help for anything visual.
Open an issue first for large changes.
