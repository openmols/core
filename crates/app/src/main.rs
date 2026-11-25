//! openmols: drug-discovery workbench.
//!
//! Usage: `openmols [project-folder | structure-file]`

mod annotations;
mod autodock_ui;
mod chat;
mod config;
mod example;
mod job_dialogs;
mod jobs_ui;
mod measure_panel;
mod settings;
mod tools;
mod viewer;
mod workspace;

use gpui::{App, AppContext, Application, Bounds, WindowBounds, WindowOptions, px, size};
use gpui_component::{Root, Theme, ThemeMode};
use std::path::PathBuf;
use workspace::Workspace;

/// Write panics to `crash.log` beside the config: a GUI app has no console.
fn install_panic_log() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let dir = config::config_dir();
        std::fs::create_dir_all(&dir).ok();
        let entry = format!(
            "{info}\n{}\n\n",
            std::backtrace::Backtrace::force_capture()
        );
        let path = dir.join("crash.log");
        let mut log = std::fs::read_to_string(&path).unwrap_or_default();
        log.push_str(&entry);
        std::fs::write(path, log).ok();
        default(info);
    }));
}

fn main() {
    install_panic_log();
    let initial: Option<PathBuf> = std::env::args_os().nth(1).map(PathBuf::from);

    Application::new().with_assets(gpui_component_assets::Assets).run(move |cx: &mut App| {
        gpui_component::init(cx);
        Theme::change(ThemeMode::Dark, None, cx);

        let bounds = Bounds::centered(None, size(px(1500.0), px(900.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(gpui::TitlebarOptions {
                    title: Some("openmols".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            |window, cx| {
                let workspace = cx.new(|cx| Workspace::new(initial, window, cx));
                cx.new(|cx| Root::new(workspace, window, cx))
            },
        )
        .expect("failed to open window");
        cx.activate(true);
    });
}
