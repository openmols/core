//! openmols: drug-discovery workbench.
//!
//! Usage: `openmols [project-folder | structure-file]`

mod chat;
mod config;
mod jobs_ui;
mod tools;
mod viewer;
mod workspace;

use gpui::{App, AppContext, Application, Bounds, WindowBounds, WindowOptions, px, size};
use gpui_component::{Root, Theme, ThemeMode};
use std::path::PathBuf;
use workspace::Workspace;

fn main() {
    let initial: Option<PathBuf> = std::env::args_os().nth(1).map(PathBuf::from);

    Application::new().run(move |cx: &mut App| {
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
