//! Settings dialog: LLM providers, external tools, general info.
//!
//! Opened from the toolbar gear (or the chat's provider chip). Edits write
//! straight into `Workspace.config` and save, so there is no Apply step.

use crate::workspace::{Workspace, muted};
use gpui::{
    AppContext, Context, Entity, IntoElement, ParentElement, Render, SharedString, Styled, Subscription, WeakEntity,
    Window, div, prelude::*, px, rgb,
};
use gpui_component::{
    Disableable, Sizable,
    button::{Button, ButtonVariants},
    input::{Input, InputEvent, InputState},
};
use llm::Provider;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Tab {
    Llm,
    Tools,
    General,
}

pub struct SettingsView {
    workspace: WeakEntity<Workspace>,
    tab: Tab,
    /// Provider being edited (index into `config.providers`).
    selected: usize,
    name: Entity<InputState>,
    base_url: Entity<InputState>,
    model: Entity<InputState>,
    api_key: Entity<InputState>,
    api_key_env: Entity<InputState>,
    models: Vec<String>,
    /// Result of the last "Test connection" / "Fetch models".
    note: Option<Result<String, String>>,
    _subscriptions: Vec<Subscription>,
}

/// Run a blocking closure on its own thread and await the result.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.await.expect("worker thread panicked")
}

impl SettingsView {
    pub fn new(
        workspace: WeakEntity<Workspace>,
        tab: Tab,
        selected: usize,
        initial: Option<Provider>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let field = |window: &mut Window, cx: &mut Context<Self>, placeholder: &'static str| {
            cx.new(|cx| InputState::new(window, cx).placeholder(placeholder))
        };
        let name = field(window, cx, "Display name");
        let base_url = field(window, cx, "https://host/v1");
        let model = field(window, cx, "model id");
        let api_key = cx.new(|cx| InputState::new(window, cx).placeholder("API key").masked(true));
        let api_key_env = field(window, cx, "ENV_VAR_NAME (optional)");

        // One subscription per field: write the value into the provider as it changes.
        type Setter = fn(&mut Provider, String);
        let fields: [(&Entity<InputState>, Setter); 5] = [
            (&name, |p, v| p.name = v),
            (&base_url, |p, v| p.base_url = v),
            (&model, |p, v| p.model = v),
            (&api_key, |p, v| p.api_key = Some(v).filter(|v| !v.is_empty())),
            (&api_key_env, |p, v| p.api_key_env = Some(v).filter(|v| !v.is_empty())),
        ];
        let subscriptions = fields
            .into_iter()
            .map(|(input, set)| {
                cx.subscribe_in(input, window, move |this, input, event: &InputEvent, _, cx| {
                    if let InputEvent::Change = event {
                        let value = input.read(cx).value().to_string();
                        this.edit_provider(cx, |p| set(p, value));
                    }
                })
            })
            .collect();

        let mut this = Self {
            workspace,
            tab,
            selected,
            name,
            base_url,
            model,
            api_key,
            api_key_env,
            models: Vec::new(),
            note: None,
            _subscriptions: subscriptions,
        };
        this.show_provider(initial, window, cx);
        this
    }

    fn provider(&self, cx: &Context<Self>) -> Option<Provider> {
        let ws = self.workspace.upgrade()?;
        ws.read(cx).config.providers.get(self.selected).cloned()
    }

    fn edit_provider(&mut self, cx: &mut Context<Self>, f: impl FnOnce(&mut Provider)) {
        let Some(ws) = self.workspace.upgrade() else { return };
        let i = self.selected;
        let changed = ws.update(cx, |ws, cx| {
            let changed = ws.config.providers.get_mut(i).map(f).is_some();
            if changed {
                ws.config.save();
                cx.notify(); // the chat chip shows provider and model
            }
            changed
        });
        if changed {
            cx.notify();
        }
    }

    /// Show the selected provider in the form. Setting an input needs a Window.
    fn load_fields(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let p = self.provider(cx);
        self.show_provider(p, window, cx);
    }

    fn show_provider(&mut self, p: Option<Provider>, window: &mut Window, cx: &mut Context<Self>) {
        let p = p.unwrap_or_else(|| Provider {
            name: String::new(),
            base_url: String::new(),
            api_key: None,
            api_key_env: None,
            model: String::new(),
        });
        let set = |input: &Entity<InputState>, value: String, window: &mut Window, cx: &mut Context<Self>| {
            input.update(cx, |i, cx| i.set_value(value, window, cx));
        };
        set(&self.name, p.name, window, cx);
        set(&self.base_url, p.base_url, window, cx);
        set(&self.model, p.model, window, cx);
        set(&self.api_key, p.api_key.unwrap_or_default(), window, cx);
        set(&self.api_key_env, p.api_key_env.unwrap_or_default(), window, cx);
        self.models.clear();
        self.note = None;
    }

    fn select(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        self.selected = index;
        self.load_fields(window, cx);
        cx.notify();
    }

    fn add_provider(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(ws) = self.workspace.upgrade() else { return };
        let index = ws.update(cx, |ws, _| {
            ws.config.providers.push(Provider {
                name: "New provider".into(),
                base_url: "http://localhost:1234/v1".into(),
                api_key: None,
                api_key_env: None,
                model: String::new(),
            });
            ws.config.save();
            ws.config.providers.len() - 1
        });
        self.select(index, window, cx);
    }

    fn remove_provider(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(ws) = self.workspace.upgrade() else { return };
        let i = self.selected;
        let index = ws.update(cx, |ws, cx| {
            // Keep at least one provider: chat always needs something to talk to.
            if ws.config.providers.len() > 1 && i < ws.config.providers.len() {
                ws.config.providers.remove(i);
                let last = ws.config.providers.len() - 1;
                ws.config.active_provider = ws.config.active_provider.min(last);
                if ws.config.active_provider > i {
                    ws.config.active_provider -= 1;
                }
                ws.config.save();
                cx.notify();
            }
            i.min(ws.config.providers.len() - 1)
        });
        self.select(index, window, cx);
    }

    fn set_active(&mut self, cx: &mut Context<Self>) {
        let Some(ws) = self.workspace.upgrade() else { return };
        let i = self.selected;
        ws.update(cx, |ws, cx| {
            ws.config.active_provider = i;
            ws.config.save();
            cx.notify();
        });
        cx.notify();
    }

    fn fetch_models(&mut self, cx: &mut Context<Self>) {
        let Some(provider) = self.provider(cx) else { return };
        self.note = Some(Ok("contacting the server…".into()));
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = blocking(move || llm::list_models(&provider)).await;
            this.update(cx, |this, cx| {
                match result {
                    Ok(models) => {
                        this.note = Some(Ok(format!("connected · {} models", models.len())));
                        this.models = models;
                    }
                    Err(e) => this.note = Some(Err(format!("{e:#}"))),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn pick_model(&mut self, model: String, window: &mut Window, cx: &mut Context<Self>) {
        self.model.update(cx, |i, cx| i.set_value(model, window, cx));
        self.models.clear();
        cx.notify();
    }

    fn label(text: &'static str) -> gpui::Div {
        div().text_xs().text_color(muted()).child(text)
    }

    fn render_llm(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let (providers, active) = self
            .workspace
            .upgrade()
            .map(|ws| {
                let c = &ws.read(cx).config;
                (c.providers.iter().map(|p| p.name.clone()).collect::<Vec<_>>(), c.active_provider)
            })
            .unwrap_or_default();

        let mut list = div().flex().flex_col().gap_1().w(px(190.0));
        for (i, name) in providers.iter().enumerate() {
            let label = if i == active { format!("● {name}") } else { name.clone() };
            list = list.child(
                div()
                    .id(("settings-provider", i))
                    .px_2()
                    .py_1()
                    .rounded_md()
                    .text_sm()
                    .cursor_pointer()
                    .when(i == self.selected, |d| d.bg(rgb(0x24324d)).text_color(rgb(0xe8eefc)))
                    .hover(|s| s.bg(rgb(0x2c3039)))
                    .child(label)
                    .on_click(cx.listener(move |this, _, window, cx| this.select(i, window, cx))),
            );
        }
        list = list
            .child(
                Button::new("settings-add-provider")
                    .label("+ Add provider")
                    .small()
                    .on_click(cx.listener(|this, _, window, cx| this.add_provider(window, cx))),
            )
            .child(div().text_xs().text_color(muted()).child("● = used by the assistant"));

        let is_active = self.selected == active;
        let mut form = div()
            .flex()
            .flex_col()
            .gap_2()
            .flex_1()
            .child(Self::label("Name"))
            .child(Input::new(&self.name).small())
            .child(Self::label("Base URL (OpenAI-compatible, with /v1)"))
            .child(Input::new(&self.base_url).small())
            .child(Self::label("API key (stored in the OS keychain)"))
            .child(Input::new(&self.api_key).small())
            .child(Self::label("…or read it from this environment variable"))
            .child(Input::new(&self.api_key_env).small())
            .child(Self::label("Model"))
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(div().flex_1().child(Input::new(&self.model).small()))
                    .child(
                        Button::new("settings-fetch-models")
                            .label("Fetch models")
                            .small()
                            .on_click(cx.listener(|this, _, _, cx| this.fetch_models(cx))),
                    ),
            );
        if !self.models.is_empty() {
            let mut models = div()
                .id("settings-models")
                .flex()
                .flex_col()
                .max_h(px(160.0))
                .overflow_y_scroll()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x343842));
            for (i, m) in self.models.iter().enumerate() {
                let model = m.clone();
                models = models.child(
                    div()
                        .id(("settings-model", i))
                        .px_2()
                        .py_1()
                        .text_xs()
                        .cursor_pointer()
                        .hover(|s| s.bg(rgb(0x2c3039)))
                        .child(m.clone())
                        .on_click(cx.listener(move |this, _, window, cx| this.pick_model(model.clone(), window, cx))),
                );
            }
            form = form.child(models);
        }
        match &self.note {
            Some(Ok(msg)) => form = form.child(div().text_xs().text_color(rgb(0x8fd19e)).child(msg.clone())),
            Some(Err(msg)) => form = form.child(div().text_xs().text_color(rgb(0xff9b94)).child(msg.clone())),
            None => {}
        }
        form = form.child(
            div()
                .flex()
                .gap_2()
                .child(
                    Button::new("settings-use")
                        .label(if is_active { "In use" } else { "Use this provider" })
                        .small()
                        .primary()
                        .disabled(is_active)
                        .on_click(cx.listener(|this, _, _, cx| this.set_active(cx))),
                )
                .child(
                    Button::new("settings-remove")
                        .label("Remove")
                        .small()
                        .disabled(providers.len() <= 1)
                        .on_click(cx.listener(|this, _, window, cx| this.remove_provider(window, cx))),
                ),
        );

        div().flex().gap_4().child(list).child(form)
    }

    fn render_tools(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let (docker, frodock) = self
            .workspace
            .upgrade()
            .map(|ws| {
                let ws = ws.read(cx);
                let docker = match &ws.docker {
                    None => "checking…".to_string(),
                    Some(Ok(v)) => format!("Docker {v} ✓"),
                    Some(Err(e)) => format!("unavailable: {e}"),
                };
                (docker, ws.config.frodock_dir.clone())
            })
            .unwrap_or_default();
        let workspace = self.workspace.clone();
        let workspace2 = self.workspace.clone();
        div()
            .flex()
            .flex_col()
            .gap_2()
            .child(Self::label("Docker"))
            .child(div().text_sm().child(docker))
            .child(
                Button::new("settings-recheck-docker")
                    .label("Check again")
                    .small()
                    .on_click(move |_, _, cx| {
                        workspace2.update(cx, |ws, cx| ws.check_docker(cx)).ok();
                    }),
            )
            .child(div().pt_2().child(Self::label("FRODOCK folder (your own copy; the Docker image is built from it)")))
            .child(div().text_sm().child(match frodock {
                Some(dir) => dir.display().to_string(),
                None => "not set".to_string(),
            }))
            .child(
                Button::new("settings-frodock")
                    .label("Choose folder…")
                    .small()
                    .on_click(move |_, _, cx| {
                        workspace.update(cx, |ws, cx| ws.pick_frodock_dir(cx)).ok();
                    }),
            )
    }

    fn render_general(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let (projects_dir, adapter, frame) = self
            .workspace
            .upgrade()
            .map(|ws| {
                let ws = ws.read(cx);
                let viewer = ws.viewer.read(cx);
                (
                    ws.config.projects_dir.display().to_string(),
                    viewer.adapter_name().to_string(),
                    viewer.last_render().as_secs_f64() * 1000.0,
                )
            })
            .unwrap_or_default();
        div()
            .flex()
            .flex_col()
            .gap_2()
            .child(Self::label("New projects are created in"))
            .child(div().text_sm().child(projects_dir))
            .child(div().pt_2().child(Self::label("Renderer")))
            .child(div().text_sm().child(format!("{adapter} · {frame:.2} ms / frame")))
            .child(div().pt_2().child(Self::label("Viewport controls")))
            .child(div().text_sm().child("drag: rotate · right/shift-drag: pan · wheel: zoom · G/R: move/rotate active"))
    }
}

impl Render for SettingsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let tab = |id: &'static str, label: &'static str, which: Tab, current: Tab, cx: &mut Context<Self>| {
            let b = Button::new(id)
                .label(label)
                .small()
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.tab = which;
                    cx.notify();
                }));
            if which == current { b.primary() } else { b.ghost() }
        };
        let current = self.tab;
        let body = match current {
            Tab::Llm => self.render_llm(cx),
            Tab::Tools => self.render_tools(cx),
            Tab::General => self.render_general(cx),
        };
        div()
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(tab("settings-tab-llm", "LLM", Tab::Llm, current, cx))
                    .child(tab("settings-tab-tools", "Tools", Tab::Tools, current, cx))
                    .child(tab("settings-tab-general", "General", Tab::General, current, cx)),
            )
            .child(body)
    }
}

impl Workspace {
    /// Open the settings dialog on `tab`.
    pub fn open_settings(&mut self, tab: Tab, window: &mut Window, cx: &mut Context<Self>) {
        use gpui_component::WindowExt;
        let weak = cx.entity().downgrade();
        let selected = self.config.active_provider;
        let initial = self.config.providers.get(selected).cloned();
        let view = cx.new(|cx| SettingsView::new(weak, tab, selected, initial, window, cx));
        window.open_dialog(cx, move |dialog, _, _| {
            dialog
                .title(SharedString::from("Settings"))
                .w(px(780.0))
                .child(view.clone())
        });
    }
}
