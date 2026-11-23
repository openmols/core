//! The LLM chat drawer: provider/model picker, conversation and the agent loop
//! (stream → execute tool calls → feed results back → repeat).

use crate::tools;
use crate::workspace::Workspace;
use futures::StreamExt;
use gpui::{
    AppContext, Context, Entity, IntoElement, ParentElement, Render, ScrollHandle, SharedString,
    Styled, Subscription, Task, WeakEntity, Window, div, prelude::*, px, rgb,
};
use gpui_component::{
    Sizable,
    button::{Button, ButtonVariants},
    input::{Input, InputEvent, InputState},
    text::TextView,
};
use llm::{Event, Message, Provider, Role};
use project::Id;
use serde_json::{Value, json};

const MAX_TOOL_ROUNDS: usize = 12;

pub struct ChatPanel {
    workspace: WeakEntity<Workspace>,
    input: Entity<InputState>,
    model_input: Entity<InputState>,
    messages: Vec<Message>,
    conversation: Option<Id>,
    streaming: String,
    busy: bool,
    error: Option<String>,
    models: Vec<String>,
    show_models: bool,
    scroll: ScrollHandle,
    task: Option<Task<()>>,
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

impl ChatPanel {
    pub fn new(workspace: WeakEntity<Workspace>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| {
            InputState::new(window, cx)
                .multi_line(true)
                .auto_grow(2, 8)
                .placeholder("Ask about the structure, or tell me what to show…  (Ctrl+Enter to send)")
        });
        let model_input = cx.new(|cx| InputState::new(window, cx).placeholder("model id"));
        let subscriptions = vec![
            cx.subscribe_in(&input, window, |this, _, event: &InputEvent, window, cx| {
                if let InputEvent::PressEnter { secondary: true } = event {
                    this.send(window, cx);
                }
            }),
            cx.subscribe_in(&model_input, window, |this, input, event: &InputEvent, _, cx| {
                if let InputEvent::Change = event {
                    let model = input.read(cx).value().to_string();
                    this.update_provider(cx, |p| p.model = model);
                }
            }),
        ];
        Self {
            workspace,
            input,
            model_input,
            messages: Vec::new(),
            conversation: None,
            streaming: String::new(),
            busy: false,
            error: None,
            models: Vec::new(),
            show_models: false,
            scroll: ScrollHandle::new(),
            task: None,
            _subscriptions: subscriptions,
        }
    }

    fn provider(&self, cx: &Context<Self>) -> Option<Provider> {
        self.workspace
            .upgrade()
            .and_then(|ws| ws.read(cx).config.provider().cloned())
    }

    fn update_provider(&mut self, cx: &mut Context<Self>, f: impl FnOnce(&mut Provider)) {
        if let Some(ws) = self.workspace.upgrade() {
            ws.update(cx, |ws, _| {
                let i = ws.config.active_provider;
                if let Some(p) = ws.config.providers.get_mut(i) {
                    f(p);
                    ws.config.save();
                }
            });
        }
    }

    fn select_provider(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(ws) = self.workspace.upgrade() {
            ws.update(cx, |ws, _| {
                ws.config.active_provider = index;
                ws.config.save();
            });
        }
        let model = self.provider(cx).map(|p| p.model).unwrap_or_default();
        self.model_input.update(cx, |i, cx| i.set_value(model, window, cx));
        self.models.clear();
        self.show_models = false;
        self.error = None;
        cx.notify();
    }

    /// Show `model` in the model field (called by the workspace after config loads).
    pub fn set_model_field(&mut self, model: String, window: &mut Window, cx: &mut Context<Self>) {
        self.model_input.update(cx, |i, cx| i.set_value(model, window, cx));
    }

    fn fetch_models(&mut self, cx: &mut Context<Self>) {
        let Some(provider) = self.provider(cx) else { return };
        self.error = None;
        cx.spawn(async move |this, cx| {
            let result = blocking(move || llm::list_models(&provider)).await;
            this.update(cx, |this, cx| {
                match result {
                    Ok(models) => {
                        this.models = models;
                        this.show_models = true;
                    }
                    Err(e) => this.error = Some(format!("{e:#}")),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn pick_model(&mut self, model: String, window: &mut Window, cx: &mut Context<Self>) {
        self.update_provider(cx, |p| p.model = model.clone());
        self.model_input.update(cx, |i, cx| i.set_value(model, window, cx));
        self.show_models = false;
        cx.notify();
    }

    /// Show a stored conversation (the workspace reads it from the project).
    /// Must not read the workspace: it is called while the workspace is updating.
    pub fn show_conversation(&mut self, conversation: Option<Id>, messages: Vec<Message>, cx: &mut Context<Self>) {
        self.task = None;
        self.busy = false;
        self.streaming.clear();
        self.error = None;
        self.conversation = conversation;
        self.messages = messages;
        self.scroll.scroll_to_bottom();
        cx.notify();
    }

    pub fn new_conversation(&mut self, cx: &mut Context<Self>) {
        self.task = None;
        self.busy = false;
        self.streaming.clear();
        self.messages.clear();
        self.conversation = None;
        self.error = None;
        cx.notify();
    }

    /// Append to the transcript and persist to the project (if one is open).
    fn push(&mut self, message: Message, cx: &mut Context<Self>) {
        if let Some(ws) = self.workspace.upgrade()
            && let Some(project) = &ws.read(cx).project
        {
            if self.conversation.is_none() && message.role == Role::User {
                let title: String = message.content.chars().take(60).collect();
                self.conversation = project.create_conversation(&title).ok().map(|c| c.id);
            }
            if let Some(id) = self.conversation {
                project
                    .append_message(id, &serde_json::to_value(&message).unwrap_or(Value::Null))
                    .ok();
            }
        }
        self.messages.push(message);
        self.scroll.scroll_to_bottom();
        cx.notify();
    }

    fn send(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let text = self.input.read(cx).value().trim().to_string();
        if text.is_empty() || self.busy {
            return;
        }
        self.input.update(cx, |i, cx| i.set_value("", window, cx));
        self.send_text(text, cx);
    }

    /// Send a message as if the user typed it.
    pub fn send_text(&mut self, text: String, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        self.error = None;
        self.push(Message::user(text), cx);
        self.run(cx);
    }

    fn stop(&mut self, cx: &mut Context<Self>) {
        self.task = None;
        self.busy = false;
        if !self.streaming.is_empty() {
            let partial = std::mem::take(&mut self.streaming);
            self.push(
                Message {
                    role: Role::Assistant,
                    content: partial,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                },
                cx,
            );
        }
        cx.notify();
    }

    fn request_messages(&self) -> Vec<Message> {
        let mut messages = vec![Message::system(tools::SYSTEM_PROMPT)];
        messages.extend(self.messages.iter().cloned());
        messages
    }

    fn run(&mut self, cx: &mut Context<Self>) {
        let Some(provider) = self.provider(cx) else {
            self.error = Some("No LLM provider configured".into());
            return;
        };
        self.busy = true;
        let workspace = self.workspace.clone();
        self.task = Some(cx.spawn(async move |this, cx| {
            let tools = tools::specs();
            for _ in 0..MAX_TOOL_ROUNDS {
                let Ok(history) = this.update(cx, |this, _| this.request_messages()) else { return };
                let (tx, mut rx) = futures::channel::mpsc::unbounded::<Event>();
                let (p, t) = (provider.clone(), tools.clone());
                // Stream on a dedicated thread; text deltas arrive through `rx`.
                let result_rx = {
                    let (done_tx, done_rx) = futures::channel::oneshot::channel();
                    std::thread::spawn(move || {
                        let result = llm::chat(&p, &history, &t, |ev| {
                            tx.unbounded_send(ev).ok();
                        });
                        let _ = done_tx.send(result);
                    });
                    done_rx
                };
                while let Some(event) = rx.next().await {
                    if let Event::Text(t) = event {
                        this.update(cx, |this, cx| {
                            this.streaming.push_str(&t);
                            this.scroll.scroll_to_bottom();
                            cx.notify();
                        })
                        .ok();
                    }
                }
                let result = result_rx.await.unwrap_or_else(|_| Err(anyhow::anyhow!("request thread died")));
                let message = match result {
                    Ok(m) => m,
                    Err(e) => {
                        this.update(cx, |this, cx| {
                            this.streaming.clear();
                            this.error = Some(format!("{e:#}"));
                            cx.notify();
                        })
                        .ok();
                        break;
                    }
                };
                let calls = message.tool_calls.clone();
                this.update(cx, |this, cx| {
                    this.streaming.clear();
                    this.push(message, cx);
                })
                .ok();
                if calls.is_empty() {
                    break;
                }
                for call in calls {
                    let args: Value = serde_json::from_str(&call.arguments).unwrap_or_else(|_| json!({}));
                    let outcome: anyhow::Result<String> = if call.name == "detect_cavities" {
                        let started = workspace
                            .update(cx, |ws, cx| ws.start_kvfinder_tool(&args, cx))
                            .unwrap_or_else(|_| Err(anyhow::anyhow!("workspace closed")));
                        match started {
                            Ok(rx) => rx.await.unwrap_or_else(|_| Err(anyhow::anyhow!("job dropped"))),
                            Err(e) => Err(e),
                        }
                    } else if call.name == "fetch_pdb" {
                        let id = args["pdb_id"].as_str().unwrap_or_default().to_string();
                        let download_id = id.clone();
                        match blocking(move || tools::download_pdb(&download_id)).await {
                            Ok(bytes) => workspace
                                .update(cx, |ws, cx| ws.add_downloaded_structure(&id, &bytes, cx))
                                .unwrap_or_else(|_| Err(anyhow::anyhow!("workspace closed"))),
                            Err(e) => Err(e),
                        }
                    } else {
                        workspace
                            .update(cx, |ws, cx| ws.run_tool(&call.name, &args, cx))
                            .unwrap_or_else(|_| Err(anyhow::anyhow!("workspace closed")))
                    };
                    let text = match outcome {
                        Ok(s) => s,
                        Err(e) => format!("error: {e:#}"),
                    };
                    this.update(cx, |this, cx| this.push(Message::tool_result(call.id, text), cx))
                        .ok();
                }
            }
            this.update(cx, |this, cx| {
                this.busy = false;
                cx.notify();
            })
            .ok();
        }));
        cx.notify();
    }

    fn render_message(&self, index: usize, message: &Message, window: &mut Window, cx: &mut Context<Self>) -> Option<gpui::AnyElement> {
        let el = match message.role {
            Role::System => return None,
            Role::User => div()
                .flex()
                .justify_end()
                .child(
                    div()
                        .max_w(px(320.0))
                        .px_3()
                        .py_2()
                        .rounded_lg()
                        .bg(rgb(0x24324d))
                        .text_sm()
                        .text_color(rgb(0xe8eefc))
                        .child(message.content.clone()),
                )
                .into_any_element(),
            Role::Assistant => {
                let mut col = div().flex().flex_col().gap_1();
                if !message.content.trim().is_empty() {
                    col = col.child(
                        div()
                            .text_sm()
                            .child(TextView::markdown(("msg", index), message.content.clone(), window, cx).selectable(true)),
                    );
                }
                for call in &message.tool_calls {
                    let args: String = call.arguments.chars().take(120).collect();
                    col = col.child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x8fb8de))
                            .child(format!("⚙ {}({})", call.name, args)),
                    );
                }
                col.into_any_element()
            }
            Role::Tool => {
                let first: String = message.content.lines().next().unwrap_or_default().chars().take(160).collect();
                let is_error = message.content.starts_with("error:");
                div()
                    .pl_3()
                    .text_xs()
                    .text_color(rgb(if is_error { 0xff9b94 } else { 0x7d818c }))
                    .child(format!("↳ {first}"))
                    .into_any_element()
            }
        };
        Some(el)
    }
}

impl Render for ChatPanel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (providers, active) = self
            .workspace
            .upgrade()
            .map(|ws| {
                let c = &ws.read(cx).config;
                (c.providers.iter().map(|p| p.name.clone()).collect::<Vec<_>>(), c.active_provider)
            })
            .unwrap_or_default();

        let mut provider_row = div().flex().flex_wrap().gap_1();
        for (i, name) in providers.into_iter().enumerate() {
            let mut b = Button::new(SharedString::from(format!("provider-{i}")))
                .label(name)
                .small()
                .on_click(cx.listener(move |this, _, window, cx| this.select_provider(i, window, cx)));
            if i == active {
                b = b.primary();
            }
            provider_row = provider_row.child(b);
        }

        let header = div()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .border_b_1()
            .border_color(rgb(0x2a2d35))
            .child(
                div()
                    .flex()
                    .justify_between()
                    .items_center()
                    .child(div().text_sm().font_weight(gpui::FontWeight::SEMIBOLD).child("Assistant"))
                    .child(
                        Button::new("new-chat")
                            .label("New chat")
                            .small()
                            .on_click(cx.listener(|this, _, _, cx| this.new_conversation(cx))),
                    ),
            )
            .child(provider_row)
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(div().flex_1().child(Input::new(&self.model_input).small()))
                    .child(
                        Button::new("models")
                            .label("Models")
                            .small()
                            .on_click(cx.listener(|this, _, _, cx| this.fetch_models(cx))),
                    ),
            )
            .when(self.show_models, |el| {
                let mut list = div()
                    .id("model-list")
                    .flex()
                    .flex_col()
                    .max_h(px(180.0))
                    .overflow_y_scroll()
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0x343842));
                for (i, m) in self.models.iter().enumerate() {
                    let model = m.clone();
                    list = list.child(
                        div()
                            .id(("model", i))
                            .px_2()
                            .py_1()
                            .text_xs()
                            .cursor_pointer()
                            .hover(|s| s.bg(rgb(0x2c3039)))
                            .child(m.clone())
                            .on_click(cx.listener(move |this, _, window, cx| this.pick_model(model.clone(), window, cx))),
                    );
                }
                el.child(list)
            });

        let mut transcript = div()
            .id("transcript")
            .flex_1()
            .flex()
            .flex_col()
            .gap_3()
            .p_3()
            .overflow_y_scroll()
            .track_scroll(&self.scroll);
        if self.messages.is_empty() && !self.busy {
            transcript = transcript.child(
                div()
                    .text_sm()
                    .text_color(rgb(0x7d818c))
                    .child("Try: “Fetch 1HSG, show the protein as cartoon and select the binding site of MK1”."),
            );
        }
        let messages = self.messages.clone();
        for (i, m) in messages.iter().enumerate() {
            if let Some(el) = self.render_message(i, m, window, cx) {
                transcript = transcript.child(el);
            }
        }
        if !self.streaming.is_empty() {
            transcript = transcript.child(
                div()
                    .text_sm()
                    .child(TextView::markdown("streaming", self.streaming.clone(), window, cx)),
            );
        } else if self.busy {
            transcript = transcript.child(div().text_xs().text_color(rgb(0x7d818c)).child("thinking…"));
        }
        if let Some(e) = &self.error {
            transcript = transcript.child(div().text_xs().text_color(rgb(0xff9b94)).child(e.clone()));
        }

        let footer = div()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .border_t_1()
            .border_color(rgb(0x2a2d35))
            .child(Input::new(&self.input))
            .child(
                div().flex().justify_end().gap_2().child(if self.busy {
                    Button::new("stop")
                        .label("Stop")
                        .small()
                        .on_click(cx.listener(|this, _, _, cx| this.stop(cx)))
                } else {
                    Button::new("send")
                        .label("Send")
                        .primary()
                        .small()
                        .on_click(cx.listener(|this, _, window, cx| this.send(window, cx)))
                }),
            );

        div()
            .flex()
            .flex_col()
            .size_full()
            .child(header)
            .child(transcript)
            .child(footer)
    }
}
