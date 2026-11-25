//! The LLM chat drawer: provider/model picker, conversation and the agent loop
//! (stream → execute tool calls → feed results back → repeat).

use crate::tools;
use crate::workspace::Workspace;
use futures::StreamExt;
use gpui::{
    AppContext, Context, Entity, IntoElement, ParentElement, Render, ScrollHandle,
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
    messages: Vec<Message>,
    conversation: Option<Id>,
    streaming: String,
    busy: bool,
    error: Option<String>,
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
        let subscriptions = vec![
            cx.subscribe_in(&input, window, |this, _, event: &InputEvent, window, cx| {
                if let InputEvent::PressEnter { secondary: true } = event {
                    this.send(window, cx);
                }
            }),
        ];
        Self {
            workspace,
            input,
            messages: Vec::new(),
            conversation: None,
            streaming: String::new(),
            busy: false,
            error: None,
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
                    // Tools that run a container finish asynchronously.
                    let outcome: anyhow::Result<String> = if call.name == "run_frodock" {
                        let started = workspace
                            .update(cx, |ws, cx| ws.start_frodock_tool(&args, cx))
                            .unwrap_or_else(|_| Err(anyhow::anyhow!("workspace closed")));
                        match started {
                            Ok(rx) => rx.await.unwrap_or_else(|_| Err(anyhow::anyhow!("job dropped"))),
                            Err(e) => Err(e),
                        }
                    } else if call.name == "run_autodock" {
                        let started = workspace
                            .update(cx, |ws, cx| ws.start_autodock_tool(&args, cx))
                            .unwrap_or_else(|_| Err(anyhow::anyhow!("workspace closed")));
                        match started {
                            Ok(rx) => rx.await.unwrap_or_else(|_| Err(anyhow::anyhow!("job dropped"))),
                            Err(e) => Err(e),
                        }
                    } else if call.name == "detect_pockets_ensemble" {
                        let started = workspace
                            .update(cx, |ws, cx| ws.start_ensemble_tool(&args, cx))
                            .unwrap_or_else(|_| Err(anyhow::anyhow!("workspace closed")));
                        match started {
                            Ok(rx) => rx.await.unwrap_or_else(|_| Err(anyhow::anyhow!("job dropped"))),
                            Err(e) => Err(e),
                        }
                    } else if call.name == "detect_cavities" {
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
                            Ok((bytes, extension)) => workspace
                                .update(cx, |ws, cx| {
                                    ws.add_downloaded_structure(&id, &bytes, extension, cx)
                                })
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
        let chip = self
            .provider(cx)
            .map(|p| {
                if p.model.is_empty() { p.name } else { format!("{} · {}", p.name, p.model) }
            })
            .unwrap_or_else(|| "No provider — open Settings".into());
        let workspace = self.workspace.clone();

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
            .child(
                Button::new("provider-chip")
                    .label(chip)
                    .small()
                    .on_click(move |_, window, cx| {
                        workspace
                            .update(cx, |ws, cx| ws.open_settings(crate::settings::Tab::Llm, window, cx))
                            .ok();
                    }),
            );

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
