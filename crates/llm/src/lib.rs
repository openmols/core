//! Chat-completion client for OpenAI-compatible endpoints.
//!
//! One wire protocol covers OpenRouter, LM Studio, Ollama (`/v1`), vLLM,
//! OpenAI itself and most hosted gateways. Native Anthropic/Gemini clients can
//! be added later behind the same [`Provider`] description.
//!
//! The client is blocking (ureq) and meant to run on a background thread; it
//! reports progress through a callback.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader};
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Provider {
    /// Display name, e.g. "OpenRouter".
    pub name: String,
    /// Base URL including the version prefix, e.g. `https://openrouter.ai/api/v1`.
    pub base_url: String,
    /// API key stored in the config file (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Environment variable to read the API key from (preferred over storing it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Model identifier as the endpoint expects it.
    pub model: String,
}

impl Provider {
    pub fn openrouter() -> Self {
        Self {
            name: "OpenRouter".into(),
            base_url: "https://openrouter.ai/api/v1".into(),
            api_key: None,
            api_key_env: Some("OPENROUTER_API_KEY".into()),
            model: "anthropic/claude-sonnet-4.5".into(),
        }
    }

    pub fn lm_studio() -> Self {
        Self {
            name: "LM Studio".into(),
            base_url: "http://localhost:1234/v1".into(),
            api_key: None,
            api_key_env: None,
            model: String::new(),
        }
    }

    pub fn ollama() -> Self {
        Self {
            name: "Ollama".into(),
            base_url: "http://localhost:11434/v1".into(),
            api_key: None,
            api_key_env: None,
            model: "qwen3".into(),
        }
    }

    pub fn defaults() -> Vec<Provider> {
        vec![Self::openrouter(), Self::lm_studio(), Self::ollama()]
    }

    fn key(&self) -> Option<String> {
        self.api_key
            .clone()
            .filter(|k| !k.is_empty())
            .or_else(|| self.api_key_env.as_ref().and_then(|v| std::env::var(v).ok()))
            .filter(|k| !k.is_empty())
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url.trim_end_matches('/'), path)
    }

    fn agent(&self) -> ureq::Agent {
        ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(10)))
            .http_status_as_error(false)
            .build()
            .into()
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON-encoded arguments, exactly as the model produced them.
    pub arguments: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: Some(call_id.into()),
            ..Self::new(Role::Tool, content)
        }
    }
    fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    fn to_wire(&self) -> Value {
        let mut m = json!({ "role": self.role, "content": self.content });
        if !self.tool_calls.is_empty() {
            m["tool_calls"] = self
                .tool_calls
                .iter()
                .map(|c| {
                    json!({
                        "id": c.id,
                        "type": "function",
                        "function": { "name": c.name, "arguments": c.arguments },
                    })
                })
                .collect();
        }
        if let Some(id) = &self.tool_call_id {
            m["tool_call_id"] = json!(id);
        }
        m
    }
}

/// A function the model may call. `parameters` is a JSON Schema object.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// A piece of assistant text.
    Text(String),
    /// Reasoning tokens, for models that expose them.
    Reasoning(String),
}

/// Stream one assistant turn. Calls `on_event` as text arrives and returns the
/// complete assistant message, including any tool calls it requested.
pub fn chat(
    provider: &Provider,
    messages: &[Message],
    tools: &[ToolSpec],
    mut on_event: impl FnMut(Event),
) -> Result<Message> {
    if provider.model.trim().is_empty() {
        bail!("no model selected for {}", provider.name);
    }
    let mut body = json!({
        "model": provider.model,
        "messages": messages.iter().map(Message::to_wire).collect::<Vec<_>>(),
        "stream": true,
    });
    if !tools.is_empty() {
        body["tools"] = tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();
    }

    let mut request = provider
        .agent()
        .post(&provider.url("chat/completions"))
        .header("Content-Type", "application/json")
        .header("X-Title", "openmols");
    if let Some(key) = provider.key() {
        request = request.header("Authorization", &format!("Bearer {key}"));
    }
    let response = request
        .send(body.to_string())
        .with_context(|| format!("connecting to {} ({})", provider.name, provider.base_url))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.into_body().read_to_string().unwrap_or_default();
        bail!("{} returned {status}: {}", provider.name, error_message(&text));
    }

    let mut text = String::new();
    // Tool calls stream as fragments keyed by index.
    let mut calls: Vec<ToolCall> = Vec::new();
    let reader = BufReader::new(response.into_body().into_reader());
    for line in reader.lines() {
        let line = line.context("reading stream")?;
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if let Some(err) = chunk.get("error") {
            bail!("{}: {}", provider.name, error_message(&err.to_string()));
        }
        let Some(delta) = chunk.pointer("/choices/0/delta") else {
            continue;
        };
        for key in ["reasoning_content", "reasoning"] {
            if let Some(r) = delta.get(key).and_then(Value::as_str)
                && !r.is_empty()
            {
                on_event(Event::Reasoning(r.to_string()));
            }
        }
        if let Some(t) = delta.get("content").and_then(Value::as_str)
            && !t.is_empty()
        {
            text.push_str(t);
            on_event(Event::Text(t.to_string()));
        }
        if let Some(deltas) = delta.get("tool_calls").and_then(Value::as_array) {
            for d in deltas {
                let index = d.get("index").and_then(Value::as_u64).unwrap_or(calls.len() as u64) as usize;
                while calls.len() <= index {
                    calls.push(ToolCall {
                        id: String::new(),
                        name: String::new(),
                        arguments: String::new(),
                    });
                }
                let call = &mut calls[index];
                if let Some(id) = d.get("id").and_then(Value::as_str) {
                    call.id = id.to_string();
                }
                if let Some(name) = d.pointer("/function/name").and_then(Value::as_str) {
                    call.name.push_str(name);
                }
                if let Some(args) = d.pointer("/function/arguments").and_then(Value::as_str) {
                    call.arguments.push_str(args);
                }
            }
        }
    }

    for (i, call) in calls.iter_mut().enumerate() {
        if call.id.is_empty() {
            call.id = format!("call_{i}");
        }
        if call.arguments.trim().is_empty() {
            call.arguments = "{}".into();
        }
    }
    calls.retain(|c| !c.name.is_empty());

    Ok(Message {
        role: Role::Assistant,
        content: text,
        tool_calls: calls,
        tool_call_id: None,
    })
}

/// List model ids offered by the endpoint (`GET /models`).
pub fn list_models(provider: &Provider) -> Result<Vec<String>> {
    let mut request = provider.agent().get(&provider.url("models"));
    if let Some(key) = provider.key() {
        request = request.header("Authorization", &format!("Bearer {key}"));
    }
    let response = request
        .call()
        .with_context(|| format!("connecting to {} ({})", provider.name, provider.base_url))?;
    let status = response.status();
    let text = response.into_body().read_to_string()?;
    if !status.is_success() {
        bail!("{} returned {status}: {}", provider.name, error_message(&text));
    }
    let v: Value = serde_json::from_str(&text)?;
    let mut ids: Vec<String> = v
        .get("data")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    ids.sort();
    Ok(ids)
}

fn error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message")
                .or_else(|| v.get("message"))
                .or_else(|| v.get("error"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| body.chars().take(300).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_format_includes_tool_calls() {
        let m = Message {
            role: Role::Assistant,
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "c1".into(),
                name: "reset_view".into(),
                arguments: "{}".into(),
            }],
            tool_call_id: None,
        };
        let w = m.to_wire();
        assert_eq!(w["tool_calls"][0]["function"]["name"], "reset_view");
        let t = Message::tool_result("c1", "ok").to_wire();
        assert_eq!(t["role"], "tool");
        assert_eq!(t["tool_call_id"], "c1");
    }

    #[test]
    fn extracts_error_messages() {
        assert_eq!(error_message(r#"{"error":{"message":"bad key"}}"#), "bad key");
        assert_eq!(error_message("plain"), "plain");
    }
}
