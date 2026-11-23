//! Check that an endpoint streams text and tool calls.
//!
//! cargo run -p llm --example probe -- <base_url> <model>

use anyhow::Result;
use llm::{Message, Provider, ToolSpec};
use serde_json::json;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let base_url = args.next().unwrap_or_else(|| "http://localhost:1234/v1".into());
    let model = args.next().unwrap_or_default();
    let provider = Provider { name: "probe".into(), base_url, api_key: None, api_key_env: Some("OPENROUTER_API_KEY".into()), model };
    let tools = vec![ToolSpec {
        name: "select_binding_site".into(),
        description: "Select the residues around a ligand".into(),
        parameters: json!({"type": "object", "properties": {"ligand": {"type": "string"}, "cutoff": {"type": "number"}}}),
    }];
    let messages = vec![
        Message::system("You control a molecular viewer through tools."),
        Message::user("Select the binding site of ligand MK1 with a 4.5 Å cutoff."),
    ];
    let reply = llm::chat(&provider, &messages, &tools, |e| eprint!("{e:?} "))?;
    println!("\ncontent: {:?}\ntool calls: {:?}", reply.content, reply.tool_calls);
    Ok(())
}
