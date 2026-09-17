# Agent Workflow

What happens from a user question to the answer.

```mermaid
flowchart TD
  A["User message<br/>POST /chat (bot) or /chat/web (browser)"] --> B{"Route"}
  B -->|"/chat"| C1["resolve_telegram_caller<br/>X-Bot-Secret + X-Telegram-Id"]
  B -->|"/chat/web"| C2["current_user_id(request)<br/>signed X-User-* headers"]
  C1 --> D["get_active_session_row(user_id)<br/>sessions WHERE is_active"]
  C2 --> D2["get_session(session_id) + ownership check"]
  D --> E{"session has<br/>provider, model, variant?"}
  D2 --> E
  E -->|no| E1["409 missing model config"]
  E -->|yes| F["resolve_model_config(user_provider_id, provider, model, variant, user_id)<br/>api/providers.py"]
  F --> F1{"resolved?"}
  F1 -->|"ValueError"| F2["400 unknown provider/model/variant"]
  F1 -->|"requires_key and no key"| F3["400 no API key on file"]
  F1 -->|ok| G["create_web_session(...) if new web session"]

  G --> H["StreamingResponse(stream(...))<br/>text/event-stream"]
  H --> I["_build_agent<br/>OpenAI / Anthropic / DeepSeek provider + settings"]
  I --> J["load_session(session_id)<br/>sessions.pydantic_message -> ModelMessagesTypeAdapter"]
  J --> K["agent.iter(prompt, message_history=history)"]

  K --> L{"node type"}
  L -->|"ModelRequestNode"| M["yield SSE tool_result for ToolReturnPart<br/>stream TextPart -> message_start / text_delta / message_end"]
  L -->|"CallToolsNode"| N["yield SSE tool for each ToolCallPart<br/>yield SSE thinking for ThinkingPart"]
  L -->|"End"| O["capture result.output"]

  N --> P{"tool name"}
  P -->|"dense / sparse / hybrid_search"| Q["Qdrant retrieval + cross-encoder rerank<br/>see 06-retrieval-pipeline.md"]
  P -->|"get_quran / get_hadith / get_tafsir / get_sunnah"| R["Qdrant scroll by payload ids"]
  P -->|"get_books_*"| S["static book/category lists in search.py"]
  Q --> L
  R --> L
  S --> L

  O --> T["finally: save_session_messages<br/>sessions.pydantic_message = all_messages"]
  T --> U["append_session_messages<br/>INSERT INTO messages (role, content, sequence)"]
  U --> V["yield SSE done {output}"]
  V --> W["Bot: markdownify + split at 4096 chars<br/>Browser: render stream"]
  M --> L

  K -.->|"ModelHTTPError"| X["yield SSE text_delta with provider message, re-raise"]
  K -.->|"CancelledError"| Y["log agent_run_cancelled"]

  subgraph ctx["Context management (capabilities)"]
    Hooks["Hooks: before_tool_execute / before_model_request logging"]
    Clear["ClearToolResults(max_tokens=70000)"]
    Summ["SummarizingCompaction(max_fraction=0.5, keep_messages=30)"]
    Ctx["ReportContextUsage -> context_usage log"]
  end
  I -.-> ctx
```

Notes:

- Tools are a `FunctionToolset` of 11 functions, all from `search.py`; the system prompt is `skills/turath-index-skill.md` (read at import, relative path).
- Two persistence targets per turn: `sessions.pydantic_message` (full pydantic-ai history, the resumption source) and `messages` rows (projected role/content, sequence computed in the INSERT subquery).
- `retries=3` on the Agent; tool errors are returned to the model as `ToolReturnPart`s (`search.py` catches `Exception` and returns `[{"error": ...}]`).
