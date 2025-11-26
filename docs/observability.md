# Observability

StoryLand AI provides two modes for observability:

## Development: ADK Web UI

For interactive development and debugging, use the built-in ADK Web UI:

```bash
# Launch via CLI
python main.py --dev

# Or directly
.venv/bin/adk web agents/
```

The Web UI provides:
- Visual agent execution flow
- Request/response inspection
- Built-in DEBUG logging
- Interactive testing

> **Note:** Plugins are NOT supported in ADK web mode.

## Production: ADK LoggingPlugin

For production runs, ADK's built-in `LoggingPlugin` automatically logs agent, tool, and model activity:

```bash
# Normal run (INFO level logging)
python main.py "1984" --author "George Orwell"

# Verbose mode (DEBUG level)
python main.py "1984" -v
```

**Sample output:**
```
🤖 AGENT STARTING
   Agent Name: workflow
   Invocation ID: e54b...
🤖 AGENT STARTING
   Agent Name: book_metadata_pipeline
🔧 TOOL STARTING
   Tool Name: search_book
🔧 TOOL COMPLETED
✅ AGENT COMPLETED
   Agent Name: book_metadata_pipeline
```

**What's logged:**
| Event | Description |
|-------|-------------|
| `🤖 AGENT STARTING/COMPLETED` | Agent execution |
| `🔧 TOOL STARTING/COMPLETED` | Tool invocations |
| `🧠 MODEL REQUEST/RESPONSE` | LLM calls |
| `❌ ERROR` | Failures |

## Configuration

```env
# .env
LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR
ENABLE_ADK_DEBUG=false         # Enable DEBUG for ADK internal logs
```

| Mode | Logging | Plugin | Use Case |
|------|---------|--------|----------|
| `--dev` | DEBUG (ADK Web) | None | Development, debugging |
| Default | INFO | LoggingPlugin | Production |
| `-v` | DEBUG | LoggingPlugin | Troubleshooting |
