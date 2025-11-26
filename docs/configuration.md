# Configuration

All configuration is managed via environment variables in the `.env` file.

## Environment Variables

```env
# Required
GOOGLE_API_KEY=your-google-ai-api-key-here

# Database (optional)
USE_DATABASE=false
DATABASE_URL=sqlite:///storyland_sessions.db

# Session (optional)
SESSION_MAX_EVENTS=20

# Context (optional)
MAX_CONTEXT_TOKENS=30000

# Model (optional)
MODEL_NAME=gemini-2.0-flash-lite

# Workflow execution (optional)
WORKFLOW_TIMEOUT=300    # Max seconds for workflow (default: 300)
AGENT_TIMEOUT=60        # Max seconds per agent (default: 60)

# Logging (optional)
LOG_LEVEL=INFO
ENABLE_ADK_DEBUG=false  # Enable DEBUG for ADK internal logs
```

## Configuration Reference

### Required Settings

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google AI API key from [AI Studio](https://aistudio.google.com/app/apikey) | `AIza...` |

### Database Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_DATABASE` | Enable SQLite persistence | `false` |
| `DATABASE_URL` | SQLite database path | `sqlite:///storyland_sessions.db` |

### Session Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `SESSION_MAX_EVENTS` | Max events to keep in session | `20` |

### Context Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_CONTEXT_TOKENS` | Max tokens for context window | `30000` |

### Model Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Gemini model to use | `gemini-2.0-flash-lite` |

### Workflow Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKFLOW_TIMEOUT` | Max seconds for entire workflow | `300` |
| `AGENT_TIMEOUT` | Max seconds per individual agent | `60` |

### Logging Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `ENABLE_ADK_DEBUG` | Enable detailed ADK internal logging | `false` |

## Getting Your API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and add it to your `.env` file

The free tier includes:
- 15 requests per minute (RPM)
- 200 requests per day
- Sufficient for development and testing
