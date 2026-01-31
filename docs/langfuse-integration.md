# Langfuse Integration - Token Usage Tracking

StoryLand AI now includes comprehensive token usage tracking via Langfuse, enabling cost monitoring and observability for Gemini API calls.

## Features

### Token Tracking
- **Input/Output Tokens**: Accurate counts from Gemini API responses
- **Per-Agent Metrics**: Track token usage for each agent call
- **Real-time Monitoring**: View token consumption as workflows execute
- **Cost Calculation**: Automatic cost estimation based on Gemini pricing

### Cost Estimation
Current pricing for Gemini 2.0 Flash (as of Jan 2026):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

### Observability Features
- **Trace Hierarchy**: Top-level traces for workflows, spans for nested agents
- **Tool Tracking**: Monitor tool execution within agent calls
- **Model Call Logging**: Capture all LLM requests and responses
- **Session Statistics**: Aggregate token usage across entire workflow

## Setup

### 1. Install Langfuse Dependency

```bash
pip install -e .
```

The `langfuse>=2.0.0` dependency is included in `pyproject.toml`.

### 2. Configure Langfuse Credentials

Add your Langfuse credentials to `.env`:

```bash
# Langfuse Configuration
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted instance
```

**To get credentials:**
1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a new project
3. Copy the API keys from project settings

### 3. Enable in Your Application

The plugin is automatically initialized if credentials are provided:

```python
from plugins.langfuse_plugin import LangfusePlugin
from common.config import load_config

config = load_config()

# Initialize plugin (auto-disabled if credentials missing)
langfuse_plugin = LangfusePlugin(
    secret_key=config.langfuse_secret_key,
    public_key=config.langfuse_public_key,
    host=config.langfuse_host,
)

# Add to Runner plugins
runner = Runner(
    agent=my_agent,
    plugins=[LoggingPlugin(), langfuse_plugin],
)
```

## Usage

### CLI

Run the CLI as usual - token stats will be displayed at the end:

```bash
python main.py "1984" --author "George Orwell"
```

**Output:**
```
📊 Context: 45 events, ~12000 tokens
💰 Token Usage: 8,432 tokens (input: 5,123, output: 3,309)
   Estimated cost: $0.001378
```

### Streamlit Demo

Token statistics are automatically tracked and included in workflow data:

```bash
streamlit run streamlit_demo.py
```

Token stats are accessible via `workflow_data["token_stats"]`:
```python
{
    "input_tokens": 5123,
    "output_tokens": 3309,
    "total_tokens": 8432,
    "cost_usd": 0.001378
}
```

### Programmatic Access

```python
# Get current session stats
stats = langfuse_plugin.get_session_stats()
print(f"Total tokens: {stats['total_tokens']}")
print(f"Cost: ${stats['cost_usd']:.6f}")

# Reset stats for new session
langfuse_plugin.reset_stats()

# Flush pending events to Langfuse
await langfuse_plugin.flush()
```

## Viewing Data in Langfuse

### Dashboard Access
1. Log in to [cloud.langfuse.com](https://cloud.langfuse.com)
2. Select your project
3. Navigate to **Traces** to see workflow executions

### What You'll See
- **Traces**: One per workflow execution (metadata_extraction, discovery, composition)
- **Generations**: Individual LLM calls with token counts and costs
- **Spans**: Tool executions and nested agent calls
- **Metadata**: Agent names, invocation IDs, model parameters

### Custom Dashboards
Create dashboards to visualize:
- Total token usage over time
- Cost per workflow phase
- Average tokens per agent
- Token efficiency trends

## Architecture

### Event Flow
```
ADK Runner
    ↓
Plugin.on_event()
    ↓
AgentStartEvent → Create Langfuse trace/span
ModelRequestEvent → Start generation tracking
ModelResponseEvent → Extract token usage, calculate cost
AgentCompleteEvent → Finalize trace with totals
```

### Token Extraction
The plugin extracts token counts from Gemini's `usage_metadata`:
```python
response.usage_metadata.prompt_token_count      # Input tokens
response.usage_metadata.candidates_token_count  # Output tokens
response.usage_metadata.total_token_count       # Total tokens
```

### Cost Calculation
```python
input_cost = (input_tokens / 1_000_000) * 0.075
output_cost = (output_tokens / 1_000_000) * 0.30
total_cost = input_cost + output_cost
```

## Optional: Disable Langfuse

To disable Langfuse without removing credentials:

**Option 1:** Don't include the plugin in Runner:
```python
runner = Runner(
    agent=my_agent,
    plugins=[LoggingPlugin()],  # Omit langfuse_plugin
)
```

**Option 2:** Remove credentials from `.env`:
```bash
# Comment out or remove these lines
# LANGFUSE_SECRET_KEY=sk-lf-xxx
# LANGFUSE_PUBLIC_KEY=pk-lf-xxx
# LANGFUSE_HOST=https://cloud.langfuse.com
```

The plugin automatically disables itself if credentials are missing.

## Troubleshooting

### Plugin Not Tracking
- **Check credentials**: Ensure `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` are set
- **Check logs**: Look for `langfuse_enabled` or `langfuse_disabled` log messages
- **Verify installation**: `pip list | grep langfuse` should show `langfuse>=2.0.0`

### Missing Token Counts
- **ADK Web mode**: Plugins are not supported in `adk web` mode (use CLI instead)
- **Gemini response format**: Ensure responses include `usage_metadata`
- **Check Langfuse logs**: Look for warnings about token extraction failures

### Zero Cost Displayed
- Token usage is tracked, but cost calculation may be $0.000000 for very small usage
- Verify pricing constants in `plugins/langfuse_plugin.py` are up to date

## Example Output

### CLI Execution
```bash
$ python main.py "Pride and Prejudice" --author "Jane Austen"

======================================================================
Creating itinerary for: Pride and Prejudice
Author: Jane Austen
======================================================================

Phase 1: Extracting book metadata...
Phase 2: Discovering travel locations...
Phase 3: Creating personalized itinerary...

📊 Context: 52 events, ~15000 tokens
💰 Token Usage: 12,845 tokens (input: 8,234, output: 4,611)
   Estimated cost: $0.002000
```

### Langfuse Trace Example
```
Trace: create_discovery_workflow
├── Generation: gemini-2.0-flash-lite_call
│   ├── Input: 2,345 tokens
│   ├── Output: 1,234 tokens
│   └── Cost: $0.000547
├── Span: tool_search_google
├── Span: city_pipeline
│   └── Generation: gemini-2.0-flash-lite_call
│       ├── Input: 1,234 tokens
│       ├── Output: 567 tokens
│       └── Cost: $0.000262
└── Total: 8,234 input, 4,611 output, $0.002000
```

## Next Steps

1. **Set up alerts**: Configure Langfuse alerts for cost thresholds
2. **Analyze patterns**: Review which agents consume the most tokens
3. **Optimize prompts**: Use Langfuse data to identify verbose prompts
4. **Track experiments**: A/B test different model configurations
5. **Monitor production**: Set up dashboards for production monitoring

## Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Gemini API Pricing](https://ai.google.dev/pricing)
- [Google ADK Plugins Guide](https://github.com/google/adk-plugins)
