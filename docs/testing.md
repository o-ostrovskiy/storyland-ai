# Testing & Evaluation

StoryLand AI includes comprehensive testing with pytest unit tests and ADK evaluation framework.

## Unit Tests (pytest)

Run unit tests locally without any API calls:

```bash
# Run all unit tests
.venv/bin/pytest tests/unit/ -v

# Run with coverage
.venv/bin/pytest tests/unit/ --cov=. --cov-report=term-missing

# Run specific test file
.venv/bin/pytest tests/unit/test_models.py -v

# Run specific test class
.venv/bin/pytest tests/unit/test_agents.py::TestEvalWorkflow -v
```

**Test coverage (125 tests total):**
| Module | Tests | Description |
|--------|-------|-------------|
| `test_models.py` | 46 | Pydantic model validation (incl. RegionAnalysis) |
| `test_tools.py` | 16 | Google Books, preferences tools |
| `test_agents.py` | 41 | Agent factory functions (three-phase & eval workflows) |
| `test_services.py` | 16 | Session service, context manager |
| `test_workflow_timeout.py` | 6 | Workflow timeout behavior |

## ADK Evaluation (CLI)

Run agent evaluation with rubric-based scoring:

```bash
# Run single test scenario (fast)
.venv/bin/adk eval agents/storyland single_test \
  --config_file_path tests/evaluation/eval_config.json \
  --print_detailed_results
```

**Note:** The eval workflow (`create_eval_workflow`) includes region analysis but auto-selects all regions since human-in-the-loop interaction is not possible in automated evaluations. The workflow still validates that region grouping works correctly.

**Evaluation rubrics:**
| Rubric | Description |
|--------|-------------|
| `book_relevance` | Locations connected to book's settings, themes, characters |
| `preference_adherence` | Respects user's budget, pace, accessibility preferences |
| `completeness` | Comprehensive itinerary with cities, landmarks, author sites |
| `actionability` | Practical details: times, transport, booking tips |
| `geographical_accuracy` | Real places correctly associated with countries |
| `engagement` | Engaging descriptions capturing literary spirit |

**Configuration files:**
- `tests/evaluation/eval_config.json` - Rubric definitions and judge model settings
- `agents/storyland/single_test.evalset.json` - Test scenario for Pride and Prejudice

## ADK Web UI Evaluation

For interactive evaluation with visual results:

```bash
# Launch ADK Web UI
.venv/bin/adk web agents/

# Open browser to http://localhost:8000
```

**Using the Eval tab:**

1. Navigate to the **Eval** tab in the Web UI
2. Select an eval set from the dropdown (e.g., `single_test`)
3. Click **Run Eval** to execute all test cases
4. View results with:
   - Pass/fail status per test case
   - Rubric scores with rationale
   - Agent response inspection
   - Turn-by-turn execution trace

**Creating custom eval sets:**

Create `.evalset.json` files in `agents/storyland/`:

```json
{
  "eval_set_id": "my_custom_eval",
  "name": "My Custom Evaluation",
  "description": "Test specific scenarios",
  "eval_cases": [
    {
      "eval_id": "test_case_1",
      "conversation": [
        {
          "invocation_id": "turn_1",
          "user_content": {
            "parts": [{"text": "Create a travel itinerary for..."}]
          }
        }
      ],
      "session_input": {
        "app_name": "storyland",
        "user_id": "eval_user",
        "state": {
          "user:preferences": {
            "budget": "moderate",
            "preferred_pace": "relaxed"
          }
        }
      }
    }
  ]
}
```

## Rate Limits

The Gemini free tier has limits (15 RPM, 200 requests/day). For evaluation:
- Use `gemini-2.5-flash-lite` for judge model (lower cost)
- Run single tests during development
- Run full suite when quota is available
