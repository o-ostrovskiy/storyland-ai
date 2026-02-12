# Testing

StoryLand AI includes comprehensive testing with pytest for unit and integration tests.

## Unit Tests

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

## Integration Tests

Run integration tests with VCR cassettes (no real API calls):

```bash
# Run all integration tests
.venv/bin/pytest tests/integration/ -v

# Run specific integration test
.venv/bin/pytest tests/integration/test_main_workflow.py -v
```

Integration tests use [VCR.py](https://vcrpy.readthedocs.io/) to record/replay HTTP interactions, eliminating the need for real API calls during testing.

## Quality Evaluation

For automated quality evaluation and monitoring, see the [Evaluation Pipeline](../evaluation/README.md).

The evaluation system:
- Runs complete workflows on test datasets
- Tracks results in Langfuse
- Generates trend reports
- Runs weekly via GitHub Actions
