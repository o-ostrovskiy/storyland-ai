# Claude AI Assistant Guidelines for StoryLand AI

## Testing Requirements

**IMPORTANT: Always run tests when finishing a task.**

1. Run `make test` (unit tests) and `make test-integration` (integration tests)
2. Fix any failures before considering the task complete
3. Report test results (passed/failed counts) to the user
4. Never skip or ignore failing tests — fix syntax errors first (they block all tests)

Other commands: `make test-all` (both), `make test-cov` (with coverage).

## Documentation Requirements

**Before every commit, verify documentation is up to date.**

If your changes affect any of the following, update the relevant docs:
- **Project structure** (new files/directories) → update the tree in `README.md`
- **CLI flags or config variables** → update Quick Start / Configuration in `README.md`
- **Agent additions or changes** → update Architecture section in `README.md` and `docs/ARCHITECTURE.md`
- **Evaluation or observability** → update `evaluation/README.md`
- **Test counts** → update the Testing section in `README.md`

Documentation files:
- `README.md` — main project docs (structure, config, architecture, testing)
- `docs/ARCHITECTURE.md` — architecture decision records
- `evaluation/README.md` — evaluation pipeline and Langfuse token/cost tracking

## Project Context

- **Project**: StoryLand AI — Literary travel itinerary generator
- **Framework**: Google ADK (Agent Development Kit)
- **Language**: Python 3.12
- **Key Dependencies**: google-adk, google-genai, pydantic
- **Testing**: pytest with VCR for integration tests

## Code Quality

- Follow existing code style and patterns
- Use structured logging via `common.logging.get_logger()`
- Validate input data using Pydantic models
- Handle errors gracefully with proper error messages

## Security: API Keys & VCR Cassettes

**CRITICAL: Never expose API keys or credentials in code or test cassettes.**

- All API keys MUST be stored in `.env` files, never hardcoded
- VCR cassettes in `tests/integration/conftest.py` must filter sensitive headers:
  ```python
  "filter_headers": ["authorization", "x-goog-api-key"]
  "filter_query_parameters": ["key"]
  ```
- **Before committing cassettes**, always verify no keys leaked:
  ```bash
  grep -r "AIza\|x-goog-api-key" tests/integration/cassettes/
  ```
  Expected: No matches (0 results).
- When adding new API integrations, add their auth headers/parameters to VCR filters and regenerate cassettes with `--vcr-record=all`
