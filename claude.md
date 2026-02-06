# Claude AI Assistant Guidelines for StoryLand AI

This document contains guidelines and requirements for Claude when working on the StoryLand AI codebase.

## Testing Requirements

**IMPORTANT: Always run tests when finishing a task.**

When you complete any code changes, bug fixes, or new features, you MUST:

1. **Run Unit Tests**: Execute `make test` to run all unit tests
2. **Run Integration Tests**: Execute `make test-integration` to run integration tests
3. **Fix Any Failures**: If tests fail, investigate and fix the issues before considering the task complete
4. **Report Results**: Inform the user of test results (passed/failed counts)

### Test Commands

```bash
# Run all unit tests
make test

# Run integration tests
make test-integration

# Run all tests (unit + integration)
make test-all

# Run tests with coverage
make test-cov
```

### When Tests Fail

- Read the error messages carefully
- Fix syntax errors first (they block all tests)
- Update tests if functionality intentionally changed
- Never skip or ignore failing tests
- Ask the user for clarification if unsure about expected behavior

## Project Context

- **Project**: StoryLand AI - Literary travel itinerary generator
- **Framework**: Google ADK (Agent Development Kit)
- **Language**: Python 3.12
- **Key Dependencies**: google-adk, google-genai, pydantic
- **Testing**: pytest with VCR for integration tests

## Code Quality Standards

- Follow existing code style and patterns
- Use structured logging via `common.logging.get_logger()`
- Write docstrings for new functions and classes
- Handle errors gracefully with proper error messages
- Validate input data using Pydantic models

## Security Guidelines

**CRITICAL: Never expose API keys or credentials in code or test cassettes.**

### API Key Protection

1. **Environment Variables Only**: All API keys MUST be stored in `.env` files, never hardcoded
2. **VCR Cassette Security**: When working with VCR cassettes for integration tests:
   - **ALWAYS** verify that `tests/integration/conftest.py` filters sensitive headers:
     ```python
     "filter_headers": ["authorization", "x-goog-api-key"]
     "filter_query_parameters": ["key"]
     ```
   - **ALWAYS** audit cassette files before committing to ensure no API keys are present
   - Run this check: `grep -r "AIza\|x-goog-api-key" tests/integration/cassettes/`
   - Expected result: No matches found

3. **Adding New API Integrations**: When adding new external APIs:
   - Identify all authentication headers/parameters
   - Add them to VCR's `filter_headers` and `filter_query_parameters`
   - Regenerate affected cassettes with `--vcr-record=all`
   - Verify no credentials in cassettes before committing

4. **Before Every Commit**:
   - Check `.env` is in `.gitignore`
   - Audit any modified cassettes for API keys
   - Never commit files containing real credentials

### VCR Configuration Details

The VCR configuration in `tests/integration/conftest.py` currently filters:
- **Headers**: `authorization`, `x-goog-api-key` (Google GenAI API)
- **Query Parameters**: `key` (Google Books API)

If VCR cassettes stop working (tests make real API calls instead of using cassettes), check:
1. API key in cassette doesn't match current `.env` → regenerate cassette
2. New authentication headers not filtered → add to `filter_headers`
3. VCR not intercepting HTTP client → verify httpcore/httpx support

## Common Maintenance Tasks

1. **Import Errors**: Check Google ADK version compatibility (currently v1.23.0)
2. **Test Failures**: Often due to changed error messages or API responses
3. **Syntax Errors**: Fix immediately as they block all tests
4. **Type Errors**: Ensure Pydantic models match actual data structures
