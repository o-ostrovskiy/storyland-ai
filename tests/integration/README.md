# Google Books API Integration Tests

## Overview

This directory contains integration tests for the Google Books API using VCR.py for recording and replaying HTTP interactions.

## Test Structure

- **test_google_books_api.py**: Core integration tests (10 tests)
  - Basic API functionality
  - API key scenarios
  - Response parsing

- **test_google_books_edge_cases.py**: Error handling and edge cases (12 tests)
  - Network errors (timeout, 500, 429, connection errors)
  - Query validation (special characters, unicode, long strings)
  - Search book edge cases

**Total**: 22 integration tests

## How VCR.py Works

1. **First Run**: Tests hit the real Google Books API and record responses to YAML cassettes
2. **Subsequent Runs**: Tests replay from cassettes (fast, deterministic, no API quota usage)
3. **Re-recording**: Delete cassettes or use `make test-integration-vcr-record`

## Running Tests

### Run all integration tests (using VCR cassettes)
```bash
make test-integration
```

### Re-record all cassettes (requires API quota)
```bash
make test-integration-vcr-record
```

### Run specific test file
```bash
pytest tests/integration/test_google_books_api.py -v
```

### Run specific test
```bash
pytest tests/integration/test_google_books_api.py::TestSearchBooksIntegration::test_search_books_success -v
```

## Important Notes

### API Quota

- The Google Books API has daily quota limits
- VCR cassettes eliminate quota usage after initial recording
- If quota is exhausted, wait until midnight PT for reset

### API Key

- Tests work with or without `GOOGLE_API_KEY` in `.env`
- With API key: Higher quota limits
- Without API key: Falls back to public quota

### Cassettes Directory

- Cassettes are stored in `tests/integration/cassettes/`
- **Commit cassettes to git** for reproducible tests in CI/CD
- API keys are automatically filtered from cassettes (security)

## Test Results

### Working Tests (5/22)
✅ Network error simulation tests (using responses library)
- `test_network_timeout_error`
- `test_api_500_error`
- `test_rate_limit_429`
- `test_connection_error`
- `test_search_book_handles_exception_gracefully`

### Pending Tests (17/22)
⏳ VCR-based tests (require initial API recording)
- Will work after recording cassettes when API quota is available
- Cassettes partially created but contain 429 errors

## Next Steps

1. **Wait for quota reset** (midnight PT) or use a fresh API key
2. **Record cassettes**: Run `make test-integration-vcr-record`
3. **Verify all tests pass**: Run `make test-integration`
4. **Commit cassettes**: Add cassettes to version control

## GitHub Actions CI/CD

### Workflow Configuration

The integration tests run automatically in GitHub Actions via `.github/workflows/integration-tests.yml`:

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Jobs:**
1. **integration-tests** (runs on every trigger)
   - Uses VCR cassettes (no API quota consumed)
   - Runs in ~2 seconds
   - Comments results on PRs

2. **re-record-cassettes** (manual trigger only)
   - Re-records all VCR cassettes
   - Consumes API quota
   - Creates artifact with new cassettes

### GitHub Secrets Setup

Add `GOOGLE_API_KEY` to your repository secrets:

1. Go to: `Settings` → `Secrets and variables` → `Actions`
2. Click `New repository secret`
3. Name: `GOOGLE_API_KEY`
4. Value: Your Google Books API key
5. Click `Add secret`

### Manual Cassette Re-recording

To manually re-record cassettes in CI:

1. Go to `Actions` tab in GitHub
2. Select `Integration Tests` workflow
3. Click `Run workflow`
4. Select branch
5. Click `Run workflow` button

This will re-record all cassettes and upload them as artifacts.

## Troubleshooting

### "429 Too Many Requests" errors
- API quota exhausted
- Wait until midnight PT for quota reset
- Or use a different Google Cloud project API key
- Delete old cassettes: `rm -rf tests/integration/cassettes/*`

### "ScopeMismatch" fixture errors
- Fixed in conftest.py
- vcr_config fixture is module-scoped
- cassette directory created during setup

### Tests hitting real API instead of cassettes
- Cassettes may not exist yet
- Delete cassettes to re-record: `rm -rf tests/integration/cassettes/*`
- Re-run with `--vcr-record=all` flag

### CI Tests Failing

**If tests fail in GitHub Actions:**
1. Check if cassettes are committed to git
2. Verify `GOOGLE_API_KEY` secret is set
3. Check workflow logs for specific errors
4. Re-record cassettes using manual workflow trigger

## Configuration

VCR configuration is in `conftest.py`:
- **Record mode**: `once` (record once, replay thereafter)
- **Match on**: method, scheme, host, port, path, query
- **Filters**: API keys and authorization headers redacted
- **Cassette location**: `tests/integration/cassettes/`

## Best Practices

1. **Commit cassettes to git** - Ensures reproducible tests in CI
2. **Don't commit API keys** - VCR automatically filters them
3. **Re-record periodically** - When API responses change
4. **Use descriptive test names** - Cassette names match test names
5. **Test both success and error cases** - Use VCR for success, responses for errors
