# GitHub Actions Setup for Integration Tests

This guide explains how to configure GitHub Actions to run integration tests automatically.

## Quick Setup

### 1. Add GitHub Secret

The integration tests workflow requires a `GOOGLE_API_KEY` secret:

1. Navigate to your repository on GitHub
2. Go to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Configure the secret:
   - **Name**: `GOOGLE_API_KEY`
   - **Secret**: Your Google Books API key (e.g., `your-google-books-api-key-here`)
5. Click **Add secret**

### 2. Commit VCR Cassettes

For the fastest CI runs, commit the VCR cassettes to git:

```bash
git add tests/integration/cassettes/
git commit -m "chore: Add VCR cassettes for integration tests"
git push
```

This ensures tests run in ~1-2 seconds without making real API calls.

### 3. Verify Workflow

The workflow will automatically run on:
- ✅ Push to `main` or `develop` branches
- ✅ Pull requests to `main` or `develop` branches
- ✅ Manual workflow dispatch (for re-recording cassettes)

## Workflow Details

### File Location
`.github/workflows/integration-tests.yml`

### Jobs

#### 1. `integration-tests` (Always Runs)
- **Purpose**: Run tests using VCR cassettes
- **Runs on**: Every push/PR
- **Duration**: ~1-2 seconds
- **API Quota**: None (uses cassettes)
- **Actions**:
  - Sets up Python 3.12
  - Installs dependencies
  - Runs `make test-integration`
  - Uploads test results as artifacts
  - Comments on PRs with results

#### 2. `re-record-cassettes` (Manual Only)
- **Purpose**: Re-record VCR cassettes when API changes
- **Runs on**: Manual workflow dispatch only
- **Duration**: ~10-15 seconds
- **API Quota**: Uses quota (records real API responses)
- **Actions**:
  - Deletes old cassettes
  - Runs `make test-integration-vcr-record`
  - Uploads new cassettes as artifacts
  - Optionally creates PR with updated cassettes

## Usage

### View Test Results

1. Go to **Actions** tab in GitHub
2. Click on the latest workflow run
3. View test results and logs

### Manually Re-record Cassettes

When the Google Books API changes and you need fresh cassettes:

1. Go to **Actions** tab
2. Select **Integration Tests** workflow
3. Click **Run workflow** dropdown
4. Select the branch (usually `main` or `develop`)
5. Click **Run workflow** button

The workflow will:
- Delete old cassettes
- Record fresh API responses
- Upload new cassettes as artifacts

You can then:
- Download the artifact
- Replace local cassettes
- Commit and push

### Download Cassette Artifacts

1. Go to the workflow run
2. Scroll to **Artifacts** section
3. Download `vcr-cassettes` artifact
4. Extract and copy to `tests/integration/cassettes/`
5. Commit and push

## Troubleshooting

### Tests Failing in CI

**Problem**: Tests pass locally but fail in CI

**Solutions**:
1. Ensure cassettes are committed to git
2. Check that `GOOGLE_API_KEY` secret is set
3. Verify Python version matches (3.12)
4. Check workflow logs for specific errors

### Secret Not Found

**Problem**: `GOOGLE_API_KEY` secret is missing

**Solution**:
```
Error: Secret GOOGLE_API_KEY is not set
```

1. Verify secret name is exactly `GOOGLE_API_KEY` (case-sensitive)
2. Check secret is set in repository (not organization) secrets
3. Ensure secret is accessible to the workflow (check permissions)

### Cassettes Out of Date

**Problem**: Tests fail because API responses changed

**Solution**:
1. Use manual workflow dispatch to re-record cassettes
2. Download new cassettes artifact
3. Commit and push updated cassettes

### Test Duration Too Long

**Problem**: Tests take >10 seconds in CI

**Likely Cause**: Tests are hitting real API instead of using cassettes

**Solution**:
1. Verify cassettes exist in `tests/integration/cassettes/`
2. Ensure cassettes are committed to git
3. Check VCR configuration in `tests/integration/conftest.py`

## Best Practices

1. **Always commit cassettes** - Enables fast, deterministic tests
2. **Re-record periodically** - Keep cassettes up-to-date with API changes
3. **Use manual trigger for re-recording** - Avoid unnecessary API quota usage
4. **Monitor workflow runs** - Check for failures and act quickly
5. **Keep API key secure** - Never commit `.env` file or expose secrets

## Security Notes

- ✅ API keys are automatically filtered from VCR cassettes
- ✅ GitHub Secrets are encrypted and masked in logs
- ✅ `.env` file is in `.gitignore` (never committed)
- ✅ Cassettes are safe to commit (no sensitive data)

## Example Workflow Output

```
✅ Integration Tests succeeded

Tests ran using VCR cassettes (no API quota consumed).

- 21 tests passed
- 1 test skipped
- Duration: 1.25s
```

## Additional Resources

- [VCR.py Documentation](https://vcrpy.readthedocs.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Integration Tests README](../tests/integration/README.md)
