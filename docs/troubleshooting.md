# Troubleshooting

## Rate Limits (429 Errors)

The workflow makes ~11 API calls per book, which can hit free tier limits (15 RPM).

**Solution:** The retry logic will handle it automatically. Wait ~60 seconds between books.

## Database Issues

```bash
# Delete database to start fresh
rm storyland_sessions.db

# Check database exists
ls -lh *.db
```

## Import Errors

```bash
# Ensure you're in project root
pwd

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"
```

## API Key Issues

```bash
# Check API key is set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OK' if os.getenv('GOOGLE_API_KEY') else 'MISSING')"
```

## Common Issues

### Virtual Environment Not Activated

Make sure you activate your virtual environment:

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Missing Dependencies

If you encounter import errors, reinstall dependencies:

```bash
pip install -e ".[dev]"
```

### Database Lock Errors

If you see "database is locked" errors, make sure no other process is using the database:

```bash
# Check for running processes
ps aux | grep python

# Kill any stale processes if needed
```

### Timeout Errors

If workflows timeout, you can adjust the timeout values in your `.env` file:

```env
WORKFLOW_TIMEOUT=300    # Increase if needed (default: 300 seconds)
AGENT_TIMEOUT=60        # Increase if needed (default: 60 seconds)
```
