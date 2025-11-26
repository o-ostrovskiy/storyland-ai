# Database Schema Reference

## ADK DatabaseSessionService Schema

The Google ADK's `DatabaseSessionService` creates this table structure:

### Sessions Table

```sql
CREATE TABLE sessions (
    app_name VARCHAR(128) NOT NULL,
    user_id VARCHAR(128) NOT NULL,
    id VARCHAR(128) NOT NULL,
    state TEXT NOT NULL,
    create_time DATETIME NOT NULL,
    update_time DATETIME NOT NULL,
    PRIMARY KEY (app_name, user_id, id)
)
```

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `app_name` | VARCHAR(128) | Application identifier (e.g., "storyland") |
| `user_id` | VARCHAR(128) | User identifier (unique per user) |
| `id` | VARCHAR(128) | Session ID (UUID) |
| `state` | TEXT | JSON-encoded session state |
| `create_time` | DATETIME | When session was created |
| `update_time` | DATETIME | When session was last modified |

### Primary Key

The composite primary key is `(app_name, user_id, id)`, which ensures:
- Multiple apps can use the same database
- Users are isolated within each app
- Each session is unique

### State JSON Structure

The `state` column contains JSON with this structure:

```json
{
  "book_title": "Gone with the Wind",
  "author": "Margaret Mitchell",
  "user:preferences": {
    "prefers_museums": true,
    "budget": "moderate",
    "favorite_genres": ["classics"]
  },
  "book_metadata": { ... },
  "final_itinerary": { ... }
}
```

**State Scopes:**
- No prefix (e.g., `book_title`) - Session-scoped (ephemeral)
- `user:` prefix (e.g., `user:preferences`) - User-scoped (persistent across sessions)
- `app:` prefix - App-scoped (global configuration)
- `temp:` prefix - Temporary (not persisted)

## Common SQL Queries

### Get All Sessions for a User

```sql
SELECT
    id,
    json_extract(state, '$.book_title') as book,
    create_time
FROM sessions
WHERE app_name = 'storyland'
  AND user_id = 'alice'
ORDER BY create_time DESC;
```

### Get User Preferences

```sql
SELECT
    user_id,
    json_extract(state, '$."user:preferences"') as preferences
FROM sessions
WHERE app_name = 'storyland'
  AND user_id = 'alice'
ORDER BY create_time DESC
LIMIT 1;
```

### Count Sessions Per User

```sql
SELECT
    user_id,
    COUNT(*) as session_count
FROM sessions
WHERE app_name = 'storyland'
GROUP BY user_id
ORDER BY session_count DESC;
```

### Get Latest Session

```sql
SELECT
    id,
    user_id,
    json_extract(state, '$.book_title') as book,
    create_time
FROM sessions
WHERE app_name = 'storyland'
ORDER BY create_time DESC
LIMIT 1;
```

### Extract Nested JSON

```sql
-- Get a specific preference
SELECT
    user_id,
    json_extract(state, '$."user:preferences".budget') as budget
FROM sessions
WHERE app_name = 'storyland';

-- Get array elements
SELECT
    user_id,
    json_extract(state, '$."user:preferences".favorite_genres') as genres
FROM sessions
WHERE app_name = 'storyland';
```

### Search in State

```sql
-- Find sessions with specific book
SELECT
    id,
    user_id,
    json_extract(state, '$.book_title') as book
FROM sessions
WHERE app_name = 'storyland'
  AND json_extract(state, '$.book_title') = 'Gone with the Wind';

-- Find users who prefer museums
SELECT DISTINCT
    user_id
FROM sessions
WHERE app_name = 'storyland'
  AND json_extract(state, '$."user:preferences".prefers_museums') = 1;
```

## Important Notes

### Column Name Differences

⚠️ **Common mistake:** The column names are different from what you might expect:

| ❌ Wrong | ✅ Correct |
|----------|-----------|
| `session_id` | `id` |
| `created_at` | `create_time` |
| `updated_at` | `update_time` |

### JSON Extraction

- Use `json_extract(state, '$.key')` for top-level keys
- Use `json_extract(state, '$."user:key"')` for keys with colons (must quote!)
- Boolean values return as `1` (true) or `0` (false) in SQLite
- Arrays return as JSON strings (e.g., `'["a","b"]'`)

### State Persistence

Only `user:` prefixed keys persist across sessions for the same user:

```python
# Session 1
state["book_title"] = "Book A"          # Ephemeral
state["user:preferences"] = {...}       # Persistent

# Session 2 (same user)
# state["book_title"] is NOT available
# state["user:preferences"] IS available
```

## Examples from Python

### Query from Python

```python
import sqlite3
import json

conn = sqlite3.connect('storyland_sessions.db')
cursor = conn.cursor()

# Get user's book history
cursor.execute("""
    SELECT
        id,
        json_extract(state, '$.book_title') as book,
        create_time
    FROM sessions
    WHERE app_name = 'storyland'
      AND user_id = ?
    ORDER BY create_time DESC
""", ('alice',))

for session_id, book, created in cursor.fetchall():
    print(f"{book} - {created}")

conn.close()
```

### Parse JSON State

```python
cursor.execute("""
    SELECT state
    FROM sessions
    WHERE app_name = 'storyland'
      AND user_id = ?
      AND id = ?
""", ('alice', 'session-id-here'))

state_json = cursor.fetchone()[0]
state = json.loads(state_json)

print(state.get('book_title'))
print(state.get('user:preferences'))
```

## Database Location

- Default: `storyland_sessions.db` in current directory
- Configured via: `DATABASE_URL` environment variable
- Format: `sqlite:///path/to/database.db`

## Backup & Maintenance

### Backup

```bash
# Backup database
cp storyland_sessions.db storyland_sessions_backup_$(date +%Y%m%d).db

# Or use SQLite backup command
sqlite3 storyland_sessions.db ".backup backup.db"
```

### Cleanup

```bash
# Delete old sessions (older than 30 days)
sqlite3 storyland_sessions.db "
DELETE FROM sessions
WHERE create_time < datetime('now', '-30 days');"

# Vacuum to reclaim space
sqlite3 storyland_sessions.db "VACUUM;"
```

### Export

```bash
# Export to CSV
sqlite3 -header -csv storyland_sessions.db \
  "SELECT * FROM sessions;" > sessions_export.csv

# Export to JSON
sqlite3 storyland_sessions.db \
  "SELECT json_object(
    'id', id,
    'user_id', user_id,
    'book', json_extract(state, '$.book_title'),
    'created', create_time
  ) FROM sessions;" > sessions.json
```

## Troubleshooting

### Table doesn't exist

```bash
# Check if table exists
sqlite3 storyland_sessions.db ".tables"

# If missing, run app once with --database flag to create it
python main.py "test" --database
```

### Invalid JSON

```bash
# Check for valid JSON in state
sqlite3 storyland_sessions.db "
SELECT id, user_id,
       CASE
         WHEN json_valid(state) THEN 'valid'
         ELSE 'INVALID'
       END as json_status
FROM sessions;"
```

### Locked database

```bash
# Check for locks
lsof storyland_sessions.db

# Close all connections and try again
```
