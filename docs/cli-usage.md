# CLI Usage

## Basic Commands

```bash
# Simple query (default user: user1)
python main.py "Gone with the Wind"

# With author
python main.py "The Nightingale" --author "Kristin Hannah"

# Custom user
python main.py "1984" --user-id alice
```

## Advanced Usage

```bash
# Enable SQLite persistence
python main.py "Pride and Prejudice" --database

# With preferences (via CLI flags)
python main.py "The Great Gatsby" --budget luxury --pace relaxed

# Family trip with database
python main.py "Harry Potter" --database --with-kids --budget moderate --museums
```

## Multi-User Support

```bash
# User "alice" explores Pride and Prejudice
python main.py "Pride and Prejudice" --user-id alice --database

# User "bob" explores Dune (completely separate data)
python main.py "Dune" --user-id bob --database

# Alice explores another book (her preferences persist!)
python main.py "Emma" --user-id alice --database
```

**Check user data:**

```bash
# See all users
sqlite3 storyland_sessions.db "SELECT DISTINCT user_id FROM sessions;"

# Count sessions per user
sqlite3 storyland_sessions.db "
SELECT user_id, COUNT(*) as session_count
FROM sessions
GROUP BY user_id;"

# View specific user's sessions
sqlite3 storyland_sessions.db "
SELECT id,
       json_extract(state, '$.book_title') as book,
       create_time
FROM sessions
WHERE user_id = 'alice';"
```

## Help

```bash
python main.py --help
```

## Features

### 1. Sessions & State Management

- **In-Memory (Default)**: Fast, perfect for development and testing
- **SQLite-Backed**: Persistent sessions across restarts
- **Multi-scoped State**: Session, user, app, and temporary scopes
- **User Preferences**: Persist across sessions with `user:` prefix

```python
# Preferences persist automatically
state["user:preferences"] = {
    "prefers_museums": True,
    "budget": "moderate",
    "favorite_genres": ["classics"]
}
```

### 2. Context Engineering

- **Sliding Window**: Keep recent N events
- **Token Estimation**: Track conversation size
- **Compaction Detection**: Know when to reduce context
- **ADK Integration**: Use built-in `GetSessionConfig`

```python
from services.context_manager import ContextManager

context_manager = ContextManager(max_events=20)
if context_manager.should_compact(session.events):
    session.events = context_manager.limit_events(session.events)
```

### 3. Personalized Itineraries

The trip composer uses user preferences to tailor recommendations:

```python
# Set preferences in session state
state["user:preferences"] = {
    "budget": "luxury",           # budget, moderate, luxury
    "preferred_pace": "relaxed",  # relaxed, moderate, fast-paced
    "prefers_museums": True,
    "travels_with_kids": False,
    "dietary_restrictions": ["vegetarian"]
}
```

**How preferences affect itineraries:**

| Preference | Impact |
|------------|--------|
| `budget: "budget"` | Free museums, walking tours, affordable cafes |
| `budget: "luxury"` | Premium experiences, fine dining, private tours |
| `preferred_pace: "relaxed"` | 2-3 stops/day, longer breaks |
| `preferred_pace: "fast-paced"` | 5+ stops/day, efficient routing |
| `prefers_museums: true` | Prioritize literary museums, archives |
| `travels_with_kids: true` | Family-friendly activities, avoid long queues |

### 4. Multi-User Support

Each user has isolated sessions and preferences:

```bash
python main.py "Pride and Prejudice" --user-id alice --database
python main.py "Dune" --user-id bob --database
```

### 5. Automated Research

- Agents handle complexity of gathering scattered information
- **Parallel Processing**: Multiple discovery agents work simultaneously
- **Structured Output**: Clear, actionable travel plans

## Using Saved Data in Future Runs

When you enable database persistence (`--database` or `USE_DATABASE=true`), all data is saved and can be reused.

### 1. User Preferences Persist

```python
# First session - set preferences
await session_service.create_session(
    app_name="storyland",
    user_id="alice",
    session_id="session-1",
    state={
        "book_title": "Pride and Prejudice",
        "user:preferences": {
            "prefers_museums": True,
            "budget": "moderate"
        }
    }
)

# Later session - preferences auto-loaded!
new_session = await session_service.create_session(
    app_name="storyland",
    user_id="alice",  # Same user
    session_id="session-2",
    state={"book_title": "Emma"}
)

prefs = new_session.state["user:preferences"]  # Available!
```

### 2. Resume Previous Sessions

```python
# Load existing session
session = await session_service.get_session(
    app_name="storyland",
    user_id="alice",
    session_id="previous-session-id"
)

# Access previous itinerary
itinerary = session.state.get("final_itinerary")
```

### 3. Query User History

```python
import sqlite3

def get_user_books(user_id: str):
    conn = sqlite3.connect('storyland_sessions.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT json_extract(state, '$.book_title') as book,
               create_time
        FROM sessions
        WHERE user_id = ?
        ORDER BY create_time DESC
    """, (user_id,))
    return cursor.fetchall()
```

### 4. Practical Demo

See **[sessions_memory_demo.ipynb](../sessions_memory_demo.ipynb)** for 11 complete scenarios covering sessions, preferences, and context management.

## Database Reference

### Schema

ADK's `DatabaseSessionService` creates:

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

**Important:** Column names are:
- `id` (not `session_id`)
- `create_time` (not `created_at`)
- `update_time` (not `updated_at`)

See **[DATABASE_SCHEMA.md](../DATABASE_SCHEMA.md)** for complete reference.

### Inspecting Database

```bash
# Open database
sqlite3 storyland_sessions.db

# Inside sqlite3:
.tables                    # List tables
.schema                    # Show structures
.mode column              # Better formatting
.headers on

# View sessions
SELECT * FROM sessions;

# View latest
SELECT * FROM sessions ORDER BY create_time DESC LIMIT 1;

# Exit
.quit
```

**Quick one-liners:**

```bash
# List tables
sqlite3 storyland_sessions.db ".tables"

# Count sessions
sqlite3 storyland_sessions.db "SELECT COUNT(*) FROM sessions;"

# Latest session
sqlite3 -column -header storyland_sessions.db \
  "SELECT id, user_id, create_time FROM sessions ORDER BY create_time DESC LIMIT 1;"
```
