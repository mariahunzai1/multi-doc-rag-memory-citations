"""
core/memory.py
==============
Phase 2C: Durable SQLite Multi-Turn Thread Memory.

Design guarantees:
- Pure SQLite standard library (sqlite3) — zero ORM dependencies.
- Persistent across application restarts, keyed by session_id.
- Thread-safe with WAL (Write-Ahead Logging) journal mode enabled.
- Automatic table & index creation on initialization.
- FIFO chronological history windowing with get_recent_history().

Rule compliance:
- Never imports fastapi, uvicorn, or streamlit.
- Python 3.11+ strict type annotations throughout.
- Consumes core/config.py settings read-only.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class SQLiteMemoryStore:
    """
    Durable multi-turn conversation memory backed by SQLite.

    Stores conversation turns in a 'messages' table with an index on session_id.
    """

    def __init__(self, db_path: str = "./rag_memory.sqlite") -> None:
        self.db_path = str(db_path)
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create table and index if they do not exist, and set WAL mode."""
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id);"
            )

    def append_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
    ) -> None:
        """
        Record a full conversational turn (user question + assistant response).

        Both messages are inserted atomically within a single transaction
        with UTC ISO-8601 timestamps.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, 'user', ?, ?);
                """,
                (session_id, user_msg, now),
            )
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, 'assistant', ?, ?);
                """,
                (session_id, assistant_msg, now),
            )

    def get_recent_history(
        self,
        session_id: str,
        max_messages: int = 6,
    ) -> list[dict[str, str]]:
        """
        Retrieve up to max_messages recent messages for the given session.

        Returns messages in chronological order (oldest to newest):
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        if max_messages <= 0:
            return []

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?;
                """,
                (session_id, max_messages),
            )
            rows = cursor.fetchall()

        # Reverse from DESC to chronological ASC
        return [
            {"role": str(row["role"]), "content": str(row["content"])}
            for row in reversed(rows)
        ]

    def clear_session(self, session_id: str) -> None:
        """Purge all conversation history for a specific session."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?;",
                (session_id,),
            )

    def list_sessions(self) -> list[str]:
        """Return all distinct active session IDs ordered by recency."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT session_id, MAX(id) AS latest_id
                FROM messages
                GROUP BY session_id
                ORDER BY latest_id DESC;
                """
            )
            return [str(row["session_id"]) for row in cursor.fetchall()]
