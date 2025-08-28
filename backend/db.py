# backend/db.py
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional

DB_PATH = Path(__file__).resolve().parent / "memory.db"


def init_db():
    """Initialize SQLite database and create tables if not exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,       -- "user" or "assistant"
            message TEXT NOT NULL,
            tags TEXT,                -- comma-separated tags
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
    print("🔹 Database initialized and ready.")


def save_message(session_id: str, role: str, message: str, tags: Optional[str] = None):
    """Save single message in memory with optional tags."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_memory (session_id, role, message, tags) VALUES (?, ?, ?, ?)",
            (session_id, role, message, tags),
        )
        conn.commit()
    print(f" Saved message | session={session_id} role={role} tags={tags or 'none'} text={message[:60]}...")


def load_history(session_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """Load conversation history for a session (last N messages)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT role, message, tags FROM chat_memory WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cur.fetchall()

    print(f" Loaded history | session={session_id} count={len(rows)}")
    # Reverse to chronological order
    return [
        {"role": row["role"], "content": row["message"], "tags": row["tags"] or ""}
        for row in rows[::-1]
    ]


def clear_session(session_id: str):
    """Delete all history for a given session."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_memory WHERE session_id = ?", (session_id,))
        deleted = cur.rowcount
        conn.commit()
    print(f" Cleared session | session={session_id} deleted={deleted}")
