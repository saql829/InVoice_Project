# backend/api/history.py
from fastapi import APIRouter, Query
from utils.db import get_db

router = APIRouter(tags=["History"])

@router.get("/history")
def get_history(session_id: str = Query(None), limit: int = 20):
    """
    Fetch conversation history.
    - If session_id provided → return only that session's history.
    - Else → return latest N conversations across sessions.
    """
    conn = get_db()
    conn.row_factory = lambda cursor, row: {
        col[0]: row[idx] for idx, col in enumerate(cursor.description)
    }
    cur = conn.cursor()

    if session_id:
        cur.execute(
            """
            SELECT role, text, topic, ts AS created_at
            FROM conversations
            WHERE session_id=?
            ORDER BY ts ASC
            LIMIT ?
            """,
            (session_id, limit),
        )
    else:
        cur.execute(
            """
            SELECT session_id, role, text, topic, ts AS created_at
            FROM conversations
            ORDER BY ts DESC
            LIMIT ?
            """,
            (limit,),
        )

    rows = cur.fetchall()
    return {"history": rows}
