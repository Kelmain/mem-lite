"""CLI functions for managing learnings."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3


def list_learnings(
    conn: sqlite3.Connection,
    category: str | None = None,
) -> list[dict]:
    """List active learnings, optionally filtered by category.

    Args:
        conn: SQLite connection with migrated schema.
        category: Optional category filter.

    Returns:
        List of learning dicts with id, category, content, confidence,
        times_seen, and is_manual fields.
    """
    query = (
        "SELECT id, category, content, confidence, times_seen, is_manual "
        "FROM learnings WHERE is_active = 1"
    )
    params: list[str] = []
    if category:
        query += " AND category = ?"
        params.append(category)
    query += " ORDER BY confidence DESC, times_seen DESC"

    cursor = conn.execute(query, params)
    return [
        {
            "id": row[0],
            "category": row[1],
            "content": row[2],
            "confidence": row[3],
            "times_seen": row[4],
            "is_manual": bool(row[5]),
        }
        for row in cursor.fetchall()
    ]


def add_learning(
    conn: sqlite3.Connection,
    category: str,
    content: str,
) -> str:
    """Add a manual learning with confidence=1.0 and is_manual=True.

    Args:
        conn: SQLite connection with migrated schema.
        category: Learning category (e.g. "convention", "gotcha").
        content: The learning text.

    Returns:
        The generated learning ID.
    """
    learning_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO learnings "
        "(id, category, content, confidence, source_session_id, source_sessions, "
        "times_seen, is_manual, is_active, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            learning_id,
            category,
            content,
            1.0,  # Manual learnings get max confidence
            "manual",
            json.dumps(["manual"]),
            1,
            1,  # is_manual = True
            1,  # is_active = True
            now,
            now,
        ),
    )
    conn.commit()
    return learning_id


def edit_learning(
    conn: sqlite3.Connection,
    learning_id: str,
    content: str,
) -> bool:
    """Edit a learning's content.

    Args:
        conn: SQLite connection with migrated schema.
        learning_id: ID of the learning to edit.
        content: New content text.

    Returns:
        True if found and updated, False otherwise.
    """
    now = datetime.now(UTC).isoformat()
    cursor = conn.execute(
        "UPDATE learnings SET content = ?, updated_at = ? WHERE id = ? AND is_active = 1",
        (content, now, learning_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def remove_learning(
    conn: sqlite3.Connection,
    learning_id: str,
) -> bool:
    """Soft-delete a learning (set is_active=0).

    Args:
        conn: SQLite connection with migrated schema.
        learning_id: ID of the learning to remove.

    Returns:
        True if found and deactivated, False otherwise.
    """
    now = datetime.now(UTC).isoformat()
    cursor = conn.execute(
        "UPDATE learnings SET is_active = 0, updated_at = ? WHERE id = ? AND is_active = 1",
        (now, learning_id),
    )
    conn.commit()
    return cursor.rowcount > 0
