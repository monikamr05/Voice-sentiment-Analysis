"""
SQLite database for prediction history.
"""

import sqlite3
from datetime import datetime

from config import DATABASE_PATH


def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not exist."""
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT,
            emotion TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(filename, filepath, emotion, sentiment, confidence):
    """Insert a new prediction record."""
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO predictions (filename, filepath, emotion, sentiment, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            filepath,
            emotion,
            sentiment,
            float(confidence),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    pred_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return pred_id


def get_all_predictions(limit=100):
    """Fetch prediction history, newest first."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_prediction_by_id(pred_id):
    """Fetch single prediction by ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM predictions WHERE id = ?", (pred_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_statistics():
    """Aggregate stats for dashboard charts."""
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    by_sentiment = conn.execute(
        """
        SELECT sentiment, COUNT(*) as count
        FROM predictions GROUP BY sentiment
        """
    ).fetchall()
    by_emotion = conn.execute(
        """
        SELECT emotion, COUNT(*) as count
        FROM predictions GROUP BY emotion
        """
    ).fetchall()
    avg_conf = conn.execute(
        "SELECT AVG(confidence) FROM predictions"
    ).fetchone()[0]
    conn.close()
    return {
        "total": total,
        "by_sentiment": {r["sentiment"]: r["count"] for r in by_sentiment},
        "by_emotion": {r["emotion"]: r["count"] for r in by_emotion},
        "avg_confidence": round(avg_conf or 0, 2),
    }
