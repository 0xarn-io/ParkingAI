"""SQLite persistence: vehicle identities + parking sessions + statistics.

Thread-safe behind a single lock (the engine writes from its worker thread,
the API reads from request threads). Keeps an in-memory cache of vehicle
fingerprints so re-ID matching doesn't hit the DB on every parking event.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import identity

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    color TEXT,
    embedding BLOB,
    first_seen REAL,
    last_seen REAL,
    visits INTEGER DEFAULT 0,
    total_parked REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id TEXT,
    vehicle_id INTEGER,
    start_ts REAL,
    end_ts REAL,
    duration REAL
);
"""


class Store:
    def __init__(self, path: str = "parkingai.db") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # in-memory fingerprint cache: vehicle_id -> embedding
        self._emb: Dict[int, np.ndarray] = {}
        for row in self._conn.execute("SELECT id, embedding FROM vehicles"):
            if row["embedding"] is not None:
                self._emb[row["id"]] = np.frombuffer(row["embedding"], dtype=np.float32)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- identity --------------------------------------------------------
    def _unique_name(self, color: str) -> str:
        for _ in range(50):
            name = identity.make_name(color)
            if not self._conn.execute("SELECT 1 FROM vehicles WHERE name=?", (name,)).fetchone():
                return name
        return f"{color} {int(time.time()) % 100000}"

    def match_or_create(self, emb: np.ndarray, color: str, reid_threshold: float,
                        now: Optional[float] = None) -> Tuple[int, str, bool]:
        """Find the closest known vehicle within threshold, else create one.

        Returns (vehicle_id, name, returning).
        """
        now = now or time.time()
        with self._lock:
            best_id, best_dist = None, 1e9
            for vid, ref in self._emb.items():
                d = identity.distance(emb, ref)
                if d < best_dist:
                    best_id, best_dist = vid, d

            if best_id is not None and best_dist <= reid_threshold:
                # returning car: nudge the stored fingerprint toward the new view
                blended = (0.7 * self._emb[best_id] + 0.3 * emb).astype(np.float32)
                self._emb[best_id] = blended
                self._conn.execute(
                    "UPDATE vehicles SET last_seen=?, embedding=? WHERE id=?",
                    (now, blended.tobytes(), best_id),
                )
                self._conn.commit()
                name = self._conn.execute(
                    "SELECT name FROM vehicles WHERE id=?", (best_id,)).fetchone()["name"]
                return best_id, name, True

            name = self._unique_name(color)
            cur = self._conn.execute(
                "INSERT INTO vehicles (name,color,embedding,first_seen,last_seen,visits,total_parked)"
                " VALUES (?,?,?,?,?,0,0)",
                (name, color, emb.astype(np.float32).tobytes(), now, now),
            )
            self._conn.commit()
            vid = int(cur.lastrowid)
            self._emb[vid] = emb.astype(np.float32)
            return vid, name, False

    def rename(self, vehicle_id: int, name: str) -> bool:
        with self._lock:
            try:
                self._conn.execute("UPDATE vehicles SET name=? WHERE id=?", (name, vehicle_id))
                self._conn.commit()
                return self._conn.total_changes > 0
            except sqlite3.IntegrityError:
                return False

    # -- sessions --------------------------------------------------------
    def open_session(self, zone_id: str, vehicle_id: Optional[int],
                     now: Optional[float] = None) -> int:
        now = now or time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (zone_id,vehicle_id,start_ts) VALUES (?,?,?)",
                (zone_id, vehicle_id, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def close_session(self, session_id: int, now: Optional[float] = None,
                      min_seconds: float = 0.0) -> Optional[float]:
        """Finish a session. Discards (deletes) sessions shorter than min_seconds."""
        now = now or time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT start_ts, vehicle_id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                return None
            duration = now - row["start_ts"]
            if duration < min_seconds:
                self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
                self._conn.commit()
                return None
            self._conn.execute(
                "UPDATE sessions SET end_ts=?, duration=? WHERE id=?", (now, duration, session_id))
            if row["vehicle_id"] is not None:
                self._conn.execute(
                    "UPDATE vehicles SET visits=visits+1, total_parked=total_parked+?, last_seen=?"
                    " WHERE id=?", (duration, now, row["vehicle_id"]))
            self._conn.commit()
            return duration

    # -- queries ---------------------------------------------------------
    def list_vehicles(self) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id,name,color,first_seen,last_seen,visits,total_parked"
                " FROM vehicles ORDER BY last_seen DESC").fetchall()
        return [dict(r) for r in rows]

    def list_sessions(self, limit: int = 50) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.id,s.zone_id,s.vehicle_id,v.name AS vehicle_name,"
                " s.start_ts,s.end_ts,s.duration"
                " FROM sessions s LEFT JOIN vehicles v ON v.id=s.vehicle_id"
                " ORDER BY s.start_ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        midnight = time.time() - (time.time() % 86400)
        with self._lock:
            vehicles = self._conn.execute("SELECT COUNT(*) c FROM vehicles").fetchone()["c"]
            parked_now = self._conn.execute(
                "SELECT COUNT(*) c FROM sessions WHERE end_ts IS NULL").fetchone()["c"]
            done = self._conn.execute(
                "SELECT COUNT(*) c, AVG(duration) a FROM sessions WHERE end_ts IS NOT NULL"
            ).fetchone()
            today = self._conn.execute(
                "SELECT COUNT(*) c FROM sessions WHERE start_ts>=?", (midnight,)).fetchone()["c"]
            per_zone = self._conn.execute(
                "SELECT zone_id, COUNT(*) sessions, AVG(duration) avg_duration,"
                " MAX(end_ts) last_departure FROM sessions WHERE end_ts IS NOT NULL"
                " GROUP BY zone_id ORDER BY zone_id").fetchall()
        return {
            "vehicles_known": vehicles,
            "currently_parked": parked_now,
            "sessions_today": today,
            "completed_sessions": done["c"],
            "avg_park_seconds": round(done["a"], 1) if done["a"] else 0.0,
            "per_zone": [dict(r) for r in per_zone],
        }
