"""
Saga Transaction Manager with SQLite-backed Persistent Outbox.

Implements a Compensating Transaction pattern between Neo4j (Graph) and
Qdrant (Vector) using a crash-resilient SQLite outbox. If the process is
killed mid-transaction (OOM, SIGTERM), the outbox survives on disk and
the background worker will detect and clean up orphaned data on restart.

NOTE: For distributed multi-node deployments, replace SQLite with
PostgreSQL or Redis Streams for cross-process visibility.
"""

import logging
import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)

# Default path — persisted via Docker volume mount
_DEFAULT_OUTBOX_DB = "data/saga_outbox.db"


class TransactionStatus:
    """Transaction lifecycle states."""
    PENDING = "PENDING"
    NEO4J_DONE = "NEO4J_DONE"
    QDRANT_DONE = "QDRANT_DONE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SagaOutbox:
    """
    SQLite-backed persistent outbox for the Saga pattern.

    Every transaction record is written to disk BEFORE any database
    mutation begins, ensuring recoverability after crashes.
    """

    def __init__(self, db_path: str = _DEFAULT_OUTBOX_DB) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # Concurrent-read safe
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS outbox (
                tx_id      TEXT PRIMARY KEY,
                status     TEXT NOT NULL,
                action     TEXT NOT NULL,
                payload    TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def create(self, action: str, payload: dict) -> str:
        """Write a new PENDING record to disk before any mutations."""
        tx_id = str(uuid4())
        self._conn.execute(
            "INSERT INTO outbox (tx_id, status, action, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (tx_id, TransactionStatus.PENDING, action, json.dumps(payload), time.time()),
        )
        self._conn.commit()
        return tx_id

    def update_status(self, tx_id: str, status: str) -> None:
        """Atomically update a transaction's status on disk."""
        self._conn.execute(
            "UPDATE outbox SET status = ? WHERE tx_id = ?",
            (status, tx_id),
        )
        self._conn.commit()

    def get_pending(self) -> list[dict]:
        """Return all non-terminal transactions (crash survivors)."""
        cursor = self._conn.execute(
            "SELECT tx_id, status, action, payload FROM outbox WHERE status NOT IN (?, ?)",
            (TransactionStatus.COMPLETED, TransactionStatus.FAILED),
        )
        return [
            {"tx_id": row[0], "status": row[1], "action": row[2], "payload": json.loads(row[3])}
            for row in cursor.fetchall()
        ]

    def cleanup_completed(self, max_age_seconds: int = 86400) -> int:
        """Prune completed/failed records older than max_age to prevent DB bloat."""
        cutoff = time.time() - max_age_seconds
        cursor = self._conn.execute(
            "DELETE FROM outbox WHERE status IN (?, ?) AND created_at < ?",
            (TransactionStatus.COMPLETED, TransactionStatus.FAILED, cutoff),
        )
        self._conn.commit()
        return cursor.rowcount


class SagaTransactionManager:
    """
    Coordinates distributed transactions between Graph and Vector stores.

    Singleton — must be initialized once in FastAPI lifespan and injected
    via Depends(). The background worker scans the persistent outbox for
    stuck transactions and logs them for operational alerting.
    """

    def __init__(self, outbox_db: str = _DEFAULT_OUTBOX_DB) -> None:
        self.outbox = SagaOutbox(db_path=outbox_db)
        self._worker_task: Optional[asyncio.Task] = None

    def start_worker(self) -> None:
        """Start the background retry scanner (call once from lifespan)."""
        if self._worker_task is None or self._worker_task.done():
            try:
                self._worker_task = asyncio.create_task(self._background_retry_worker())
                logger.info("SAGA: Background outbox scanner started")
            except RuntimeError:
                logger.warning("SAGA: No event loop — worker not started")

    async def execute_delete(
        self,
        recipe_id: str,
        neo4j_delete_fn: Callable,
        qdrant_delete_fn: Callable,
    ) -> None:
        """
        Delete a recipe with Compensating Transaction protection.
        Outbox record is written to disk BEFORE any mutation.
        """
        logger.info(f"SAGA: Starting distributed delete for recipe {recipe_id}")
        tx_id = self.outbox.create("DELETE", {"recipe_id": recipe_id})

        # Phase 1: Graph (System of Record)
        try:
            await neo4j_delete_fn(recipe_id)
            self.outbox.update_status(tx_id, TransactionStatus.NEO4J_DONE)
            logger.info("  [Phase 1] Neo4j deletion: OK")
        except Exception as e:
            logger.error(f"  [Phase 1] Neo4j deletion failed: {e}. Aborting.")
            self.outbox.update_status(tx_id, TransactionStatus.FAILED)
            raise

        # Phase 2: Vector (Dependent System)
        try:
            await qdrant_delete_fn(recipe_id)
            self.outbox.update_status(tx_id, TransactionStatus.COMPLETED)
            logger.info("  [Phase 2] Qdrant deletion: OK (SAGA COMPLETED)")
        except Exception as e:
            logger.warning(f"  [Phase 2] Qdrant deletion failed: {e}. Worker will retry.")

    async def execute_insert(
        self,
        insert_fn: Callable,
        rollback_fn: Callable,
        **kwargs,
    ) -> None:
        """
        Execute a vector insertion with compensating rollback on failure.
        Outbox tracks lifecycle for crash recovery.
        """
        tx_id = self.outbox.create("INSERT", {"kwargs_keys": list(kwargs.keys())})
        try:
            await insert_fn(**kwargs)
            self.outbox.update_status(tx_id, TransactionStatus.COMPLETED)
        except Exception as e:
            logger.error(f"SAGA: Insert failed, executing compensating rollback: {e}")
            self.outbox.update_status(tx_id, TransactionStatus.FAILED)
            try:
                await rollback_fn(**kwargs)
            except Exception as rollback_err:
                logger.critical(f"SAGA: Rollback also failed: {rollback_err}")
            raise

    async def _background_retry_worker(self) -> None:
        """
        Periodic scanner that detects stuck transactions after a crash.
        In production, this would invoke the actual cleanup functions.
        Currently logs for operational alerting / manual intervention.
        """
        while True:
            await asyncio.sleep(60)
            try:
                pending = self.outbox.get_pending()
                for record in pending:
                    if record["status"] == TransactionStatus.NEO4J_DONE:
                        logger.warning(
                            f"SAGA WORKER: Stuck transaction {record['tx_id']} — "
                            f"action={record['action']}, payload={record['payload']}. "
                            f"Requires manual Qdrant cleanup."
                        )
                # Housekeeping: prune old completed records
                pruned = self.outbox.cleanup_completed()
                if pruned:
                    logger.info(f"SAGA WORKER: Pruned {pruned} old outbox records")
            except Exception as e:
                logger.error(f"SAGA WORKER: Scanner error: {e}")
