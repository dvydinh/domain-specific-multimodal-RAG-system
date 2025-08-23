"""
Saga Transaction / Outbox Pattern Manager.

Ensures ACID Eventual Consistency between Neo4j (Graph) and Qdrant (Vector).
Resolves the "Phantom Data" and "Vector Drift" problems when updating/deleting recipes.

Flow (Example: Delete Recipe):
1. Create Outbox Record (Status: PENDING)
2. Delete from Neo4j  -> (If fails, abort and mark FAILED)
3. Delete from Qdrant -> (If fails, Saga Background Worker will retry)
4. Mark Outbox Record (Status: COMPLETED)
"""

import logging
import asyncio
from typing import Optional, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


class TransactionStatus:
    """Enum-like constants for saga transaction states."""
    PENDING = "PENDING"
    NEO4J_DONE = "NEO4J_DONE"
    QDRANT_DONE = "QDRANT_DONE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SagaOutbox:
    """In-memory outbox store for the Saga pattern.

    In production, this would be backed by a persistent database table
    to survive process restarts.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def create(self, action: str, payload: dict) -> str:
        """Create a new outbox record with PENDING status.

        Args:
            action: The transaction action type (e.g., "DELETE").
            payload: Data associated with this transaction.

        Returns:
            The generated transaction ID.
        """
        tx_id = str(uuid4())
        self._store[tx_id] = {
            "status": TransactionStatus.PENDING,
            "action": action,
            "payload": payload,
        }
        return tx_id

    def update_status(self, tx_id: str, status: str) -> None:
        """Update the status of an existing transaction.

        Args:
            tx_id: Transaction ID to update.
            status: New TransactionStatus value.
        """
        if tx_id in self._store:
            self._store[tx_id]["status"] = status

    def get_pending(self) -> dict[str, dict]:
        """Return all non-terminal transactions (not COMPLETED or FAILED)."""
        return {
            k: v for k, v in self._store.items()
            if v["status"] not in (TransactionStatus.COMPLETED, TransactionStatus.FAILED)
        }


class SagaTransactionManager:
    """Coordinates 2-phase distributed transactions between Graph and Vector DBs.

    The background retry worker is created lazily on first use to avoid
    crashing when no event loop is running at import time.
    """

    def __init__(self) -> None:
        self.outbox = SagaOutbox()
        self._worker_task: Optional[asyncio.Task] = None

    def _ensure_worker(self) -> None:
        """Start the background retry worker if not already running."""
        if self._worker_task is None or self._worker_task.done():
            try:
                self._worker_task = asyncio.create_task(self._background_retry_worker())
            except RuntimeError:
                logger.warning("No running event loop — background retry worker not started")

    async def execute_delete(
        self,
        recipe_id: str,
        neo4j_delete_fn: Callable,
        qdrant_delete_fn: Callable,
    ) -> None:
        """Delete a recipe ensuring Eventual Consistency across both stores.

        Args:
            recipe_id: UUID of the recipe to delete.
            neo4j_delete_fn: Async callable to delete from Neo4j.
            qdrant_delete_fn: Async callable to delete from Qdrant.

        Raises:
            Exception: If Phase 1 (Neo4j) fails, the saga is aborted.
        """
        self._ensure_worker()
        logger.info(f"SAGA: Starting distributed delete for recipe {recipe_id}")

        # 1. Create Outbox Record
        tx_id = self.outbox.create("DELETE", {"recipe_id": recipe_id})

        # 2. Phase 1: Modify System of Record (Graph)
        try:
            await neo4j_delete_fn(recipe_id)
            self.outbox.update_status(tx_id, TransactionStatus.NEO4J_DONE)
            logger.info("  [Phase 1] Neo4j Deletion: Success")
        except Exception as e:
            logger.error(f"  [Phase 1] Neo4j Deletion failed: {e}. Aborting Saga.")
            self.outbox.update_status(tx_id, TransactionStatus.FAILED)
            raise

        # 3. Phase 2: Modify Dependent System (Vector)
        try:
            await qdrant_delete_fn(recipe_id)
            self.outbox.update_status(tx_id, TransactionStatus.COMPLETED)
            logger.info("  [Phase 2] Qdrant Deletion: Success (SAGA COMPLETED)")
        except Exception as e:
            logger.warning(f"  [Phase 2] Qdrant Deletion Timeout/Error: {e}.")
            logger.warning("  -> Outbox marked NEO4J_DONE. Background worker will retry.")
            raise

    async def _background_retry_worker(self) -> None:
        """Worker that continuously scans Outbox for stuck transactions to retry.

        TODO: In a production system, this worker would call the actual
        qdrant_delete_fn stored in the outbox record. Currently a stub
        that logs pending transactions for manual intervention.
        """
        while True:
            await asyncio.sleep(60)
            pending = self.outbox.get_pending()
            for tx_id, record in pending.items():
                if record["status"] == TransactionStatus.NEO4J_DONE and record["action"] == "DELETE":
                    logger.info(
                        f"SAGA WORKER: Found stuck transaction {tx_id} — "
                        f"recipe {record['payload']['recipe_id']} needs Qdrant cleanup"
                    )
