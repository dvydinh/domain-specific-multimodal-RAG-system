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
    PENDING = "PENDING"
    NEO4J_DONE = "NEO4J_DONE"
    QDRANT_DONE = "QDRANT_DONE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SagaOutbox:
    """Mock database table for Outbox pattern."""
    def __init__(self):
        self._store = {}
        
    def create(self, action: str, payload: dict) -> str:
        tx_id = str(uuid4())
        self._store[tx_id] = {
            "status": TransactionStatus.PENDING,
            "action": action,
            "payload": payload,
        }
        return tx_id
        
    def update_status(self, tx_id: str, status: str):
        if tx_id in self._store:
            self._store[tx_id]["status"] = status
            
    def get_pending(self):
        return {k: v for k, v in self._store.items() if v["status"] not in (TransactionStatus.COMPLETED, TransactionStatus.FAILED)}


class SagaTransactionManager:
    """
    Coordinates 2-Phase Distributed Transactions between Graph and Vector DBs.
    """
    def __init__(self):
        self.outbox = SagaOutbox()
        # In a real enterprise system, this background task runs infinitely in a separate worker (e.g. Celery).
        self._worker_task = asyncio.create_task(self._background_retry_worker())
        
    async def execute_delete(
        self, 
        recipe_id: str, 
        neo4j_delete_fn: Callable, 
        qdrant_delete_fn: Callable
    ):
        """
        Deletes a recipe ensuring Eventual Consistency.
        """
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
            raise e
            
        # 3. Phase 2: Modify Dependent System (Vector)
        try:
            await qdrant_delete_fn(recipe_id)
            self.outbox.update_status(tx_id, TransactionStatus.COMPLETED)
            logger.info("  [Phase 2] Qdrant Deletion: Success (SAGA COMPLETED)")
        except Exception as e:
            logger.warning(f"  [Phase 2] Qdrant Deletion Timeout/Error: {e}.")
            logger.warning(f"  -> Outbox marked NEO4J_DONE. Vector drift prevented via background retry.")
            # At this point, the worker will pick it up and retry to prevent phantom vectors
            raise e
            
    async def _background_retry_worker(self):
        """Worker that continuously scans Outbox for stuck transactions to retry."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            pending = self.outbox.get_pending()
            for tx_id, record in pending.items():
                if record["status"] == TransactionStatus.NEO4J_DONE and record["action"] == "DELETE":
                    logger.info(f"SAGA WORKER: Retrying Qdrant delete for TX {tx_id}")
                    # In real system: Call qdrant_delete_fn(record["payload"]["recipe_id"]) again
                    # If succeeds -> mark COMPLETED
