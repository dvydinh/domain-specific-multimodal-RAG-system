"""
Database cleanup utility — wipes Neo4j and Qdrant for fresh ingestion.

Usage: python -m scripts.clean_db
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import get_settings
from neo4j import GraphDatabase
from qdrant_client import QdrantClient


def clean_db():
    settings = get_settings()

    # Neo4j
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()
    print("Cleaned Neo4j")

    # Qdrant
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    for collection in [settings.qdrant_text_collection, settings.qdrant_image_collection]:
        try:
            client.delete_collection(collection)
            print(f"Cleaned Qdrant collection: {collection}")
        except Exception as e:
            print(f"Collection '{collection}' not found or already clean: {e}")


if __name__ == "__main__":
    clean_db()
