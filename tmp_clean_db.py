import asyncio
from backend.config import get_settings
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

def clean_db():
    settings = get_settings()
    
    # Neo4j
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Cleaned Neo4j")
    
    # Qdrant
    try:
        qclient = QdrantClient(url=settings.qdrant_host, port=settings.qdrant_port)
    except:
        qclient = QdrantClient(url="localhost", port=6333)

    try:
        qclient.delete_collection("recipe_text")
        print("Cleaned Qdrant recipe_text")
    except Exception as e:
        print(f"Qdrant collection clear error: {e}")

if __name__ == "__main__":
    clean_db()
