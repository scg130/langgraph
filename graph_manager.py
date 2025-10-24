from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()


NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class Neo4jManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))


    def close(self):
        self.driver.close()


    def create_entity(self, label: str, props: dict):
        with self.driver.session() as session:
            session.run(f"MERGE (n:{label} {{id: $id}}) SET n += $props", id=props.get("id"), props=props)


    def create_relation(self, src_label, src_id, rel, dst_label, dst_id):
        with self.driver.session() as session:
            session.run(f"MATCH (a:{src_label} {{id:$src_id}}),(b:{dst_label} {{id:$dst_id}}) MERGE (a)-[r:{rel}]->(b)", src_id=src_id, dst_id=dst_id)


    def run_query(self, cypher: str, params: dict = None):
        with self.driver.session() as session:
            res = session.run(cypher, params or {})
            return [r.data() for r in res]


neo4j_mgr = Neo4jManager()