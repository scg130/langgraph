from neo4j import GraphDatabase


class Neo4jManager:
    def __init__(self, url: str, user: str, password: str):
        self.driver = GraphDatabase.driver(url, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_entity(self, label: str, props: dict):
        with self.driver.session() as session:
            session.run(
                f"MERGE (n:{label} {{id: $id}}) SET n += $props", id=props.get("id"), props=props)

    def create_relation(self, src_label, src_id, rel, dst_label, dst_id):
        with self.driver.session() as session:
            session.run(
                f"MATCH (a:{src_label} {{id:$src_id}}),(b:{dst_label} {{id:$dst_id}}) MERGE (a)-[r:{rel}]->(b)", src_id=src_id, dst_id=dst_id)

    def run_query(self, cypher: str, params: dict = None):
        with self.driver.session() as session:
            res = session.run(cypher, params or {})
            return [r.data() for r in res]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
