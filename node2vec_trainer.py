import json
import argparse
import networkx as nx
from neo4j import GraphDatabase
from node2vec import Node2Vec

class Node2VecTrainer:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def _fetch_graph_to_networkx(self):
        """Neo4j에서 TTP 관계를 가져와 NetworkX 그래프로 변환합니다."""
        query = """
        MATCH (t1:TTP)-[r:PRECEDES|SIMILAR_TO]->(t2:TTP)
        RETURN t1.tid AS source, t2.tid AS target, type(r) AS rel_type
        """
        print("Fetching graph data from Neo4j...")
        graph = nx.DiGraph()
        with self.driver.session(database=self.database) as session:
            results = session.run(query)
            for record in results:
                # 관계 유형에 따라 가중치 부여 (PRECEDES가 더 중요)
                weight = 1.0 if record["rel_type"] == "PRECEDES" else 0.5
                graph.add_edge(record["source"], record["target"], weight=weight)
        
        print(f"Built NetworkX graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        return graph

    def train_and_save_embeddings(self, output_file: str, dimensions: int, walk_length: int, num_walks: int):
        """Node2Vec 모델을 훈련하고 임베딩을 파일로 저장합니다."""
        graph = self._fetch_graph_to_networkx()
        
        print("Initializing Node2Vec...")
        # is_directed=True, weight_key='weight'를 통해 가중 방향 그래프로 설정
        node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4, quiet=True, weight_key='weight')
        print("Training Node2Vec model (this may take a while)...")
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # 결과를 딕셔너리 형태로 변환
        embeddings = {node: model.wv[node].tolist() for node in model.wv.index_to_key}

        print(f"Training complete. Saving {len(embeddings)} embeddings to '{output_file}'...")
        with open(output_file, 'w') as f:
            json.dump(embeddings, f)
        print("Embeddings saved successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Node2Vec model on the TTP graph.")
    parser.add_argument("--output-file", type=str, default="ttp_embeddings.json", help="Path to save the generated TTP embeddings.")
    parser.add_argument("--dimensions", type=int, default=128, help="Dimensionality of the embeddings.")
    parser.add_argument("--walk-length", type=int, default=30, help="Length of each random walk.")
    parser.add_argument("--num-walks", type=int, default=200, help="Number of random walks per node.")
    args = parser.parse_args()

    trainer = Node2VecTrainer(uri="bolt://localhost:7687", user="neo4j", password="password")
    trainer.train_and_save_embeddings(args.output_file, args.dimensions, args.walk_length, args.num_walks)
    trainer.close()