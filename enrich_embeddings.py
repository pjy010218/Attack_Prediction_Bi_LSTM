import json
import argparse
import numpy as np
from neo4j import GraphDatabase
from tqdm import tqdm

class EmbeddingEnricher:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.embeddings = {}

    def close(self):
        self.driver.close()

    def load_embeddings(self, embeddings_file: str):
        """사전 훈련된 Node2Vec 임베딩을 로드합니다."""
        print(f"Loading base embeddings from '{embeddings_file}'...")
        with open(embeddings_file, 'r') as f:
            self.embeddings = json.load(f)
        print(f"Loaded {len(self.embeddings)} TTP embeddings.")

    def create_enriched_embeddings(self, output_file: str):
        """
        각 TTP의 이웃 노드 임베딩을 집계하여 '그래프 문맥 벡터'를 생성하고,
        기존 임베딩과 결합하여 보강된 임베딩을 만듭니다.
        """
        if not self.embeddings:
            print("Error: Base embeddings not loaded.")
            return

        enriched_embeddings = {}
        
        print("Enriching embeddings with graph context...")
        with self.driver.session(database=self.database) as session:
            # 모든 TTP에 대해 이웃 노드들을 조회
            for ttp_id, ttp_embedding in tqdm(self.embeddings.items()):
                query = """
                MATCH (t:TTP {tid: $tid})
                // PRECEDES 또는 SIMILAR_TO 관계로 연결된 모든 이웃 노드 조회
                OPTIONAL MATCH (t)-[:PRECEDES|SIMILAR_TO]-(neighbor:TTP)
                RETURN collect(DISTINCT neighbor.tid) AS neighbors
                """
                result = session.run(query, tid=ttp_id).single()
                neighbor_ttps = result['neighbors'] if result else []
                
                neighbor_embeddings = []
                for neighbor_id in neighbor_ttps:
                    if neighbor_id in self.embeddings:
                        neighbor_embeddings.append(self.embeddings[neighbor_id])
                
                # 1. 이웃 노드들의 임베딩 평균을 내어 '그래프 문맥 벡터' 생성
                if neighbor_embeddings:
                    graph_context_vector = np.mean(neighbor_embeddings, axis=0).tolist()
                else:
                    # 이웃이 없는 경우, 0으로 채운 벡터 사용
                    embedding_dim = len(ttp_embedding)
                    graph_context_vector = [0.0] * embedding_dim
                
                # 2. 기존 임베딩과 그래프 문맥 벡터를 결합(concatenate)
                enriched_vector = ttp_embedding + graph_context_vector
                enriched_embeddings[ttp_id] = enriched_vector

        print(f"\nEnrichment complete. New vector dimension: {len(enriched_vector)}")
        with open(output_file, 'w') as f:
            json.dump(enriched_embeddings, f)
        print(f"Saved {len(enriched_embeddings)} enriched embeddings to '{output_file}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enrich Node2Vec embeddings with GNN-like context.")
    parser.add_argument("--embeddings-file", type=str, default="ttp_embeddings.json", help="Path to the base TTP embeddings file.")
    parser.add_argument("--output-file", type=str, default="ttp_enriched_embeddings.json", help="Path to save the enriched embeddings.")
    args = parser.parse_args()

    enricher = EmbeddingEnricher(uri="bolt://localhost:7687", user="neo4j", password="password")
    enricher.load_embeddings(args.embeddings_file)
    enricher.create_enriched_embeddings(args.output_file)
    enricher.close()