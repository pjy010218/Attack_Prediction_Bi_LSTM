import torch
from sentence_transformers import SentenceTransformer, util
from neo4j import GraphDatabase
from itertools import combinations
from tqdm import tqdm

class TtpSimilarityCalculator:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device} for embedding model.")
        self.model = SentenceTransformer('all-mpnet-base-v2', device=self.device)

    def close(self):
        self.driver.close()

    def calculate_and_store_similarities(self, threshold=0.75, batch_size=500):
        """
        Calculates semantic similarity between all TTPs and stores relationships in Neo4j.
        """
        print("Fetching TTP data from Neo4j...")
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (t:TTP) WHERE t.description IS NOT NULL RETURN t.id AS id, t.description AS description")
            ttp_data = [record.data() for record in result]

        if len(ttp_data) < 2:
            print("Not enough TTPs to calculate similarities.")
            return

        print(f"Found {len(ttp_data)} TTPs with descriptions. Generating embeddings...")
        
        ttp_ids = [t['id'] for t in ttp_data]
        ttp_descs = [t['description'] for t in ttp_data]
        
        embeddings = self.model.encode(
            ttp_descs, 
            convert_to_tensor=True, 
            batch_size=32, 
            show_progress_bar=True
        )

        print("Embeddings generated. Calculating similarity scores...")
        # 모든 TTP 쌍 간의 코사인 유사도 계산
        cosine_scores = util.cos_sim(embeddings, embeddings)

        print("Preparing relationships to create...")
        relationships_to_create = []
        # combinations를 사용하여 중복 없이 모든 쌍을 확인 (i, j) where i < j
        for i, j in tqdm(combinations(range(len(ttp_data)), 2), total=len(ttp_data)*(len(ttp_data)-1)//2):
            score = cosine_scores[i][j].item()
            if score >= threshold:
                relationships_to_create.append({
                    'id1': ttp_ids[i],
                    'id2': ttp_ids[j],
                    'score': round(score, 4)
                })

        if not relationships_to_create:
            print("No new similar relationships found above the threshold.")
            return

        print(f"\nFound {len(relationships_to_create)} new :SIMILAR_TO relationships. Writing to database in batches...")
        
        # 데이터베이스에 관계 생성
        cypher = """
        UNWIND $rels AS rel
        MATCH (t1:TTP {id: rel.id1})
        MATCH (t2:TTP {id: rel.id2})
        MERGE (t1)-[r:SIMILAR_TO]-(t2)
        SET r.score = rel.score
        """
        with self.driver.session(database=self.database) as session:
            total_rows = len(relationships_to_create)
            for i in range(0, total_rows, batch_size):
                batch = relationships_to_create[i:i + batch_size]
                session.run(cypher, rels=batch)
                print(f"  - Wrote batch {i // batch_size + 1} / {(total_rows + batch_size - 1) // batch_size} ({len(batch)} items)")
        
        print("✅ Similarity relationship creation complete.")


if __name__ == '__main__':
    # Neo4j 연결 정보는 실제 환경에 맞게 수정해주세요.
    calculator = TtpSimilarityCalculator(uri="bolt://localhost:7687", user="neo4j", password="password")
    # 임계값(threshold)은 필요에 따라 조절
    calculator.calculate_and_store_similarities(threshold=0.75)
    calculator.close()