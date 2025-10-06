#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVE-TTP Linker for creating semantic relationships between CVEs and TTPs.
Uses sentence-transformers to calculate similarity and creates EXPLOITED_BY relationships.
"""

import torch
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
from torch.cpu import is_available
from torch.cuda import device
from tqdm import tqdm

# Links CVEs and TTPs based on semantic similarity of their descriptions.
# Uses sentence-transformers to create embeddings and calculates cosine similarity.
class CVETTPLinker:
    # Initialize the CVE-TTP Linker.
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        
        # Args:
            # uri: Neo4j connection URI
            # user: Neo4j username
            # password: Neo4j password
            # database: Neo4j database name

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.similarity_threshold = 0.6
        self.top_k = 3

        if torch.cuda.is_available():
            print(f"Selected: GPU")
            
    # Close the Neo4j connection.
    def close(self):
        self.driver.close()

    def _fetch_data(self, cypher_query: str):
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]

    # Retrieve all CVE descriptions from Neo4j.
    def get_cve_descriptions(self) -> List[Tuple[str, str]]:
        
        # Returns:
           # List of tuples containing (cve_id, description)
        
        cypher = "MATCH (c:CVE) RETURN c.id as id, c.description as description"
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher)
            return [(record["id"], record["description"]) for record in result]
    
    # Retrieve all TTP descriptions from Neo4j.
    def get_ttp_descriptions(self) -> List[Tuple[str, str]]:
    
        # Returns:
            # List of tuples containing (ttp_id, description)
        
        cypher = "MATCH (t:TTP) RETURN t.id as id, t.description as description"
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher)
            return [(record["id"], record["description"]) for record in result]

    # Create embeddings for a list of descriptions using sentence-transformers.
    def create_embeddings(self, descriptions: List[str]) -> np.ndarray:    
        # Args:
            # descriptions: List of text descriptions
            
        # Returns:
            # numpy array of embeddings
        
        # Filter out empty or None descriptions
        valid_descriptions = [desc for desc in descriptions if desc and desc.strip()]
        
        if not valid_descriptions:
            return np.array([])
        
        # Create embeddings
        embeddings = self.model.encode(valid_descriptions, convert_to_tensor=False)
        return embeddings

    # Calculate cosine similarity between CVE and TTP embeddings.
    def calculate_similarities(self, cve_embeddings: np.ndarray, ttp_embeddings: np.ndarray) -> np.ndarray:

        # Args:
            # cve_embeddings: Array of CVE embeddings
            # ttp_embeddings: Array of TTP embeddings
            
        # Returns:
            # Similarity matrix where [i][j] is similarity between CVE i and TTP j
        
        if cve_embeddings.size == 0 or ttp_embeddings.size == 0:
            return np.array([])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(cve_embeddings, ttp_embeddings)
        return similarities

    # Find top-k most similar TTPs for each CVE above the similarity threshold.
    def find_top_similar_ttps(self, similarities: np.ndarray, cve_ids: List[str], 
                             ttp_ids: List[str]) -> List[Dict[str, Any]]:
        # Args:
            # similarities: Similarity matrix
            # cve_ids: List of CVE IDs
            # ttp_ids: List of TTP IDs
            
        # Returns:
            # List of dictionaries containing CVE-TTP pairs with similarity scores
        
        relationships = []
        
        for i, cve_id in enumerate(cve_ids):
            if i >= similarities.shape[0]:
                break
                
            # Get similarities for this CVE
            cve_similarities = similarities[i]
            
            # Create list of (ttp_index, similarity_score) pairs
            ttp_scores = [(j, score) for j, score in enumerate(cve_similarities) 
                         if score >= self.similarity_threshold]
            
            # Sort by similarity score (descending) and take top-k
            ttp_scores.sort(key=lambda x: x[1], reverse=True)
            top_ttps = ttp_scores[:self.top_k]
            
            # Add relationships for top TTPs
            for ttp_idx, score in top_ttps:
                if ttp_idx < len(ttp_ids):
                    relationships.append({
                        'cve_id': cve_id,
                        'ttp_id': ttp_ids[ttp_idx],
                        'similarity_score': float(score)
                    })
        
        return relationships

    # Create EXPLOITED_BY relationships in Neo4j for CVE-TTP pairs.
    def create_exploited_by_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        
        # Args:
            # relationships: List of dictionaries containing CVE-TTP pairs and similarity scores
        
        if not relationships:
            print("No relationships to create.")
            return
        
        cypher = (
            "UNWIND $relationships AS rel "
            "MATCH (c:CVE {id: rel.cve_id}) "
            "MATCH (t:TTP {id: rel.ttp_id}) "
            "MERGE (c)-[r:EXPLOITED_BY]->(t) "
            "SET r.similarity_score = rel.similarity_score, "
            "r.created_at = datetime()"
        )
        
        with self.driver.session(database=self.database) as session:
            session.run(cypher, relationships=relationships)
            print(f"Created {len(relationships)} EXPLOITED_BY relationships.")

    # Main method to link CVEs to TTPs based on semantic similarity.
    def link_cves_to_ttps(self, threshold=0.6, cve_batch_size=1024):
        print("Fetching data from Neo4j...")
        cve_data = self._fetch_data("MATCH (c:CVE) WHERE c.description IS NOT NULL RETURN c.id AS id, c.description AS description")
        ttp_data = self._fetch_data("MATCH (t:TTP) WHERE t.description IS NOT NULL RETURN t.id AS id, t.description AS description")

        if not cve_data or not ttp_data:
            print("Not enough data to create links.")
            return

        print(f"Found {len(cve_data)} CVEs and {len(ttp_data)} TTPs with descriptions.")

        # TTP 임베딩은 수가 적으므로 한 번에 생성
        print("Creating TTP embeddings...")
        ttp_descs = [t['description'] for t in ttp_data]
        ttp_embeddings = self.model.encode(ttp_descs, convert_to_tensor=True, show_progress_bar=True)

        # CVE 데이터를 배치로 나누어 처리
        total_cve_batches = (len(cve_data) + cve_batch_size - 1) // cve_batch_size

        for i in range(0, len(cve_data), cve_batch_size):
            batch_num = (i // cve_batch_size) + 1
            print(f"\n--- Processing CVE Batch {batch_num}/{total_cve_batches} ---")
            
            cve_batch = cve_data[i:i + cve_batch_size]
            cve_descs_batch = [c['description'] for c in cve_batch]

            # 1. CVE 임베딩을 배치 단위로 생성
            print(f"Creating embeddings for {len(cve_batch)} CVEs...")
            cve_embeddings_batch = self.model.encode(
                cve_descs_batch, 
                convert_to_tensor=True, 
                batch_size=64,
                show_progress_bar=True
            )

            # 2. 코사인 유사도 계산
            print("Calculating similarities...")
            cosine_scores = util.cos_sim(cve_embeddings_batch, ttp_embeddings)

            # 3. 관계를 생성할 데이터 준비
            relationships_to_create = []
            for j in range(len(cve_batch)):
                # 각 CVE에 대해 가장 유사한 TTP 찾기
                # 여기서는 상위 3개를 찾도록 예시를 구성
                top_results = torch.topk(cosine_scores[j], k=3)
                cve_id = cve_batch[j]['id']
                
                for score, idx in zip(top_results[0], top_results[1]):
                    if score.item() >= threshold:
                        ttp_id = ttp_data[idx]['id']
                        relationships_to_create.append({
                            'cve_id': cve_id,
                            'ttp_id': ttp_id,
                            'score': round(score.item(), 4)
                        })
            
            # 4. 데이터베이스에 관계 생성 (배치 단위)
            if relationships_to_create:
                print(f"Found {len(relationships_to_create)} new relationships. Writing to database...")
                cypher = """
                UNWIND $rels AS rel
                MATCH (c:CVE {id: rel.cve_id})
                MATCH (t:TTP {id: rel.ttp_id})
                MERGE (c)-[r:IS_EXPLOITED_BY]->(t)
                SET r.score = rel.score, r.source = 'embedding_v1', r.weight = 1.0
                """
                with self.driver.session(database=self.database) as session:
                    session.run(cypher, rels=relationships_to_create)
            else:
                print("No new relationships found in this batch.")

        print("\n✅ All CVEs processed.")

# Main function to run the CVE-TTP linking process.
def main():
    
    # Configuration - update these values
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"
    database = "neo4j"
    
    try:
        linker = CVETTPLinker(uri, user, password, database)
        linker.link_cves_to_ttps(threshold=0.6)
    except Exception as e:
        print(f"Error during linking process: {e}")
    finally:
        linker.close()


if __name__ == "__main__":
    main()
