import json
import random
import math
import argparse
from tqdm import tqdm
from typing import List, Dict, Any

import numpy as np
from neo4j import GraphDatabase

class ThreatVectorizer:
    """
    TTP 시퀀스를 기계 학습 모델이 사용할 수 있는 벡터 시퀀스로 변환하는 클래스.
    리팩토링된 최종 버전 (19+2 = 21차원 벡터 생성)
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.vector_dim = 19  # [수정] 기본 벡터 차원을 19로 설정
        self._one_hot_tactics_map = {
            'reconnaissance': 0, 'resource-development': 1, 'initial-access': 2,
            'execution': 3, 'persistence': 4, 'privilege-escalation': 5,
            'defense-evasion': 6, 'credential-access': 7, 'discovery': 8,
            'lateral-movement': 9, 'collection': 10, 'command-and-control': 11,
            'exfiltration': 12, 'impact': 13
        }
        self._one_hot_platforms_map = {
            'windows': 0, 'linux': 1, 'macos': 2, 'network': 3,
            'aws': 4, 'gcp': 5, 'azure': 6, 'office-365': 7
        }
        self.top_25_cwes = [
            "CWE-79", "CWE-787", "CWE-89", "CWE-352", "CWE-22", "CWE-125",
            "CWE-78", "CWE-416", "CWE-862", "CWE-434", "CWE-94", "CWE-20",
            "CWE-77", "CWE-287", "CWE-269", "CWE-502", "CWE-200", "CWE-863",
            "CWE-918", "CWE-119", "CWE-476", "CWE-798", "CWE-190", "CWE-400",
            "CWE-306" 
        ]

    def close(self):
        self.driver.close()

    @staticmethod
    def _log1p(value: float) -> float:
        return float(np.log1p(value))

    @staticmethod
    def _mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _std(values: List[float]) -> float:
        return float(np.std(values)) if len(values) >= 2 else 0.0

    def _one_hot_encode(self, values: List[str], mapping: Dict[str, int], dim: int) -> List[float]:
        vec = [0.0] * dim
        for val in values:
            if val in mapping:
                vec[mapping[val]] = 1.0
        return vec

    def _one_hot_tactics(self, tactics: List[str]) -> List[float]:
        return self._one_hot_encode(tactics, self._one_hot_tactics_map, 14)

    def _one_hot_platforms(self, platforms: List[str]) -> List[float]:
        return self._one_hot_encode(platforms, self._one_hot_platforms_map, 8)

    def _fetch_ttp_context(self, tid: str) -> tuple:
        """TTP에 필요한 핵심 컨텍스트 정보만 조회하여 튜플로 반환합니다."""
        cypher = """
        MATCH (t:TTP {tid: $tid})
        
        OPTIONAL MATCH (t)-[:BELONGS_TO]->(tac:Tactic)
        WITH t, collect(DISTINCT tac.name) AS tactics
        
        OPTIONAL MATCH (t)<-[:IS_EXPLOITED_BY]-(c:CVE)
        WITH t, tactics, collect(DISTINCT {id: c.id, score: c.cvss_score}) AS cve_scores

        OPTIONAL MATCH (t)<-[:IS_EXPLOITED_BY]-(cve:CVE)-[:IS_INSTANCE_OF]->(w:CWE)
        
        RETURN
            tactics,
            cve_scores,
            collect(DISTINCT {id: w.id, score: CASE WHEN w.id IN $top_25_cwes THEN 10.0 ELSE 5.0 END}) AS cwe_scores,
            coalesce(t.betweenness_centrality, 0.0) AS centrality
        """
        with self.driver.session(database=self.database) as session:
            record = session.run(cypher, tid=tid, top_25_cwes=self.top_25_cwes).single()
            if not record:
                return [], [], [], 0.0

            tactics = record.get("tactics", [])
            cve_scores = record.get("cve_scores", [])
            cwe_scores = record.get("cwe_scores", [])
            centrality = record.get("centrality", 0.0)
            
            return tactics, cve_scores, cwe_scores, centrality

    def vectorize_ttp(self, tid: str) -> List[float]:
        """하나의 TTP ID에 대한 19차원 벡터를 생성합니다."""
        context_tuple = self._fetch_ttp_context(tid)
        if not context_tuple:
            return [0.0] * self.vector_dim
            
        tactics, cve_scores, cwe_scores, centrality = context_tuple

        # 14차원: Tactics 원-핫 인코딩
        tactic_vector = self._one_hot_tactics(tactics)

        # 5차원: 수치형 특성 집계
        cve_s = [rec.get('score', 0.0) for rec in cve_scores if rec and rec.get('score') is not None]
        cwe_s = [rec.get('score', 0.0) for rec in cwe_scores if rec and rec.get('score') is not None]

        numerical_vector = [
            self._mean(cve_s),
            self._std(cve_s),
            self._mean(cwe_s),
            self._std(cwe_s),
            float(centrality or 0.0)
        ]

        # 모든 벡터를 합쳐 19차원 벡터 반환
        return tactic_vector + numerical_vector

    def vectorize_sequence(self, ttp_ids_or_tids: List[str]) -> List[List[float]]:
        """TTP 시퀀스를 21차원 벡터 시퀀스로 변환합니다."""
        sequence_vectors = []
        seq_len = len(ttp_ids_or_tids)
        max_possible_len = 15.0

        for i, tid in enumerate(ttp_ids_or_tids):
            try:
                # 1. TTP 고유의 19차원 벡터를 가져옵니다.
                ttp_vector = self.vectorize_ttp(tid)

                # 2. 2차원의 위치 정보를 계산합니다.
                absolute_step = i / max_possible_len
                relative_step = i / (seq_len - 1) if seq_len > 1 else 0.0

                # 3. 두 벡터를 합쳐 최종 21차원 벡터를 만듭니다.
                ttp_vector.extend([absolute_step, relative_step])
                sequence_vectors.append(ttp_vector)

            except Exception as e:
                print(f"Error vectorizing {tid}: {e}. Appending zero vector.")
                sequence_vectors.append([0.0] * (self.vector_dim + 2))
                
        return sequence_vectors

def load_json(path: str):
    """
    지정된 경로의 JSON 파일을 읽어 파이썬 객체로 반환합니다.
    """
    with open("attack_sequences_augmented.json", "r", encoding="utf-8") as f:
        return json.load(f)     

def main():
    parser = argparse.ArgumentParser(description="Vectorize TTP sequences into feature vectors.")
    parser.add_argument("--file", type=str, required=True, help="Path to JSON file of TTP ID sequences")
    parser.add_argument("--output", type=str, default="vectorized_sequences.json", help="Path to save output vectors")
    args = parser.parse_args()

    sequences = load_json(args.file)
    print(f"Found {len(sequences)} sequences to vectorize.")

    vect = ThreatVectorizer(uri="bolt://localhost:7687", user="neo4j", password="password")
    
    all_vectors = []
    # tqdm을 사용하여 진행 상황 표시
    try:
        iterable = tqdm(sequences, desc="Vectorizing sequences")
    except ImportError:
        iterable = sequences

    for seq in iterable:
        vectors = vect.vectorize_sequence(seq)
        all_vectors.append(vectors)
    
    with open(args.output, "w") as f:
        json.dump(all_vectors, f)

    print(f"\nSaved {len(all_vectors)} vectorized sequences to '{args.output}'.")
    vect.close()

if __name__ == "__main__":
    main()