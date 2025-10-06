import json
import random
import math
import re
import argparse
from typing import List, Dict, Any, Set, Optional
from neo4j import GraphDatabase
from pathlib import Path
import numpy as np
from tqdm import tqdm

def get_actor_ttps_from_layers(layer_dir: Path) -> Dict[str, Set[str]]:
    """
    지정된 디렉터리의 모든 ATT&CK Navigator 레이어 JSON 파일을 파싱하여
    각 Actor(group_id)가 사용하는 TTP 세트를 정확하게 추출합니다.
    다양한 JSON 구조에 대응할 수 있도록 안정성을 높였습니다.
    """
    actor_profiles: Dict[str, Set[str]] = {}
    
    json_files = list(layer_dir.glob("*.json"))
    if not json_files:
        print(f"Warning: No JSON files found in directory '{layer_dir}'")
        return {}

    for layer_file in json_files:
        try:
            with layer_file.open("r", encoding="utf-8") as f:
                layer = json.load(f)

            actor_id = None
            
            # 1차 시도: 'metadata' 필드에서 group_id 찾기
            metadata = layer.get("metadata", [])
            for item in metadata:
                if item.get("name", "").lower() == "group_id":
                    actor_id = item.get("value")
                    break
            
            # 2차 시도: 1차 시도가 실패하면 'name' 필드에서 파싱 (예: "Ke3chang (G0004)")
            if not actor_id:
                name = layer.get("name", "")
                match = re.search(r'\((G\d{4})\)', name)
                if match:
                    actor_id = match.group(1)

            if not actor_id:
                continue # 두 방법 모두 실패하면 이 파일은 건너뜀

            # TTP ID 추출
            ttps = {tech["techniqueID"] for tech in layer.get("techniques", []) if tech.get("techniqueID")}
            
            if actor_id in actor_profiles:
                actor_profiles[actor_id].update(ttps)
            else:
                actor_profiles[actor_id] = ttps
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse layer file '{layer_file.name}'. Error: {e}. Skipping.")
            
    return actor_profiles

class SequenceGenerator:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
        self.phase1_tactics = {"reconnaissance", "resource-development", "initial-access"}
        self.phase2_tactics = {
            "execution", "persistence", "privilege-escalation", "defense-evasion",
            "credential-access", "discovery", "lateral-movement", "collection"
        }
        self.phase3_tactics = {"command-and-control", "exfiltration", "impact"}
        
        # 편향 가중치 설정
        self.weights = {
            "tactic": 2.5,  # 전술 흐름의 중요도를 가장 높게 설정
            "centrality": 1.5
        }
        self.temperature = 0.5

    def close(self):
        self.driver.close()

    def _get_ttp_tactic_map(self, ttp_ids: Set[str]) -> Dict[str, str]:
        """주어진 TTP 목록에 대한 Tactic 맵을 미리 생성합니다."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                UNWIND $ttp_ids AS tid
                MATCH (t:TTP {tid: tid})-[:BELONGS_TO]->(tac:Tactic)
                RETURN t.tid AS ttp_id, tac.name AS tactic
            """, ttp_ids=list(ttp_ids))
            return {r["ttp_id"]: r["tactic"] for r in result}
    
    def _get_phase(self, tactic: str) -> int:
        """Tactic 이름에 해당하는 킬 체인 단계를 반환합니다."""
        if tactic in self.phase1_tactics: return 1
        if tactic in self.phase2_tactics: return 2
        if tactic in self.phase3_tactics: return 3
        return 0

    def generate_sequence(self, actor_ttp_ids: Set[str], min_length: int = 8, max_length: int = 15, target_ttp: Optional[str] = None) -> List[str] | None:
        if not actor_ttp_ids: return None
        
        ttp_tactic_map = self._get_ttp_tactic_map(actor_ttp_ids)
        if not ttp_tactic_map: return None

        if target_ttp and target_ttp in actor_ttp_ids:
            start_node = target_ttp
        else:
            initial_access_ttps = [ttp for ttp, tactic in ttp_tactic_map.items() if tactic == 'initial-access']
            if not initial_access_ttps: return None
            start_node = random.choice(initial_access_ttps)

        current_ttp = start_node
        sequence = [current_ttp]
        remaining_ttps = actor_ttp_ids - {current_ttp}
        walk_length = random.randint(min_length - 1, max_length - 1)

        for _ in range(walk_length):
            if not remaining_ttps: break

            current_tactic = ttp_tactic_map.get(current_ttp)
            if not current_tactic: break
            current_phase = self._get_phase(current_tactic)

            with self.driver.session(database=self.database) as session:
                candidates_result = session.run("""
                    MATCH (current:TTP {tid: $current_tid})
                    OPTIONAL MATCH (current)-[:PRECEDES]->(candidate:TTP)
                    WHERE candidate.tid IN $remaining_ttps
                    RETURN collect(DISTINCT candidate.tid) AS candidates
                """, current_tid=current_ttp, remaining_ttps=list(remaining_ttps))
                
                candidates_ids = candidates_result.single()['candidates']
                
                # PRECEDES 관계가 없는 경우, 같은 페이즈 내 다른 TTP를 후보로 추가
                if not candidates_ids:
                    fallback_candidates = [
                        ttp for ttp in remaining_ttps 
                        if self._get_phase(ttp_tactic_map.get(ttp, '')) == current_phase
                    ]
                    candidates_ids.extend(fallback_candidates)
                
                if not candidates_ids: break

                # 후보들의 중심성 값을 한번에 조회
                centrality_result = session.run("""
                    UNWIND $candidates AS tid
                    MATCH (t:TTP {tid: tid})
                    RETURN t.tid AS ttp_id, coalesce(t.betweenness_centrality, 0.0) AS centrality
                """, candidates=candidates_ids)
                candidates_info = {r['ttp_id']: {'centrality': r['centrality']} for r in centrality_result}

            scores, valid_candidates = [], []
            for cand_id in candidates_ids:
                cand_tactic = ttp_tactic_map.get(cand_id)
                if not cand_tactic: continue
                
                cand_phase = self._get_phase(cand_tactic)
                
                # 전술 흐름 점수 계산 (Phase 기반)
                tactic_score = 0.0
                if cand_phase > current_phase: tactic_score = 1.0  # 진행
                elif cand_phase == current_phase: tactic_score = 0.5 # 유지
                else: tactic_score = -1.0 # 역행 (패널티)

                # 중심성 점수 (로그 스케일링으로 값 안정화)
                centrality_score = math.log1p(candidates_info.get(cand_id, {}).get('centrality', 0.0))
                
                final_score = (tactic_score * self.weights['tactic'] +
                               centrality_score * self.weights['centrality'])
                
                scores.append(final_score)
                valid_candidates.append(cand_id)

            if not valid_candidates: break

            # Softmax 확률 계산
            scores_with_temp = np.asarray(scores) / self.temperature
            probabilities = np.exp(scores_with_temp) / np.sum(np.exp(scores_with_temp))
            next_ttp = np.random.choice(valid_candidates, p=probabilities)
            
            sequence.append(next_ttp)
            current_ttp = next_ttp
            if current_ttp in remaining_ttps:
                remaining_ttps.remove(current_ttp)
        
        return sequence if len(sequence) >= min_length else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plausible attack sequences from actor profiles.")
    parser.add_argument("--layer-dir", type=str, default="layers", help="Directory containing the actor layer .json files.")
    parser.add_argument("--num-sequences", type=int, default=1000, help="Total number of sequences to generate.")
    parser.add_argument("--min-length", type=int, default=6, help="Minimum length of a valid sequence.")
    parser.add_argument("--max-length", type=int, default=15, help="Maximum length of a sequence.")
    parser.add_argument("--output-file", type=str, default="attack_sequences.json", help="Output file to save sequences.")
    args = parser.parse_args()

    actor_profiles = get_actor_ttps_from_layers(Path(args.layer_dir))
    generator = SequenceGenerator(uri="bolt://localhost:7687", user="neo4j", password="password")
    
    all_generated_sequences = []
    print(f"Generating up to {args.num_sequences} sequences for all actors...")
    
    # 모든 Actor에 대해 시퀀스 생성
    for actor_id, actor_ttps in actor_profiles.items():
        print(f"  -> Generating for actor {actor_id}...")
        for _ in range(args.num_sequences // len(actor_profiles)): # Actor별로 할당량만큼 생성
            seq = generator.generate_sequence(actor_ttps, min_length=args.min_length, max_length=args.max_length)
            if seq:
                all_generated_sequences.append(seq)

    print(f"\nSuccessfully generated {len(all_generated_sequences)} valid sequences (length >= {args.min_length}).")
    
    with open(args.output_file, 'w') as f:
        json.dump(all_generated_sequences, f)
    print(f"Saved sequences to '{args.output_file}'.")

    generator.close()