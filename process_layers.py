import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from stix2 import parse # stix2 라이브러리 임포트

# Tactic의 논리적 순서
TACTIC_ORDER = [
    "reconnaissance", "resource-development", "initial-access", "execution",
    "persistence", "privilege-escalation", "defense-evasion", "credential-access",
    "discovery", "lateral-movement", "collection", "command-and-control",
    "exfiltration", "impact"
]

def build_ttp_to_tactic_map(stix_file_path: Path) -> Dict[str, str]:
    """
    enterprise-attack.json 파일을 파싱하여
    TTP ID(TID)를 Tactic 이름에 매핑하는 딕셔너리를 생성합니다.
    """
    print(f"Building TTP-to-Tactic map from '{stix_file_path.name}'...")
    ttp_map = {}
    try:
        with open(stix_file_path, 'r', encoding='utf-8') as f:
            bundle = parse(f.read(), allow_custom=True)
        
        for obj in bundle.objects:
            if obj.get('type') == 'attack-pattern' and obj.get('external_references'):
                tid = ""
                # external_references에서 TID (예: T1113) 추출
                for ref in obj.get('external_references', []):
                    if ref.get('source_name') == 'mitre-attack':
                        tid = ref.get('external_id')
                        break
                
                # kill_chain_phases에서 Tactic 이름 추출
                if tid and obj.get('kill_chain_phases'):
                    for phase in obj.get('kill_chain_phases', []):
                        if phase.get('kill_chain_name') == 'mitre-attack':
                            ttp_map[tid] = phase.get('phase_name')
                            break
        print(f"Map built successfully with {len(ttp_map)} entries.")
        return ttp_map
    except Exception as e:
        print(f"Error building TTP-to-Tactic map: {e}")
        return {}


def process_layer_file(file_path: Path, ttp_tactic_map: Dict[str, str]) -> List[str]:
    """하나의 Navigator Layer JSON 파일을 처리하여 정렬된 TTP 시퀀스를 반환합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    actor_techniques: List[Tuple[str, str]] = []
    
    for tech in data.get('techniques', []):
        # --- Fixed part: 'enabled' 대신 'score'의 존재 여부로 판단 ---
        # 'score' 키가 존재하고 0보다 크면 사용된 기술로 간주
        if tech.get('score', 0) > 0:
            tech_id = tech.get('techniqueID')
            if tech_id:
                # 미리 만들어둔 맵에서 Tactic 정보를 조회
                tactic = ttp_tactic_map.get(tech_id)
                if tactic:
                    actor_techniques.append((tech_id, tactic))
    
    if not actor_techniques:
        return []

    def get_tactic_index(tactic_name: str) -> int:
        try:
            return TACTIC_ORDER.index(tactic_name)
        except ValueError:
            return 99

    sorted_techniques = sorted(actor_techniques, key=lambda x: get_tactic_index(x[1]))
    
    return [tech_id for tech_id, tactic in sorted_techniques]

def main():
    parser = argparse.ArgumentParser(description="Process MITRE ATT&CK Navigator layer files to generate attack sequences.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the layer .json files.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output JSON file for vectorizer.py.")
    # --- Fixed part: enterprise-attack.json 파일 경로를 인자로 받도록 추가 ---
    parser.add_argument("--stix-file", type=str, required=True, help="Path to the enterprise-attack.json STIX file.")
    args = parser.parse_args()

    # --- Fixed part: TTP-Tactic 맵을 먼저 생성 ---
    ttp_tactic_map = build_ttp_to_tactic_map(Path(args.stix_file))
    if not ttp_tactic_map:
        print("Could not build TTP-Tactic map. Exiting.")
        return

    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    all_sequences = []
    json_files = list(input_path.glob('*.json'))
    print(f"\nFound {len(json_files)} layer files to process...")

    for file_path in json_files:
        sequence = process_layer_file(file_path, ttp_tactic_map)
        if sequence:
            all_sequences.append(sequence)
    
    print(f"\nSuccessfully generated {len(all_sequences)} attack sequences.")

    if all_sequences:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            # 이제 단일 시퀀스가 아닌, 모든 시퀀스의 리스트를 저장
            json.dump(all_sequences, f, indent=4)
        print(f"Saved all {len(all_sequences)} sequences to '{args.output_file}'.")
    else:
        print("No valid sequences were generated.")

if __name__ == "__main__":
    main()