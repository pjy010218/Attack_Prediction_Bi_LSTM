import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Set
from collections import defaultdict

# 이 스크립트는 sequence_generator_v2.py와 같은 디렉토리에 있어야 합니다.
from sequence_generator import SequenceGenerator, get_actor_ttps_from_layers

def generate_seeded_scenarios(
    real_sequences_file: str,
    output_file: str,
    actor_layers_dir: str,
    num_per_seed: int,
    min_len: int,
    max_len: int
):
    """
    실제 시퀀스의 시작 TTP를 '시드'로 사용하여, 각 시드마다
    지정된 수의 합성 시나리오를 생성합니다.
    """
    print("--- Synthetic Scenario Generation for Validation ---")
    
    # 1. 실제 시퀀스 파일 로드 및 시드 TTP 추출
    try:
        with open(real_sequences_file, 'r', encoding='utf-8') as f:
            real_sequences = json.load(f)
    except FileNotFoundError:
        print(f"Error: Real-world sequence file not found at '{real_sequences_file}'")
        return
        
    # 각 실제 시퀀스의 첫 번째 TTP를 시드로 사용
    seed_ttps = [seq[0] for seq in real_sequences if seq]
    if not seed_ttps:
        print("Error: No valid sequences found in the real-world data file to use as seeds.")
        return
    print(f"Extracted {len(seed_ttps)} seed TTPs from real-world data.")

    # 2. Actor 프로필 및 TTP-Actor 역인덱스 생성
    print(f"Loading actor profiles from '{actor_layers_dir}'...")
    actor_profiles = get_actor_ttps_from_layers(Path(actor_layers_dir))
    if not actor_profiles:
        print("Error: No actor profiles found.")
        return
        
    ttp_to_actors = defaultdict(list)
    for actor_id, ttps in actor_profiles.items():
        for ttp in ttps:
            ttp_to_actors[ttp].append(actor_id)

    # 3. 시나리오 생성기 초기화
    generator = SequenceGenerator(uri="bolt://localhost:7687", user="neo4j", password="password")
    
    # 4. 각 시드 TTP에 대해 지정된 수만큼 시나리오 생성
    synthetic_scenarios = []
    print(f"Generating {num_per_seed} scenarios for each of the {len(seed_ttps)} seeds...")
    
    for seed_ttp in tqdm(seed_ttps, desc="Processing Seeds"):
        actors_who_use_seed = ttp_to_actors.get(seed_ttp)
        if not actors_who_use_seed:
            print(f"\nWarning: No actor found who uses seed TTP '{seed_ttp}'. Skipping.")
            continue

        generated_count = 0
        max_attempts = num_per_seed * 20 # 무한 루프 방지
        attempts = 0
        while generated_count < num_per_seed and attempts < max_attempts:
            attempts += 1
            actor_id = random.choice(actors_who_use_seed)
            actor_ttps = actor_profiles[actor_id]
            
            # target_ttp 인자를 사용하여 시드 TTP에서 생성을 시작
            seq = generator.generate_sequence(
                actor_ttps,
                min_length=min_len,
                max_length=max_len,
                target_ttp=seed_ttp 
            )
            
            if seq:
                synthetic_scenarios.append(seq)
                generated_count += 1
    
    generator.close()

    # 5. 생성된 시퀀스 파일로 저장
    print(f"\nSuccessfully generated a total of {len(synthetic_scenarios)} synthetic scenarios.")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(synthetic_scenarios, f)
    print(f"Saved synthetic scenarios for validation to '{output_file}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic scenarios based on seed TTPs from a real-world dataset.")
    parser.add_argument("--real-file", type=str, required=True, help="Path to the JSON file of real-world sequences to use as seeds.")
    parser.add_argument("--output-file", type=str, default="synthetic_scenarios_for_validation.json", help="Path to save the generated synthetic scenarios.")
    parser.add_argument("--actor-layers", type=str, required=True, help="Directory containing the actor layer .json files.")
    parser.add_argument("--scenarios-per-seed", type=int, default=100, help="Number of synthetic scenarios to generate for each seed TTP.")
    parser.add_argument("--min-length", type=int, default=4, help="Minimum length of a valid sequence.")
    parser.add_argument("--max-length", type=int, default=4, help="Maximum length of a valid sequence.")

    args = parser.parse_args()
    random.seed(42)
    
    generate_seeded_scenarios(
        args.real_file,
        args.output_file,
        args.actor_layers,
        args.scenarios_per_seed,
        args.min_length,
        args.max_length
    )