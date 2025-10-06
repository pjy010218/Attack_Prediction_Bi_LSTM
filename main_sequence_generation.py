import json
from pathlib import Path
from tqdm import tqdm
from sequence_generator import SequenceGenerator, get_actor_ttps_from_layers

def main():
    # 설정값
    LAYER_DIR = "layers"
    OUTPUT_FILE = "attack_sequences.json"
    SEQUENCES_PER_ACTOR = 50 # 각 공격 그룹당 생성할 시나리오 수

    # 1. Layer 파일에서 공격 그룹별 TTP 목록 로드
    print(f"Loading actor profiles from '{LAYER_DIR}' directory...")
    actor_profiles = get_actor_ttps_from_layers(Path(LAYER_DIR))
    print(f"Loaded {len(actor_profiles)} actor profiles.")

    # 2. SequenceGenerator 인스턴스화
    generator = SequenceGenerator(uri="bolt://localhost:7687", user="neo4j", password="password")

    all_sequences = []
    print(f"\nGenerating {SEQUENCES_PER_ACTOR} sequences for each actor...")

    # 3. 각 공격 그룹에 대해 지정된 횟수만큼 시퀀스 생성
    for actor_id, actor_ttps in tqdm(actor_profiles.items(), desc="Generating Sequences"):
        for _ in range(SEQUENCES_PER_ACTOR):
            seq = generator.generate_sequence(actor_ttps)
            if seq and len(seq) > 1: # 길이가 1 이상인 유의미한 시퀀스만 추가
                all_sequences.append(seq)

    generator.close()

    # 4. 생성된 모든 시퀀스를 파일로 저장
    print(f"\nSuccessfully generated a total of {len(all_sequences)} attack sequences.")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_sequences, f)
    print(f"All sequences saved to '{OUTPUT_FILE}'.")


if __name__ == '__main__':
    main()