import json
import random
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
from sequence_generator import SequenceGenerator, get_actor_ttps_from_layers

def augment_dataset(
    input_file: str,
    output_file: str,
    actor_layers_dir: str,
    target_count: int,
    min_len: int,
    max_len: int
):
    """
    데이터셋을 분석하여 TTP별 시퀀스 수가 부족한 경우,
    목표치에 도달하도록 시퀀스를 추가로 생성하여 데이터셋을 증강합니다.
    """
    # 1. 기존 시퀀스 및 Actor 프로필 로드
    print(f"Loading existing sequences from '{input_file}'...")
    with open(input_file, 'r', encoding='utf-8') as f:
        sequences = json.load(f)
    print(f"Loaded {len(sequences)} sequences.")

    print("Loading actor profiles...")
    actor_profiles = get_actor_ttps_from_layers(Path(actor_layers_dir))
    
    # TTP별로 어떤 Actor가 사용하는지 역인덱스 생성 (효율성)
    ttp_to_actors = defaultdict(list)
    for actor_id, ttps in actor_profiles.items():
        for ttp in ttps:
            ttp_to_actors[ttp].append(actor_id)
    all_known_ttps = set(ttp_to_actors.keys())

    print(f"\n--- DEBUG STEP 1: Initial State ---")
    print(f"Loaded profiles for {len(actor_profiles)} actors, covering {len(all_known_ttps)} unique TTPs.")

    # 2. 현재 데이터셋의 TTP별 시퀀스 수 분석
    ttp_counts = Counter()
    for seq in sequences:
        for ttp in set(seq):
            ttp_counts[ttp] += 1

    print(f"\n--- DEBUG STEP 2: TTP Count Analysis ---")
    print(f"Counted sequence appearances for {len(ttp_counts)} unique TTPs found in the input file.")
    if ttp_counts:
        print(f"Top 5 MOST common TTPs: {ttp_counts.most_common(5)}")
        print(f"Top 5 LEAST common TTPs: {ttp_counts.most_common()[-5:]}")
    else:
        print("Warning: No TTPs were counted from the input file.")
            
    # 3. 목표치에 미달하는 TTP 목록 (부족분) 계산
    deficiencies = {}
    print(f"\n--- DEBUG STEP 3: Checking Deficiencies (Target Count: {target_count}) ---")
    checked_count = 0
    for ttp in sorted(list(all_known_ttps)):
        count = ttp_counts.get(ttp, 0)

        if checked_count < 5:
            print(f"  - Checking '{ttp}': Found in {count} sequences. Is {count} < {target_count}? -> {count < target_count}")

        if count < target_count:
            deficiencies[ttp] = target_count - count
        checked_count += 1

    if not deficiencies:
        print("All TTPs meet the target count. No augmentation needed.")
        return

    print(f"Found {len(deficiencies)} TTPs that need more sequences.")

    # 4. 부족분을 채우기 위한 목표지향적 시퀀스 생성
    print("Starting targeted sequence generation to augment dataset...")
    generator = SequenceGenerator(uri="bolt://localhost:7687", user="neo4j", password="password")
    newly_generated_sequences = []
    
    # tqdm을 사용하여 진행 상황 표시
    pbar = tqdm(deficiencies.items(), desc="Augmenting dataset")
    for target_ttp, num_needed in pbar:
        pbar.set_postfix_str(f"Generating for {target_ttp}")
        
        actors_who_use_target = ttp_to_actors.get(target_ttp)
        if not actors_who_use_target:
            continue # 해당 TTP를 사용하는 Actor가 없으면 생성 불가

        generated_count = 0
        max_attempts = num_needed * 20 # 무한 루프 방지
        attempts = 0
        while generated_count < num_needed and attempts < max_attempts:
            attempts += 1
            # 해당 TTP를 사용하는 Actor 중 하나를 무작위로 선택
            actor_id = random.choice(actors_who_use_target)
            actor_ttps = actor_profiles[actor_id]
            
            # target_ttp에서 시작하는 시퀀스 생성
            seq = generator.generate_sequence(
                actor_ttps, 
                min_length=min_len, 
                max_length=max_len, 
                target_ttp=target_ttp
            )
            
            if seq:
                newly_generated_sequences.append(seq)
                generated_count += 1

    generator.close()
    print(f"Generated {len(newly_generated_sequences)} new sequences.")

    # 5. 기존 시퀀스와 새로 생성된 시퀀스를 합치고 저장
    final_sequences = sequences + newly_generated_sequences
    random.shuffle(final_sequences)
    
    print(f"Created a new augmented dataset with {len(final_sequences)} sequences.")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_sequences, f)
    print(f"Saved augmented sequences to '{output_file}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Augment a sequence dataset to ensure minimum counts for all TTPs.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the original sequence JSON file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the new augmented sequence file.")
    parser.add_argument("--actor-layers", type=str, required=True, help="Directory containing the actor layer .json files.")
    parser.add_argument("--target-count", type=int, default=200, help="The minimum number of sequences each TTP should appear in.")
    parser.add_argument("--min-length", type=int, default=6, help="Minimum length of a generated sequence.")
    parser.add_argument("--max-length", type=int, default=15, help="Maximum length of a generated sequence.")
    
    args = parser.parse_args()
    random.seed(42)
    augment_dataset(args.input_file, args.output_file, args.actor_layers, args.target_count, args.min_length, args.max_length)