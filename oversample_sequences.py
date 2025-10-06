import json
import random
import argparse
from collections import Counter
from tqdm import tqdm

def oversample_sequences(
    input_file: str,
    output_file: str,
    top_n: int,
    target_size: int
):
    """
    소수 클래스(Top-N에 속하지 않는 TTP)가 포함된 시퀀스를 오버샘플링하여
    목표 데이터셋 크기를 달성합니다.
    """
    print(f"Loading sequences from '{input_file}'...")
    with open(input_file, 'r', encoding='utf-8') as f:
        sequences = json.load(f)
    
    current_size = len(sequences)
    print(f"Loaded {current_size} sequences.")

    if current_size >= target_size:
        print("Current dataset size is already greater than or equal to the target size. No oversampling needed.")
        return

    # 1. 전체 TTP의 등장 빈도를 계산하여 Top-N (다수 클래스) TTP 식별
    ttp_counts = Counter(ttp for seq in sequences for ttp in seq)
    majority_ttps = {item[0] for item in ttp_counts.most_common(top_n)}
    print(f"Identified Top-{top_n} majority TTPs (e.g., {list(majority_ttps)[:5]}...).")

    # 2. 다수 클래스 시퀀스와 소수 클래스 시퀀스를 분리
    majority_sequences = []
    minority_sequences = []
    for seq in sequences:
        is_minority = True
        for ttp in set(seq):
            if ttp in majority_ttps:
                is_minority = False
                break
        if is_minority:
            minority_sequences.append(seq)
        else:
            majority_sequences.append(seq)
            
    print(f"Found {len(majority_sequences)} sequences containing at least one majority TTP.")
    print(f"Found {len(minority_sequences)} sequences containing only minority TTPs.")

    if not minority_sequences:
        print("Warning: No minority sequences found to oversample. Cannot reach target size.")
        return

    # 3. 목표 크기에 도달하기 위해 필요한 샘플 수 계산
    num_to_generate = target_size - current_size
    print(f"Need to generate {num_to_generate} more sequences by oversampling minority class sequences.")

    # 4. 소수 클래스 시퀀스 중에서 무작위로 복제하여 추가 (중복 허용)
    # random.choices는 중복을 허용하여 리스트에서 여러 항목을 뽑음
    oversampled_part = random.choices(minority_sequences, k=num_to_generate)
    
    # 5. 기존 시퀀스와 새로 생성된 시퀀스를 합치고 순서를 섞음
    final_sequences = sequences + oversampled_part
    random.shuffle(final_sequences)

    print(f"\nCreated a new balanced dataset with {len(final_sequences)} sequences.")

    # 6. 새로운 시퀀스 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_sequences, f)
    print(f"Saved oversampled sequences to '{output_file}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Oversample sequences containing minority class TTPs.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the sequence JSON file to be oversampled.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the new oversampled sequence file.")
    parser.add_argument("--top-n", type=int, default=20, help="The number of most frequent TTPs to be considered 'majority class'.")
    parser.add_argument("--target-size", type=int, required=True, help="The desired total number of sequences in the final dataset.")
    
    args = parser.parse_args()
    random.seed(42) # 재현성을 위한 시드 고정
    oversample_sequences(args.input_file, args.output_file, args.top_n, args.target_size)