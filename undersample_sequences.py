import json
import random
import argparse
from collections import Counter, defaultdict

def undersample_sequences(
    input_file: str,
    output_file: str,
    majority_ttps: list[str],
    limit: int
):
    """
    다수 클래스 TTP가 포함된 시퀀스를 언더샘플링하여 균형잡힌 데이터셋을 생성합니다.
    """
    print(f"Loading sequences from '{input_file}'...")
    with open(input_file, 'r', encoding='utf-8') as f:
        sequences = json.load(f)
    print(f"Loaded {len(sequences)} total sequences.")

    # 1. 각 시퀀스가 어떤 TTP를 포함하는지 인덱싱
    #    TTP -> [이 TTP를 포함하는 시퀀스의 인덱스 목록]
    ttp_to_seq_indices = defaultdict(list)
    for i, seq in enumerate(sequences):
        for ttp in set(seq): # 시퀀스 내 중복 TTP는 한 번만 카운트
            ttp_to_seq_indices[ttp].append(i)

    # 2. 다수 클래스 TTP가 포함된 시퀀스와 그렇지 않은 시퀀스를 분리
    majority_indices = set()
    for ttp in majority_ttps:
        indices = ttp_to_seq_indices.get(ttp, [])
        print(f"Found {len(indices)} sequences containing majority TTP '{ttp}'.")
        # 해당 TTP가 포함된 시퀀스 인덱스들을 모두 합침
        majority_indices.update(indices)

    # 전체 시퀀스 인덱스 집합
    all_indices = set(range(len(sequences)))
    # 다수 클래스를 포함하지 않는 '소수' 시퀀스 인덱스들
    minority_indices = all_indices - majority_indices
    
    print(f"Total sequences with at least one majority TTP: {len(majority_indices)}")
    print(f"Total sequences with only minority TTPs: {len(minority_indices)}")

    # 3. 다수 클래스 시퀀스들을 'limit' 개수만큼 무작위로 샘플링
    if len(majority_indices) > limit:
        print(f"Undersampling majority sequences from {len(majority_indices)} down to {limit}...")
        sampled_majority_indices = set(random.sample(list(majority_indices), limit))
    else:
        print("No undersampling needed for majority class as it's below the limit.")
        sampled_majority_indices = majority_indices

    # 4. 최종 데이터셋 구성: 소수 시퀀스 + 샘플링된 다수 시퀀스
    final_indices_to_keep = sorted(list(minority_indices | sampled_majority_indices))
    
    undersampled_sequences = [sequences[i] for i in final_indices_to_keep]

    print(f"\nCreated a new balanced dataset with {len(undersampled_sequences)} sequences.")

    # 5. 새로운 시퀀스 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(undersampled_sequences, f)
    print(f"Saved undersampled sequences to '{output_file}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Undersample sequences containing majority class TTPs.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the original sequence JSON file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the new undersampled sequence file.")
    parser.add_argument("--majority-ttps", nargs='+', required=True, help="List of majority TTP IDs to undersample (e.g., T1059.003 T1027.009).")
    parser.add_argument("--limit", type=int, required=True, help="The maximum number of sequences to keep for any sequence containing a majority TTP.")
    
    args = parser.parse_args()
    random.seed(42) # 재현성을 위한 시드 고정
    undersample_sequences(args.input_file, args.output_file, args.majority_ttps, args.limit)