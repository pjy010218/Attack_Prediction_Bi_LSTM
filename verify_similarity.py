import json
import numpy as np
from tqdm import tqdm
import argparse

def levenshtein_distance(seq1, seq2):
    """
    두 시퀀스 간의 레벤슈타인 편집 거리를 계산합니다.
    편집 거리가 낮을수록 두 시퀀스는 더 유사합니다.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            # 두 TTP가 같으면 비용은 0, 다르면 1
            cost = 0 if seq1[x - 1] == seq2[y - 1] else 1
            matrix[x, y] = min(
                matrix[x - 1, y] + 1,        # 삭제(Deletion)
                matrix[x, y - 1] + 1,        # 삽입(Insertion)
                matrix[x - 1, y - 1] + cost  # 대체(Substitution)
            )
    return matrix[size_x - 1, size_y - 1]

def calculate_similarity(real_sequences_file: str, synthetic_sequences_file: str):
    """
    실제 시퀀스 데이터셋과 합성 시퀀스 데이터셋 간의
    평균 최소 편집 거리를 계산하여 유사도를 측정합니다.
    """
    try:
        with open(real_sequences_file, 'r', encoding='utf-8') as f:
            real_sequences = json.load(f)
        with open(synthetic_sequences_file, 'r', encoding='utf-8') as f:
            synthetic_sequences = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find a file. {e}")
        return

    if not real_sequences or not synthetic_sequences:
        print("Error: One of the sequence files is empty.")
        return

    total_min_distances = []
    
    print("Calculating similarity between real and synthetic sequences...")
    # 각 실제 시퀀스에 대해 루프를 돕니다.
    for real_seq in tqdm(real_sequences, desc="Processing real sequences"):
        min_distance_for_real_seq = float('inf')
        
        # 현재 실제 시퀀스와 가장 유사한(편집 거리가 가장 낮은) 합성 시퀀스를 찾습니다.
        for synthetic_seq in synthetic_sequences:
            # 시퀀스 길이 차이가 너무 크면 비교에서 제외 (성능 최적화)
            if abs(len(real_seq) - len(synthetic_seq)) > 10:
                continue

            distance = levenshtein_distance(real_seq, synthetic_seq)
            if distance < min_distance_for_real_seq:
                min_distance_for_real_seq = distance
        
        total_min_distances.append(min_distance_for_real_seq)

    # 모든 실제 시퀀스에 대한 '최소 편집 거리'들의 평균을 계산합니다.
    average_min_distance = np.mean(total_min_distances)
    
    print("\n--- Similarity Verification Result ---")
    print(f"Number of real-world sequences analyzed: {len(real_sequences)}")
    print(f"Number of synthetic sequences compared against: {len(synthetic_sequences)}")
    print(f"Average Minimum Edit Distance: {average_min_distance:.2f}")
    print("\nInterpretation: A lower score indicates higher similarity between the synthetic and real-world scenarios.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify similarity between real and synthetic attack scenarios.")
    parser.add_argument("--real-file", type=str, required=True, help="Path to the JSON file of real-world sequences.")
    parser.add_argument("--synthetic-file", type=str, required=True, help="Path to the JSON file of synthetic sequences.")
    
    args = parser.parse_args()
    
    calculate_similarity(args.real_file, args.synthetic_file)