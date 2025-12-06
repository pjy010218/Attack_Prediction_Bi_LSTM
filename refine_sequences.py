import json
from collections import Counter
import numpy as np

def refine_dataset(
    input_file: str, 
    output_file: str, 
    min_unique_ratio: float = 0.4,  # 길이 대비 고유 TTP 비율 (낮을수록 반복 심함)
    max_duplicate_count: int = 50   # 동일 시퀀스 최대 허용 개수
):
    print(f"--- 데이터 정제 시작: {input_file} ---")
    
    with open(input_file, 'r') as f:
        sequences = json.load(f)
    
    print(f"원본 시퀀스 개수: {len(sequences)}")
    
    refined_sequences = []
    dropped_by_loop = 0
    
    # 1. 다양성(Entropy) 필터링: 단순 루프 제거
    # 예: [A, B, C, A, B, C, A, B, C] -> 길이 9, 고유값 3 -> 비율 0.33 < 0.4 (제거)
    temp_sequences = []
    for seq in sequences:
        if len(seq) == 0: continue
        
        unique_count = len(set(seq))
        unique_ratio = unique_count / len(seq)
        
        # 길이가 아주 짧으면(예: 4 미만) 비율이 낮아도 허용 (예: [A, B, A] -> 0.66)
        # 길이가 길면서 비율이 낮은 경우만 필터링
        if len(seq) > 6 and unique_ratio < min_unique_ratio:
            dropped_by_loop += 1
        else:
            temp_sequences.append(seq)
            
    print(f"-> 단순 반복(Loop) 시퀀스 제거: {dropped_by_loop}개")
    
    # 2. 빈도 제한 (Frequency Capping): 과도한 중복 제거
    # 리스트는 해싱이 안되므로 튜플로 변환하여 카운팅
    seq_tuples = [tuple(seq) for seq in temp_sequences]
    counts = Counter(seq_tuples)
    
    final_sequences = []
    dropped_by_freq = 0
    
    # 이미 추가한 시퀀스 개수를 추적
    added_counts = {seq: 0 for seq in counts}
    
    for seq in temp_sequences:
        seq_t = tuple(seq)
        if added_counts[seq_t] < max_duplicate_count:
            final_sequences.append(seq)
            added_counts[seq_t] += 1
        else:
            dropped_by_freq += 1
            
    print(f"-> 과도한 중복 시퀀스 제거(Top {max_duplicate_count}개 제한): {dropped_by_freq}개")
    print(f"최종 남은 시퀀스: {len(final_sequences)}")
    
    if final_sequences:
        print(f"예시 시퀀스: {final_sequences[0]}")

    with open(output_file, 'w') as f:
        json.dump(final_sequences, f, indent=2)
    print(f"저장 완료: {output_file}")

if __name__ == "__main__":
    # 이전 단계에서 생성한 파일 입력
    INPUT_FILE = "uwf_smart_sequences.json"
    OUTPUT_FILE = "uwf_refined_sequences.json"
    
    # min_unique_ratio: 0.3~0.5 추천 (높을수록 엄격하게 반복 제거)
    refine_dataset(INPUT_FILE, OUTPUT_FILE, min_unique_ratio=0.4, max_duplicate_count=50)