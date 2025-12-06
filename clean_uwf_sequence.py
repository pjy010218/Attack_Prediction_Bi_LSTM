import json
from itertools import groupby

def clean_sequences(input_file, output_file, min_len=2):
    print(f"Loading sequences from {input_file}...")
    with open(input_file, 'r') as f:
        sequences = json.load(f)
    
    cleaned_sequences = []
    total_original_events = 0
    total_cleaned_events = 0
    
    for seq in sequences:
        total_original_events += len(seq)
        
        # itertools.groupby를 사용하여 연속된 중복 제거
        # 예: ['T1110', 'T1110', 'T1059', 'T1059'] -> ['T1110', 'T1059']
        deduplicated_seq = [key for key, group in groupby(seq)]
        
        if len(deduplicated_seq) >= min_len:
            cleaned_sequences.append(deduplicated_seq)
            total_cleaned_events += len(deduplicated_seq)
            
    print(f"--- Cleaning Report ---")
    print(f"Original sequences: {len(sequences)}")
    print(f"Cleaned sequences: {len(cleaned_sequences)} (removed {len(sequences) - len(cleaned_sequences)} short seqs)")
    print(f"Total events: {total_original_events} -> {total_cleaned_events} (Reduction: {100 - (total_cleaned_events/total_original_events*100):.2f}%)")
    
    # 예시 출력
    if cleaned_sequences:
        print(f"Sample cleaned sequence: {cleaned_sequences[0]}")

    with open(output_file, 'w') as f:
        json.dump(cleaned_sequences, f, indent=2)
    print(f"Saved to {output_file}")

# 실행
clean_sequences('uwf_real_sequences.json', 'uwf_real_sequences_clean.json')