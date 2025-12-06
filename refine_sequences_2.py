import json

input_file = "uwf_refined_sequences.json"
output_file = "uwf_final_sequences.json"

with open(input_file, 'r') as f:
    sequences = json.load(f)

# 길이가 4 이상인 시퀀스만 남김
filtered_sequences = [seq for seq in sequences if len(seq) >= 4]

print(f"Original: {len(sequences)} -> Filtered: {len(filtered_sequences)}")
print(f"Removed {len(sequences) - len(filtered_sequences)} short sequences.")

with open(output_file, 'w') as f:
    json.dump(filtered_sequences, f, indent=2)