import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 1. 데이터 로드
with open('D:\\Github\\TK_Graph\\attack_sequences_augmented.json', 'r') as f:
    sequences = json.load(f)

# 2. 시퀀스 길이 분석
lengths = [len(seq) for seq in sequences]
print(f"--- 시퀀스 길이 분석 ---")
print(f"총 시퀀스 수: {len(lengths)}")
print(f"최소/최대/평균 길이: {np.min(lengths)} / {np.max(lengths)} / {np.mean(lengths):.2f}")

plt.hist(lengths, bins=30)
plt.title("Sequence Length Distribution")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()

# 3. TTP 빈도 분석
all_ttps = [ttp for seq in sequences for ttp in seq]
ttp_counts = Counter(all_ttps)
print(f"\n--- TTP 빈도 분석 ---")
print(f"고유 TTP 수 (Vocab size): {len(ttp_counts)}")
print("가장 많이 등장한 TTP Top 10:")
for ttp, count in ttp_counts.most_common(10):
    print(f"{ttp}: {count}회")

# 4. "다음 단계" 예측 가능성 분석 (예시: 'T1059.003' 다음 TTP)
target_ttp = 'T1059.003' # 분석하고 싶은 TTP
next_ttps = []
for seq in sequences:
    for i, ttp in enumerate(seq[:-1]):
        if ttp == target_ttp:
            next_ttps.append(seq[i+1])

if next_ttps:
    next_ttp_counts = Counter(next_ttps)
    print(f"\n--- '{target_ttp}' 다음 TTP 분석 ---")
    print(f"다음에 등장한 고유 TTP 수: {len(next_ttp_counts)}")
    print("가장 많이 등장한 다음 TTP Top 5:")
    for ttp, count in next_ttp_counts.most_common(5):
        print(f"{ttp}: {count}회")