import json
import argparse
import numpy as np
from tqdm import tqdm

def vectorize_from_embeddings(
    sequences_file: str,
    embeddings_file: str,
    output_file: str
):
    """
    사전 훈련된 임베딩을 사용하여 TTP 시퀀스를 벡터 시퀀스로 변환합니다.
    """
    print(f"Loading attack sequences from '{sequences_file}'...")
    with open(sequences_file, 'r') as f:
        sequences = json.load(f)
        
    print(f"Loading TTP embeddings from '{embeddings_file}'...")
    with open(embeddings_file, 'r') as f:
        embeddings = json.load(f)
    
    # 임베딩 차원 확인 (위치 정보 2차원 추가됨)
    embedding_dim = len(next(iter(embeddings.values())))
    vector_dim = embedding_dim + 2
    print(f"Embedding dimension: {embedding_dim}, Final vector dimension will be: {vector_dim}")

    final_vectorized_sequences = []
    max_possible_len = 0
    for seq in sequences:
        if len(seq) > max_possible_len:
            max_possible_len = len(seq)

    print("Vectorizing sequences using embeddings...")
    for seq in tqdm(sequences):
        vector_sequence = []
        seq_len = len(seq)
        for i, ttp in enumerate(seq):
            # 1. TTP 임베딩 벡터 조회
            ttp_embedding = embeddings.get(ttp)
            if ttp_embedding is None:
                # 임베딩이 없는 경우 (드문 경우), 0으로 채운 벡터 사용
                ttp_embedding = [0.0] * embedding_dim

            # 2. 2차원의 위치 정보 계산
            absolute_step = i / max_possible_len
            relative_step = i / (seq_len - 1) if seq_len > 1 else 0.0
            
            # 3. 임베딩 벡터와 위치 정보를 합쳐 최종 벡터 생성
            final_vector = ttp_embedding + [absolute_step, relative_step]
            vector_sequence.append(final_vector)
        final_vectorized_sequences.append(vector_sequence)

    print(f"Vectorization complete. Saving to '{output_file}'...")
    with open(output_file, 'w') as f:
        json.dump(final_vectorized_sequences, f)
    print("Successfully saved vectorized sequences.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vectorize TTP sequences using pre-trained Node2Vec embeddings.")
    parser.add_argument("--sequences-file", type=str, required=True, help="Path to the attack sequence JSON file.")
    parser.add_argument("--embeddings-file", type=str, default="ttp_enriched_embeddings.json", help="Path to the TTP embeddings JSON file.")
    parser.add_argument("--output-file", type=str, default="vectorized_sequences.json", help="Path to save the final vectorized sequences.")
    args = parser.parse_args()
    vectorize_from_embeddings(args.sequences_file, args.embeddings_file, args.output_file)