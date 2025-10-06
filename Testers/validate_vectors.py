import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def analyze_vector_sparsity(all_vectors, vector_array):
        """벡터의 희소성을 분석하고 출력합니다."""
        print("\n--- 1단계: 벡터 희소성 분석 ---")
        
        total_elements = vector_array.size
        zero_elements = np.count_nonzero(vector_array == 0)
        overall_sparsity = (zero_elements / total_elements) * 100
        
        print(f"\nOverall Statistics:")
        print(f"Total number of vectors: {len(all_vectors)}")
        print(f"Vector dimension: {vector_array.shape[1]}")
        print(f"Overall sparsity (percentage of zeros): {overall_sparsity:.2f}%")
        
        sparsity_per_dimension = []
        for i in range(vector_array.shape[1]):
            dim_zeros = np.count_nonzero(vector_array[:, i] == 0)
            dim_sparsity = (dim_zeros / len(all_vectors)) * 100
            sparsity_per_dimension.append(dim_sparsity)
            
        # [수정] 벡터 차원에 맞게 Description 목록을 동적으로 생성
        embedding_dim = vector_array.shape[1] - 2
        descriptions = [f"Embedding Dim {i}" for i in range(embedding_dim)] + \
                       ["Positional: Abs. Step", "Positional: Rel. Step"]

        sparsity_df = pd.DataFrame({
            "Dimension": list(range(vector_array.shape[1])),
            "Description": descriptions,
            "Sparsity (%)": sparsity_per_dimension
        })
        
        print("\nSparsity per Dimension:")
        pd.set_option('display.max_rows', None)
        print(sparsity_df.to_string(index=False))
        pd.reset_option('display.max_rows')

def test_feature_validity(vector_array):
    """수치형/위치형 특성의 유효성을 검증하는 실험을 수행합니다."""
    print("\n\n--- 2단계: 특징 유효성 검증 실험 ---")
    
    # 데이터 준비: X에는 수치/위치 특성, y에는 전술(Tactic) 레이블
    # Tactic 정보는 0~13번 차원에 원-핫 인코딩되어 있음
    # 수치/위치 정보는 14~20번 차원에 있음
    X = vector_array[:, 14:]
    y = np.argmax(vector_array[:, :14], axis=1) # 원-핫 인코딩에서 레이블 추출

    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("\nTraining a RandomForestClassifier to predict Tactic from other features...")
    # 모델 훈련
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Experiment Results ---")
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

def main():
    """메인 실행 함수"""
    filepath = "D:\\Github\\TK_Graph\\vectorized_sequences.json"
    print(f"Loading data from '{filepath}'...")
    try:
        with open(filepath, 'r') as f:
            sequences = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: File not found at '{filepath}'.")
        print("Please make sure this script is in the same directory as your vector file.")
        return

    all_vectors = [vector for seq in sequences for vector in seq]
    if not all_vectors:
        print("No vectors found in the file.")
        return
    
    vector_array = np.array(all_vectors)

    # 1단계 실행
    analyze_vector_sparsity(all_vectors, vector_array)
    
    # 2단계 실행
    test_feature_validity(vector_array)

if __name__ == "__main__":
    main()