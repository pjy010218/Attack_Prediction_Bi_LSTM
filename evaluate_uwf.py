import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, accuracy_score

# Focal Loss 함수 (모델 로드용)
def sparse_categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    num_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    cross_entropy = -y_true_one_hot * K.log(y_pred)
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
    return K.sum(loss, axis=-1)

def evaluate_on_real_data(model_dir, vector_file):
    print(f"--- Evaluating Model on Real Data: {vector_file} ---")

    # 1. 모델 및 단어장 로드
    print("Loading model and vocab...")
    try:
        model = tf.keras.models.load_model(
            os.path.join(model_dir, "best_model.keras"),
            custom_objects={'sparse_categorical_focal_loss': sparse_categorical_focal_loss}
        )
        with open(os.path.join(model_dir, "vocab.json"), 'r') as f:
            vocab_data = json.load(f)
            token_to_id = vocab_data['token_to_id']
            id_to_token = vocab_data['id_to_token']
    except Exception as e:
        print(f"Error loading model/vocab: {e}")
        return

    # 모델이 기대하는 입력 길이 확인 (예: 14)
    expected_timesteps = model.input_shape[1]
    vector_dim = model.input_shape[2]
    print(f"Model expects input shape: (Batch, {expected_timesteps}, {vector_dim})")

    # 2. 데이터 로드
    print("Loading vector data...")
    with open(vector_file, 'r') as f:
        vector_sequences = json.load(f)

    # 정답 라벨 로드 (uwf_vectors.json -> uwf_final_sequences.json)
    # 파일명 규칙에 따라 적절히 수정 필요
    sequence_file = vector_file.replace("_vectors", "_final_sequences") 
    try:
        with open(sequence_file, 'r') as f:
            id_sequences = json.load(f)
    except FileNotFoundError:
        print(f"Error: 정답 시퀀스 파일 '{sequence_file}'을 찾을 수 없습니다.")
        return

    y_true_all = []
    y_pred_all = []
    valid_samples = 0

    print("Running predictions...")
    for i, (vec_seq, id_seq) in enumerate(zip(vector_sequences, id_sequences)):
        if len(vec_seq) < 2: continue

        # 입력(X): 마지막 시점 제외
        curr_X = np.array(vec_seq[:-1], dtype=np.float32)
        
        # 정답(y): 첫 시점 제외 (다음 단계를 예측해야 하므로)
        curr_y_labels = id_seq[1:]
        curr_y_ids = []
        for ttp in curr_y_labels:
            curr_y_ids.append(token_to_id.get(ttp, -1)) # Vocab에 없으면 -1

        # -------------------------------------------------------
        # [수정] 패딩(Padding) 로직 추가
        # -------------------------------------------------------
        current_len = len(curr_X)
        
        # 모델 입력용 배열 생성 (0으로 초기화)
        # 학습 시 Post-padding(데이터 앞, 0 뒤)을 썼으므로 동일하게 맞춤
        padded_X = np.zeros((1, expected_timesteps, vector_dim), dtype=np.float32)
        
        if current_len <= expected_timesteps:
            # 길이가 짧으면 앞에서부터 채움 (나머지는 0)
            padded_X[0, :current_len, :] = curr_X
        else:
            # 길이가 길면 뒷부분(최신 데이터)만 남기고 자르거나, 앞부분을 사용
            # 보통 LSTM은 최근 문맥이 중요하므로 뒷부분을 사용하는 것이 좋으나,
            # 여기서는 단순하게 앞에서부터 자름 (또는 필요시 수정)
            padded_X[0, :, :] = curr_X[:expected_timesteps, :]
            # 정답 라벨도 길이에 맞춰 잘라줘야 함
            curr_y_ids = curr_y_ids[:expected_timesteps]

        # 예측 수행
        pred_probs = model.predict(padded_X, verbose=0)
        
        # 패딩된 부분은 예측 결과에서 제외하고, 실제 데이터 길이만큼만 가져옴
        # (긴 데이터를 잘랐다면 expected_timesteps만큼, 짧으면 current_len만큼)
        valid_len = min(current_len, expected_timesteps)
        pred_ids = np.argmax(pred_probs, axis=-1)[0][:valid_len]

        # 결과 수집
        for true_id, pred_id in zip(curr_y_ids, pred_ids):
            if true_id != -1: # Unknown 정답은 평가 제외
                y_true_all.append(true_id)
                y_pred_all.append(pred_id)
        
        valid_samples += 1
        if valid_samples % 100 == 0:
            print(f"Processed {valid_samples} sequences...")

    # 4. 결과 출력
    if not y_true_all:
        print("No valid predictions made.")
        return

    print("\n--- Evaluation Results ---")
    acc = accuracy_score(y_true_all, y_pred_all)
    print(f"Total Valid Predictions (Time steps): {len(y_true_all)}")
    print(f"Accuracy: {acc:.4f}")
    
    # 분류 리포트 (클래스가 많으므로 주요 클래스만 보거나 전체 출력)
    unique_labels = sorted(list(set(y_true_all)))
    target_names = [id_to_token[str(i)] for i in unique_labels]
    
    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all, labels=unique_labels, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    MODEL_DIR = "models"
    VECTOR_FILE = "uwf_vectors.json" 
    
    evaluate_on_real_data(MODEL_DIR, VECTOR_FILE)