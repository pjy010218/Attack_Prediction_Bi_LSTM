import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def sparse_categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    희소(sparse) 레이블을 지원하는 Focal Loss 함수.
    y_true는 정수 인덱스 형태여야 합니다.
    """
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    
    num_classes = tf.shape(y_pred)[-1]
    
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    
    cross_entropy = -y_true_one_hot * K.log(y_pred)
    
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
    
    return K.sum(loss, axis=-1)

def diagnose_model(model_dir="models"):
    """
    훈련된 모델의 성능을 심층적으로 진단합니다.
    1. 혼동 행렬(Confusion Matrix) 생성
    2. 예측 오류 정성적 분석(Qualitative Error Analysis) 리포트 생성
    """
    print("--- Starting Model Diagnosis ---")
    
    # 1. 필요 파일 로드
    print("Loading model, validation data, and vocabulary...")
    try:
        # [수정 2] custom_objects 인자를 사용하여 Focal Loss 함수를 Keras에 알려줌
        model = tf.keras.models.load_model(
            os.path.join(model_dir, "best_model.keras"),
            custom_objects={'sparse_categorical_focal_loss': sparse_categorical_focal_loss}
        )
        vocab_data = json.load(open(os.path.join(model_dir, "vocab.json")))
        val_data = np.load(os.path.join(model_dir, "validation_data.npz"))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Make sure all necessary files are in '{model_dir}'.")
        return

    inv_vocab = vocab_data['id_to_token']
    X_val, y_val, weights_val = val_data['X_val'], val_data['y_val'], val_data['weights_val']

    # 2. 모델 예측 수행
    print("Generating predictions on validation data...")
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    # 3. 패딩 제외한 실제 데이터만 추출
    mask = weights_val > 0
    y_true_flat = y_val[mask]
    y_pred_flat = y_pred[mask]

    # --- 진단 1: 혼동 행렬 분석 ---
    print("\n--- Diagnosis 1: Generating Confusion Matrix ---")
    
    top_k = 20
    top_classes = [item[0] for item in Counter(y_true_flat).most_common(top_k)]
    top_class_names = [inv_vocab.get(str(i), str(i)) for i in top_classes]

    # [ ▼ 수정된 부분 시작 ▼ ]
    # Top 20 클래스에 해당하는 데이터만 필터링
    mask_top_k = np.isin(y_true_flat, top_classes)
    y_true_filtered = y_true_flat[mask_top_k]
    y_pred_filtered = y_pred_flat[mask_top_k]

    # 실제 TTP ID를 0~19 범위의 새 ID로 재매핑
    label_to_new_id = {label: new_id for new_id, label in enumerate(top_classes)}
    
    y_true_remapped = np.array([label_to_new_id[label] for label in y_true_filtered])
    # 예측된 값 중 Top 20에 없는 값은 -1로 처리하여 무시하도록 함
    y_pred_remapped = np.array([label_to_new_id.get(label, -1) for label in y_pred_filtered])

    # -1로 처리된 예측은 혼동 행렬 계산에서 제외
    valid_indices = y_pred_remapped != -1
    y_true_final = y_true_remapped[valid_indices]
    y_pred_final = y_pred_remapped[valid_indices]

    if top_classes:
        cm = tf.math.confusion_matrix(y_true_final, y_pred_final, num_classes=len(top_classes)).numpy()
        cm_df = pd.DataFrame(cm, index=top_class_names, columns=top_class_names)
        
        plt.figure(figsize=(18, 15))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for Top {top_k} Most Frequent TTPs')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        print(f"Saved confusion matrix plot to 'confusion_matrix.png'")
    else:
        print("Could not generate confusion matrix: No top classes found in validation data.")


    # --- 진단 2: 예측 오류 정성적 분석 ---
    print("\n--- Diagnosis 2: Generating Qualitative Error Analysis Report ---")
    report_path = "error_analysis_report.txt"
    num_errors_to_report = 10
    errors_found = 0

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Qualitative Error Analysis Report\n")
        f.write("="*40 + "\n\n")

        for i in range(X_val.shape[0]):
            if errors_found >= num_errors_to_report:
                break
            
            for t in range(X_val.shape[1]):
                if errors_found >= num_errors_to_report:
                    break

                if weights_val[i, t] > 0 and y_val[i, t] != y_pred[i, t]:
                    errors_found += 1
                    
                    f.write(f"--- Error Example #{errors_found} ---\n")
                    
                    # -1은 패딩이므로 제외
                    input_seq_indices = [idx for idx in y_val[i, :t] if idx != -1]
                    input_seq_ttps = [inv_vocab.get(str(idx), "PAD") for idx in input_seq_indices]
                    f.write(f"Input Sequence TTPs:\n")
                    for ttp in input_seq_ttps:
                        f.write(f"  - {ttp}\n")
                    f.write("-"*40 + "\n")
                    
                    true_label_id = y_val[i, t]
                    pred_label_id = y_pred[i, t]
                    f.write(f"Correct Next TTP: {inv_vocab.get(str(true_label_id))}\n")
                    f.write(f"Model's Prediction: {inv_vocab.get(str(pred_label_id))}\n")
                    f.write("-"*40 + "\n")
                    
                    top_5_indices = np.argsort(y_pred_probs[i, t])[-5:][::-1]
                    f.write("Model's Top 5 Predictions (Confidence):\n")
                    for k, idx in enumerate(top_5_indices):
                        prob = y_pred_probs[i, t, idx]
                        ttp_id = inv_vocab.get(str(idx))
                        f.write(f"  {k+1}. {ttp_id} ({prob:.2%})")
                        if idx == true_label_id:
                            f.write("  <-- Correct answer was here\n")
                        else:
                            f.write("\n")
                    f.write("\n" + "="*40 + "\n\n")

    print(f"Saved qualitative error analysis report to '{report_path}'")


if __name__ == '__main__':
    diagnose_model()