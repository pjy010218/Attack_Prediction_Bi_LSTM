#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow/Keras 기반의 LSTM 모델 훈련 스크립트 (모든 개선 사항 적용)
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def _parse_args() -> argparse.Namespace:
    """스크립트 실행을 위한 인자(argument)를 파싱합니다."""
    parser = argparse.ArgumentParser(description="Train LSTM to predict next TTP from vector sequences")
    parser.add_argument("--vectors", type=str, required=True, help="Path to JSON file of vector sequences")
    parser.add_argument("--ids", type=str, required=True, help="Path to JSON file of TTP ID sequences")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model & artifacts")
    # [수정] 리팩토링된 벡터 차원(258)으로 기본값 변경
    parser.add_argument("--vector_dim", type=int, default=258, help="Dimensionality of input vectors") # 130 -> 258
    # [수정] 모델 용량 증가 (전략 1)
    parser.add_argument("--lstm_units", type=int, default=256, help="Number of LSTM units")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout on LSTM")
    parser.add_argument("--recurrent_dropout", type=float, default=0.0, help="Recurrent dropout on LSTM")
    # [수정] 조기 종료를 위해 epochs 넉넉하게 설정
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def _load_json(path: str):
    """JSON 파일을 로드합니다."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_vocab(id_sequences: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """TTP ID 시퀀스로부터 단어장(vocab)을 생성합니다."""
    vocab: Dict[str, int] = {}
    for seq in id_sequences:
        for tid in seq:
            if tid not in vocab:
                vocab[tid] = len(vocab)
    inv_vocab: Dict[int, str] = {idx: tid for tid, idx in vocab.items()}
    return vocab, inv_vocab

def _prepare_dataset(
    vector_sequences: List[List[List[float]]],
    id_sequences: List[List[str]],
    vocab: Dict[str, int],
    vector_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """가변 길이 시퀀스를 훈련에 사용할 수 있도록 패딩된 배열로 변환합니다."""
    X_seqs: List[np.ndarray] = []
    y_seqs: List[np.ndarray] = []
    for vecs, ids in zip(vector_sequences, id_sequences):
        min_len = min(len(vecs), len(ids))
        vecs, ids = vecs[:min_len], ids[:min_len]
        if len(vecs) < 2:
            continue
        X_arr = np.asarray(vecs[:-1], dtype=np.float32)
        y_arr = np.asarray([vocab[tid] for tid in ids[1:]], dtype=np.int64)
        if X_arr.shape[-1] != vector_dim:
            raise ValueError(f"Vector dim mismatch: expected {vector_dim}, got {X_arr.shape[-1]}")
        X_seqs.append(X_arr)
        y_seqs.append(y_arr)

    if not X_seqs:
        raise ValueError("No usable sequences found (need at least one with length >= 2)")

    max_len = max(x.shape[0] for x in X_seqs)
    num_seq = len(X_seqs)
    X = np.zeros((num_seq, max_len, vector_dim), dtype=np.float32)
    y = np.zeros((num_seq, max_len), dtype=np.int64)
    sample_weights = np.zeros((num_seq, max_len), dtype=np.float32)

    for i, (x_seq, y_seq) in enumerate(zip(X_seqs, y_seqs)):
        L = x_seq.shape[0]
        X[i, :L, :] = x_seq
        y[i, :L] = y_seq
        sample_weights[i, :L] = 1.0

    return X, y, sample_weights

def sparse_categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    희소(sparse) 레이블을 지원하는 Focal Loss 함수.
    y_true는 정수 인덱스 형태여야 합니다.
    """
    # y_pred가 NaN이나 Inf가 되지 않도록 안정성을 위해 작은 값을 더하고 빼줍니다.
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    
    # 클래스 개수를 y_pred의 마지막 차원에서 가져옵니다.
    num_classes = tf.shape(y_pred)[-1]
    
    # y_true (정수 인덱스)를 원-핫 인코딩 형태로 변환합니다.
    # 예: y_true(shape=[32, 14]) -> y_true_one_hot(shape=[32, 14, 496])
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    
    # 각 클래스에 대한 cross-entropy 계산
    cross_entropy = -y_true_one_hot * K.log(y_pred)
    
    # 각 클래스에 대한 focal loss 계산
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
    
    # 배치 내 각 샘플에 대한 최종 손실을 계산 (클래스 차원에 대해 합산)
    # sample_weight가 이 결과를 기반으로 타임스텝별 가중치를 적용하게 됩니다.
    return K.sum(loss, axis=-1)

def _build_model(input_timesteps: int, vector_dim: int, vocab_size: int, lstm_units: int, dropout: float, recurrent_dropout: float):
    """
    안정성이 검증된 2-Layer Stacked LSTM 모델을 생성하고 컴파일합니다.
    """
    inputs = layers.Input(shape=(input_timesteps, vector_dim), name="input_vectors")
    x = layers.Masking(mask_value=0.0, name="masking")(inputs)
    
    # 첫 번째 LSTM 레이어
    x = layers.Bidirectional(layers.LSTM(
        lstm_units,
        return_sequences=True,  # 다음 LSTM 레이어와 연결하기 위해 필수
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        name="lstm_1",
    ), name="bidirectional_1")(x)
    
    # 과적합 방지를 위한 Dropout 레이어
    x = layers.Dropout(dropout, name="dropout_1")(x)

    # 두 번째 LSTM 레이어
    x = layers.Bidirectional(layers.LSTM(
        lstm_units,
        return_sequences=True,  # 마지막 Dense 레이어가 모든 타임스텝에 적용되므로 True 유지
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        name="lstm_2",
    ), name="bidirectional_2")(x)
    
    # 최종 분류기는 두 번째 LSTM의 출력을 바로 입력으로 사용
    outputs = layers.Dense(vocab_size, activation="softmax", name="classifier")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Focal Loss를 사용하여 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=sparse_categorical_focal_loss,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )
    return model

def main() -> None:
    """메인 실행 로직"""
    args = _parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # 데이터 로드
    vector_sequences = _load_json(args.vectors)
    id_sequences = _load_json(args.ids)

    # 단어장 생성
    vocab, inv_vocab = _build_vocab(id_sequences)
    vocab_size = len(vocab)

    # 데이터셋 준비
    X, y, sample_weights = _prepare_dataset(vector_sequences, id_sequences, vocab, args.vector_dim)

    # [추가] 훈련/검증 데이터 분할 및 StandardScaler 적용
    print("Splitting data and applying StandardScaler...")
    indices = np.arange(X.shape[0])
    X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
        X, y, sample_weights, test_size=args.validation_split, random_state=args.seed
    )
    
    print("Saving validation set for later evaluation...")
    np.savez(
        os.path.join(args.output_dir, "validation_data.npz"),
        X_val=X_val,
        y_val=y_val,
        weights_val=weights_val,
    )

    scaler = StandardScaler()

    num_samples_train, timesteps, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features)
    scaler.fit(X_train_reshaped)
    X_train = scaler.transform(X_train_reshaped).reshape(num_samples_train, timesteps, num_features)
    
    num_samples_val = X_val.shape[0]
    X_val_reshaped = X_val.reshape(-1, num_features)
    X_val = scaler.transform(X_val_reshaped).reshape(num_samples_val, timesteps, num_features)
    print("Data scaling complete.")

    # [추가] 클래스 가중치 계산 (전략 2)
    print("Calculating class weights to handle data imbalance...")
    y_train_flat = y_train.flatten()
    weights_train_flat = weights_train.flatten()
    active_labels = y_train_flat[weights_train_flat > 0]
    classes = np.unique(active_labels)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=active_labels)
    class_weight_dict = dict(zip(classes, weights))
    print("Class weights calculated.")

    print("Combining class weights and sample weights...")
    y_train_flat = y_train.flatten()
    weights_train_flat = weights_train.flatten()
    active_labels = y_train_flat[weights_train_flat > 0]

    # 1. 클래스별 가중치 계산
    classes = np.unique(active_labels)
    class_weights_array = class_weight.compute_class_weight('balanced', classes=classes, y=active_labels)
    class_weight_dict = dict(zip(classes, class_weights_array))
    
    # 2. 훈련 데이터(y_train)의 각 레이블에 해당하는 클래스 가중치를 매핑
    #    - 모든 위치에 기본값 0.0으로 채워진 배열 생성
    sample_weights_final = np.zeros_like(y_train, dtype=np.float32)
    #    - 각 클래스에 해당하는 위치를 찾아 해당 클래스의 가중치 값으로 채움
    for class_idx, weight in class_weight_dict.items():
        sample_weights_final[y_train == class_idx] = weight

    # 3. 기존의 패딩 마스크(weights_train)와 곱하여 패딩된 부분은 0으로 만듦
    #    - 이로써 '마스크'와 '중요도' 기능이 하나로 통합됨
    sample_weights_final = sample_weights_final * weights_train
    print("Final sample weights calculated.")

    # 모델 생성
    timesteps = X.shape[1]
    model = _build_model(timesteps, args.vector_dim, vocab_size, args.lstm_units, args.dropout, args.recurrent_dropout)

    # [추가] 콜백(Callback) 정의 (전략 3)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6)

    callbacks = [early_stopping, reduce_lr]

    # 모델 훈련
    print("\nStarting training...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val, weights_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
        )

    # 결과물 저장
    os.makedirs(args.output_dir, exist_ok=True)
    model.save(os.path.join(args.output_dir, "best_model.keras")) # .keras 형식으로 저장 권장
    with open(os.path.join(args.output_dir, "vocab.json"), "w") as f:
        json.dump({"token_to_id": vocab, "id_to_token": inv_vocab}, f, indent=2)
    
    print(f"\nTraining finished. Best model saved to '{args.output_dir}'.")

if __name__ == "__main__":
    main()