import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import callbacks
from models.lstm_model import build_lstm_model


# 전처리 함수
def relative_coordinates(sample):
    s = np.array(sample, dtype=np.float32).reshape(30, 42, 3)
    left  = s[:, :21, :]
    right = s[:, 21:, :]

    left_ok  = np.any(np.abs(left).sum(axis=2) > 1e-6, axis=1)
    right_ok = np.any(np.abs(right).sum(axis=2) > 1e-6, axis=1)

    ref = np.zeros((30, 1, 3), dtype=np.float32)
    ref[left_ok] = left[left_ok, 0:1, :]
    use_right = (~left_ok) & right_ok
    ref[use_right] = right[use_right, 0:1, :]

    rel = s - ref
    return rel.reshape(30, 126)

def is_bad_sample(sample, max_empty_ratio=0.2):
    s = np.array(sample, dtype=np.float32).reshape(30, 42, 3)
    both_zero = np.all(np.abs(s) < 1e-6, axis=(1,2))
    return (both_zero.mean() > max_empty_ratio)


# 데이터 로딩 및 전처리
dataset_dirs = [os.path.join('..', '..', 'data', 'dataset9')]
classes = ['help','dangerous','careful','hello','lose','card','balance','deficit','subway']

X, y = [], []
for d in dataset_dirs:
    for fname in os.listdir(d):
        if not fname.endswith('.npy'):
            continue
        path = os.path.join(d, fname)
        data = np.load(path)

        if data.shape == (30, 126):
            data = data.reshape(30, 42, 3)
        elif data.shape == (30, 42, 3):
            pass
        else:
            continue

        if is_bad_sample(data):
            continue

        data = relative_coordinates(data)  # (30,126)
        X.append(data)

        lab = None
        low = fname.lower()
        for i, c in enumerate(classes):
            if c in low:
                lab = i
                break
        if lab is None:
            continue
        y.append(lab)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 모델 생성 및 컴파일
model = build_lstm_model(input_shape=(30, 126), num_classes=len(classes))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=7, restore_best_weights=True
)

#  클래스 불균형 보정
cls_vals = np.unique(y_train)
cw = compute_class_weight(class_weight='balanced', classes=cls_vals, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(cls_vals, cw)}

# 모델 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=16,
    callbacks=[early_stop],
    class_weight=class_weight
)

# 평가 및 저장
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ LSTM 모델 정확도: {acc:.4f}")

os.makedirs(os.path.join('..', '..', 'trained_models'), exist_ok=True)
model.save(os.path.join('..', '..', 'trained_models', 'dynamic_gesture_model3.h5'))
print("✅ 모델 저장 완료 ➜ trained_models/dynamic_gesture_model3.h5")