import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models.cnn_model import build_cnn_model

# 전처리 함수
def _normalize_array(arr):
    arr = np.array(arr)
    if arr.ndim == 2:
        if arr.shape[1] == 63:
            arr = arr.reshape(-1, 21, 3)
        elif arr.shape[1] == 126:
            arr = arr.reshape(-1, 42, 3)
    elif arr.ndim == 3:
        if arr.shape == (21,3) or arr.shape == (42,3):
            arr = arr.reshape(1, *arr.shape)
    return arr.astype(np.float32)

def preprocess_static_mixed(npy_path, enforce_lr_by_x=True):
    raw = np.load(npy_path, allow_pickle=True)
    data = _normalize_array(raw)
    X_list, M_list = [], []

    for sample in data:
        sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)

        if sample.shape == (21,3):  # 한손
            wrist = sample[0]
            rel = sample - wrist
            left, right = rel, np.zeros_like(rel)
            mask = np.array([1]*21 + [0]*21, dtype=np.float32)
            combined = np.concatenate([left, right], axis=0)

        elif sample.shape == (42,3):  # 양손
            left = sample[:21].copy()
            right = sample[21:].copy()
            if enforce_lr_by_x:
                if left[0,0] > right[0,0]:
                    left, right = right, left
            wrist = left[0]
            left_rel  = left - wrist
            right_rel = right - wrist
            combined = np.concatenate([left_rel, right_rel], axis=0)
            mask = np.ones(42, dtype=np.float32)
        else:
            continue

        X_list.append(combined)
        M_list.append(mask)

    X = np.stack(X_list).astype(np.float32)
    M = np.stack(M_list).astype(np.float32)
    return X, M

# 데이터 로드
npy_dir = os.path.join('..','..','data', 'dataset')
label_dict = {'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8}

X_list, M_list, y_list = [], [], []

for name, idx in label_dict.items():
    path = os.path.join(npy_dir, f'landmarks_{name}.npy')
    if os.path.exists(path):
        Xc, Mc = preprocess_static_mixed(path, enforce_lr_by_x=True)
        X_list.append(Xc)
        M_list.append(Mc)
        y_list += [idx]*len(Xc)
    else:
        print(f"❌ 파일 없음: {path}")

X = np.concatenate(X_list, axis=0)
masks = np.concatenate(M_list, axis=0)
y = np.array(y_list)

num_classes = len(label_dict)

X_train, X_test, M_train, M_test, y_train, y_test = train_test_split(
    X, masks, y, test_size=0.2, random_state=42, stratify=y
)

y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test,  num_classes)

# 모델 생성
model = build_cnn_model(input_shape=(42,3), mask_shape=(42,), num_classes=num_classes)
model.summary()

# 콜백 & 학습
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    [X_train, M_train], y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[es, rlr]
)

# 평가 및 저장
loss, acc = model.evaluate([X_test, M_test], y_test)
print(f"✅ 모델 정확도: {acc:.4f}")

os.makedirs(os.path.join('..', '..', 'trained_models'), exist_ok=True)
model.save(os.path.join('..', '..', 'trained_models', 'cnn_model_bothhands.keras'))
print("✅ 모델 저장 완료 ➜ trained_models/cnn_model_bothhands.keras")