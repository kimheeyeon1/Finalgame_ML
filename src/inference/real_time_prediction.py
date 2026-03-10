# TensorFlow/Keras 모델 load
import os
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import mediapipe as mp


# 클래스 맵 
cnn_class_map = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9'}
lstm_class_map = {0: 'help', 1: 'dangerous', 2: 'careful', 3: 'hello', 4: 'lose', 5:'card', 6:'balance',7:'deficit',8:'subway'}

# 모델 경로 
cnn_model_path = "../../trained_models/cnn_model.h5"
lstm_model_path = "../../trained_models/lstm_model.h5"

if not os.path.exists(cnn_model_path) or not os.path.exists(lstm_model_path):
    raise FileNotFoundError("❌ 모델 파일을 model/ 폴더에 넣어주세요.")

# 모델 로드 
static_model = load_model(cnn_model_path, compile=False)
dynamic_model = load_model(lstm_model_path, compile=False)

# Mediapipe 세팅 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 절대좌표 수집(왼/오 슬롯 고정) 
def process_hands_absolute(hands_landmarks):
    raw = []
    wrists_x = []
    for hand in hands_landmarks:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
        coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        raw.append(coords)
        wrists_x.append(coords[0, 0])

    if len(raw) == 0:
        left = np.zeros((21,3), dtype=np.float32)
        right = np.zeros((21,3), dtype=np.float32)
        mask = np.zeros(42, dtype=np.float32)
    elif len(raw) == 1:
        left = raw[0]
        right = np.zeros((21,3), dtype=np.float32)
        mask = np.array([1]*21 + [0]*21, dtype=np.float32)
    else:
        i_left, i_right = (0,1) if wrists_x[0] <= wrists_x[1] else (1,0)
        left, right = raw[i_left], raw[i_right]
        mask = np.array([1]*21 + [1]*21, dtype=np.float32)

    combined_abs = np.concatenate([left, right], axis=0).reshape(-1)
    combined_abs = np.nan_to_num(combined_abs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return combined_abs, mask

# 상대좌표 변환 (동적 수어) 
def relative_coordinates_dynamic(seq):
    seq = np.array(seq, dtype=np.float32).reshape(30,42,3)
    wrist = seq[:,0:1,:]
    rel = seq - wrist
    return rel.reshape(30,126)

# 트래젝토리 변화량 계산 
def compute_trajectory_variance(sequence):
    diffs = np.diff(sequence, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    return float(np.mean(norms)) if norms.size else 0.0

# 예측 함수 
def predict(sequence, mask, threshold=0.05, lstm_confidence_threshold=0.8):
    traj_var = compute_trajectory_variance(sequence)

    if traj_var < threshold:
        # 정적 예측 (CNN)
        static_input = sequence[0].reshape(1,42,3).astype(np.float32)
        mask_input = mask.reshape(1,42).astype(np.float32)
        mask_input[:,21:] = 0.0
        probs = static_model.predict([static_input, mask_input], verbose=0)[0]
        confidence = float(np.max(probs))
        label = int(np.argmax(probs))
        return cnn_class_map.get(label, "unknown"), "STATIC-CNN", traj_var, confidence
    else:
        # 동적 예측 (LSTM)
        probs = dynamic_model.predict(sequence.reshape(1,30,126).astype(np.float32), verbose=0)[0]
        confidence = float(np.max(probs))
        label = int(np.argmax(probs))
        label_text = lstm_class_map.get(label, "unknown")

        if label_text == "help" and confidence < 0.95:
            return "Waiting...", "DYNAMIC-LSTM", traj_var, confidence
        elif confidence < lstm_confidence_threshold:
            return "Waiting...", "DYNAMIC-LSTM", traj_var, confidence

        return label_text, "DYNAMIC-LSTM", traj_var, confidence

# 웹캠 시작
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("❌ 웹캠 열기 실패")

seq = []
last_label = "Waiting..."
last_model_type = ""
last_traj_score = 0.0
last_update_time = 0

STREAK_N = 2
_current = None
_streak = 0
accepted_tokens = []

print("🎥 실시간 수어 인식 시작 (q: 종료, c: 문장 초기화)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 캡처 실패")
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        landmarks_list = [hand.landmark for hand in results.multi_hand_landmarks]
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        combined_abs, mask = process_hands_absolute(landmarks_list)
        if combined_abs.shape == (126,):
            seq.append(combined_abs)

        if len(seq) == 30:
            try:
                seq_stack = np.stack(seq, axis=0).astype(np.float32)
                seq_np = relative_coordinates_dynamic(seq_stack)
                mask_np = np.array(mask, dtype=np.float32)

                label_text, model_type, traj_var, confidence = predict(seq_np, mask_np)
                last_label = label_text
                last_model_type = model_type
                last_traj_score = traj_var
                last_update_time = time.time()

                # -------- 단순 연속 단어 기록 --------
                if label_text != "Waiting...":
                    if label_text == _current:
                        _streak += 1
                    else:
                        _current = label_text
                        _streak = 1

                    if _streak >= STREAK_N:
                        if not accepted_tokens or accepted_tokens[-1] != label_text:
                            accepted_tokens.append(label_text)
                else:
                    _current = None
                    _streak = 0

            except Exception as e:
                print("예측 실패:", e)
            finally:
                seq = []
    else:
        seq = []
        _current = None
        _streak = 0

    # 화면 출력
    if time.time() - last_update_time < 2.0:
        cv2.putText(frame, f"{last_label} ({last_model_type})", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(frame, f"Trajectory Var: {last_traj_score:.5f}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
    else:
        cv2.putText(frame, "Waiting for gesture...", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    # 기록된 단어 출력 
    best_sentence = " ".join(accepted_tokens)
    cv2.putText(frame, best_sentence, (10,95),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow("Real-time Gesture Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        accepted_tokens.clear()
        _current = None
        _streak = 0

cap.release()
cv2.destroyAllWindows() 