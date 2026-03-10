# Finalgame_ML
수화 인식 AI 기반 청각장애인 지하철 민원 보조 웹서비스 - ML 모듈 

## 프로젝트 개요
본 프로젝트는 지하철 개찰구에서 청각장애인이 수화를 통해 민원을 처리할 수 있도록 지원하는 웹서비스의 ML 모듈입니다.  
정적 수화는 CNN, 동적 수화는 LSTM 모델을 사용하며, 손 동작의 trajectory 변화량에 따라 적합한 모델을 선택하여 실시간 예측을 수행합니다.

---
## 파일 구조
```bash
finalgame_ml/
├─ data/ # 데이터셋
│ ├─ dataset # 정적 수화 데이터
│ └─ dataset9 # 동적 수화 데이터
├─ src/ # 실제 코드
│ ├─ models/ # 모델 정의
│ │ ├─ cnn_model.py
│ │ └─ lstm_model.py
│ ├─ train/ # 학습 코드
│ │ ├─ train_cnn.py
│ │ └─ train_lstm.py
│ └─ inference/ # 실시간 추론 코드
│ └─ real_time_prediction.py
├─ trained_models/ # 학습 완료된 모델 파일
│ ├─ cnn_model_bothhands.keras
│ └─ dynamic_gesture_model3.h5
├─ results/ # 평가 및 시각화 자료
│ 
└─ requirements.txt
```
