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
│ ├─ cnn_mode.h5
│ └─ lstm_model.h5
└─ requirements.txt
```
---
## 실행 방법
```bash
# 1. 저장소 클론 후 이동
git clone https://github.com/kimheeyeon1/Finalgame_ML.git
cd Finalgame_ML

# 2. 가상환경 생성/활성화
python -m venv venv
venv\Scripts\activate

# 3. 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# 4. 모델 파일 준비
cnn_model.h5와 lstm_model.h5를 trained_models/ 폴더에 넣습니다.
경로 예시:
Finalgame_ML/
├─ trained_models/
│  ├─ cnn_model.h5
│  └─ lstm_model.h5

5. 실행
python src/inference/real_time_inference.py

6. 사용 방법
웹캠이 켜지고 실시간 수어 인식 시작
화면 표시:
인식된 단어 + 모델 유형
트래젝토리 변화량
기록된 단어(문장)
단축키:
q : 프로그램 종료
c : 문장 초기화
```
