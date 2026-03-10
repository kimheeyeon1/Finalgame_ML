from tensorflow.keras import layers, models

def build_lstm_model(input_shape=(30, 126), num_classes=9):
    
    #LSTM 모델 정의 함수
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(48, return_sequences=True),
        layers.Dropout(0.4),
        layers.LSTM(48),
        layers.Dropout(0.4),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model