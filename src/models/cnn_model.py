import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Multiply, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def build_cnn_model(input_shape=(42,3), mask_shape=(42,), num_classes=9):
    
    #CNN 모델 정의 (정적 수어, 한손/양손 혼합)
    
    inp_data = Input(shape=input_shape, name='input_data')
    inp_mask = Input(shape=mask_shape, name='input_mask')

    mask_exp = Reshape((42,1), name='expand_last_dim')(inp_mask)
    masked   = Multiply()([inp_data, mask_exp])  # 존재하는 손만 보이게

    x = Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.001))(masked)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.5)(x)

    x = Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)

    out = Dense(num_classes, activation='softmax')(x)

    model = Model([inp_data, inp_mask], out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model