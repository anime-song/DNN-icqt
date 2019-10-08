import keras
from keras import layers as L


def _add_lstm_layer(
        input_layer,
        n_layer,
        hidden_size,
        dropout=0.1,
        bidirectional=False):
    layer = input_layer

    for i in range(n_layer):
        lstm = L.LSTM(
            hidden_size,
            recurrent_dropout=dropout,
            return_sequences=True)

        if bidirectional:
            layer = L.Bidirectional(lstm)(layer)
        else:
            layer = lstm(layer)

    return layer


def inverse_CQT(stft_shape, cqt_shape, hidden_size=256, n_layer=2):

    lstm_hidden_size = hidden_size // 2

    input_stft = L.Input(shape=stft_shape, name="STFT_INPUT")
    input_cqt = L.Input(shape=cqt_shape, name="CQT_INPUT")
    
    x1 = input_stft
    x1 = L.Dense(hidden_size)(x1)
    x1 = L.BatchNormalization()(x1)
    x1 = L.Activation("relu")(x1)

    x2 = input_cqt
    x2 = L.Dense(hidden_size)(x2)
    x2 = L.BatchNormalization()(x2)
    x2 = L.Activation("relu")(x2)

    x = L.Concatenate()([x1, x2])

    x = L.Dense(hidden_size)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("tanh")(x)

    lstm = _add_lstm_layer(x, n_layer, lstm_hidden_size, bidirectional=True, dropout=0.4)
    x = L.Concatenate()([x, lstm])

    x = L.Dense(hidden_size)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    x = L.Dense(stft_shape[1])(x)
    x = L.BatchNormalization()(x)
    x = L.Lambda(lambda x: (x + 1) * 1)(x)
    x = L.ReLU(max_value=1.0, name="mask_layer")(x)

    x = L.Multiply()([x, input_stft])

    return keras.Model(inputs=[input_stft, input_cqt], outputs=[x])
