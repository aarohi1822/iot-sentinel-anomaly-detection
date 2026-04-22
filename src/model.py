from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_autoencoder(
    window_size: int,
    n_features: int,
    latent_dim: int = 32,
    learning_rate: float = 0.001,
) -> keras.Model:
    inputs = keras.Input(shape=(window_size, n_features))
    encoded = layers.LSTM(latent_dim, activation="tanh", return_sequences=False)(inputs)
    decoded = layers.RepeatVector(window_size)(encoded)
    decoded = layers.LSTM(latent_dim, activation="tanh", return_sequences=True)(decoded)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(decoded)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse")
    return model


def load_trained_model(path: str, window_size: int, n_features: int):
    model = build_lstm_autoencoder(window_size, n_features)
    model.load_weights(path.replace(".keras", ".weights.h5"))
    return model

