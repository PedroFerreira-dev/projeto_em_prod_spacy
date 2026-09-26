import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    GlobalAveragePooling1D,
    Dense
)


def carregar_dados(caminho="dados.csv"):
    return pd.read_csv(caminho)


def preparar_dados(dados):
    dados = dados[
        dados["classe"].isin(["positivo", "negativo"])
    ].copy()

    textos = dados["texto"].astype(str).tolist()

    classes = (
        dados["classe"]
        .map({
            "negativo": 0,
            "positivo": 1
        })
        .astype("float32")
        .tolist()
    )

    return textos, classes


def criar_modelo():
    vetorizar = TextVectorization(
        max_tokens=1000,
        output_mode="int",
        output_sequence_length=20
    )

    modelo = Sequential([
        vetorizar,
        Embedding(
            input_dim=1000,
            output_dim=16
        ),
        GlobalAveragePooling1D(),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


def treinar_modelo(textos, classes):

    textos = tf.constant(
        textos,
        dtype=tf.string
    )

    classes = tf.constant(
        classes,
        dtype=tf.float32
    )

    modelo = criar_modelo()

    # Cria o vocabulário usando os textos de treinamento
    modelo.layers[0].adapt(textos)

    modelo.fit(
        textos,
        classes,
        epochs=20,
        verbose=0
    )

    return modelo
