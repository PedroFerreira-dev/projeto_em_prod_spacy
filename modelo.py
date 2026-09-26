import pandas as pd
import tensorflow as tf


def carregar_dados():
    """
    Carrega os dados existentes no arquivo CSV.
    """

    dados = pd.read_csv("dados.csv")

    return dados


def preparar_dados(dados):
    """
    Prepara os textos e transforma as classes
    'positivo' e 'negativo' em números.
    """

    mapa_classes = {
        "negativo": 0,
        "positivo": 1
    }

    # Mantém somente as classes que o modelo
    # consegue classificar.
    dados = dados[
        dados["classe"].isin(["positivo", "negativo"])
    ].copy()

    dados["classe"] = dados["classe"].map(mapa_classes)

    textos = dados["texto"].astype(str).values
    classes = dados["classe"].values

    return textos, classes


def criar_modelo(textos):
    """
    Cria a rede neural responsável pela
    classificação do sentimento.
    """

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=1000,
        output_mode="int",
        output_sequence_length=20
    )

    vectorizer.adapt(textos)

    tamanho_vocabulario = len(
        vectorizer.get_vocabulary()
    )

    modelo = tf.keras.Sequential([

        vectorizer,

        tf.keras.layers.Embedding(
            input_dim=tamanho_vocabulario,
            output_dim=16
        ),

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    return modelo


def treinar_modelo(textos, classes):
    """
    Cria, compila e treina o modelo.
    """

    modelo = criar_modelo(textos)

    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    modelo.fit(
        textos,
        classes,
        epochs=20,
        verbose=0
    )

    return modelo