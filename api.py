import tensorflow as tf

from flask import Flask, request, jsonify

from processamento_texto import (
    tokenizar_texto,
    remover_stopwords
)

from analise import (
    contar_frequencia,
    identificar_palavras_negativas,
    detectar_palavras_chave,
    classificar_sentimento,
    classificar_setor
)

from modelo import (
    carregar_dados,
    preparar_dados,
    treinar_modelo
)


# ============================================================
# CONFIGURAÇÃO DO FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# PREPARAÇÃO DO MODELO
# ============================================================

def preparar_modelo():

    dados = carregar_dados()

    textos, classes = preparar_dados(dados)

    modelo = treinar_modelo(
        textos,
        classes
    )

    return modelo


modelo = preparar_modelo()


# ============================================================
# ROTA PRINCIPAL
# ============================================================

@app.route("/")
def inicio():

    return jsonify({
        "mensagem": "API de análise de textos funcionando"
    })


# ============================================================
# ROTA DE ANÁLISE
# ============================================================

@app.route("/analisar", methods=["POST"])
def analisar():

    dados = request.get_json()

    if not dados:
        return jsonify({
            "erro": "Nenhum JSON foi enviado."
        }), 400

    texto = dados.get("texto")

    if not texto:
        return jsonify({
            "erro": "O campo 'texto' é obrigatório."
        }), 400

    # ========================================================
    # 1. TOKENIZAÇÃO
    # ========================================================

    tokens = tokenizar_texto(texto)

    # ========================================================
    # 2. REMOÇÃO DE STOPWORDS
    # ========================================================

    palavras_relevantes = remover_stopwords(tokens)

    # ========================================================
    # 3. FREQUÊNCIA
    # ========================================================

    frequencia = contar_frequencia(
        palavras_relevantes
    )

    # ========================================================
    # 4. PALAVRAS NEGATIVAS
    # ========================================================

    negativas = identificar_palavras_negativas(
        palavras_relevantes
    )

    # ========================================================
    # 5. SENTIMENTO POR REGRAS
    # ========================================================

    sentimento = classificar_sentimento(
        palavras_relevantes
    )

    # ========================================================
    # 6. PALAVRAS-CHAVE
    # ========================================================

    palavras_chave = detectar_palavras_chave(
        palavras_relevantes
    )

    # ========================================================
    # 7. PALAVRAS MAIS FREQUENTES
    # ========================================================

    palavras_frequentes = frequencia.most_common(5)

    # ========================================================
    # 8. SETOR
    # ========================================================

    setor = classificar_setor(
        palavras_relevantes
    )

    # ========================================================
    # 9. TEXTO NORMALIZADO
    # ========================================================

    texto_normalizado = " ".join(
        palavras_relevantes
    )

    # ========================================================
    # 10. TENSORFLOW
    # ========================================================

    entrada = tf.constant(
        [texto],
        dtype=tf.string
    )

    previsao = modelo(
        entrada,
        training=False
    ).numpy()[0][0]

    if previsao >= 0.5:
        resultado_modelo = "Positivo"
    else:
        resultado_modelo = "Negativo"

    # ========================================================
    # RESPOSTA DA API
    # ========================================================

    return jsonify({

        "texto_original": texto,

        "tokenizacao": tokens,

        "palavras_relevantes": palavras_relevantes,

        "frequencia": dict(frequencia),

        "palavras_negativas": negativas,

        "sentimento_regras": sentimento,

        "palavras_chave": palavras_chave,

        "palavras_frequentes": palavras_frequentes,

        "setor": setor,

        "texto_normalizado": texto_normalizado,

        "tensorflow": {
            "sentimento": resultado_modelo,
            "probabilidade_positiva": float(previsao)
        }
    })


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )