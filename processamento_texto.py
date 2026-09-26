import re

import spacy


# ============================================================
# CARREGAMENTO DO MODELO spaCy
# ============================================================

nlp = spacy.load(
    "pt_core_news_sm"
)


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar_texto(texto):
    """
    Converte o texto para letras minúsculas
    e remove pontuações.
    """

    texto = texto.lower()

    texto = re.sub(
        r"[^\w\s]",
        "",
        texto
    )

    return texto


# ============================================================
# TOKENIZAÇÃO
# ============================================================

def tokenizar_texto(texto):
    """
    Divide o texto em palavras utilizando spaCy.
    """

    texto = normalizar_texto(texto)

    documento = nlp(texto)

    tokens = []

    for token in documento:

        if token.is_space:
            continue

        tokens.append(token.text)

    return tokens


# ============================================================
# REMOÇÃO DE STOPWORDS
# ============================================================

def remover_stopwords(tokens):
    """
    Remove palavras muito comuns da língua portuguesa
    utilizando spaCy.
    """

    palavras_relevantes = []

    for palavra in tokens:

        if not nlp.vocab[palavra].is_stop:

            palavras_relevantes.append(
                palavra
            )

    return palavras_relevantes