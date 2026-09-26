import re
import spacy


# ============================================================
# CARREGAMENTO DO MODELO spaCy
# ============================================================

nlp = spacy.load("pt_core_news_sm")


def normalizar_texto(texto):
    """
    Converte o texto para letras minúsculas
    e remove pontuações.
    """

    # Converte todas as letras para minúsculas.
    texto = texto.lower()

    # Remove pontuações e mantém letras,
    # números e espaços.
    texto = re.sub(r"[^\w\s]", "", texto)

    return texto


def tokenizar_texto(texto):
    """
    Divide o texto em palavras utilizando spaCy.
    """

    # Primeiro normalizamos o texto.
    texto = normalizar_texto(texto)

    # O spaCy transforma o texto em um Doc.
    documento = nlp(texto)

    # Criamos uma lista contendo os tokens.
    tokens = []

    for token in documento:

        # Ignora espaços.
        if token.is_space:
            continue

        tokens.append(token.text)

    return tokens


def remover_stopwords(tokens):
    """
    Remove palavras muito comuns da língua portuguesa
    utilizando a lista de stopwords do spaCy.
    """

    palavras_relevantes = []

    for palavra in tokens:

        # O spaCy possui sua própria lista de stopwords.
        if not nlp.vocab[palavra].is_stop:
            palavras_relevantes.append(palavra)

    return palavras_relevantes