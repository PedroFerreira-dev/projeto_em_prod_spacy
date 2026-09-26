from collections import Counter


# ============================================================
# PALAVRAS NEGATIVAS
# ============================================================

PALAVRAS_NEGATIVAS = {
    "ruim",
    "péssimo",
    "péssima",
    "erro",
    "insatisfeito",
    "insatisfeita",
    "problema",
    "quebrado",
    "quebrada"
}


# ============================================================
# PALAVRAS POSITIVAS
# ============================================================

PALAVRAS_POSITIVAS = {
    "bom",
    "boa",
    "excelente",
    "ótimo",
    "ótima",
    "satisfeito",
    "satisfeita",
    "gostei",
    "rápido",
    "perfeito"
}


# ============================================================
# PALAVRAS-CHAVE
# ============================================================

PALAVRAS_CHAVE = {
    "cancelar",
    "erro",
    "pagamento"
}


def contar_frequencia(tokens):
    """
    Conta quantas vezes cada palavra aparece.
    """

    frequencia = Counter(tokens)

    return frequencia


def identificar_palavras_negativas(tokens):
    """
    Identifica palavras negativas presentes no texto.
    """

    encontradas = []

    for palavra in tokens:

        if palavra in PALAVRAS_NEGATIVAS:
            encontradas.append(palavra)

    return encontradas


def detectar_palavras_chave(tokens):
    """
    Identifica palavras importantes para direcionamento
    do atendimento.
    """

    encontradas = []

    for palavra in tokens:

        if palavra in PALAVRAS_CHAVE:
            encontradas.append(palavra)

    return encontradas


def classificar_sentimento(tokens):
    """
    Faz uma classificação simples de sentimento
    usando regras condicionais.
    """

    pontos_positivos = 0
    pontos_negativos = 0

    for palavra in tokens:

        if palavra in PALAVRAS_POSITIVAS:
            pontos_positivos += 1

        if palavra in PALAVRAS_NEGATIVAS:
            pontos_negativos += 1

    if pontos_positivos > pontos_negativos:
        return "Positivo"

    if pontos_negativos > pontos_positivos:
        return "Negativo"

    return "Neutro"


def classificar_setor(tokens):
    """
    Classifica a mensagem entre suporte técnico
    e financeiro.
    """

    palavras_tecnicas = {
        "erro",
        "sistema",
        "senha",
        "login",
        "bug",
        "aplicativo"
    }

    palavras_financeiras = {
        "pagamento",
        "boleto",
        "cobrança",
        "cobrar",
        "pix",
        "cartão"
    }

    encontrou_tecnico = False
    encontrou_financeiro = False

    for palavra in tokens:

        if palavra in palavras_tecnicas:
            encontrou_tecnico = True

        if palavra in palavras_financeiras:
            encontrou_financeiro = True

    if encontrou_tecnico and encontrou_financeiro:
        return "Suporte técnico e financeiro"

    if encontrou_tecnico:
        return "Suporte técnico"

    if encontrou_financeiro:
        return "Financeiro"

    return "Não identificado"