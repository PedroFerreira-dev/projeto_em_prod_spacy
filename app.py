import requests
import streamlit as st


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Análise de Textos",
    page_icon="📝",
    layout="wide"
)


# ============================================================
# TÍTULO
# ============================================================

st.title("📝 Sistema de Análise de Textos")

st.write(
    """
    Sistema de análise de mensagens de clientes
    utilizando spaCy, TensorFlow, Flask e Streamlit.
    """
)


# ============================================================
# CAMPO DE TEXTO
# ============================================================

texto = st.text_area(
    "Digite uma mensagem:",
    placeholder=(
        "Exemplo: Meu pagamento apresentou erro "
        "e quero cancelar a compra."
    )
)


# ============================================================
# BOTÃO
# ============================================================

if st.button("Analisar texto"):

    if not texto.strip():

        st.warning(
            "Digite uma mensagem para realizar a análise."
        )

    else:

        try:

            # Envia o texto para o Flask.
            resposta = requests.post(
                "http://127.0.0.1:5000/analisar",
                json={
                    "texto": texto
                }
            )

            # Verifica se a API respondeu corretamente.
            if resposta.status_code != 200:

                st.error(
                    "Erro ao realizar a análise."
                )

            else:

                resultado = resposta.json()

                # ====================================================
                # 1. TOKENIZAÇÃO
                # ====================================================

                st.subheader("1. Tokenização")

                st.write(
                    resultado["tokenizacao"]
                )

                # ====================================================
                # 2. STOPWORDS
                # ====================================================

                st.subheader(
                    "2. Remoção de stopwords"
                )

                st.write(
                    resultado["palavras_relevantes"]
                )

                # ====================================================
                # 3. FREQUÊNCIA
                # ====================================================

                st.subheader(
                    "3. Frequência das palavras"
                )

                st.write(
                    resultado["frequencia"]
                )

                # ====================================================
                # 4. NEGATIVAS
                # ====================================================

                st.subheader(
                    "4. Palavras negativas"
                )

                negativas = resultado[
                    "palavras_negativas"
                ]

                if negativas:

                    st.error(
                        f"Palavras negativas encontradas: "
                        f"{negativas}"
                    )

                else:

                    st.success(
                        "Nenhuma palavra negativa encontrada."
                    )

                # ====================================================
                # 5. SENTIMENTO
                # ====================================================

                st.subheader(
                    "5. Sentimento"
                )

                st.info(
                    "Classificação: "
                    f"{resultado['sentimento_regras']}"
                )

                # ====================================================
                # 6. PALAVRAS-CHAVE
                # ====================================================

                st.subheader(
                    "6. Palavras-chave"
                )

                palavras_chave = resultado[
                    "palavras_chave"
                ]

                if palavras_chave:

                    st.write(
                        palavras_chave
                    )

                else:

                    st.write(
                        "Nenhuma palavra-chave encontrada."
                    )

                # ====================================================
                # 7. PALAVRAS MAIS FREQUENTES
                # ====================================================

                st.subheader(
                    "7. Palavras mais frequentes"
                )

                palavras_frequentes = resultado[
                    "palavras_frequentes"
                ]

                if palavras_frequentes:

                    for palavra, quantidade in palavras_frequentes:

                        st.write(
                            f"**{palavra}** → "
                            f"{quantidade} ocorrência(s)"
                        )

                else:

                    st.write(
                        "Nenhuma palavra encontrada."
                    )

                # ====================================================
                # 8. SETOR
                # ====================================================

                st.subheader(
                    "8. Setor"
                )

                st.info(
                    "Setor identificado: "
                    f"{resultado['setor']}"
                )

                # ====================================================
                # 9. TEXTO NORMALIZADO
                # ====================================================

                st.subheader(
                    "9. Texto normalizado"
                )

                st.code(
                    resultado["texto_normalizado"]
                )

                # ====================================================
                # 10. TENSORFLOW
                # ====================================================

                st.subheader(
                    "10. Análise com TensorFlow"
                )

                tensorflow = resultado[
                    "tensorflow"
                ]

                st.write(
                    "Sentimento previsto pelo modelo: "
                    f"**{tensorflow['sentimento']}**"
                )

                st.write(
                    "Probabilidade positiva: "
                    f"**{tensorflow['probabilidade_positiva']:.2%}**"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Não foi possível conectar ao Flask. "
                "Verifique se a API está sendo executada."
            )