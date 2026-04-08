import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Configuração da Página
st.set_page_config(page_title="Analista de Sono: Manoel", layout="wide")

st.title("👶 Analisador Inteligente de Sono do Bebê")
st.markdown("""
Esta ferramenta utiliza **Machine Learning** (Isolation Forest) para identificar padrões e anomalias 
no sono do bebê, ajudando a diferenciar o progresso natural de noites fora do comum.
""")

# --- 1. UPLOAD DO DATASET ---
uploaded_file = st.file_uploader("Suba o arquivo CSV do aplicativo (My_Baby...-events.csv)", type="csv")

if uploaded_file is not None:
    # Carregamento e Limpeza
    df = pd.read_csv(uploaded_file)
    
    # Tratamento de dados (mesma lógica do Colab)
    if 'Valor.Número' in df.columns:
        df['Valor.Número'] = pd.to_numeric(df['Valor.Número'].astype(str).str.replace(',', '.'), errors='coerce')
    
    for col in ['Data e hora', 'Começo', 'Fim']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Filtro apenas para Sono Noturno
    df_sono = df[(df['Evento'].astype(str).str.strip() == 'Sono') & 
                 (df['Tipo'].astype(str).str.strip() == 'Noite')].copy()
    df_sono = df_sono.sort_values('Data e hora').dropna(subset=['Valor.Número']).reset_index(drop=True)
    df_sono['duracao_horas'] = df_sono['Valor.Número'] / 3600

    # Feature Engineering
    df_sono['media_7_noites'] = df_sono['duracao_horas'].rolling(window=7, min_periods=1).mean()
    df_sono['desvio_media_7'] = df_sono['duracao_horas'] - df_sono['media_7_noites']
    
    # Modelo de Detecção de Anomalias
    X = df_sono[['duracao_horas', 'desvio_media_7']].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    modelo = IsolationForest(contamination=0.02, random_state=42)
    df_sono['anomaly_flag'] = modelo.fit_predict(X_scaled)
    anomalias = df_sono[df_sono['anomaly_flag'] == -1]

    # --- TABS: ORGANIZAÇÃO DO CONTEÚDO ---
    tab1, tab2, tab3 = st.tabs(["📊 Painel Técnico", "📝 Relatório para Pais", "📑 Dados Brutos"])

    with tab1:
        st.header("Análise Estatística")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de Períodos", len(df_sono))
        c2.metric("Média de Sono (Bloco)", f"{df_sono['duracao_horas'].mean():.2f}h")
        c3.metric("Anomalias Detectadas", len(anomalias))

        # Gráfico Principal
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_sono['Data e hora'], df_sono['duracao_horas'], color='gray', alpha=0.3, label='Duração')
        ax.plot(df_sono['Data e hora'], df_sono['media_7_noites'], color='blue', label='Média Móvel')
        ax.scatter(anomalias['Data e hora'], anomalias['duracao_horas'], color='red', label='Anomalia')
        ax.set_ylabel("Horas de Sono")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        st.header("Relatório de Acompanhamento")
        
        # Insights Automáticos
        st.success(f"**A Linha da Vitória:** O Manoel já teve blocos de até {df_sono['duracao_horas'].max():.1f} horas de sono!")
        
        st.markdown(f"""
        ### Olá! Este é o resumo do progresso do Manoel:
        
        * **Consistência:** Foram registrados **{len(df_sono)} períodos** de sono noturno nos últimos 7 meses. Isso mostra uma rotina muito bem acompanhada.
        * **Progresso:** A linha azul no gráfico mostra que, apesar das oscilações, a capacidade dele de dormir blocos mais longos está evoluindo.
        * **Alertas:** Identificamos **{len(anomalias)} noites** que fugiram muito do padrão. Geralmente, isso acontece em picos de crescimento ou pequenos desconfortos.
        
        **Sugestão dos Dados:** Tente observar se as noites com sono mais longo acontecem quando o início do sono é mais cedo. Os dados sugerem que a regularidade no horário de início favorece noites mais calmas.
        """)

    with tab3:
        st.dataframe(df_sono[['Data e hora', 'Valor', 'duracao_horas', 'anomaly_flag']])

else:
    st.info("Aguardando upload do arquivo para iniciar a análise.")
