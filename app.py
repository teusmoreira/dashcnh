import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from io import BytesIO

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise de Sentimentos – CNH",
    page_icon="🚗",
    layout="wide",
)

# ── Léxico de sentimentos em português ─────────────────────────────────────
POSITIVOS = {
    "maravilhoso","ótimo","excelente","parabéns","perfeito","bom","boa","bons","boas",
    "fantástico","incrível","legal","show","top","melhor","melhorou","melhorando",
    "facilitou","facilitar","facilita","facilitando","fácil","acessível","acessibilidade",
    "barato","economizando","economia","economizar","reduziu","reduzir","redução",
    "benefício","beneficiou","beneficia","beneficiar","aprovado","aprovar","aprovação",
    "glória","amei","adorei","gostei","gosto","amo","adoro","adoro","ótima","certo",
    "correto","verdade","verdadeiro","justo","sucesso","vitória","ganho","ganhar",
    "ajudou","ajudar","ajuda","apoio","apoiar","graças","obrigado","parabens",
    "eficiente","eficiência","moderno","modernização","progresso","avanço","avanços",
    "bênção","deus","alegria","feliz","felicidade","satisfeito","satisfação",
    "valeu","vai","vantagem","simplificou","simplificar","desburocratizou","boa notícia",
    "aprovamos","aprovei","ganhamos","ganhou","baratear",
}

NEGATIVOS = {
    "caro","cara","caros","absurdo","ridículo","lamentável","vergonha","vergonhoso",
    "ruim","péssimo","horrível","terrível","pior","piora","piorou","piorando",
    "injusto","injustiça","problema","problemas","dificuldade","difícil","complicado",
    "complicação","burocracia","burocrático","exploração","explorar","exploram","exploram",
    "roubando","roubo","rouba","cobram","cobrança","extorção","extorsão","abusivo",
    "abuso","prejudica","prejudicou","prejudicial","prejudicando","perdemos","perder",
    "perdeu","péssima","golpe","mentira","mentiroso","enganação","enganar","engana",
    "lamentavelmente","triste","tristeza","raiva","indignado","indignação","revoltante",
    "revoltado","revolta","protesto","protestamos","não","nunca","jamais","impossível",
    "caiu","cai","corrompido","corrupto","corrupção","decepção","decepcionado","decepcionante",
    "insatisfeito","insatisfação","desnecessário","inútil","falhou","falha","falso",
    "aumentar","aumento","aumentaram","aumentou","taxa","taxas","imposto","impostos",
    "desvantagem","complicou","complicar","problemático","erro","erros","errou",
}

STOPWORDS_PT = {
    "de","da","do","das","dos","em","na","no","nas","nos","a","o","as","os",
    "que","e","é","para","por","com","se","não","um","uma","uns","umas",
    "mais","mas","ou","como","ao","aos","à","às","pelo","pela","pelos","pelas",
    "este","esta","estes","estas","esse","essa","esses","essas","isso","isto",
    "aquele","aquela","aqui","lá","já","só","ainda","então","quando","onde",
    "muito","pouco","bem","mal","também","nem","até","sobre","entre","depois",
    "antes","agora","hoje","vai","ter","ser","estar","foi","são","está","tem",
    "eu","ele","ela","nós","você","eles","elas","me","te","se","nos","vos",
    "meu","minha","teu","tua","seu","sua","nosso","nossa","meus","minhas",
    "ia","ir","fui","fez","fazer","fica","ficar","pode","poder","quer","querer",
    "pra","pro","pras","pros","né","aqui","lhe","lhes","quem","qual","quais",
    "tá","ta","vou","tô","to","ai","ah","rs","kkk","kk","k","rsrs","haha",
    "https","www","youtube","com","watch","br","tv","video","canal",
}

# ── Funções de processamento ────────────────────────────────────────────────
def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"[^\w\sáéíóúâêîôûãõàèìòùäëïöüç]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def classificar_sentimento(texto):
    if not isinstance(texto, str) or len(texto.strip()) < 3:
        return "Neutro"
    palavras = set(limpar_texto(texto).split())
    pos = len(palavras & POSITIVOS)
    neg = len(palavras & NEGATIVOS)
    if pos > neg:
        return "Positivo"
    elif neg > pos:
        return "Negativo"
    else:
        return "Neutro"

def score_sentimento(texto):
    if not isinstance(texto, str):
        return 0
    palavras = limpar_texto(texto).split()
    pos = sum(1 for w in palavras if w in POSITIVOS)
    neg = sum(1 for w in palavras if w in NEGATIVOS)
    total = pos + neg
    if total == 0:
        return 0
    return (pos - neg) / total

def extrair_palavras(textos, excluir_stopwords=True):
    todas = []
    for t in textos:
        if isinstance(t, str):
            palavras = limpar_texto(t).split()
            if excluir_stopwords:
                palavras = [p for p in palavras if p not in STOPWORDS_PT and len(p) > 2]
            todas.extend(palavras)
    return todas

def gerar_wordcloud(palavras, colormap, titulo):
    freq = Counter(palavras)
    if not freq:
        return None
    wc = WordCloud(
        width=700, height=380,
        background_color="white",
        colormap=colormap,
        max_words=80,
        collocations=False,
        prefer_horizontal=0.7,
    ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(titulo, fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig

# ── Carregamento dos dados ──────────────────────────────────────────────────
@st.cache_data
def carregar_dados(path):
    df = pd.read_excel(path)
    df = df[["tema", "video_id", "comentario", "likes", "autor", "fonte"]].copy()
    df["comentario"] = df["comentario"].fillna("").astype(str)
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    df["texto_limpo"] = df["comentario"].apply(limpar_texto)
    df["sentimento"] = df["comentario"].apply(classificar_sentimento)
    df["score"] = df["comentario"].apply(score_sentimento)
    df["n_palavras"] = df["texto_limpo"].apply(lambda x: len(x.split()))
    return df

df = carregar_dados("dataset_cnh.xlsx")

# ── Sidebar / Filtros ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Flag_of_Brazil.svg/200px-Flag_of_Brazil.svg.png", width=60)
    st.title("🚗 Filtros")

    sentimentos_opcoes = ["Todos"] + sorted(df["sentimento"].unique().tolist())
    filtro_sent = st.selectbox("Sentimento", sentimentos_opcoes)

    fontes = ["Todas"] + [f.split("v=")[-1] if "v=" in str(f) else str(f) for f in df["fonte"].dropna().unique()]
    fontes_raw = ["Todas"] + list(df["fonte"].dropna().unique())
    idx_fonte = st.selectbox("Vídeo (fonte)", range(len(fontes)), format_func=lambda i: fontes[i])

    min_likes = int(df["likes"].min())
    max_likes = int(df["likes"].max())
    if max_likes > min_likes:
        faixa_likes = st.slider("Mínimo de likes", min_likes, max_likes, min_likes)
    else:
        faixa_likes = min_likes

    st.markdown("---")
    st.caption(f"Total de comentários: **{len(df)}**")

# Aplicar filtros
dff = df.copy()
if filtro_sent != "Todos":
    dff = dff[dff["sentimento"] == filtro_sent]
if idx_fonte > 0:
    dff = dff[dff["fonte"] == fontes_raw[idx_fonte]]
dff = dff[dff["likes"] >= faixa_likes]

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#1a1a2e;'>
  🚗 Análise de Sentimentos — Novas Regras da CNH
</h1>
<p style='text-align:center; color:#555; font-size:16px;'>
  Análise de comentários públicos sobre as mudanças no processo de habilitação no Brasil
</p>
<hr>
""", unsafe_allow_html=True)

# ── KPIs ────────────────────────────────────────────────────────────────────
total = len(dff)
pos = (dff["sentimento"] == "Positivo").sum()
neg = (dff["sentimento"] == "Negativo").sum()
neu = (dff["sentimento"] == "Neutro").sum()
pct_pos = pos / total * 100 if total else 0
pct_neg = neg / total * 100 if total else 0
pct_neu = neu / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("💬 Comentários", f"{total:,}")
c2.metric("😊 Positivos", f"{pos:,}", f"{pct_pos:.1f}%")
c3.metric("😠 Negativos", f"{neg:,}", f"{pct_neg:.1f}%")
c4.metric("😐 Neutros",   f"{neu:,}", f"{pct_neu:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ── Linha 1: Pizza + Barras por Fonte ──────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Distribuição de Sentimentos")
    fig_pie = px.pie(
        names=["Positivo", "Negativo", "Neutro"],
        values=[pos, neg, neu],
        color=["Positivo", "Negativo", "Neutro"],
        color_discrete_map={"Positivo": "#2ecc71", "Negativo": "#e74c3c", "Neutro": "#95a5a6"},
        hole=0.45,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=320)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    st.subheader("🎥 Sentimentos por Vídeo")
    df_fonte = (
        df.groupby(["fonte", "sentimento"])
        .size().reset_index(name="count")
    )
    df_fonte["video"] = df_fonte["fonte"].apply(
        lambda x: ("v=" + x.split("v=")[-1])[:20] if isinstance(x, str) and "v=" in x else str(x)[:20]
    )
    fig_bar = px.bar(
        df_fonte, x="video", y="count", color="sentimento",
        color_discrete_map={"Positivo": "#2ecc71", "Negativo": "#e74c3c", "Neutro": "#95a5a6"},
        barmode="stack",
        labels={"count": "Comentários", "video": "Vídeo", "sentimento": "Sentimento"},
    )
    fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, legend_title="")
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Linha 2: Score médio + Top palavras ────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("📈 Score de Sentimento por Vídeo")
    df_score = df.groupby("fonte")["score"].mean().reset_index()
    df_score["video"] = df_score["fonte"].apply(
        lambda x: ("v=" + x.split("v=")[-1])[:20] if isinstance(x, str) and "v=" in x else str(x)[:20]
    )
    df_score["cor"] = df_score["score"].apply(lambda s: "#2ecc71" if s > 0 else ("#e74c3c" if s < 0 else "#95a5a6"))
    fig_score = go.Figure(go.Bar(
        x=df_score["video"], y=df_score["score"],
        marker_color=df_score["cor"],
        text=df_score["score"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
    ))
    fig_score.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_score.update_layout(
        yaxis_title="Score médio", xaxis_title="Vídeo",
        margin=dict(t=10, b=10, l=10, r=10), height=320
    )
    st.plotly_chart(fig_score, use_container_width=True)

with col_d:
    st.subheader("🔠 Top 15 Palavras Mais Frequentes")
    palavras_all = extrair_palavras(dff["comentario"])
    freq_df = pd.DataFrame(Counter(palavras_all).most_common(15), columns=["Palavra", "Frequência"])
    fig_freq = px.bar(
        freq_df, x="Frequência", y="Palavra", orientation="h",
        color="Frequência", color_continuous_scale="Blues",
    )
    fig_freq.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=10), height=320
    )
    st.plotly_chart(fig_freq, use_container_width=True)

# ── Linha 3: Nuvens de Palavras ─────────────────────────────────────────────
st.markdown("---")
st.subheader("☁️ Nuvem de Palavras por Sentimento")

tab_pos, tab_neg, tab_neu, tab_all = st.tabs(["😊 Positivos", "😠 Negativos", "😐 Neutros", "🌐 Todos"])

def mostrar_nuvem(sent_filtro, colormap, titulo, tab):
    with tab:
        sub = dff[dff["sentimento"] == sent_filtro] if sent_filtro else dff
        palavras = extrair_palavras(sub["comentario"])
        if palavras:
            fig = gerar_wordcloud(palavras, colormap, titulo)
            if fig:
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
                st.image(buf.getvalue(), use_container_width=True)
                plt.close(fig)
        else:
            st.info("Nenhum comentário nesta categoria com os filtros atuais.")

mostrar_nuvem("Positivo", "Greens",  "Palavras nos Comentários Positivos", tab_pos)
mostrar_nuvem("Negativo", "Reds",    "Palavras nos Comentários Negativos", tab_neg)
mostrar_nuvem("Neutro",   "Blues",   "Palavras nos Comentários Neutros",   tab_neu)
mostrar_nuvem(None,       "plasma",  "Palavras em Todos os Comentários",   tab_all)

# ── Linha 4: Likes por sentimento + Tamanho dos comentários ────────────────
st.markdown("---")
col_e, col_f = st.columns(2)

with col_e:
    st.subheader("👍 Likes por Sentimento")
    df_likes = dff.groupby("sentimento")["likes"].agg(["sum","mean","max"]).reset_index()
    df_likes.columns = ["Sentimento", "Total de Likes", "Média de Likes", "Máx. Likes"]
    fig_likes = px.bar(
        df_likes, x="Sentimento", y="Total de Likes", color="Sentimento",
        color_discrete_map={"Positivo": "#2ecc71", "Negativo": "#e74c3c", "Neutro": "#95a5a6"},
        text="Total de Likes",
    )
    fig_likes.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig_likes.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
    st.plotly_chart(fig_likes, use_container_width=True)

with col_f:
    st.subheader("📝 Comprimento dos Comentários por Sentimento")
    fig_box = px.box(
        dff, x="sentimento", y="n_palavras", color="sentimento",
        color_discrete_map={"Positivo": "#2ecc71", "Negativo": "#e74c3c", "Neutro": "#95a5a6"},
        labels={"n_palavras": "Nº de palavras", "sentimento": "Sentimento"},
        points="outliers",
    )
    fig_box.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
    st.plotly_chart(fig_box, use_container_width=True)

# ── Linha 5: Top comentários com mais likes ─────────────────────────────────
st.markdown("---")
st.subheader("🏆 Comentários com Mais Likes")

tab_t1, tab_t2, tab_t3 = st.tabs(["😊 Mais curtidos Positivos", "😠 Mais curtidos Negativos", "📋 Todos (top 10)"])

def mostrar_top(df_sub, tab, n=10):
    with tab:
        top = df_sub.nlargest(n, "likes")[["comentario","likes","sentimento","autor"]].reset_index(drop=True)
        top.index += 1
        cores = {"Positivo": "🟢", "Negativo": "🔴", "Neutro": "⚪"}
        for _, row in top.iterrows():
            icone = cores.get(row["sentimento"], "⚪")
            st.markdown(
                f"**{icone} {row['sentimento']}** · 👍 {int(row['likes'])} likes  \n"
                f"> {row['comentario'][:250]}{'...' if len(row['comentario'])>250 else ''}",
                unsafe_allow_html=False,
            )
            st.markdown("---")

mostrar_top(dff[dff["sentimento"]=="Positivo"], tab_t1)
mostrar_top(dff[dff["sentimento"]=="Negativo"], tab_t2)
mostrar_top(dff, tab_t3)

# ── Linha 6: Comparativo de palavras-chave ──────────────────────────────────
st.markdown("---")
st.subheader("🔍 Palavras-Chave: Positivos vs Negativos")

col_g, col_h = st.columns(2)

with col_g:
    pal_pos = extrair_palavras(df[df["sentimento"]=="Positivo"]["comentario"])
    top_pos = pd.DataFrame(Counter(pal_pos).most_common(12), columns=["Palavra","Freq"])
    fig_pp = px.bar(top_pos, x="Freq", y="Palavra", orientation="h",
                    color_discrete_sequence=["#2ecc71"],
                    title="Top palavras — Positivos")
    fig_pp.update_layout(yaxis={"categoryorder":"total ascending"},
                         margin=dict(t=40,b=10,l=10,r=10), height=340, showlegend=False)
    st.plotly_chart(fig_pp, use_container_width=True)

with col_h:
    pal_neg = extrair_palavras(df[df["sentimento"]=="Negativo"]["comentario"])
    top_neg = pd.DataFrame(Counter(pal_neg).most_common(12), columns=["Palavra","Freq"])
    fig_pn = px.bar(top_neg, x="Freq", y="Palavra", orientation="h",
                    color_discrete_sequence=["#e74c3c"],
                    title="Top palavras — Negativos")
    fig_pn.update_layout(yaxis={"categoryorder":"total ascending"},
                         margin=dict(t=40,b=10,l=10,r=10), height=340, showlegend=False)
    st.plotly_chart(fig_pn, use_container_width=True)

# ── Explorador de dados ──────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🗂️ Explorar base de dados filtrada"):
    cols_show = ["comentario","sentimento","score","likes","autor"]
    st.dataframe(
        dff[cols_show].sort_values("likes", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=350,
    )
    csv = dff[cols_show].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV", csv, "comentarios_sentimentos.csv", "text/csv")

# ── Rodapé ───────────────────────────────────────────────────────────────────
st.markdown("""
<br>
<p style='text-align:center; color:#aaa; font-size:12px;'>
  Análise de Sentimentos · Novas Regras CNH Brasil · Dados coletados do YouTube
</p>
""", unsafe_allow_html=True)