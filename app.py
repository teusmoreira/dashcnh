"""
Análise de Sentimentos — Novas Regras da CNH no Brasil
Dataset: raspagem_cnh.xlsx  (YouTube Data API v3)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from io import BytesIO

# ── Página ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise de Sentimentos – CNH",
    page_icon="🚗",
    layout="wide",
)

# ── Léxico de sentimentos em português (contexto CNH/governo) ───────────────
POSITIVOS = {
    "maravilhoso","ótimo","excelente","parabéns","perfeito","bom","boa","bons","boas",
    "fantástico","incrível","legal","show","top","melhor","melhorou","melhorando",
    "facilitou","facilitar","facilita","facilitando","fácil","acessível","acessibilidade",
    "barato","economizando","economia","economizar","reduziu","reduzir","redução",
    "benefício","beneficiou","beneficia","beneficiar","aprovado","aprovar","aprovação",
    "glória","amei","adorei","gostei","gosto","amo","adoro","ótima","certo",
    "correto","verdade","verdadeiro","justo","sucesso","vitória","ganho","ganhar",
    "ajudou","ajudar","ajuda","apoio","apoiar","graças","obrigado","parabens",
    "eficiente","eficiência","moderno","modernização","progresso","avanço","avanços",
    "alegria","feliz","felicidade","satisfeito","satisfação","valeu","vantagem",
    "simplificou","simplificar","desburocratizou","aprovamos","aprovei","baratear",
    "lenda","incrivel","demais","otimo","facil","rapido","rápido","pratico","prático",
    "simples","funciona","funcionou","funcionando","recomendo","recomendado",
    "perfeito","excelente","maravilha","felizmente","gratuito","grátis","economico",
}

NEGATIVOS = {
    "caro","cara","caros","absurdo","ridículo","lamentável","vergonha","vergonhoso",
    "ruim","péssimo","horrível","terrível","pior","piora","piorou","piorando",
    "injusto","injustiça","problema","problemas","dificuldade","difícil","complicado",
    "complicação","burocracia","burocrático","exploração","explorar","exploram",
    "roubando","roubo","rouba","cobram","cobrança","extorção","extorsão","abusivo",
    "abuso","prejudica","prejudicou","prejudicial","prejudicando","perdemos","perder",
    "perdeu","péssima","golpe","mentira","mentiroso","enganação","enganar","engana",
    "lamentavelmente","triste","tristeza","raiva","indignado","indignação","revoltante",
    "revoltado","revolta","protesto","protestamos","impossível","corrompido","corrupto",
    "corrupção","decepção","decepcionado","decepcionante","insatisfeito","insatisfação",
    "desnecessário","inútil","falhou","falha","falso","aumentar","aumento","aumentaram",
    "aumentou","taxa","taxas","imposto","impostos","desvantagem","complicou","complicar",
    "problemático","erro","erros","errou","perigoso","perigo","acidente","acidentes",
    "inseguro","insegurança","irresponsável","irresponsabilidade","descaso","negligência",
    "fraude","fraudulento","irregular","ilegal","burocratico","demora","demorou",
    "lento","travou","trava","bugou","bug","pessimo","horrivel","terrivel","ridiculo",
    "absurda","vergonhosa","injusta","caríssimo","caríssima","salgado","inacreditável",
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
    "pra","pro","pras","pros","né","lhe","lhes","quem","qual","quais",
    "tá","ta","vou","tô","to","ai","ah","rs","kkk","kk","k","rsrs","haha","kkkkk",
    "https","www","youtube","com","watch","br","tv","video","canal","link",
    "sim","nao","nada","tudo","cada","todo","toda","todos","todas","outro","outra",
    "então","logo","pois","porém","contudo","entanto","todavia","porque",
    "vcs","vc","gente","cara","pessoa","pessoal","galera","povo","brasil","brasileiro",
    "ter","tive","tinha","tenho","temos","voce","ele","ela","eles","elas","esse","essa",
}

# ── Funções ─────────────────────────────────────────────────────────────────
def limpar_texto(texto):
    if not isinstance(texto, str) or texto.strip() == "":
        return ""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"[^\w\sáéíóúâêîôûãõàèìòùäëïöüç]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()

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
    return "Neutro"

def score_sentimento(texto):
    if not isinstance(texto, str):
        return 0.0
    palavras = limpar_texto(texto).split()
    pos = sum(1 for w in palavras if w in POSITIVOS)
    neg = sum(1 for w in palavras if w in NEGATIVOS)
    total = pos + neg
    return round((pos - neg) / total, 3) if total else 0.0

def extrair_palavras(textos):
    todas = []
    for t in textos:
        if isinstance(t, str):
            palavras = limpar_texto(t).split()
            palavras = [p for p in palavras if p not in STOPWORDS_PT and len(p) > 2]
            todas.extend(palavras)
    return todas

def gerar_wordcloud_fig(palavras, colormap, titulo):
    freq = Counter(palavras)
    if not freq:
        return None
    wc = WordCloud(
        width=720, height=380,
        background_color="white",
        colormap=colormap,
        max_words=100,
        collocations=False,
        prefer_horizontal=0.75,
    ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if titulo:
        ax.set_title(titulo, fontsize=13, fontweight="bold", pad=8)
    fig.tight_layout()
    return fig

def fig_para_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

# ── Carregamento ─────────────────────────────────────────────────────────────
@st.cache_data
def carregar_dados(path):
    df = pd.read_excel(path)
    df["comentario"]       = df["comentario"].fillna("").astype(str)
    df["comentario_limpo"] = df["comentario_limpo"].fillna("").astype(str)
    df["likes_comentario"] = pd.to_numeric(df["likes_comentario"], errors="coerce").fillna(0).astype(int)
    df["respostas"]        = pd.to_numeric(df["respostas"], errors="coerce").fillna(0).astype(int)
    df["nome_canal"]       = df["nome_canal"].fillna("Desconhecido")
    df["termo_busca"]      = df["termo_busca"].fillna("Outros")
    df["autor"]            = df["autor"].fillna("Anônimo")
    df["sentimento"]       = df["comentario"].apply(classificar_sentimento)
    df["score"]            = df["comentario"].apply(score_sentimento)
    df["n_palavras"]       = df["comentario_limpo"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    return df

df = carregar_dados("raspagem_cnh.xlsx")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Flag_of_Brazil.svg/200px-Flag_of_Brazil.svg.png", width=55)
    st.title("🚗 Filtros")

    filtro_sent  = st.selectbox("Sentimento", ["Todos", "Positivo", "Negativo", "Neutro"])

    canais = ["Todos os canais"] + sorted(df["nome_canal"].unique().tolist())
    filtro_canal = st.selectbox("Canal", canais)

    termos = ["Todos os termos"] + sorted(df["termo_busca"].unique().tolist())
    filtro_termo = st.selectbox("Termo de busca", termos)

    max_likes    = int(df["likes_comentario"].max())
    filtro_likes = st.slider("Mínimo de likes no comentário", 0, max_likes, 0)
    filtro_palavras = st.slider("Mínimo de palavras no comentário", 0, 50, 0)

    st.markdown("---")
    st.caption(f"Base total: **{len(df):,}** comentários")
    st.caption(f"Vídeos: **{df['link_video'].nunique()}**")
    st.caption(f"Canais: **{df['nome_canal'].nunique()}**")

# Aplicar filtros
dff = df.copy()
if filtro_sent  != "Todos":           dff = dff[dff["sentimento"] == filtro_sent]
if filtro_canal != "Todos os canais": dff = dff[dff["nome_canal"] == filtro_canal]
if filtro_termo != "Todos os termos": dff = dff[dff["termo_busca"] == filtro_termo]
dff = dff[dff["likes_comentario"] >= filtro_likes]
dff = dff[dff["n_palavras"] >= filtro_palavras]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:white;'>
  🚗 Análise de Sentimentos — Novas Regras da CNH
</h1>
<p style='text-align:center; color:white; font-size:20px;'>
  Comentários coletados do YouTube via API · Mudanças no processo de habilitação no Brasil
</p>
<hr style='margin-top:0'>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
total       = len(dff)
pos         = (dff["sentimento"] == "Positivo").sum()
neg         = (dff["sentimento"] == "Negativo").sum()
neu         = (dff["sentimento"] == "Neutro").sum()
pct_p       = pos / total * 100 if total else 0
pct_n       = neg / total * 100 if total else 0
pct_u       = neu / total * 100 if total else 0
total_likes = int(dff["likes_comentario"].sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💬 Comentários",   f"{total:,}")
k2.metric("😊 Positivos",     f"{pos:,}",  f"{pct_p:.1f}%")
k3.metric("😠 Negativos",     f"{neg:,}",  f"{pct_n:.1f}%")
k4.metric("😐 Neutros",       f"{neu:,}",  f"{pct_u:.1f}%")
k5.metric("👍 Total de Likes",f"{total_likes:,}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Linha 1: Pizza + Sentimento por Termo ─────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 Distribuição de Sentimentos")
    fig_pie = px.pie(
        names=["Positivo","Negativo","Neutro"], values=[pos, neg, neu],
        color=["Positivo","Negativo","Neutro"],
        color_discrete_map={"Positivo":"#2ecc71","Negativo":"#e74c3c","Neutro":"#95a5a6"},
        hole=0.45,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10), height=320)
    st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    st.subheader("🔎 Sentimentos por Termo de Busca")
    df_termo = dff.groupby(["termo_busca","sentimento"]).size().reset_index(name="count")
    df_termo["termo_curto"] = df_termo["termo_busca"].apply(lambda x: x[:35]+"..." if len(x)>35 else x)
    fig_termo = px.bar(
        df_termo, x="count", y="termo_curto", color="sentimento", orientation="h",
        color_discrete_map={"Positivo":"#2ecc71","Negativo":"#e74c3c","Neutro":"#95a5a6"},
        barmode="stack",
        labels={"count":"Comentários","termo_curto":"Termo","sentimento":"Sentimento"},
    )
    fig_termo.update_layout(
        yaxis={"categoryorder":"total ascending"},
        legend_title="", margin=dict(t=10,b=10,l=10,r=10), height=320
    )
    st.plotly_chart(fig_termo, use_container_width=True)

# ── Linha 2: Sentimento por Canal + Likes por Canal ───────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.subheader("📺 Sentimentos por Canal (Top 10)")
    top_canais = dff["nome_canal"].value_counts().head(10).index.tolist()
    df_canal = (
        dff[dff["nome_canal"].isin(top_canais)]
        .groupby(["nome_canal","sentimento"]).size().reset_index(name="count")
    )
    fig_canal = px.bar(
        df_canal, x="count", y="nome_canal", color="sentimento", orientation="h",
        color_discrete_map={"Positivo":"#2ecc71","Negativo":"#e74c3c","Neutro":"#95a5a6"},
        barmode="stack",
        labels={"count":"Comentários","nome_canal":"Canal"},
    )
    fig_canal.update_layout(
        yaxis={"categoryorder":"total ascending"},
        legend_title="", margin=dict(t=10,b=10,l=10,r=10), height=340
    )
    st.plotly_chart(fig_canal, use_container_width=True)

with c4:
    st.subheader("👍 Total de Likes por Canal (Top 10)")
    df_lk = (
        dff.groupby("nome_canal")["likes_comentario"].sum()
        .reset_index().sort_values("likes_comentario", ascending=False).head(10)
    )
    fig_lk = px.bar(
        df_lk, x="likes_comentario", y="nome_canal", orientation="h",
        color="likes_comentario", color_continuous_scale="Blues",
        labels={"likes_comentario":"Total de Likes","nome_canal":"Canal"},
    )
    fig_lk.update_layout(
        yaxis={"categoryorder":"total ascending"},
        coloraxis_showscale=False, margin=dict(t=10,b=10,l=10,r=10), height=340
    )
    st.plotly_chart(fig_lk, use_container_width=True)

# ── Linha 3: Score por canal + Engajamento scatter ────────────────────────────
c5, c6 = st.columns(2)

with c5:
    st.subheader("📈 Score Médio de Sentimento por Canal")
    df_sc = (
        dff.groupby("nome_canal")["score"].mean()
        .reset_index().sort_values("score", ascending=False).head(15)
    )
    df_sc["cor"] = df_sc["score"].apply(
        lambda s: "#2ecc71" if s > 0 else ("#e74c3c" if s < 0 else "#95a5a6")
    )
    fig_sc = go.Figure(go.Bar(
        x=df_sc["score"], y=df_sc["nome_canal"], orientation="h",
        marker_color=df_sc["cor"],
        text=df_sc["score"].apply(lambda x: f"{x:.3f}"), textposition="outside",
    ))
    fig_sc.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_sc.update_layout(
        xaxis_title="Score médio",
        yaxis={"categoryorder":"total ascending"},
        margin=dict(t=10,b=10,l=10,r=10), height=380
    )
    st.plotly_chart(fig_sc, use_container_width=True)

with c6:
    st.subheader("💬 Engajamento: Likes vs Respostas")
    df_eng = dff[dff["likes_comentario"] > 0].nlargest(300, "likes_comentario")
    fig_eng = px.scatter(
        df_eng, x="likes_comentario", y="respostas",
        color="sentimento", size="likes_comentario",
        color_discrete_map={"Positivo":"#2ecc71","Negativo":"#e74c3c","Neutro":"#95a5a6"},
        hover_data={"comentario":True,"nome_canal":True},
        labels={"likes_comentario":"Likes","respostas":"Respostas"},
        size_max=30, opacity=0.7,
    )
    fig_eng.update_layout(legend_title="", margin=dict(t=10,b=10,l=10,r=10), height=380)
    st.plotly_chart(fig_eng, use_container_width=True)

# ── Linha 4: Top palavras + Box plot ─────────────────────────────────────────
c7, c8 = st.columns(2)

with c7:
    st.subheader("🔠 Top 15 Palavras Mais Frequentes")
    palavras_all = extrair_palavras(dff["comentario"])
    freq_df = pd.DataFrame(Counter(palavras_all).most_common(15), columns=["Palavra","Frequência"])
    fig_freq = px.bar(
        freq_df, x="Frequência", y="Palavra", orientation="h",
        color="Frequência", color_continuous_scale="Blues",
    )
    fig_freq.update_layout(
        yaxis={"categoryorder":"total ascending"},
        coloraxis_showscale=False, margin=dict(t=10,b=10,l=10,r=10), height=340
    )
    st.plotly_chart(fig_freq, use_container_width=True)

with c8:
    st.subheader("📝 Comprimento dos Comentários por Sentimento")
    fig_box = px.box(
        dff, x="sentimento", y="n_palavras", color="sentimento",
        color_discrete_map={"Positivo":"#2ecc71","Negativo":"#e74c3c","Neutro":"#95a5a6"},
        labels={"n_palavras":"Nº de palavras","sentimento":"Sentimento"},
        points="outliers",
    )
    fig_box.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10), height=340)
    st.plotly_chart(fig_box, use_container_width=True)

# ── Nuvem de palavras — três colunas na mesma aba ─────────────────────────────
st.markdown("---")
st.subheader("☁️ Nuvem de Palavras por Sentimento")

wc_col1, wc_col2, wc_col3 = st.columns(3)

for col, sent_filtro, cmap, rotulo in [
    (wc_col1, "Positivo", "Greens", "😊 Positivos"),
    (wc_col2, "Negativo", "Reds",   "😠 Negativos"),
    (wc_col3, "Neutro",   "Blues",  "😐 Neutros"),
]:
    with col:
        st.markdown(f"**{rotulo}**")
        sub      = dff[dff["sentimento"] == sent_filtro]
        palavras = extrair_palavras(sub["comentario"])
        if palavras:
            fig = gerar_wordcloud_fig(palavras, cmap, "")
            if fig:
                st.image(fig_para_bytes(fig), use_container_width=True)
        else:
            st.info("Nenhum comentário nesta categoria com os filtros atuais.")

# ── Comparativo Positivos vs Negativos ────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Palavras-Chave: Positivos vs Negativos")

c9, c10 = st.columns(2)
with c9:
    pal_pos = extrair_palavras(dff[dff["sentimento"]=="Positivo"]["comentario"])
    top_pos = pd.DataFrame(Counter(pal_pos).most_common(12), columns=["Palavra","Freq"])
    fig_pp  = px.bar(top_pos, x="Freq", y="Palavra", orientation="h",
                     color_discrete_sequence=["#2ecc71"], title="Top palavras — Positivos")
    fig_pp.update_layout(yaxis={"categoryorder":"total ascending"},
                         margin=dict(t=40,b=10,l=10,r=10), height=360, showlegend=False)
    st.plotly_chart(fig_pp, use_container_width=True)

with c10:
    pal_neg = extrair_palavras(dff[dff["sentimento"]=="Negativo"]["comentario"])
    top_neg = pd.DataFrame(Counter(pal_neg).most_common(12), columns=["Palavra","Freq"])
    fig_pn  = px.bar(top_neg, x="Freq", y="Palavra", orientation="h",
                     color_discrete_sequence=["#e74c3c"], title="Top palavras — Negativos")
    fig_pn.update_layout(yaxis={"categoryorder":"total ascending"},
                         margin=dict(t=40,b=10,l=10,r=10), height=360, showlegend=False)
    st.plotly_chart(fig_pn, use_container_width=True)

# ── Top comentários com mais likes ────────────────────────────────────────────
st.markdown("---")
st.subheader("🏆 Comentários com Mais Likes")

tab_t1, tab_t2, tab_t3 = st.tabs(["😊 Positivos","😠 Negativos","📋 Geral (Top 10)"])
COR = {"Positivo":"🟢","Negativo":"🔴","Neutro":"⚪"}

def mostrar_top(df_sub, tab, n=10):
    with tab:
        top = df_sub.nlargest(n, "likes_comentario")[
            ["comentario","likes_comentario","respostas","sentimento","autor","nome_canal","link_video"]
        ].reset_index(drop=True)
        for _, row in top.iterrows():
            icone = COR.get(row["sentimento"], "⚪")
            texto = row["comentario"][:300] + ("..." if len(row["comentario"]) > 300 else "")
            st.markdown(
                f"**{icone} {row['sentimento']}** · 👍 {int(row['likes_comentario'])} likes "
                f"· 💬 {int(row['respostas'])} respostas · 📺 {row['nome_canal']}  \n> {texto}"
            )
            st.markdown("---")

mostrar_top(dff[dff["sentimento"]=="Positivo"], tab_t1)
mostrar_top(dff[dff["sentimento"]=="Negativo"], tab_t2)
mostrar_top(dff, tab_t3)

# ── Explorador de dados ────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🗂️ Explorar base de dados filtrada"):
    cols_show = ["comentario","sentimento","score","likes_comentario","respostas",
                 "autor","nome_canal","termo_busca","link_video"]
    st.dataframe(
        dff[cols_show].sort_values("likes_comentario", ascending=False).reset_index(drop=True),
        use_container_width=True, height=380,
    )
    csv = dff[cols_show].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("⬇️ Baixar CSV filtrado", csv, "comentarios_sentimentos.csv", "text/csv")

# ── Rodapé ─────────────────────────────────────────────────────────────────────
st.markdown("""
<br>
<p style='text-align:center; color:#bbb; font-size:12px;'>
  Análise de Sentimentos · Novas Regras CNH Brasil · Dados coletados do YouTube via API
</p>
""", unsafe_allow_html=True)
