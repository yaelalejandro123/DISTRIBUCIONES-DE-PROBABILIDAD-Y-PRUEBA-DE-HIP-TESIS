"""
app.py — Interfaz principal de la aplicación de análisis estadístico.
"""

import streamlit as st
from data_loader import cargar_datos, generar_datos_sinteticos, obtener_columnas_numericas
from visualization import mostrar_distribucion
from stats_tests import ejecutar_prueba_z, visualizar_prueba_z
from ai_module import analizar_con_ia

# ─────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis Estadístico Interactivo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fondo general */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #1e2230;
}
section[data-testid="stSidebar"] * {
    color: #c8ccd8 !important;
}

/* Títulos principales */
h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem !important;
    color: #7eb8f7 !important;
    letter-spacing: -1px;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #1e3a5f;
}
h2 {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.15rem !important;
    color: #a8c8f0 !important;
    margin-top: 1.5rem !important;
}
h3 {
    color: #7eb8f7 !important;
    font-size: 1rem !important;
}

/* Métricas */
[data-testid="stMetric"] {
    background: #13161e;
    border: 1px solid #1e2a3d;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #7b8aaa !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #7eb8f7 !important; font-family: 'Space Mono', monospace !important; }

/* Cajas de resultados */
.result-box {
    background: #13161e;
    border: 1px solid #1e2a3d;
    border-left: 4px solid #7eb8f7;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 10px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #c8ccd8;
}
.result-box.rechazar {
    border-left-color: #f77e7e;
}
.result-box.no-rechazar {
    border-left-color: #7ef7a8;
}

/* Botones */
.stButton > button {
    background: #1a2d4a !important;
    color: #7eb8f7 !important;
    border: 1px solid #2a4a7a !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #2a4a7a !important;
    border-color: #7eb8f7 !important;
}

/* Inputs */
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background: #13161e !important;
    border-color: #1e2a3d !important;
    color: #e8eaf0 !important;
}

/* Separadores */
hr { border-color: #1e2230 !important; }

/* AI box */
.ai-box {
    background: linear-gradient(135deg, #0d1a2d, #13161e);
    border: 1px solid #2a4a7a;
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 12px;
    line-height: 1.75;
    color: #c8ccd8;
    font-size: 0.9rem;
}

/* Tablas */
.stDataFrame { background: #13161e !important; }

/* Info/Warning/Success */
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("# 📊 Análisis Estadístico Interactivo")
st.markdown("*Prueba Z · Visualización · Análisis con IA*")
st.markdown("---")


# ─────────────────────────────────────────────
# SIDEBAR — Controles globales
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    # Fuente de datos
    st.markdown("**Fuente de datos**")
    fuente = st.radio(
        "Selecciona origen:",
        ["Generar datos sintéticos", "Subir archivo CSV"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Parámetros prueba Z
    st.markdown("**Parámetros — Prueba Z**")

    mu_0 = st.number_input("Media hipotética (μ₀)", value=0.0, step=0.1, format="%.4f")
    sigma = st.number_input("Desv. estándar poblacional (σ)", value=1.0, min_value=0.0001, step=0.1, format="%.4f")
    alpha = st.select_slider(
        "Nivel de significancia (α)",
        options=[0.01, 0.025, 0.05, 0.10],
        value=0.05
    )
    tipo_prueba = st.selectbox(
        "Tipo de prueba",
        ["Bilateral", "Cola izquierda", "Cola derecha"]
    )

    st.markdown("---")
    st.markdown("🤖 **Análisis con IA habilitado**")
    st.caption("Powered by OpenRouter · Mistral 7B")


# ─────────────────────────────────────────────
# PASO 1 — Carga de datos
# ─────────────────────────────────────────────
st.markdown("## 1 · Datos")

df = None

if fuente == "Subir archivo CSV":
    archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if archivo is not None:
        df, error = cargar_datos(archivo)
        if error:
            st.error(f"❌ Error al cargar el archivo: {error}")
        else:
            st.success(f"✅ Archivo cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
            with st.expander("Vista previa del dataset"):
                st.dataframe(df.head(20), use_container_width=True)
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        n_muestras = st.number_input("Número de muestras", min_value=30, max_value=10000, value=200, step=10)
    with col2:
        media_real = st.number_input("Media real", value=0.0, step=0.1, format="%.2f")
    with col3:
        std_real = st.number_input("Desv. estándar real", value=1.0, min_value=0.01, step=0.1, format="%.2f")

    if st.button("🎲 Generar datos sintéticos"):
        df = generar_datos_sinteticos(n=n_muestras, media=media_real, std=std_real)
        st.session_state["df"] = df
        st.success(f"✅ Generados {n_muestras} datos con distribución Normal(μ={media_real}, σ={std_real})")

    # Recuperar de session_state si ya fue generado
    if df is None and "df" in st.session_state:
        df = st.session_state["df"]

# ─────────────────────────────────────────────
# PASO 2 — Selección de columna
# ─────────────────────────────────────────────
columna_seleccionada = None
datos = None

if df is not None:
    st.markdown("## 2 · Variable")
    columnas_num = obtener_columnas_numericas(df)

    if not columnas_num:
        st.error("❌ No se encontraron columnas numéricas en el dataset.")
    else:
        columna_seleccionada = st.selectbox("Selecciona la variable a analizar:", columnas_num)
        datos = df[columna_seleccionada].dropna()
        st.info(f"📌 Variable seleccionada: **{columna_seleccionada}** · n = {len(datos)}")


# ─────────────────────────────────────────────
# PASO 3 — Visualización de la distribución
# ─────────────────────────────────────────────
if datos is not None and len(datos) >= 2:
    st.markdown("## 3 · Distribución")
    mostrar_distribucion(datos, nombre_columna=columna_seleccionada)


# ─────────────────────────────────────────────
# PASO 4 & 5 — Prueba Z
# ─────────────────────────────────────────────
if datos is not None and len(datos) >= 30:
    st.markdown("## 4 · Prueba Z")

    if st.button("▶ Ejecutar Prueba Z"):
        resultado = ejecutar_prueba_z(
            datos=datos,
            mu_0=mu_0,
            sigma=sigma,
            alpha=alpha,
            tipo_prueba=tipo_prueba
        )
        st.session_state["resultado_z"] = resultado
        st.session_state["params_z"] = {
            "mu_0": mu_0, "sigma": sigma, "alpha": alpha,
            "tipo_prueba": tipo_prueba, "columna": columna_seleccionada
        }

    # Mostrar resultados si existen
    if "resultado_z" in st.session_state:
        r = st.session_state["resultado_z"]
        p = st.session_state["params_z"]

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Media muestral (x̄)", f"{r['media_muestral']:.4f}")
        with col2:
            st.metric("Tamaño de muestra (n)", f"{r['n']}")
        with col3:
            st.metric("Estadístico Z", f"{r['z_stat']:.4f}")
        with col4:
            st.metric("p-value", f"{r['p_value']:.4f}")

        # Decisión
        clase_box = "rechazar" if r["rechazar_h0"] else "no-rechazar"
        icono = "❌" if r["rechazar_h0"] else "✅"
        st.markdown(f"""
        <div class="result-box {clase_box}">
        <b>{icono} Decisión:</b> {r['decision']}<br><br>
        <b>Región crítica:</b> {r['region_critica']}<br>
        <b>Interpretación:</b> {r['interpretacion']}
        </div>
        """, unsafe_allow_html=True)

        # Gráfica de la prueba
        st.markdown("## 5 · Visualización de la Prueba")
        visualizar_prueba_z(
            z_stat=r["z_stat"],
            alpha=alpha,
            tipo_prueba=tipo_prueba,
            p_value=r["p_value"]
        )

        # ─────────────────────────────────────────────
        # PASO 6 — Análisis con IA
        # ─────────────────────────────────────────────
        st.markdown("## 6 · Análisis con IA")

        if st.button("🤖 Generar análisis con IA"):
            with st.spinner("Consultando al modelo..."):
                respuesta, error_ia = analizar_con_ia(
                    media_muestral=r["media_muestral"],
                    mu_0=p["mu_0"],
                    n=r["n"],
                    sigma=p["sigma"],
                    alpha=p["alpha"],
                    tipo_prueba=p["tipo_prueba"],
                    z_stat=r["z_stat"],
                    p_value=r["p_value"],
                    decision=r["decision"]
                )
            if error_ia:
                st.error(f"❌ {error_ia}")
            else:
                st.markdown(f'<div class="ai-box">🤖 <b>Análisis:</b><br><br>{respuesta}</div>', unsafe_allow_html=True)

elif datos is not None and len(datos) < 30:
    st.warning("⚠️ La prueba Z requiere al menos 30 observaciones (n ≥ 30). Aumenta el tamaño de la muestra.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#3a4a6a; font-size:0.75rem; font-family:Space Mono,monospace;'>"
    "Análisis Estadístico Interactivo · Prueba Z · Python + Streamlit</p>",
    unsafe_allow_html=True
)
