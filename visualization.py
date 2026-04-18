"""
visualization.py — Funciones de visualización estadística.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import streamlit as st


# ── Paleta de colores coherente con el tema dark ──
COLOR_HIST    = "#3a7bd5"
COLOR_KDE     = "#7eb8f7"
COLOR_BOXPLOT = "#3a7bd5"
COLOR_OUTLIER = "#f77e7e"
COLOR_MEAN    = "#f7c97e"
COLOR_NORMAL_CURVE = "#7ef7c8"
COLOR_REJECT  = "#f77e7e"
COLOR_ACCEPT  = "#3a7bd5"
BG_COLOR      = "#0d0f14"
GRID_COLOR    = "#1e2230"
TEXT_COLOR    = "#c8ccd8"


def _aplicar_estilo_dark(fig, ax_list):
    """Aplica el tema oscuro a una figura de matplotlib."""
    fig.patch.set_facecolor(BG_COLOR)
    for ax in ax_list:
        ax.set_facecolor("#13161e")
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)


def mostrar_distribucion(datos: pd.Series, nombre_columna: str = "Variable"):
    """
    Muestra histograma con KDE, boxplot y estadísticos descriptivos.

    Args:
        datos: Serie de pandas con los datos numéricos.
        nombre_columna: Nombre de la variable para los títulos.
    """
    datos_arr = datos.dropna().values

    # ── Estadísticos descriptivos ──
    media       = np.mean(datos_arr)
    mediana     = np.median(datos_arr)
    std_muestral = np.std(datos_arr, ddof=1)
    sesgo       = stats.skew(datos_arr)
    curtosis    = stats.kurtosis(datos_arr)
    n           = len(datos_arr)

    # Test de normalidad (Shapiro si n<=5000, sino D'Agostino)
    if n <= 5000:
        stat_norm, p_norm = stats.shapiro(datos_arr[:5000])
        test_name = "Shapiro-Wilk"
    else:
        stat_norm, p_norm = stats.normaltest(datos_arr)
        test_name = "D'Agostino"

    # Detección de outliers (IQR)
    q1, q3 = np.percentile(datos_arr, [25, 75])
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    n_outliers = np.sum((datos_arr < limite_inf) | (datos_arr > limite_sup))

    # ── Métricas en columnas de Streamlit ──
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Media (x̄)", f"{media:.4f}")
    with col2:
        st.metric("Mediana", f"{mediana:.4f}")
    with col3:
        st.metric("Desv. Est. (s)", f"{std_muestral:.4f}")
    with col4:
        sesgo_label = "positivo ▶" if sesgo > 0.5 else ("negativo ◀" if sesgo < -0.5 else "simétrico ≈")
        st.metric("Sesgo", f"{sesgo:.4f}", delta=sesgo_label, delta_color="off")
    with col5:
        st.metric("Outliers (IQR)", f"{n_outliers}")

    # ── Figura: Histograma + KDE + Boxplot ──
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1,
        figsize=(9, 6),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08}
    )
    _aplicar_estilo_dark(fig, [ax_hist, ax_box])

    # Histograma
    n_bins = min(50, max(10, int(np.sqrt(n))))
    ax_hist.hist(
        datos_arr, bins=n_bins,
        color=COLOR_HIST, alpha=0.6, edgecolor="#1e2a3d",
        density=True, label="Histograma"
    )

    # KDE
    kde = stats.gaussian_kde(datos_arr)
    x_range = np.linspace(datos_arr.min() - std_muestral, datos_arr.max() + std_muestral, 300)
    ax_hist.plot(x_range, kde(x_range), color=COLOR_KDE, linewidth=2.5, label="KDE")

    # Curva normal teórica
    y_normal = stats.norm.pdf(x_range, media, std_muestral)
    ax_hist.plot(x_range, y_normal, color=COLOR_NORMAL_CURVE, linewidth=1.5,
                 linestyle="--", alpha=0.8, label="Normal teórica")

    # Líneas de media y mediana
    ax_hist.axvline(media, color=COLOR_MEAN, linewidth=1.8, linestyle="-", label=f"Media = {media:.3f}")
    ax_hist.axvline(mediana, color="#c87ef7", linewidth=1.5, linestyle=":", label=f"Mediana = {mediana:.3f}")

    ax_hist.set_ylabel("Densidad", fontsize=10)
    ax_hist.set_title(f"Distribución de '{nombre_columna}'  (n={n})", fontsize=12, pad=10)
    ax_hist.tick_params(labelbottom=False)

    legend = ax_hist.legend(
        fontsize=8, framealpha=0.2,
        facecolor="#13161e", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR
    )

    # Boxplot
    bp = ax_box.boxplot(
        datos_arr, vert=False, patch_artist=True,
        flierprops=dict(marker="o", markerfacecolor=COLOR_OUTLIER, markersize=4, alpha=0.7),
        medianprops=dict(color=COLOR_MEAN, linewidth=2),
        whiskerprops=dict(color=TEXT_COLOR, linewidth=1.2),
        capprops=dict(color=TEXT_COLOR, linewidth=1.2),
        boxprops=dict(facecolor=COLOR_BOXPLOT, alpha=0.4, linewidth=1.2, edgecolor=COLOR_KDE)
    )
    ax_box.set_xlabel(nombre_columna, fontsize=10)
    ax_box.set_yticks([])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Info de normalidad ──
    normalidad_msg = (
        f"✅ Los datos **no rechazan** la normalidad según {test_name} "
        f"(p={p_norm:.4f} > α=0.05)."
        if p_norm > 0.05
        else
        f"⚠️ Los datos **no siguen** una distribución normal según {test_name} "
        f"(p={p_norm:.4f} ≤ α=0.05)."
    )
    sesgo_msg = (
        "La distribución tiene **sesgo positivo** (cola a la derecha)." if sesgo > 0.5
        else "La distribución tiene **sesgo negativo** (cola a la izquierda)." if sesgo < -0.5
        else "La distribución es aproximadamente **simétrica**."
    )

    with st.expander("📋 Diagnóstico de la distribución"):
        st.markdown(normalidad_msg)
        st.markdown(sesgo_msg)
        st.markdown(f"Curtosis = **{curtosis:.4f}** "
                    f"({'leptocúrtica (colas pesadas)' if curtosis > 0 else 'platicúrtica (colas ligeras)'}).")
        if n_outliers > 0:
            st.markdown(f"Se detectaron **{n_outliers} outliers** por el método IQR "
                        f"(límites: [{limite_inf:.3f}, {limite_sup:.3f}]).")
        else:
            st.markdown("No se detectaron outliers por el método IQR.")
