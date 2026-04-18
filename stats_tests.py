"""
stats_tests.py — Prueba Z para una muestra con varianza poblacional conocida.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
import streamlit as st


# ── Colores ──
BG_COLOR   = "#0d0f14"
GRID_COLOR = "#1e2230"
TEXT_COLOR = "#c8ccd8"
COLOR_CURVE   = "#7eb8f7"
COLOR_REJECT  = "#f77e7e"
COLOR_ACCEPT  = "#3a7bd5"
COLOR_Z_LINE  = "#f7c97e"


def ejecutar_prueba_z(
    datos,
    mu_0: float,
    sigma: float,
    alpha: float,
    tipo_prueba: str
) -> dict:
    """
    Realiza la prueba Z de una muestra con varianza poblacional conocida.

    Args:
        datos: pd.Series o array con los datos muestrales.
        mu_0: Media hipotética bajo H0.
        sigma: Desviación estándar poblacional conocida.
        alpha: Nivel de significancia.
        tipo_prueba: "Bilateral", "Cola izquierda" o "Cola derecha".

    Returns:
        Diccionario con todos los resultados de la prueba.
    """
    datos_arr = np.array(datos, dtype=float)
    n = len(datos_arr)
    media_muestral = np.mean(datos_arr)
    error_estandar = sigma / np.sqrt(n)

    # Estadístico Z
    z_stat = (media_muestral - mu_0) / error_estandar

    # p-value y región crítica según tipo de prueba
    if tipo_prueba == "Bilateral":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        z_critico = stats.norm.ppf(1 - alpha / 2)
        rechazar_h0 = abs(z_stat) > z_critico
        region_critica = f"|Z| > {z_critico:.4f}  →  Z < {-z_critico:.4f} ó Z > {z_critico:.4f}"

    elif tipo_prueba == "Cola izquierda":
        p_value = stats.norm.cdf(z_stat)
        z_critico = stats.norm.ppf(alpha)
        rechazar_h0 = z_stat < z_critico
        region_critica = f"Z < {z_critico:.4f}"

    else:  # Cola derecha
        p_value = 1 - stats.norm.cdf(z_stat)
        z_critico = stats.norm.ppf(1 - alpha)
        rechazar_h0 = z_stat > z_critico
        region_critica = f"Z > {z_critico:.4f}"

    # Decisión en texto
    if rechazar_h0:
        decision = (
            f"Se RECHAZA H₀ (Z = {z_stat:.4f}, p = {p_value:.4f} < α = {alpha}). "
            "Hay evidencia estadística suficiente en contra de la hipótesis nula."
        )
    else:
        decision = (
            f"No se rechaza H₀ (Z = {z_stat:.4f}, p = {p_value:.4f} ≥ α = {alpha}). "
            "No hay evidencia estadística suficiente para rechazar la hipótesis nula."
        )

    # Interpretación contextual
    interpretacion = _generar_interpretacion(
        rechazar_h0, tipo_prueba, media_muestral, mu_0, alpha
    )

    return {
        "media_muestral": media_muestral,
        "n": n,
        "error_estandar": error_estandar,
        "z_stat": z_stat,
        "p_value": p_value,
        "z_critico": z_critico if tipo_prueba != "Bilateral" else z_critico,
        "rechazar_h0": rechazar_h0,
        "decision": decision,
        "region_critica": region_critica,
        "interpretacion": interpretacion,
        "tipo_prueba": tipo_prueba,
        "alpha": alpha,
    }


def _generar_interpretacion(
    rechazar: bool,
    tipo: str,
    x_bar: float,
    mu_0: float,
    alpha: float
) -> str:
    """Genera un texto de interpretación en español."""
    hipotesis = {
        "Bilateral":      f"H₀: μ = {mu_0}  vs  H₁: μ ≠ {mu_0}",
        "Cola izquierda": f"H₀: μ ≥ {mu_0}  vs  H₁: μ < {mu_0}",
        "Cola derecha":   f"H₀: μ ≤ {mu_0}  vs  H₁: μ > {mu_0}",
    }
    base = hipotesis.get(tipo, "")
    diferencia = x_bar - mu_0

    if rechazar:
        return (
            f"{base}. "
            f"Con α = {alpha}, la media muestral (x̄ = {x_bar:.4f}) difiere "
            f"{'significativamente de' if tipo == 'Bilateral' else 'en la dirección esperada de'} "
            f"μ₀ = {mu_0} (diferencia = {diferencia:+.4f})."
        )
    else:
        return (
            f"{base}. "
            f"Con α = {alpha}, la diferencia observada entre x̄ = {x_bar:.4f} "
            f"y μ₀ = {mu_0} (diferencia = {diferencia:+.4f}) "
            "no es estadísticamente significativa."
        )


def visualizar_prueba_z(
    z_stat: float,
    alpha: float,
    tipo_prueba: str,
    p_value: float
):
    """
    Grafica la curva normal estándar con la región de rechazo y el estadístico Z.

    Args:
        z_stat: Valor del estadístico Z calculado.
        alpha: Nivel de significancia.
        tipo_prueba: "Bilateral", "Cola izquierda" o "Cola derecha".
        p_value: p-value de la prueba.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor("#13161e")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.6)

    # Rango del eje X
    x_min, x_max = -4.2, 4.2
    x = np.linspace(x_min, x_max, 500)
    y = stats.norm.pdf(x)

    # Curva normal
    ax.plot(x, y, color=COLOR_CURVE, linewidth=2.5, zorder=3, label="N(0,1)")
    ax.fill_between(x, y, alpha=0.07, color=COLOR_CURVE)

    # ── Región de rechazo ──
    if tipo_prueba == "Bilateral":
        z_c = stats.norm.ppf(1 - alpha / 2)
        # Cola izquierda
        x_ri = np.linspace(x_min, -z_c, 200)
        ax.fill_between(x_ri, stats.norm.pdf(x_ri), color=COLOR_REJECT, alpha=0.4, label=f"Región de rechazo (α={alpha})")
        # Cola derecha
        x_rd = np.linspace(z_c, x_max, 200)
        ax.fill_between(x_rd, stats.norm.pdf(x_rd), color=COLOR_REJECT, alpha=0.4)
        # Líneas críticas
        ax.axvline(-z_c, color=COLOR_REJECT, linewidth=1.5, linestyle="--", alpha=0.8)
        ax.axvline(z_c, color=COLOR_REJECT, linewidth=1.5, linestyle="--", alpha=0.8,
                   label=f"Z crítico = ±{z_c:.3f}")

    elif tipo_prueba == "Cola izquierda":
        z_c = stats.norm.ppf(alpha)
        x_ri = np.linspace(x_min, z_c, 200)
        ax.fill_between(x_ri, stats.norm.pdf(x_ri), color=COLOR_REJECT, alpha=0.4, label=f"Región de rechazo (α={alpha})")
        ax.axvline(z_c, color=COLOR_REJECT, linewidth=1.5, linestyle="--", alpha=0.8,
                   label=f"Z crítico = {z_c:.3f}")

    else:  # Cola derecha
        z_c = stats.norm.ppf(1 - alpha)
        x_rd = np.linspace(z_c, x_max, 200)
        ax.fill_between(x_rd, stats.norm.pdf(x_rd), color=COLOR_REJECT, alpha=0.4, label=f"Región de rechazo (α={alpha})")
        ax.axvline(z_c, color=COLOR_REJECT, linewidth=1.5, linestyle="--", alpha=0.8,
                   label=f"Z crítico = {z_c:.3f}")

    # ── Estadístico Z calculado ──
    z_clipped = np.clip(z_stat, x_min + 0.1, x_max - 0.1)
    ax.axvline(z_clipped, color=COLOR_Z_LINE, linewidth=2.2, linestyle="-", zorder=5,
               label=f"Z calc. = {z_stat:.4f}")

    # Punto sobre la curva
    y_z = stats.norm.pdf(z_clipped)
    ax.scatter([z_clipped], [y_z], color=COLOR_Z_LINE, s=60, zorder=6)

    # Anotación
    offset_x = 0.15 if z_clipped < 3 else -0.6
    ax.annotate(
        f"Z = {z_stat:.3f}\np = {p_value:.4f}",
        xy=(z_clipped, y_z),
        xytext=(z_clipped + offset_x, y_z + 0.04),
        fontsize=8.5,
        color=COLOR_Z_LINE,
        arrowprops=dict(arrowstyle="->", color=COLOR_Z_LINE, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e2a3d", edgecolor=COLOR_Z_LINE, alpha=0.8)
    )

    # ── Etiquetas y leyenda ──
    ax.set_xlabel("Z", fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel("Densidad", fontsize=10, color=TEXT_COLOR)
    ax.set_title(f"Prueba Z — {tipo_prueba}  (α = {alpha})", fontsize=12, color=TEXT_COLOR, pad=10)
    ax.set_xlim(x_min, x_max)

    legend = ax.legend(
        fontsize=8.5, framealpha=0.2,
        facecolor="#13161e", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
