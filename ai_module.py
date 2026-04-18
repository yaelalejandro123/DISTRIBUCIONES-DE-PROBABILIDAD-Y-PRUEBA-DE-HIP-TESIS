"""
ai_module.py — Integración con OpenRouter para análisis estadístico.

La API key se carga desde el archivo .env (OPENROUTER_API_KEY).
Los usuarios NO necesitan ingresar ninguna key — está configurada globalmente.
Solo se envía un resumen estadístico, NUNCA datos crudos.
"""

import os
import requests


def _cargar_env():
    """Carga variables del archivo .env sin dependencias externas."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

_cargar_env()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "mistralai/mistral-7b-instruct"


def _construir_prompt(
    media_muestral: float,
    mu_0: float,
    n: int,
    sigma: float,
    alpha: float,
    tipo_prueba: str,
    z_stat: float,
    p_value: float,
    decision: str
) -> str:
    """Construye el prompt con solo el resumen estadístico."""
    return f"""Se realizó una prueba Z de una muestra con los siguientes parámetros estadísticos:

- Media muestral (x̄) = {media_muestral:.4f}
- Media hipotética bajo H₀ (μ₀) = {mu_0}
- Tamaño de muestra (n) = {n}
- Desviación estándar poblacional conocida (σ) = {sigma}
- Nivel de significancia (α) = {alpha}
- Tipo de prueba = {tipo_prueba}
- Estadístico Z calculado = {z_stat:.4f}
- p-value = {p_value:.4f}
- Decisión estadística: {decision}

Por favor responde en español, de forma clara y estructurada:
1. ¿Se rechaza H₀? Explica la razón basándote en el estadístico Z y el p-value.
2. ¿Son razonables los supuestos de esta prueba (n ≥ 30, σ conocida)?
3. Interpreta el resultado en lenguaje accesible para un estudiante de estadística.
4. ¿Qué implicaciones prácticas tiene esta conclusión?

Sé preciso, pedagógico y conciso. Máximo 250 palabras."""


def analizar_con_ia(
    media_muestral: float,
    mu_0: float,
    n: int,
    sigma: float,
    alpha: float,
    tipo_prueba: str,
    z_stat: float,
    p_value: float,
    decision: str
) -> tuple[str | None, str | None]:
    """
    Envía el resumen estadístico a OpenRouter y retorna el análisis.
    No requiere API key del usuario — usa la key del servidor (.env).

    Returns:
        (respuesta_texto, None) si fue exitoso.
        (None, mensaje_error) si ocurrió un error.
    """
    if not OPENROUTER_API_KEY:
        return None, "API key no configurada. Agrega OPENROUTER_API_KEY en el archivo .env"

    prompt = _construir_prompt(
        media_muestral=media_muestral, mu_0=mu_0, n=n, sigma=sigma,
        alpha=alpha, tipo_prueba=tipo_prueba, z_stat=z_stat,
        p_value=p_value, decision=decision
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://estadistica-interactiva.app",
        "X-Title": "Análisis Estadístico Interactivo"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un profesor de estadística universitario. "
                    "Explicas pruebas de hipótesis de forma clara, precisa y pedagógica en español."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 600,
        "temperature": 0.4,
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            try:
                texto = data["choices"][0]["message"]["content"]
                return texto.strip(), None
            except (KeyError, IndexError) as e:
                return None, f"Respuesta inesperada del modelo: {e}"

        elif response.status_code == 401:
            return None, "Error 401: API key inválida o expirada. Actualiza el archivo .env"
        elif response.status_code == 402:
            return None, "Error 402: Créditos insuficientes en OpenRouter."
        elif response.status_code == 429:
            return None, "Error 429: Límite de solicitudes alcanzado. Intenta en unos momentos."
        else:
            return None, f"Error HTTP {response.status_code}: {response.text[:300]}"

    except requests.exceptions.ConnectionError:
        return None, "Error de conexión. Verifica tu conexión a internet."
    except requests.exceptions.Timeout:
        return None, "Tiempo de espera agotado (30s). Intenta de nuevo."
    except Exception as e:
        return None, f"Error inesperado: {e}"
