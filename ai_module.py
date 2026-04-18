"""
ai_module.py — Integración con Google Gemini para análisis estadístico.

Solo se envía un resumen estadístico, NO los datos crudos.
"""

import requests
import json


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


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
    """
    Construye el prompt que se enviará a Gemini.
    Solo incluye el resumen estadístico, no datos individuales.

    Returns:
        String con el prompt completo.
    """
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

Por favor responde lo siguiente de forma clara, estructurada y en español:
1. ¿Se rechaza H₀? Explica la razón basándote en el estadístico Z y el p-value.
2. ¿Son razonables los supuestos de esta prueba (n ≥ 30, σ conocida)?
3. Interpreta el resultado en lenguaje accesible para un estudiante de estadística.
4. ¿Qué implicaciones prácticas tiene esta conclusión?

Sé preciso, pedagógico y conciso. Máximo 250 palabras."""


def analizar_con_gemini(
    api_key: str,
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
    Envía el resumen estadístico a Gemini y retorna su análisis.

    Args:
        api_key: Clave de API de Google Gemini.
        (Resto de parámetros): Resultados de la prueba Z.

    Returns:
        (respuesta_texto, None) si fue exitoso.
        (None, mensaje_error) si ocurrió un error.
    """
    if not api_key or not api_key.strip():
        return None, "API Key vacía o inválida."

    prompt = _construir_prompt(
        media_muestral=media_muestral,
        mu_0=mu_0,
        n=n,
        sigma=sigma,
        alpha=alpha,
        tipo_prueba=tipo_prueba,
        z_stat=z_stat,
        p_value=p_value,
        decision=decision
    )

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 600,
        }
    }

    headers = {"Content-Type": "application/json"}
    url_con_key = f"{GEMINI_URL}?key={api_key.strip()}"

    try:
        response = requests.post(url_con_key, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            # Extraer texto de la respuesta
            try:
                texto = data["candidates"][0]["content"]["parts"][0]["text"]
                return texto.strip(), None
            except (KeyError, IndexError) as e:
                return None, f"Respuesta inesperada de Gemini: {e}. Respuesta: {json.dumps(data)[:300]}"

        elif response.status_code == 400:
            return None, "Error 400: Solicitud inválida. Verifica el formato del prompt."
        elif response.status_code == 403:
            return None, "Error 403: API Key inválida o sin permisos para Gemini."
        elif response.status_code == 429:
            return None, "Error 429: Límite de solicitudes alcanzado. Intenta en unos minutos."
        else:
            return None, f"Error HTTP {response.status_code}: {response.text[:300]}"

    except requests.exceptions.ConnectionError:
        return None, "Error de conexión. Verifica tu conexión a internet."
    except requests.exceptions.Timeout:
        return None, "Tiempo de espera agotado. La API tardó demasiado en responder."
    except requests.exceptions.RequestException as e:
        return None, f"Error al conectar con Gemini: {e}"
    except Exception as e:
        return None, f"Error inesperado: {e}"
