"""
data_loader.py — Carga y generación de datos para el análisis estadístico.
"""

import pandas as pd
import numpy as np
from io import StringIO


def cargar_datos(archivo) -> tuple[pd.DataFrame | None, str | None]:
    """
    Carga un archivo CSV subido desde Streamlit.

    Args:
        archivo: Objeto de archivo de st.file_uploader.

    Returns:
        (DataFrame, None) si la carga fue exitosa.
        (None, mensaje_error) si ocurrió un error.
    """
    try:
        contenido = archivo.read().decode("utf-8")
        df = pd.read_csv(StringIO(contenido))

        if df.empty:
            return None, "El archivo CSV está vacío."

        if df.shape[1] < 1:
            return None, "El archivo no contiene columnas válidas."

        return df, None

    except UnicodeDecodeError:
        return None, "El archivo no tiene codificación UTF-8 válida."
    except pd.errors.ParserError as e:
        return None, f"Error al parsear el CSV: {e}"
    except Exception as e:
        return None, f"Error inesperado: {e}"


def generar_datos_sinteticos(n: int = 200, media: float = 0.0, std: float = 1.0) -> pd.DataFrame:
    """
    Genera un DataFrame con datos sintéticos de distribución normal.

    Args:
        n: Número de muestras.
        media: Media de la distribución.
        std: Desviación estándar de la distribución.

    Returns:
        DataFrame con columna 'valor'.
    """
    np.random.seed(42)
    datos = np.random.normal(loc=media, scale=std, size=n)
    return pd.DataFrame({"valor": datos})


def obtener_columnas_numericas(df: pd.DataFrame) -> list[str]:
    """
    Filtra y retorna solo las columnas numéricas de un DataFrame.

    Args:
        df: DataFrame de entrada.

    Returns:
        Lista de nombres de columnas numéricas.
    """
    columnas_num = df.select_dtypes(include=[np.number]).columns.tolist()
    return columnas_num
