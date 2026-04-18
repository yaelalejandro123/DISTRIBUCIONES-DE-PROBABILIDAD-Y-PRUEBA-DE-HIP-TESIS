# 📊 Análisis Estadístico Interactivo — Prueba Z

Aplicación Streamlit para análisis estadístico con prueba Z e integración con Google Gemini AI.

## Estructura del proyecto

```
stats_app/
├── app.py            ← Interfaz principal (Streamlit)
├── data_loader.py    ← Carga y generación de datos
├── visualization.py  ← Histograma, KDE, Boxplot
├── stats_tests.py    ← Prueba Z + visualización
├── ai_module.py      ← Integración con Google Gemini
├── requirements.txt
└── README.md
```

## Instalación y ejecución

### 1. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación
```bash
streamlit run app.py
```

La app se abre en: http://localhost:8501

## Uso

1. **Datos**: Sube un CSV o genera datos sintéticos normales.
2. **Variable**: Selecciona la columna numérica a analizar.
3. **Visualización**: Revisa histograma, KDE, boxplot y estadísticos.
4. **Prueba Z**: Configura μ₀, σ, α y tipo de prueba en el sidebar.
5. **Resultados**: Ejecuta la prueba y obtén Z, p-value y decisión.
6. **IA**: Ingresa tu API Key de Gemini para análisis interpretativo.

## API Key de Google Gemini

1. Ve a: https://aistudio.google.com/app/apikey
2. Crea o copia tu API Key.
3. Pégala en el campo del sidebar (se oculta como contraseña).
