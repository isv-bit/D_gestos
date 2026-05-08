```python
import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# =========================================
# CONFIGURACIÓN DE PÁGINA
# =========================================
st.set_page_config(
    page_title="Reconocimiento de Imágenes",
    page_icon="🤖",
    layout="wide"
)

# =========================================
# ESTILOS PERSONALIZADOS
# =========================================
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #0f172a, #1e1b4b, #111827);
    color: white;
}

h1 {
    color: #38bdf8 !important;
    text-align: center;
    font-size: 3rem !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: white !important;
}

.stMarkdown p {
    color: #dbeafe;
    font-size: 17px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #312e81);
    border-right: 2px solid #38bdf8;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: white !important;
}

div[data-testid="stCameraInput"] {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

.css-1v0mbdj img {
    border-radius: 20px;
    border: 3px solid #38bdf8;
    box-shadow: 0px 0px 20px rgba(56,189,248,0.4);
}

.stAlert {
    border-radius: 15px;
}

.result-box {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    padding: 20px;
    border-radius: 18px;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
    box-shadow: 0px 0px 20px rgba(59,130,246,0.5);
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# MODELO
# =========================================
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# =========================================
# TÍTULO
# =========================================
st.title("🤖 Reconocimiento de Imágenes")

st.markdown("""
<div style='text-align:center; color:#cbd5e1; font-size:18px; margin-bottom:25px;'>
Sistema de reconocimiento utilizando un modelo entrenado en 
<b>Teachable Machine</b> e inteligencia artificial.
</div>
""", unsafe_allow_html=True)

# =========================================
# INFORMACIÓN PYTHON
# =========================================
st.info(f"🐍 Versión de Python: {platform.python_version()}")

# =========================================
# IMAGEN PRINCIPAL
# =========================================
image = Image.open('OIG5.jpg')
st.image(image, width=400)

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:

    st.title("⚙️ Información")

    st.subheader("Modelo IA")

    st.markdown("""
    Esta aplicación utiliza un modelo entrenado en 
    <b>Teachable Machine</b> para reconocer imágenes 
    tomadas desde la cámara 📸.
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    🔹 Detecta posiciones automáticamente.<br><br>
    🔹 Procesamiento en tiempo real.<br><br>
    🔹 Compatible con cámara web.
    """, unsafe_allow_html=True)

# =========================================
# CÁMARA
# =========================================
img_file_buffer = st.camera_input("📸 Toma una Foto")

# =========================================
# PROCESAMIENTO
# =========================================
if img_file_buffer is not None:

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Leer imagen
    img = Image.open(img_file_buffer)

    # Redimensionar
    newsize = (224, 224)
    img = img.resize(newsize)

    # Convertir a array
    img_array = np.array(img)

    # Normalizar imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

    # Cargar datos
    data[0] = normalized_image_array

    # Predicción
    prediction = model.predict(data)

    st.subheader("📊 Resultado de la predicción")

    # RESULTADOS
    if prediction[0][0] > 0.5:
        st.markdown(f"""
        <div class="result-box">
        ⬅️ Izquierda <br>
        Probabilidad: {prediction[0][0]:.2f}
        </div>
        """, unsafe_allow_html=True)

    if prediction[0][1] > 0.5:
        st.markdown(f"""
        <div class="result-box">
        ⬆️ Arriba <br>
        Probabilidad: {prediction[0][1]:.2f}
        </div>
        """, unsafe_allow_html=True)

# =========================================
# FOOTER
# =========================================
st.markdown("---")

st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:15px'>
🧠 Inteligencia Artificial + Streamlit + Keras
</div>
""", unsafe_allow_html=True)
```
