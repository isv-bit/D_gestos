import streamlit as st
import cv2
import numpy as np
#from PIL import Image
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
    background: linear-gradient(135deg, #0f172a, #1e293b, #111827);
    color: white;
}

/* TITULO */
h1 {
    text-align: center;
    color: #38bdf8 !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
}

/* SUBTITULOS */
h2, h3 {
    color: #ffffff !important;
}

/* TEXO */
.stMarkdown p {
    color: #dbeafe;
    font-size: 16px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1e3a8a);
    border-right: 2px solid #38bdf8;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: white !important;
}

/* CAMARA */
div[data-testid="stCameraInput"] {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* IMAGEN */
.css-1v0mbdj img {
    border-radius: 20px;
    border: 3px solid #38bdf8;
    box-shadow: 0px 0px 25px rgba(56,189,248,0.4);
}

/* RESULTADOS */
.resultado {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    color: white;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
    box-shadow: 0px 0px 20px rgba(59,130,246,0.5);
}

/* ALERTAS */
.stAlert {
    border-radius: 15px;
}

/* ESPACIADO */
.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# VERSIÓN PYTHON
# =========================================
st.info(f"🐍 Versión de Python: {platform.python_version()}")

# =========================================
# MODELO
# =========================================
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# =========================================
# TITULO
# =========================================
st.title("🤖 Reconocimiento de Imágenes")

st.markdown("""
<div style='text-align:center; color:#cbd5e1; font-size:18px; margin-bottom:30px;'>
Aplicación de inteligencia artificial entrenada con 
<b>Teachable Machine</b> para reconocer imágenes en tiempo real.
</div>
""", unsafe_allow_html=True)

# =========================================
# IMAGEN PRINCIPAL
# =========================================
image = Image.open('gestos.jpg')
st.image(image, width=400)

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:

    st.title("⚙️ Información")

    st.subheader("Modelo IA")

    st.markdown("""
    Usa un modelo entrenado en 
    <b>Teachable Machine</b> para identificar imágenes 
    tomadas desde tu cámara 📸.
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    🔹 Reconocimiento automático.<br><br>
    🔹 Procesamiento en tiempo real.<br><br>
    🔹 Compatible con cámara web.
    """, unsafe_allow_html=True)

# =========================================
# CAMARA
# =========================================
img_file_buffer = st.camera_input("📸 Toma una Foto")

# =========================================
# PREDICCIÓN
# =========================================
if img_file_buffer is not None:

    # To read image file buffer with OpenCV:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)

    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Run the inference
    prediction = model.predict(data)

    print(prediction)

    st.subheader("📊 Resultado del reconocimiento")

    if prediction[0][0] > 0.5:
        st.markdown(f"""
        <div class="resultado">
        ⬅️ Izquierda <br>
        Probabilidad: {prediction[0][0]:.2f}
        </div>
        """, unsafe_allow_html=True)

    if prediction[0][1] > 0.5:
        st.markdown(f"""
        <div class="resultado">
        ⬆️ Arriba <br>
        Probabilidad: {prediction[0][1]:.2f}
        </div>
        """, unsafe_allow_html=True)

    #if prediction[0][2]>0.5:
    # st.header('Derecha, con Probabilidad: '+str( prediction[0][2]))

# =========================================
# FOOTER
# =========================================
st.markdown("---")

st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:15px'>
🧠 Inteligencia Artificial + Streamlit + Keras
</div>
""", unsafe_allow_html=True)
