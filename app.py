import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.datasets import fashion_mnist, mnist

st.set_page_config(page_title="Clasificador Inteligente", page_icon="👗")

st.title("🧠 Sistema Inteligente de Clasificación")

FASHION_CLASS_NAMES = [
    "Camiseta/top",
    "Pantalón",
    "Jersey",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla deportiva",
    "Short",
    "Botín"
]

MNIST_CLASS_NAMES = [str(i) for i in range(10)]


@st.cache_resource(show_spinner="Cargando/Entrenando modelos...")
def get_models():
    fashion_path = Path("fashion_model.joblib")
    mnist_path = Path("mnist_model.joblib")
    
    try:
        import joblib
        if fashion_path.exists() and mnist_path.exists():
            return joblib.load(fashion_path), joblib.load(mnist_path)
    except:
        pass
    
    (x_train_f, y_train_f), (_, _) = fashion_mnist.load_data()
    X_train_f_flat = x_train_f.reshape(x_train_f.shape[0], -1)
    
    fashion_model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=50, multi_class="multinomial", solver="saga", n_jobs=-1, random_state=42))
    ])
    fashion_model.fit(X_train_f_flat, y_train_f)
    
    (x_train_m, y_train_m), (_, _) = mnist.load_data()
    X_train_m_flat = x_train_m.reshape(x_train_m.shape[0], -1)
    
    mnist_model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=50, multi_class="multinomial", solver="saga", n_jobs=-1, random_state=42))
    ])
    mnist_model.fit(X_train_m_flat, y_train_m)
    
    return fashion_model, mnist_model


def preprocess_img(image, invert=True):
    img = image.convert("L")
    img = img.resize((28, 28))
    arr = np.array(img) / 255.0
    if invert:
        arr = 1.0 - arr
    arr = arr.flatten().reshape(1, -1)
    return arr


option = st.radio("¿Qué quieres clasificar?", ["👗 Ropa (Fashion MNIST)", "🔢 Números (MNIST)"])
uploaded_file = st.file_uploader("Sube tu imagen", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", width=200)
    
    if st.button("Predecir"):
        with st.spinner("Cargando modelos y clasificando..."):
            try:
                fashion_model, mnist_model = get_models()
                
                if "Ropa" in option:
                    model = fashion_model
                    class_names = FASHION_CLASS_NAMES
                    img_arr = preprocess_img(image, invert=True)
                else:
                    model = mnist_model
                    class_names = MNIST_CLASS_NAMES
                    img_arr = preprocess_img(image, invert=False)
                
                pred_idx = int(model.predict(img_arr)[0])
                proba = model.predict_proba(img_arr)[0]
                
                st.success(f"**Resultado:** {class_names[pred_idx]}")
                st.info(f"**Confianza:** {proba[pred_idx] * 100:.2f}%")
                
                st.subheader("Top 3")
                top = sorted(enumerate(proba), key=lambda x: x[1], reverse=True)[:3]
                for i, (idx, p) in enumerate(top):
                    st.write(f"{i+1}. {class_names[idx]}: {p*100:.2f}%")
            except Exception as e:
                st.error(f"Error: {e}")
