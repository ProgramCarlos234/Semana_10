import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import fashion_mnist, mnist

st.set_page_config(
    page_title="Clasificador Inteligente",
    page_icon="👗",
    layout="centered"
)

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


@st.cache_resource
def get_fashion_model():
    model_path = Path("fashion_model.keras")
    if model_path.exists():
        return load_model(model_path)
    
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(10, activation="softmax")
    ])
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
    model.save(model_path)
    return model


@st.cache_resource
def get_mnist_model():
    model_path = Path("mnist_model.keras")
    if model_path.exists():
        return load_model(model_path)
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(10, activation="softmax")
    ])
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
    model.save(model_path)
    return model


def preprocess_img(image, invert=True):
    img = image.convert("L")
    img = img.resize((28, 28))
    arr = np.array(img) / 255.0
    if invert:
        arr = 1.0 - arr
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr


option = st.radio("¿Qué quieres clasificar?", ["👗 Ropa (Fashion MNIST)", "🔢 Números (MNIST)"])

uploaded_file = st.file_uploader("Sube tu imagen", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", width=200)
    
    if st.button("Predecir"):
        with st.spinner("Clasificando..."):
            try:
                if "Ropa" in option:
                    model = get_fashion_model()
                    class_names = FASHION_CLASS_NAMES
                    img_arr = preprocess_img(image, invert=True)
                else:
                    model = get_mnist_model()
                    class_names = MNIST_CLASS_NAMES
                    img_arr = preprocess_img(image, invert=False)
                
                probs = model.predict(img_arr, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                
                st.success(f"**Resultado:** {class_names[pred_idx]}")
                st.info(f"**Confianza:** {probs[pred_idx] * 100:.2f}%")
                
                st.subheader("Top 3")
                top = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
                for i, (idx, p) in enumerate(top):
                    st.write(f"{i+1}. {class_names[idx]}: {p*100:.2f}%")
            except Exception as e:
                st.error(f"Error: {e}")
