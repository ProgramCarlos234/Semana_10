import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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


@st.cache_resource
def get_models():
    fashion_path = Path("fashion_model.keras")
    mnist_path = Path("mnist_model.keras")
    
    if fashion_path.exists() and mnist_path.exists():
        return load_model(fashion_path), load_model(mnist_path)
    
    (x_train_f, y_train_f), (_, _) = fashion_mnist.load_data()
    x_train_f = x_train_f.astype("float32") / 255.0
    x_train_f = np.expand_dims(x_train_f, -1)
    
    fashion_model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(10, activation="softmax")
    ])
    fashion_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    fashion_model.fit(x_train_f, y_train_f, epochs=3, batch_size=64, verbose=0)
    fashion_model.save(fashion_path)
    
    (x_train_m, y_train_m), (_, _) = mnist.load_data()
    x_train_m = x_train_m.astype("float32") / 255.0
    x_train_m = np.expand_dims(x_train_m, -1)
    
    mnist_model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(10, activation="softmax")
    ])
    mnist_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    mnist_model.fit(x_train_m, y_train_m, epochs=3, batch_size=64, verbose=0)
    mnist_model.save(mnist_path)
    
    return fashion_model, mnist_model


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
                    img_arr = preprocess_img(image, invert=True)
                
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
