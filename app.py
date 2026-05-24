import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

from config.settings import CLASS_NAMES, MODEL_PATH
from src.predictor import load_class_names_for_model
from src.preprocessing import preprocess_user_image

st.set_page_config(
    page_title="Clasificador Inteligente",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        .stApp {
            background-color: #1a365d !important;
        }
        .css-18e3th9 {
            background-color: #1a365d !important;
        }
        .css-1d391kg {
            background-color: #1a365d !important;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
        .stFileUploader label, .stProgress, .stSuccess, .stInfo, .stError, .stWarning {
            color: white !important;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white !important;
            font-size: 18px;
            padding: 10px 24px;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #45a049;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Sistema Inteligente de Clasificación")

tab1, tab2 = st.tabs(["👗 Ropa (Fashion MNIST)", "🔢 Números (MNIST)"])

with tab1:
    st.header("Clasificación de Ropa")
    st.write("Carga una imagen de una prenda y pulsa Predecir.")
    
    uploaded_file_fashion = st.file_uploader(
        "Selecciona una imagen de ropa",
        type=["png", "jpg", "jpeg", "bmp"],
        key="fashion"
    )
    
    if uploaded_file_fashion is not None:
        image = Image.open(uploaded_file_fashion)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagen cargada")
            st.image(image, width=300)
            
            predict_btn_fashion = st.button("Predecir", key="predict_fashion")
        
        with col2:
            st.subheader("Resultado")
            
            if not MODEL_PATH.exists():
                st.error("No hay modelos entrenados. Por favor ejecuta `python main.py train` primero.")
            elif predict_btn_fashion:
                with st.spinner("Clasificando la imagen..."):
                    try:
                        model = load_model(MODEL_PATH)
                        class_names = load_class_names_for_model(MODEL_PATH)
                        
                        temp_path = Path("temp_image.png")
                        image.save(temp_path)
                        processed_image = preprocess_user_image(temp_path)
                        temp_path.unlink(missing_ok=True)
                        
                        probabilities = model.predict(processed_image, verbose=0)[0]
                        predicted_index = int(np.argmax(probabilities))
                        
                        ranked_predictions = sorted(
                            [
                                {
                                    "indice": class_index,
                                    "clase": class_names[class_index],
                                    "probabilidad": float(probability),
                                }
                                for class_index, probability in enumerate(probabilities)
                            ],
                            key=lambda item: item["probabilidad"],
                            reverse=True,
                        )
                        
                        prediction = {
                            "clase_predicha": class_names[predicted_index],
                            "probabilidad": float(probabilities[predicted_index]),
                            "top_3": ranked_predictions[:3],
                        }
                        
                        st.success(f"**Clase predicha:** {prediction['clase_predicha']}")
                        st.info(f"**Confianza:** {prediction['probabilidad'] * 100:.2f}%")
                        
                        st.subheader("Top 3 probabilidades")
                        for i, item in enumerate(prediction["top_3"]):
                            st.progress(item["probabilidad"], text=f"{i+1}. {item['clase']} - {item['probabilidad'] * 100:.2f}%")
                    except Exception as e:
                        st.error(f"Error al clasificar: {e}")

with tab2:
    st.header("Clasificación de Números")
    st.write("Carga una imagen de un número (0-9) y pulsa Predecir.")
    
    uploaded_file_mnist = st.file_uploader(
        "Selecciona una imagen de un número",
        type=["png", "jpg", "jpeg", "bmp"],
        key="mnist"
    )
    
    if uploaded_file_mnist is not None:
        image = Image.open(uploaded_file_mnist)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagen cargada")
            st.image(image, width=300)
            
            predict_btn_mnist = st.button("Predecir", key="predict_mnist")
        
        with col2:
            st.subheader("Resultado")
            
            if predict_btn_mnist:
                with st.spinner("Clasificando el número..."):
                    try:
                        mnist_model_path = Path("models/trained/mnist_cnn.keras")
                        
                        if not mnist_model_path.exists():
                            st.info("Entrenando modelo MNIST por primera vez...")
                            (x_train, y_train), (x_test, y_test) = mnist.load_data()
                            x_train = x_train.astype("float32") / 255.0
                            x_test = x_test.astype("float32") / 255.0
                            x_train = np.expand_dims(x_train, -1)
                            x_test = np.expand_dims(x_test, -1)
                            
                            from tensorflow.keras import Sequential
                            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
                            
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
                            
                            model.compile(
                                optimizer="adam",
                                loss="sparse_categorical_crossentropy",
                                metrics=["accuracy"]
                            )
                            
                            model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
                            
                            mnist_model_path.parent.mkdir(parents=True, exist_ok=True)
                            model.save(mnist_model_path)
                        else:
                            model = load_model(mnist_model_path)
                        
                        img = image.convert("L")
                        img = img.resize((28, 28))
                        img_array = np.array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = np.expand_dims(img_array, axis=-1)
                        
                        probabilities = model.predict(img_array, verbose=0)[0]
                        predicted_index = int(np.argmax(probabilities))
                        mnist_class_names = [str(i) for i in range(10)]
                        
                        ranked_predictions = sorted(
                            [
                                {
                                    "indice": class_index,
                                    "clase": mnist_class_names[class_index],
                                    "probabilidad": float(probability),
                                }
                                for class_index, probability in enumerate(probabilities)
                            ],
                            key=lambda item: item["probabilidad"],
                            reverse=True,
                        )
                        
                        prediction = {
                            "clase_predicha": mnist_class_names[predicted_index],
                            "probabilidad": float(probabilities[predicted_index]),
                            "top_3": ranked_predictions[:3],
                        }
                        
                        st.success(f"**Número predicho:** {prediction['clase_predicha']}")
                        st.info(f"**Confianza:** {prediction['probabilidad'] * 100:.2f}%")
                        
                        st.subheader("Top 3 probabilidades")
                        for i, item in enumerate(prediction["top_3"]):
                            st.progress(item["probabilidad"], text=f"{i+1}. {item['clase']} - {item['probabilidad'] * 100:.2f}%")
                    except Exception as e:
                        st.error(f"Error al clasificar: {e}")
