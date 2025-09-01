import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

# Load model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model("app/model.keras")
        return model
    except:
        st.error("Model file not found. Please ensure 'app/model.keras' exists.")
        return None

model = load_trained_model()

# Disease names
disease_names = ['Acitinic Keratosis', 'Basal Cell Carcinoma', 'Melanoma', 
                 'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis']

def preprocess_image(image):
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to 224x224
    image_resized = cv2.resize(image_rgb, (224, 224))
    # Normalize
    image_normalized = image_resized / 255.0
    return image_normalized

def predict_image(image):
    if model is None:
        return None, None
    
    # Preprocess
    processed_img = preprocess_image(image)
    processed_img = np.expand_dims(processed_img, axis=0)
    
    # Predict
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    
    return disease_names[predicted_class], confidence

# Main app
st.title("🔬 Skin Cancer Detection")
st.markdown("Upload an image of a skin lesion to get a prediction")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, channels="BGR", use_column_width=True)
    
    # Predict
    if st.button("🔍 Predict", type="primary"):
        with st.spinner("Analyzing image..."):
            predicted_class, confidence = predict_image(image)
            
            if predicted_class:
                with col2:
                    st.subheader("Prediction Results")
                    
                    # Display result
                    st.success(f"**Predicted Class:** {predicted_class}")
                    st.info(f"**Confidence:** {confidence:.2%}")
                    
                    # Progress bar
                    st.progress(confidence)
                    
                    # Additional info
                    st.markdown("---")
                    st.markdown("**Note:** This is for educational purposes only. Please consult a healthcare professional for medical diagnosis.")
            else:
                st.error("Unable to make prediction. Please check if the model file is available.")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application uses deep learning to classify skin lesions into 6 categories:
    
    - **Acitinic Keratosis**
    - **Basal Cell Carcinoma**
    - **Melanoma**
    - **Nevus**
    - **Pigmented Benign Keratosis**
    - **Seborrheic Keratosis**
    
    **Model:** ResNet-based CNN
    **Input Size:** 224x224 pixels
    """)
    
    st.markdown("---")
    st.markdown("**Disclaimer:** This tool is for educational purposes only and should not replace professional medical diagnosis.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • [GitHub Repository](https://github.com/dudejahemani/Skin-Cancer-Detection)")
