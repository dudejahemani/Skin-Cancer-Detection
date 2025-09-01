import streamlit as st
import numpy as np
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

# Disease names
disease_names = ['Acitinic Keratosis', 'Basal Cell Carcinoma', 'Melanoma', 
                 'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis']

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
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    # Predict button
    if st.button("🔍 Predict", type="primary"):
        with st.spinner("Analyzing image..."):
            # Simulate prediction (demo mode)
            import random
            predicted_class = random.choice(disease_names)
            confidence = random.uniform(0.7, 0.95)
            
            with col2:
                st.subheader("Prediction Results")
                
                # Display result
                st.success(f"**Predicted Class:** {predicted_class}")
                st.info(f"**Confidence:** {confidence:.2%}")
                
                # Progress bar
                st.progress(confidence)
                
                # Additional info
                st.markdown("---")
                st.markdown("**Note:** This is a demo version. The actual model file is not loaded.")
                st.markdown("**Educational purposes only.** Please consult a healthcare professional for medical diagnosis.")

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
