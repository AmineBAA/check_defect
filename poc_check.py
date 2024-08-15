import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

# Function to load the ONNX model
def load_model(onnx_model_path):
    return ort.InferenceSession(onnx_model_path)

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image).astype(np.float32)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions using the ONNX model
def predict(session, image):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: image})
    return result[0]

# Streamlit app
st.title("Image Classification with ONNX Model")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the ONNX model
    session = load_model('my_model.onnx')
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make a prediction
    prediction = predict(session, preprocessed_image)
    
    # Interpret the prediction
    class_labels = ["Class 0", "Class 1"]  # Update with your actual class labels
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    st.write(f"Predicted Class: {class_labels[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
