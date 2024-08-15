import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to load the TFLite model
def load_model(tflite_model_path):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image).astype(np.float32)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions using the TFLite model
def predict(interpreter, image):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    interpreter.invoke()
    
    # Get the results from the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Streamlit app
st.title("Image Classification with TFLite Model")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the TFLite model
    interpreter = load_model('my_model.tflite')
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make a prediction
    prediction = predict(interpreter, preprocessed_image)
    
    # Interpret the prediction
    class_labels = ["Class 0", "Class 1"]  # Update with your actual class labels
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    st.write(f"Predicted Class: {class_labels[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")

# Run the Streamlit app with:
# streamlit run app.py
