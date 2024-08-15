import streamlit as st
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# Function to load the TFLite model
def load_model(tflite_model_path):
    # Load the TFLite model and allocate tensors
    interpreter = Interpreter(model_path=tflite_model_path)
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

# 
