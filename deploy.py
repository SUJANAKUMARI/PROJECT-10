import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load trained YOLO model
model = YOLO(r"C:\Users\Sujana\Desktop\PRO\best.pt")

def detect_text(image_path):
    image = cv2.imread(image_path)
    results = model(image_path)
    extracted_texts = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            
            if conf > 0.5:  # Confidence threshold
                cropped = image[y1:y2, x1:x2]  # Crop the detected region
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                text = pytesseract.image_to_string(gray, config='--psm 6')  # Extract text using OCR
                
                if text.strip():
                    extracted_texts.append(text.strip())
                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, text.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return extracted_texts, image

# Streamlit UI
st.title("YOLO OCR Text Extraction")
st.write("Upload an image to extract text.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    
    # Save uploaded file temporarily
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    
    # Perform text detection and OCR
    extracted_texts, processed_image = detect_text(temp_path)
    
    # Convert processed image to PIL format for Streamlit
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    processed_pil = Image.fromarray(processed_image)
    
    st.image(processed_pil, caption="Processed Image with Bounding Boxes", use_column_width=True)
    
    st.write("### Extracted Text:")
    if extracted_texts:
        st.write(", ".join(extracted_texts))
    else:
        st.write("No readable text detected in the image.")
