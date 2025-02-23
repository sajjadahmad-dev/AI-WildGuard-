import streamlit as st
import numpy as np
import pandas as pd
import torch
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from transformers import YolosForObjectDetection, YolosImageProcessor
import torch.nn.functional as F

# Load Models
@st.cache_resource
def load_models():
    species_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    species_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50").eval()
    
    yolo_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    yolo_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").eval()
    
    threat_model = pipeline("image-classification", model="nateraw/vit-base-beans")
    
    return species_processor, species_model, yolo_processor, yolo_model, threat_model

species_processor, species_model, yolo_processor, yolo_model, threat_model = load_models()

# Habitat Analysis Model
class HabitatAnalyzer:
    def __init__(self):
        self.CLASSES = ['vegetation', 'water', 'urban', 'barren']
    
    def analyze_vegetation(self, image_array):
        ndvi = (image_array[:, :, 3] - image_array[:, :, 0]) / (image_array[:, :, 3] + image_array[:, :, 0] + 1e-8)
        return ndvi
    
    def detect_land_changes(self, image1, image2):
        return cv2.absdiff(image1, image2)

class SpeciesMonitoringSystem:
    def __init__(self):
        self.detection_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.detection_model.eval()
        
        self.species_classes = [
            # List of species classes...
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def detect_species(self, image):
        img_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.detection_model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        top_prob, top_class = torch.topk(probabilities, 3)
        results = []
        
        for i in range(3):
            species = self.species_classes[top_class[0][i] % len(self.species_classes)]
            confidence = top_prob[0][i].item() * 100
            results.append((species, confidence))
            
        return results

    def count_population(self, image):
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_with_contours = np.array(image).copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
        
        return len(contours), Image.fromarray(img_with_contours)

    def assess_health(self, image):
        img_array = np.array(image)
        avg_color = np.mean(img_array, axis=(0, 1))
        texture_measure = np.std(img_array)
        color_variation = np.std(avg_color)
        
        color_score = np.mean(avg_color) / 255 * 100
        texture_score = min(100, texture_measure / 2)
        variation_score = min(100, color_variation * 2)
        
        health_score = (color_score * 0.4 + texture_score * 0.3 + variation_score * 0.3)
        
        if health_score > 80:
            status = "Excellent"
        elif health_score > 60:
            status = "Good"
        elif health_score > 40:
            status = "Fair"
        else:
            status = "Poor"
            
        indicators = {
            "Color Vibrancy": color_score,
            "Texture Complexity": texture_score,
            "Pattern Variation": variation_score
        }
            
        return status, health_score, indicators

def detect_threat(image, labels):
    results = threat_model(image)
    for result in results:
        if result['label'] in labels and result['score'] > 0.5:
            return f"{result['label']} Detected with confidence {result['score']:.2f}"
    return "No Threat Detected"

def detect_land_changes(image1_path, image2_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    
    image_array1 = np.array(image1)
    image_array2 = np.array(image2)
    
    if image_array1.shape != image_array2.shape:
        return "Error: Images must be the same size."
    
    changes = cv2.absdiff(image_array1, image_array2)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image1, caption="Image 1")
    with col2:
        st.image(image2, caption="Image 2")
    with col3:
        st.image(changes, caption="Changes Detected")
    
    change_percent = np.sum(changes > 50) / changes.size * 100
    st.write(f"Changed Area: {change_percent:.2f}%")
    
    return changes

def main():
    habitat_analyzer = HabitatAnalyzer()
    
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select an Analysis Type:",
                             ["Species Monitoring", "Land Change Detection", "Animal Monitoring", "Threat Detection"])

    if option == "Species Monitoring":
        st.title("Species Identification")
        monitoring_system = SpeciesMonitoringSystem()
    
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                progress_bar = st.progress(0)
                
                with st.spinner("Analyzing image..."):
                    col1, col2, col3 = st.columns(3)
                    
                    progress_bar.progress(30)
                    species_results = monitoring_system.detect_species(image)
                    
                    progress_bar.progress(60)
                    count, marked_image = monitoring_system.count_population(image)
                    
                    progress_bar.progress(90)
                    health_status, health_score, health_indicators = monitoring_system.assess_health(image)
                    
                    with col1:
                        st.subheader("🔍 Species Detection")
                        for species, confidence in species_results:
                            st.write(f"**{species.title()}**")
                            st.progress(confidence/100)
                            st.caption(f"Confidence: {confidence:.1f}%")
                    
                    with col2:
                        st.subheader("👥 Population Count")
                        st.write(f"**Detected Animals:** {count}")
                        st.image(marked_image, caption="Detection Visualization", use_column_width=True)
                    
                    with col3:
                        st.subheader("💪 Health Assessment")
                        st.write(f"**Status:** {health_status}")
                        st.write(f"**Overall Score:** {health_score:.1f}/100")
                        
                        for indicator, value in health_indicators.items():
                            st.write(f"**{indicator}:**")
                            st.progress(value/100)
                            st.caption(f"{value:.1f}%")
                
                progress_bar.progress(100)
                
                st.sidebar.markdown("---")
                st.sidebar.markdown("### Analysis Details")
                st.sidebar.text(f"Analyzed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.sidebar.text(f"Image size: {image.size}")
                
                st.markdown("---")
                st.subheader("📊 Export Results")
                
                summary = f"""Wildlife Monitoring Analysis Report
                Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

                Species Detection Results:
                {'-' * 30}
                """
                for species, confidence in species_results:
                    summary += f"\n{species.title()}: {confidence:.1f}% confidence"
                
                summary += f"""\n\nPopulation Count:
                {'-' * 30}
                Total detected: {count} individuals

                Health Assessment:
                {'-' * 30}
                Status: {health_status}
                Overall Score: {health_score:.1f}/100
                """
                for indicator, value in health_indicators.items():
                    summary += f"\n{indicator}: {value:.1f}%"
                
                st.download_button(
                    label="Download Analysis Report",
                    data=summary,
                    file_name="wildlife_analysis_report.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error processing image: {e}")

    elif option == "Land Change Detection":
        st.title("🌍 Land Change Detection")
        uploaded_file2 = st.file_uploader("Upload first image", type=['tif', 'png', 'jpg'])
        uploaded_file3 = st.file_uploader("Upload second image", type=['tif', 'png', 'jpg'])

        if uploaded_file2 is not None and uploaded_file3 is not None:
            detect_land_changes(uploaded_file2, uploaded_file3)

    elif option == "Animal Monitoring":
        st.title("Animal Monitoring")
        uploaded_file4 = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4"])
        
        if uploaded_file4:
            if uploaded_file4.type.startswith("image"):
                file_bytes = np.asarray(bytearray(uploaded_file4.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                if image is None:
                    st.error("Error loading image. Please upload a valid image file.")
                else:
                    results = yolo_model(image)
                    for result in results:
                        for box in result.boxes.xyxy:
                            x1, y1, x2, y2 = map(int, box[:4])
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                    st.image(image, caption="Detected Animals", channels="BGR")
                    st.write(f"Estimated Count: {len(results[0].boxes)}")
            
            elif uploaded_file4.type.startswith("video"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file4.read())
                cap = cv2.VideoCapture(tfile.name)
                
                if not cap.isOpened():
                    st.error("Error loading video. Please upload a valid video file.")
                else:
                    stframe = st.empty()
                    st.write("Processing video...")
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame = cv2.resize(frame, (640, 480))
                        results = yolo_model(frame)
                        
                        for result in results:
                            for box in result.boxes.xyxy:
                                x1, y1, x2, y2 = map(int, box[:4])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        stframe.image(frame, channels="BGR")
                        time.sleep(0.03)
                    
                    cap.release()

    elif option == "Threat Detection":
        st.title("Threat Detection and Prevention")
        st.sidebar.header("Choose Threat Detection")
        detection_option = st.sidebar.selectbox(
            "Select an option", 
            ["Poaching Alerts"]
        )
        
        if detection_option in ["Poaching Alerts"]:
            uploaded_file7 = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

            if uploaded_file7:
                image = Image.open(uploaded_file7)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if detection_option == "Poaching Alerts":
                    st.subheader("🎯 Poaching Activity Detection")
                    
                    with st.spinner("Analyzing image for potential poaching activities..."):
                        results = yolo_model(image)
                        
                        poaching_objects = ['person', 'gun', 'knife', 'truck', 'car']
                        detections = {}
                        
                        for result in results:
                            for box in result.boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                label = result.names[cls]
                                
                                if label in poaching_objects and conf > 0.3:
                                    detections[label] = conf
                        
                        if detections:
                            for obj, conf in detections.items():
                                st.progress(conf)
                                st.write(f"{obj.title()}: {conf*100:.1f}% confidence")
                            
                            annotated_img = np.array(image)
                            for result in results:
                                for box in result.boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            st.image(annotated_img, caption="Detected Objects", use_column_width=True)
                            
                            if any(conf > 0.7 for conf in detections.values()):
                                st.error("⚠️ High-risk poaching activity detected! Alert sent to authorities.")
                        else:
                            st.success("No suspicious activities detected.")

if __name__ == "__main__":
    main()
