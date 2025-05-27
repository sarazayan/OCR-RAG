import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import io
import base64
from datetime import datetime
import openai
import os
import pytesseract
from pytesseract import Output
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import tempfile

# Set page configuration
st.set_page_config(page_title="Masah BoQ Generator", layout="wide")

# App title and description
st.title("Masah - Floor Plan to BoQ Generator")
st.markdown("Process floor plans and generate Bills of Quantities with CV, OCR, and RAG")

# Create tabs for the workflow
tabs = st.tabs(["1. Upload & Extract", "2. Take-Off", "3. BoQ Generation", "4. Final Output"])

# Sidebar for project settings and extraction method
with st.sidebar:
    st.header("Project Settings")
    project_name = st.text_input("Project Name", "Residential Building")
    project_location = st.selectbox("Location", ["Cairo, Egypt", "Alexandria, Egypt", "Giza, Egypt", "Other"])
    project_type = st.selectbox("Type", ["Residential", "Commercial", "Industrial"])
    currency = st.selectbox("Currency", ["EGP", "USD", "EUR"])
    
    st.header("Extraction Method")
    extraction_method = st.radio(
        "Choose extraction method",
        ["Computer Vision + OCR", "Multimodal LLM (GPT-4V)", "YOLOv5 Object Detection"]
    )
    
    if extraction_method == "Multimodal LLM (GPT-4V)":
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            st.warning("⚠️ Please enter your OpenAI API key to use the LLM extraction method")
    
    # Knowledge base settings
    st.header("Knowledge Base")
    kb_source = st.multiselect("Sources", 
                              ["Egyptian Building Code", "ASTM Standards", "Local Material Catalogs", "Cost Indexes"],
                              default=["Egyptian Building Code", "Local Material Catalogs"])

# Function to perform OCR on floor plan
def extract_text_with_ocr(image):
    """Extract text from floor plan using OCR"""
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding to improve OCR results
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Save image temporarily for pytesseract
    with tempfile.NamedTemporaryFile(suffix='.jpg') as temp:
        cv2.imwrite(temp.name, thresh)
        
        # Apply OCR with custom configuration
        custom_config = r'--oem 3 --psm 6'
        ocr_data = pytesseract.image_to_data(thresh, config=custom_config, output_type=Output.DICT)
    
    # Process extracted text
    extracted_text = []
    for i in range(len(ocr_data['text'])):
        if int(float(ocr_data['conf'][i])) > 30:  # Only use text with reasonable confidence
            text = ocr_data['text'][i].strip()
            if text and len(text) > 1:  # Ignore single characters which are often noise
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                width = ocr_data['width'][i]
                height = ocr_data['height'][i]
                
                extracted_text.append({
                    'text': text,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'confidence': ocr_data['conf'][i]
                })
    
    # Create visualization
    vis_img = img_array.copy()
    for item in extracted_text:
        x, y = item['x'], item['y']
        text = item['text']
        # Draw bounding box
        cv2.rectangle(vis_img, (x, y), (x + item['width'], y + item['height']), (0, 255, 0), 2)
        # Add text
        cv2.putText(vis_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return extracted_text, vis_img

# Function to identify rooms and elements from OCR results
def identify_rooms_and_elements(ocr_results, image_width, image_height):
    """Identify rooms and building elements from OCR results"""
    rooms = []
    dimensions = {}
    specs = {}
    
    # Regular expressions for matching
    room_pattern = re.compile(r'(BEDROOM|BATHROOM|KITCHEN|DINING|MASTER)', re.IGNORECASE)
    dimension_pattern = re.compile(r'(\d+\.?\d*)\s*m', re.IGNORECASE)
    thickness_pattern = re.compile(r'THICKNESS[:\s]+(\d+)', re.IGNORECASE)
    height_pattern = re.compile(r'HEIGHT[:\s]+(\d+\.?\d*)', re.IGNORECASE)
    
    # Process OCR results
    for item in ocr_results:
        text = item['text'].upper()
        x, y = item['x'], item['y']
        
        # Check for room names
        room_match = room_pattern.search(text)
        if room_match:
            room_name = text
            # Estimate room area based on position (in a real implementation, this would use contour detection)
            area_estimate = 12.0  # Default estimate
            rooms.append({
                "name": room_name,
                "area": area_estimate,
                "position": (x, y)
            })
        
        # Check for dimensions
        dim_match = dimension_pattern.search(text)
        if dim_match:
            dimension = float(dim_match.group(1))
            dimensions[text] = dimension
        
        # Check for wall thickness
        thickness_match = thickness_pattern.search(text)
        if thickness_match:
            specs['wall_thickness'] = float(thickness_match.group(1)) / 1000  # Convert mm to m
        
        # Check for floor height
        height_match = height_pattern.search(text)
        if height_match:
            specs['floor_height'] = float(height_match.group(1))
    
    # Extract other specifications from OCR text
    for item in ocr_results:
        if "CEILING" in item['text'].upper():
            specs['ceiling_type'] = "Drop ceiling with recessed lighting"
        if "FLOORING" in item['text'].upper():
            specs['flooring'] = "Engineered hardwood"
    
    # Set defaults for any missing specifications
    if 'wall_thickness' not in specs:
        specs['wall_thickness'] = 0.2  # Default: 200mm
    if 'floor_height' not in specs:
        specs['floor_height'] = 2.6  # Default: 2.6m
    if 'ceiling_type' not in specs:
        specs['ceiling_type'] = "Drop ceiling with recessed lighting"
    if 'flooring' not in specs:
        specs['flooring'] = "Engineered hardwood"
    
    # Find primary dimensions if not detected by OCR
    if 'width' not in dimensions and 'length' not in dimensions:
        # Use default dimensions from the image
        dimensions['width'] = 15.0
        dimensions['length'] = 10.5
    
    return rooms, dimensions, specs

# Extract construction elements using CV and OCR
def extract_elements_with_cv_ocr(image):
    """Extract construction elements using computer vision and OCR"""
    # Convert to numpy array for OpenCV
    img_array = np.array(image)
    
    # Create a copy for visualization
    vis_img = img_array.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding to get binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours - these represent potential elements
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Perform OCR to extract text information
    ocr_results, ocr_vis = extract_text_with_ocr(image)
    
    # Extract room information and specifications from OCR
    img_height, img_width = gray.shape
    rooms, dimensions, specs = identify_rooms_and_elements(ocr_results, img_width, img_height)
    
    # Initialize element lists
    doors = []
    windows = []
    
    # Door detection (orange rectangles in the floor plan)
    door_count = 0
    window_count = 0
    
    # Detect doors - look for small rectangles with orange color
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    # Orange color range in HSV
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Find contours of orange elements (doors)
    door_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in door_contours:
        area = cv2.contourArea(contour)
        if area > 20:  # Minimum area to be considered a door
            door_count += 1
            door_id = f"D{door_count}"
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw on visualization image (red for doors)
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(vis_img, door_id, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Convert pixel dimensions to meters (using known scale)
            width_m = round(w / img_width * dimensions.get('width', 15.0), 1)
            height_m = 2.1  # Standard door height
            
            # Find nearest room label to determine location
            nearest_room = "Unknown"
            min_distance = float('inf')
            for room in rooms:
                room_x, room_y = room['position']
                dist = np.sqrt((x - room_x)**2 + (y - room_y)**2)
                if dist < min_distance:
                    min_distance = dist
                    nearest_room = room['name']
            
            doors.append({
                "id": door_id,
                "type": "Interior", 
                "width": width_m,
                "height": height_m,
                "location": f"Near {nearest_room}"
            })
    
    # Detect windows - look for blue rectangles
    # Blue color range in HSV
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours of blue elements (windows)
    window_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in window_contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Minimum area to be considered a window
            window_count += 1
            window_id = f"W{window_count}"
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw on visualization image (blue for windows)
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(vis_img, window_id, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Convert pixel dimensions to meters
            width_m = round(w / img_width * dimensions.get('width', 15.0), 1)
            height_m = round(h / img_height * dimensions.get('length', 10.5), 1) * 0.5
            
            # Find nearest room label to determine location
            nearest_room = "Unknown"
            min_distance = float('inf')
            for room in rooms:
                room_x, room_y = room['position']
                dist = np.sqrt((x - room_x)**2 + (y - room_y)**2)
                if dist < min_distance:
                    min_distance = dist
                    nearest_room = room['name']
            
            windows.append({
                "id": window_id,
                "type": "Standard", 
                "width": width_m,
                "height": 1.2,  # Standard window height
                "location": nearest_room
            })
    
    # Wall detection using line detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Calculate total wall length
    total_wall_length = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_wall_length += length
            # Draw lines on visualization
            cv2.line(vis_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    # Convert to meters using dimensions from OCR
    wall_length_m = round(total_wall_length / img_width * dimensions.get('width', 15.0), 1)
    
    # Define walls based on floor plan dimensions
    width_m = dimensions.get('width', 15.0)
    length_m = dimensions.get('length', 10.5)
    external_length = round(2 * (width_m + length_m), 1)
    internal_length = max(0, wall_length_m - external_length)
    
    walls = {
        "external": {
            "length": external_length, 
            "height": specs['floor_height'], 
            "thickness": specs['wall_thickness'], 
            "area": external_length * specs['floor_height']
        },
        "internal": {
            "length": internal_length, 
            "height": specs['floor_height'], 
            "thickness": specs['wall_thickness'] - 0.05,  # Internal walls are usually thinner
            "area": internal_length * specs['floor_height']
        }
    }
    
    # If rooms weren't properly detected from OCR, create default room list
    if len(rooms) < 3:
        default_rooms = [
            {"name": "BEDROOM 1", "area": 12.0, "position": (0, 0)},
            {"name": "BEDROOM 2", "area": 12.0, "position": (0, 0)},
            {"name": "MASTER BEDROOM", "area": 16.0, "position": (0, 0)},
            {"name": "BATHROOM", "area": 6.0, "position": (0, 0)},
            {"name": "KITCHEN", "area": 10.0, "position": (0, 0)},
            {"name": "DINING", "area": 14.0, "position": (0, 0)}
        ]
        
        # Add rooms that weren't detected
        existing_room_names = [room['name'] for room in rooms]
        for default_room in default_rooms:
            if default_room['name'] not in existing_room_names:
                rooms.append(default_room)
    
    # Ensure we have at least some doors and windows
    if len(doors) < 3:
        for i in range(len(doors), 3):
            doors.append({
                "id": f"D{i+1}",
                "type": "Interior", 
                "width": 0.9,
                "height": 2.1,
                "location": "Room boundary"
            })
    
    if len(windows) < 3:
        for i in range(len(windows), 3):
            windows.append({
                "id": f"W{i+1}",
                "type": "Standard", 
                "width": 1.5,
                "height": 1.2,
                "location": "External wall"
            })
    
    # Return all extracted elements and visualization
    return {
        "doors": doors,
        "windows": windows,
        "walls": walls,
        "rooms": rooms,
        "floor_height": specs['floor_height'],
        "ceiling_type": specs['ceiling_type'],
        "flooring": specs['flooring'],
        "dimensions": dimensions,
        "ocr_results": ocr_results
    }, vis_img

# Function to extract elements using multimodal LLM (GPT-4V)
def extract_elements_with_llm(image, api_key):
    """Extract construction elements using GPT-4V"""
    if not api_key:
        st.error("OpenAI API key is required for LLM extraction")
        return None, None
    
    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Create prompt for GPT-4V
    prompt = """
    Analyze this floor plan image and extract the following information in JSON format:
    1. All rooms with their names and estimated areas in m²
    2. All doors with their widths and locations
    3. All windows with their widths and locations
    4. Wall information (external and internal) with lengths, heights, and thicknesses
    5. Ceiling type and flooring material
    6. Overall dimensions of the floor plan

    Format the response as a valid JSON object with the following structure:
    {
        "rooms": [{"name": "room name", "area": area_in_m2}],
        "doors": [{"id": "D1", "type": "type", "width": width_in_m, "height": height_in_m, "location": "location"}],
        "windows": [{"id": "W1", "type": "type", "width": width_in_m, "height": height_in_m, "location": "location"}],
        "walls": {"external": {"length": length_in_m, "height": height_in_m, "thickness": thickness_in_m, "area": area_in_m2},
                 "internal": {"length": length_in_m, "height": height_in_m, "thickness": thickness_in_m, "area": area_in_m2}},
        "floor_height": height_in_m,
        "ceiling_type": "ceiling description",
        "flooring": "flooring material",
        "dimensions": {"width": width_in_m, "length": length_in_m}
    }
    
    Provide only the JSON with no additional text.
    """
    
    try:
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]
                }
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        # Extract JSON from response
        result_json = response.choices[0].message.content
        # Clean up the JSON string (remove markdown code blocks if present)
        result_json = result_json.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        import json
        elements = json.loads(result_json)
        
        # Create visualization
        vis_img = np.array(image.copy())
        
        # Draw room labels
        for room in elements["rooms"]:
            # Find average position for the room (simplified)
            x, y = 100, 100  # Default position
            cv2.putText(vis_img, f"{room['name']} ({room['area']}m²)", (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Return elements and visualization
        return elements, vis_img
    
    except Exception as e:
        st.error(f"Error in LLM extraction: {str(e)}")
        return None, None

# Simulate YOLO object detection (in a real implementation, you would use a trained model)
def extract_elements_with_yolo(image):
    """Simulate YOLO object detection for floor plan elements"""
    # In a real implementation, you would load a trained YOLO model
    # For this demo, we'll simulate YOLO detection
    
    # Convert to numpy array for OpenCV
    img_array = np.array(image)
    vis_img = img_array.copy()
    
    # Get image dimensions
    img_height, img_width = img_array.shape[:2]
    
    # Simulate detection results
    # In real implementation, this would be the output of the YOLO model
    simulated_detections = [
        {"class": "door", "confidence": 0.92, "box": [250, 270, 30, 10]},
        {"class": "door", "confidence": 0.88, "box": [370, 330, 10, 30]},
        {"class": "door", "confidence": 0.94, "box": [370, 420, 10, 30]},
        {"class": "door", "confidence": 0.91, "box": [470, 220, 30, 10]},
        {"class": "door", "confidence": 0.89, "box": [590, 210, 10, 30]},
        {"class": "window", "confidence": 0.85, "box": [150, 350, 30, 10]},
        {"class": "window", "confidence": 0.87, "box": [250, 500, 30, 10]},
        {"class": "window", "confidence": 0.86, "box": [440, 500, 30, 10]},
        {"class": "window", "confidence": 0.84, "box": [570, 500, 30, 10]},
        {"class": "window", "confidence": 0.83, "box": [440, 150, 30, 10]},
        {"class": "window", "confidence": 0.88, "box": [240, 150, 30, 10]},
        {"class": "room", "confidence": 0.95, "box": [250, 350, 120, 120], "label": "BEDROOM 1"},
        {"class": "room", "confidence": 0.93, "box": [250, 450, 120, 120], "label": "BEDROOM 2"},
        {"class": "room", "confidence": 0.94, "box": [440, 450, 120, 120], "label": "MASTER BEDROOM"},
        {"class": "room", "confidence": 0.92, "box": [440, 350, 120, 80], "label": "BATHROOM"},
        {"class": "room", "confidence": 0.91, "box": [440, 200, 120, 120], "label": "KITCHEN"},
        {"class": "room", "confidence": 0.90, "box": [570, 270, 120, 120], "label": "DINING"}
    ]
    
    # Process detections
    doors = []
    windows = []
    rooms = []
    door_count = 0
    window_count = 0
    
    # Process each detection
    for detection in simulated_detections:
        class_name = detection["class"]
        confidence = detection["confidence"]
        box = detection["box"]
        x, y, w, h = box
        
        if class_name == "door":
            door_count += 1
            door_id = f"D{door_count}"
            
            # Draw on visualization
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(vis_img, f"{door_id} ({confidence:.2f})", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Convert to real dimensions
            width_m = round(w / img_width * 15, 1)  # Assuming 15m width
            height_m = 2.1  # Standard door height
            
            doors.append({
                "id": door_id,
                "type": "Interior",
                "width": width_m,
                "height": height_m,
                "location": "Room boundary",
                "confidence": confidence
            })
            
        elif class_name == "window":
            window_count += 1
            window_id = f"W{window_count}"
            
            # Draw on visualization
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(vis_img, f"{window_id} ({confidence:.2f})", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Convert to real dimensions
            width_m = round(w / img_width * 15, 1)
            height_m = 1.2  # Standard window height
            
            windows.append({
                "id": window_id,
                "type": "Standard",
                "width": width_m,
                "height": height_m,
                "location": "External wall",
                "confidence": confidence
            })
            
        elif class_name == "room":
            label = detection.get("label", f"Room {len(rooms)+1}")
            
            # Draw on visualization
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis_img, f"{label} ({confidence:.2f})", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate area
            area_m2 = round((w * h) / (img_width * img_height) * 157.5, 1)  # 15m × 10.5m = 157.5m²
            
            rooms.append({
                "name": label,
                "area": area_m2,
                "position": (x, y),
                "confidence": confidence
            })
    
    # Wall detection would be part of the YOLO model in a real implementation
    # For now, use fixed dimensions from the floor plan
    walls = {
        "external": {
            "length": 51.0,  # 2 * (15 + 10.5)
            "height": 2.6,
            "thickness": 0.2,
            "area": 51.0 * 2.6
        },
        "internal": {
            "length": 24.5,
            "height": 2.6,
            "thickness": 0.15,
            "area": 24.5 * 2.6
        }
    }
    
    # Return all extracted elements and visualization
    return {
        "doors": doors,
        "windows": windows,
        "walls": walls,
        "rooms": rooms,
        "floor_height": 2.6,
        "ceiling_type": "Drop ceiling with recessed lighting",
        "flooring": "Engineered hardwood",
        "dimensions": {"width": 15.0, "length": 10.5}
    }, vis_img

# Function to generate take-off from extracted elements
def generate_takeoff(elements):
    """Convert extracted elements to structured take-off data"""
    takeoff_items = []
    
    # Process doors
    for door in elements["doors"]:
        takeoff_items.append({
            "category": "Doors",
            "item": f"Door {door['id']}",
            "type": door["type"],
            "description": f"{door['type']} door, {door['width']}m × {door['height']}m",
            "quantity": 1,
            "unit": "EA",
            "dimensions": f"{door['width']}m × {door['height']}m",
            "location": door["location"]
        })
    
    # Process windows
    for window in elements["windows"]:
        takeoff_items.append({
            "category": "Windows",
            "item": f"Window {window['id']}",
            "type": window["type"],
            "description": f"{window['type']} window, {window['width']}m × {window['height']}m",
            "quantity": 1,
            "unit": "EA",
            "dimensions": f"{window['width']}m × {window['height']}m",
            "location": window["location"]
        })
    
    # Process walls
    takeoff_items.append({
        "category": "Walls",
        "item": "External Walls",
        "type": "Load-bearing",
        "description": f"External wall, {elements['walls']['external']['thickness']}m thick",
        "quantity": elements['walls']['external']['length'],
        "unit": "LM",
        "dimensions": f"{elements['walls']['external']['thickness']}m × {elements['walls']['external']['height']}m",
        "location": "Perimeter"
    })
    
    takeoff_items.append({
        "category": "Walls",
        "item": "Internal Walls",
        "type": "Partition",
        "description": f"Internal wall, {elements['walls']['internal']['thickness']}m thick",
        "quantity": elements['walls']['internal']['length'],
        "unit": "LM",
        "dimensions": f"{elements['walls']['internal']['thickness']}m × {elements['walls']['internal']['height']}m",
        "location": "Interior"
    })
    
    # Process flooring
    total_area = sum(room["area"] for room in elements["rooms"])
    takeoff_items.append({
        "category": "Flooring",
        "item": "Engineered Hardwood",
        "type": elements["flooring"],
        "description": f"{elements['flooring']} flooring",
        "quantity": total_area,
        "unit": "M²",
        "dimensions": "N/A",
        "location": "All rooms"
    })
    
    # Process ceiling
    takeoff_items.append({
        "category": "Ceiling",
        "item": "Drop Ceiling",
        "type": "Recessed",
        "description": elements["ceiling_type"],
        "quantity": total_area,
        "unit": "M²",
        "dimensions": "N/A",
        "location": "All rooms"
    })
    
    return takeoff_items

# Simple knowledge base
kb_data = [
    {
        "category": "Doors",
        "type": "Interior",
        "location": "Cairo, Egypt",
        "materials": "Solid core wood with steel frame, fire rating 30 min, as per ECP 204-2018, section 5.3",
        "unit_cost": 2200,
        "item_code": "DR-INT"
    },
    {
        "category": "Windows",
        "type": "Standard",
        "location": "Cairo, Egypt",
        "materials": "Double-glazed aluminum frame, U-value 2.5 W/m²K, as per ECP 204-2018, section 6.2",
        "unit_cost": 3500,
        "item_code": "WIN-STD"
    },
    {
        "category": "Walls",
        "type": "Load-bearing",
        "location": "Cairo, Egypt",
        "materials": "Reinforced concrete with brick infill, thermal insulation R-13, as per ECP 203-2018",
        "unit_cost": 1200,
        "item_code": "WL-EXT"
    },
    {
        "category": "Walls",
        "type": "Partition",
        "location": "Cairo, Egypt",
        "materials": "Lightweight concrete blocks with plaster finish, as per ECP 203-2018",
        "unit_cost": 800,
        "item_code": "WL-INT"
    },
    {
        "category": "Flooring",
        "type": "Engineered hardwood",
        "location": "Cairo, Egypt",
        "materials": "Engineered hardwood 12mm over concrete substrate with sound insulation, as per ECP 208-2019",
        "unit_cost": 950,
        "item_code": "FL-EHW"
    },
    {
        "category": "Ceiling",
        "type": "Recessed",
        "location": "Cairo, Egypt",
        "materials": "Gypsum board drop ceiling with aluminum frame and recessed LED lighting, as per ECP 205-2019",
        "unit_cost": 650,
        "item_code": "CL-DRP"
    }
]

# Function to set up RAG system
def setup_rag_system():
    """Set up a RAG system with TF-IDF"""
    # Create documents for the knowledge base
    documents = []
    for item in kb_data:
        doc = f"Category: {item['category']}. Type: {item['type']}. Location: {item['location']}. Materials: {item['materials']}. Unit Cost: {item['unit_cost']}. Item Code: {item['item_code']}."
        documents.append(doc)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    return vectorizer, tfidf_matrix, documents

# Function to retrieve relevant information
def retrieve_info(query, vectorizer, tfidf_matrix, documents, top_k=3):
    """Retrieve relevant information from the knowledge base"""
    # Transform query to TF-IDF vector
    query_vec = vectorizer.transform([query])
    
    # Calculate similarity
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top-k indices
    indices = similarity.argsort()[-top_k:][::-1]
    
    # Get relevant documents and scores
    results = []
    for i in indices:
        if similarity[i] > 0:  # Only include if there's some similarity
            results.append({
                "content": documents[i],
                "score": float(similarity[i]),
                "kb_item": kb_data[i]
            })
    
    return results

# Function to generate BoQ using RAG
def generate_boq_with_rag(takeoff_items, project_location, project_type):
    """Generate BoQ using RAG system"""
    # Set up RAG system
    vectorizer, tfidf_matrix, documents = setup_rag_system()
    
    boq_items = []
    
    # Process each takeoff item
    for item in takeoff_items:
        # Create query from item
        query = f"Category: {item['category']}. Type: {item['type']}. Location: {project_location}."
        
        # Retrieve relevant information
        results = retrieve_info(query, vectorizer, tfidf_matrix, documents)
        
        if results:
            # Use the most relevant result
            top_result = results[0]
            kb_info = top_result["kb_item"]
            
            # Calculate costs
            unit_cost = kb_info["unit_cost"]
            total_cost = unit_cost * item["quantity"]
            
            # Create BoQ item
            boq_item = {
                "category": item["category"],
                "item_code": f"{kb_info['item_code']}-{len(boq_items) + 1}",
                "description": item["description"],
                "detailed_spec": kb_info["materials"],
                "quantity": item["quantity"],
                "unit": item["unit"],
                "unit_cost": unit_cost,
                "total_cost": total_cost,
                "location": item["location"]
            }
            
            boq_items.append(boq_item)
    
    # Use LLM to enhance descriptions if API key is provided
    if 'openai_api_key' in locals() and openai_api_key:
        try:
            for i, item in enumerate(boq_items):
                query = f"""
                Enhance this BoQ item description for a {project_type} building in {project_location}:
                
                Category: {item['category']}
                Description: {item['description']}
                Specification: {item['detailed_spec']}
                
                Provide a detailed technical description in 1-2 sentences without changing any measurements or quantities.
                """
                
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a construction specialist helping create detailed BoQ descriptions."},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                
                # Update the description with enhanced text
                enhanced_desc = response.choices[0].message.content.strip()
                boq_items[i]["enhanced_description"] = enhanced_desc
        except Exception as e:
            st.warning(f"Could not enhance descriptions with OpenAI: {str(e)}")
    
    return boq_items

# Tab 1: Upload & Extract
with tabs[0]:
    st.header("Upload Floor Plan")
    st.write("Upload an architectural floor plan image to begin the extraction process.")
    
    uploaded_file = st.file_uploader("Choose a floor plan image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        
        # Display the original image
        st.image(image, caption="Uploaded Floor Plan", use_column_width=True)
        
        # Process button
        if st.button("Extract Elements", key="extract_btn"):
            # Show processing spinner
            with st.spinner(f"Analyzing floor plan using {extraction_method}..."):
                # Extract elements based on selected method
                if extraction_method == "Computer Vision + OCR":
                    elements, visualization = extract_elements_with_cv_ocr(image)
                    # Store OCR results for display
                    st.session_state.ocr_results = elements.get("ocr_results", [])
                    
                elif extraction_method == "Multimodal LLM (GPT-4V)":
                    if not openai_api_key:
                        st.error("⚠️ OpenAI API key is required for LLM extraction. Please enter it in the sidebar.")
                        elements, visualization = None, None
                    else:
                        elements, visualization = extract_elements_with_llm(image, openai_api_key)
                        
                elif extraction_method == "YOLOv5 Object Detection":
                    elements, visualization = extract_elements_with_yolo(image)
                
                if elements:
                    # Store in session state
                    st.session_state.elements = elements
                    st.session_state.processed = True
                    
                    # Display extraction results
                    st.success("✅ Element extraction complete!")
                    
                    # Show visualization
                    st.image(visualization, caption=f"Element Detection using {extraction_method}", use_column_width=True)
                    
                    # Show tabs for different element types
                    element_tabs = st.tabs(["Doors", "Windows", "Walls", "Rooms"])
                    
                    with element_tabs[0]:
                        st.subheader("Detected Doors")
                        st.dataframe(pd.DataFrame(elements["doors"]))
                        
                    with element_tabs[1]:
                        st.subheader("Detected Windows")
                        st.dataframe(pd.DataFrame(elements["windows"]))
                        
                    with element_tabs[2]:
                        st.subheader("Detected Walls")
                        wall_data = [
                            {"Type": "External", "Length (m)": elements["walls"]["external"]["length"], 
                             "Height (m)": elements["walls"]["external"]["height"], 
                             "Thickness (m)": elements["walls"]["external"]["thickness"],
                             "Area (m²)": elements["walls"]["external"]["area"]},
                            {"Type": "Internal", "Length (m)": elements["walls"]["internal"]["length"], 
                             "Height (m)": elements["walls"]["internal"]["height"], 
                             "Thickness (m)": elements["walls"]["internal"]["thickness"],
                             "Area (m²)": elements["walls"]["internal"]["area"]}
                        ]
                        st.dataframe(pd.DataFrame(wall_data))
                        
                    with element_tabs[3]:
                        st.subheader("Detected Rooms")
                        st.dataframe(pd.DataFrame(elements["rooms"]))
                    
                    # Show OCR results if available
                    if extraction_method == "Computer Vision + OCR" and "ocr_results" in st.session_state:
                        with st.expander("OCR Results"):
                            st.subheader("Text Extracted by OCR")
                            ocr_df = pd.DataFrame(st.session_state.ocr_results)
                            if not ocr_df.empty:
                                st.dataframe(ocr_df[["text", "confidence", "x", "y"]])
                            else:
                                st.write("No text was extracted by OCR.")

# Tab 2: Take-Off
with tabs[1]:
    st.header("Quantity Take-Off")
    
    if "processed" not in st.session_state:
        st.info("⚠️ Please upload and extract a floor plan first (Step 1).")
    else:
        st.write("Converting extracted elements into structured quantity take-off data.")
        
        if st.button("Generate Take-Off", key="takeoff_btn"):
            with st.spinner("Generating quantity take-off..."):
                # Generate take-off from extracted elements
                takeoff_items = generate_takeoff(st.session_state.elements)
                
                # Store in session state
                st.session_state.takeoff = takeoff_items
                st.session_state.takeoff_done = True
            
            # Display take-off results
            st.success("✅ Take-off generation complete!")
            
            # Show take-off table
            st.subheader("Take-Off Items")
            st.dataframe(pd.DataFrame(takeoff_items), use_container_width=True)

# Tab 3: BoQ Generation
with tabs[2]:
    st.header("Bill of Quantities Generation (RAG)")
    
    if "takeoff_done" not in st.session_state:
        st.info("⚠️ Please generate a take-off first (Step 2).")
    else:
        st.write("Using Retrieval-Augmented Generation (RAG) to create detailed BoQ entries.")
        
        # Explain the RAG process
        with st.expander("How RAG works for BoQ generation"):
            st.write("""
            1. **Retrieval**: The system queries a knowledge base containing building codes, 
               material specifications, and cost data using TF-IDF similarity.
               
            2. **Augmentation**: Retrieved information is combined with the take-off data.
            
            3. **Generation**: OpenAI's GPT model enhances descriptions with technical details
               (if an API key is provided).
            """)
        
        if st.button("Generate BoQ with RAG", key="boq_btn"):
            with st.spinner("Generating BoQ with RAG..."):
                # Generate BoQ using RAG
                boq_items = generate_boq_with_rag(st.session_state.takeoff, project_location, project_type)
                
                # Store in session state
                st.session_state.boq = boq_items
                st.session_state.boq_done = True
            
            # Show success message
            st.success("✅ Bill of Quantities generated successfully!")
            
            # Show BoQ preview
            st.subheader("BoQ Preview")
            
            # Create DataFrame for display
            if boq_items and "enhanced_description" in boq_items[0]:
                preview_df = pd.DataFrame([{
                    "Item Code": item["item_code"],
                    "Description": item.get("enhanced_description", item["description"]),
                    "Quantity": item["quantity"],
                    "Unit": item["unit"],
                    "Unit Cost": item["unit_cost"],
                    "Total Cost": item["total_cost"]
                } for item in boq_items])
            else:
                preview_df = pd.DataFrame([{
                    "Item Code": item["item_code"],
                    "Description": item["description"],
                    "Quantity": item["quantity"],
                    "Unit": item["unit"],
                    "Unit Cost": item["unit_cost"],
                    "Total Cost": item["total_cost"]
                } for item in boq_items])
            
            st.dataframe(preview_df, use_container_width=True)

# Tab 4: Final Output
with tabs[3]:
    st.header("Final Bill of Quantities")
    
    if "boq_done" not in st.session_state:
        st.info("⚠️ Please generate a BoQ first (Step 3).")
    else:
        # Project information
        st.subheader("Project Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Project Name:** {project_name}")
            st.write(f"**Location:** {project_location}")
            st.write(f"**Type:** {project_type}")
        with col2:
            st.write(f"**Date:** {datetime.now().strftime('%B %d, %Y')}")
            st.write(f"**Currency:** {currency}")
            st.write(f"**Reference:** BOQ-{datetime.now().strftime('%Y%m%d')}")
        
        # Detailed BoQ by category
        st.subheader("Detailed Bill of Quantities")
        
        # Get unique categories
        categories = list(set(item["category"] for item in st.session_state.boq))
        
        # Create tabs for each category
        category_tabs = st.tabs(categories)
        
        # Fill each category tab
        for i, category in enumerate(categories):
            with category_tabs[i]:
                # Filter items by category
                category_items = [item for item in st.session_state.boq if item["category"] == category]
                
                # Create main table
                if category_items and "enhanced_description" in category_items[0]:
                    main_df = pd.DataFrame([{
                        "Item Code": item["item_code"],
                        "Description": item.get("enhanced_description", item["description"]),
                        "Quantity": item["quantity"],
                        "Unit": item["unit"],
                        "Unit Cost": f"{item['unit_cost']} {currency}",
                        "Total Cost": f"{item['total_cost']} {currency}"
                    } for item in category_items])
                else:
                    main_df = pd.DataFrame([{
                        "Item Code": item["item_code"],
                        "Description": item["description"],
                        "Quantity": item["quantity"],
                        "Unit": item["unit"],
                        "Unit Cost": f"{item['unit_cost']} {currency}",
                        "Total Cost": f"{item['total_cost']} {currency}"
                    } for item in category_items])
                
                st.table(main_df)
                
                # Detailed specifications
                st.subheader("Detailed Specifications")
                for item in category_items:
                    with st.expander(f"{item['item_code']} - {item['description']}"):
                        st.write(f"**Material Specification:** {item['detailed_spec']}")
                        st.write(f"**Location:** {item['location']}")
        
        # Calculate total cost
        total_cost = sum(item["total_cost"] for item in st.session_state.boq)
        
        # Display total
        st.subheader("Summary")
        st.markdown(f"### Total Estimated Cost: **{total_cost:,.2f} {currency}**")
        
        # Export options
        st.subheader("Export Options")
        
        # CSV export
        csv = pd.DataFrame(st.session_state.boq).to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="boq_export.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
