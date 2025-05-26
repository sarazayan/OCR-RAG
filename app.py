import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import random
import base64
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Masah BoQ Generator", layout="wide")

# App title and description
st.title("Masah - Floor Plan to BoQ Generator")
st.markdown("""
This application processes architectural floor plans to automatically extract construction 
elements, generate quantity take-offs, and create detailed Bills of Quantities using 
computer vision and RAG technology.
""")

# Create tabs for the workflow
tabs = st.tabs(["1. Upload & Extract", "2. Take-Off", "3. BoQ Generation", "4. Final Output"])

# Sidebar for project settings
with st.sidebar:
    st.header("Project Settings")
    project_name = st.text_input("Project Name", "Residential Building")
    project_location = st.selectbox("Location", ["Cairo, Egypt", "Alexandria, Egypt", "Giza, Egypt", "Other"])
    project_type = st.selectbox("Type", ["Residential", "Commercial", "Industrial"])
    currency = st.selectbox("Currency", ["EGP", "USD", "EUR"])
    
    # Knowledge base settings
    st.header("Knowledge Base")
    kb_source = st.multiselect("Sources", 
                               ["Egyptian Building Code", "ASTM Standards", "Local Material Catalogs", "Cost Indexes"],
                               default=["Egyptian Building Code", "Local Material Catalogs"])

# Function to simulate CV extraction from floor plan
def extract_elements(image):
    """Extract construction elements using CV (simulated)"""
    # Convert image to grayscale for processing
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Simulate element detection (in a real app, this would use trained CV models)
    # Detect doors (orange elements in the floor plan)
    doors = [
        {"id": "D1", "type": "Interior", "width": 0.9, "height": 2.1, "location": "Bedroom 1-Hall"},
        {"id": "D2", "type": "Interior", "width": 0.9, "height": 2.1, "location": "Bedroom 2-Hall"},
        {"id": "D3", "type": "Interior", "width": 0.9, "height": 2.1, "location": "Master Bedroom-Hall"},
        {"id": "D4", "type": "Interior", "width": 0.8, "height": 2.1, "location": "Bathroom-Hall"},
        {"id": "D5", "type": "Interior", "width": 1.0, "height": 2.1, "location": "Kitchen-Dining"}
    ]
    
    # Detect windows (blue elements in the floor plan)
    windows = [
        {"id": "W1", "type": "Standard", "width": 1.5, "height": 1.2, "location": "Bedroom 1"},
        {"id": "W2", "type": "Standard", "width": 1.5, "height": 1.2, "location": "Bedroom 2"},
        {"id": "W3", "type": "Standard", "width": 1.5, "height": 1.2, "location": "Master Bedroom"},
        {"id": "W4", "type": "Standard", "width": 1.5, "height": 1.2, "location": "Kitchen"},
        {"id": "W5", "type": "Standard", "width": 1.5, "height": 1.2, "location": "Dining"}
    ]
    
    # Detect walls (from the floor plan's measurements - 15m × 10.5m with 200mm thickness)
    walls = {
        "external": {"length": 41.0, "height": 2.6, "thickness": 0.2, "area": 106.6},
        "internal": {"length": 24.5, "height": 2.6, "thickness": 0.15, "area": 63.7}
    }
    
    # Detect rooms and calculate areas
    rooms = [
        {"name": "Bedroom 1", "area": 12.0},
        {"name": "Bedroom 2", "area": 12.0},
        {"name": "Master Bedroom", "area": 16.0},
        {"name": "Bathroom", "area": 6.0},
        {"name": "Kitchen", "area": 10.0},
        {"name": "Dining", "area": 14.0}
    ]
    
    # Return extracted elements and the processed image
    return {
        "doors": doors,
        "windows": windows,
        "walls": walls,
        "rooms": rooms,
        "floor_height": 2.6,
        "ceiling_type": "Drop ceiling with recessed lighting",
        "flooring": "Engineered hardwood"
    }, gray

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

# Function to generate BoQ using RAG (simulated)
def generate_boq(takeoff_items, project_location, project_type):
    """Generate BoQ entries using RAG from the knowledge base"""
    boq_items = []
    
    # Simulated knowledge base (in a real app, this would query a vector DB)
    kb = {
        "Doors": {
            "Cairo, Egypt": {
                "Interior": {
                    "unit_cost": 2200,
                    "installation": "As per ECP 204-2018, section 5.3",
                    "materials": "Solid core wood with steel frame, fire rating 30 min",
                    "item_code": "DR-INT"
                }
            }
        },
        "Windows": {
            "Cairo, Egypt": {
                "Standard": {
                    "unit_cost": 3500,
                    "installation": "As per ECP 204-2018, section 6.2",
                    "materials": "Double-glazed aluminum frame, U-value 2.5 W/m²K",
                    "item_code": "WIN-STD"
                }
            }
        },
        "Walls": {
            "Cairo, Egypt": {
                "Load-bearing": {
                    "unit_cost": 1200,
                    "installation": "As per ECP 203-2018",
                    "materials": "Reinforced concrete with brick infill, thermal insulation R-13",
                    "item_code": "WL-EXT"
                },
                "Partition": {
                    "unit_cost": 800,
                    "installation": "As per ECP 203-2018",
                    "materials": "Lightweight concrete blocks with plaster finish",
                    "item_code": "WL-INT"
                }
            }
        },
        "Flooring": {
            "Cairo, Egypt": {
                "Engineered hardwood": {
                    "unit_cost": 950,
                    "installation": "As per ECP 208-2019",
                    "materials": "Engineered hardwood 12mm over concrete substrate with sound insulation",
                    "item_code": "FL-EHW"
                }
            }
        },
        "Ceiling": {
            "Cairo, Egypt": {
                "Recessed": {
                    "unit_cost": 650,
                    "installation": "As per ECP 205-2019",
                    "materials": "Gypsum board drop ceiling with aluminum frame and recessed LED lighting",
                    "item_code": "CL-DRP"
                }
            }
        }
    }
    
    # Generate BoQ items using the knowledge base
    for item in takeoff_items:
        category = item["category"]
        item_type = item["type"]
        
        # Check if we have data in our KB
        if (category in kb and 
            project_location in kb[category] and 
            item_type in kb[category][project_location]):
            
            kb_info = kb[category][project_location][item_type]
            
            # Calculate costs
            unit_cost = kb_info["unit_cost"]
            total_cost = unit_cost * item["quantity"]
            
            # Generate a unique item code
            item_code = f"{kb_info['item_code']}-{random.randint(100, 999)}"
            
            # Create BoQ item with detailed information
            boq_item = {
                "category": category,
                "item_code": item_code,
                "description": item["description"],
                "detailed_spec": kb_info["materials"],
                "installation_guide": kb_info["installation"],
                "quantity": item["quantity"],
                "unit": item["unit"],
                "unit_cost": unit_cost,
                "total_cost": total_cost,
                "location": item["location"]
            }
            
            boq_items.append(boq_item)
    
    return boq_items

# Function to visualize detected elements (simulated)
def visualize_detection(gray_image, elements):
    """Create a visualization of the detected elements"""
    # Convert grayscale to RGB for visualization
    vis_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    
    # In a real app, this would draw bounding boxes or highlights
    # For this demo, we'll just return the original image
    return vis_img

# Tab 1: Upload & Extract
with tabs[0]:
    st.header("Upload Floor Plan")
    st.write("Upload an architectural floor plan image to begin the CV extraction process.")
    
    uploaded_file = st.file_uploader("Choose a floor plan image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        
        # Display the original image
        st.image(image, caption="Uploaded Floor Plan", use_column_width=True)
        
        # Process button
        if st.button("Extract Elements", key="extract_btn"):
            # Show processing spinner
            with st.spinner("Analyzing floor plan with computer vision..."):
                # Extract elements using CV
                elements, processed_img = extract_elements(image)
                
                # Visualize the detection
                visualization = visualize_detection(processed_img, elements)
                
                # Store in session state
                st.session_state.elements = elements
                st.session_state.processed = True
            
            # Display extraction results
            st.success("✅ Element extraction complete!")
            
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
            
            # Navigation button
            st.write("Continue to the next step to generate a take-off.")

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
            
            # Navigation
            st.write("Continue to the next step to generate the Bill of Quantities.")

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
            1. **Retrieval**: The system queries a knowledge base containing local building codes, 
               material specifications, cost catalogs, and installation guidelines relevant to the project location.
               
            2. **Augmentation**: Retrieved information is combined with the take-off data and project context.
            
            3. **Generation**: An LLM generates detailed BoQ entries with accurate descriptions, 
               specifications, and cost estimates grounded in the retrieved information.
            """)
        
        if st.button("Generate BoQ with RAG", key="boq_btn"):
            # Progress bar to visualize the RAG process
            progress = st.progress(0)
            status_text = st.empty()
            
            # Simulate the RAG process steps
            for i in range(101):
                progress.progress(i)
                if i < 25:
                    status_text.write("🔍 Retrieving building codes and regulations...")
                elif i < 50:
                    status_text.write("📋 Retrieving material specifications...")
                elif i < 75:
                    status_text.write("💰 Retrieving cost data and catalogs...")
                else:
                    status_text.write("✍️ Generating detailed BoQ entries...")
                
                # Add a small delay to simulate processing
                if i % 10 == 0:
                    import time
                    time.sleep(0.1)
            
            # Generate BoQ using RAG
            boq_items = generate_boq(st.session_state.takeoff, project_location, project_type)
            
            # Store in session state
            st.session_state.boq = boq_items
            st.session_state.boq_done = True
            
            # Clear progress indicators
            status_text.empty()
            
            # Show success message
            st.success("✅ Bill of Quantities generated successfully!")
            
            # Show BoQ preview
            st.subheader("BoQ Preview")
            preview_df = pd.DataFrame([{
                "Item Code": item["item_code"],
                "Description": item["description"],
                "Quantity": item["quantity"],
                "Unit": item["unit"],
                "Unit Cost": item["unit_cost"],
                "Total Cost": item["total_cost"]
            } for item in boq_items])
            
            st.dataframe(preview_df, use_container_width=True)
            
            # Navigation
            st.write("Continue to the final step to view the complete Bill of Quantities.")

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
            st.write(f"**Reference:** BOQ-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}")
        
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
                        st.write(f"**Installation Guide:** {item['installation_guide']}")
                        st.write(f"**Location:** {item['location']}")
        
        # Calculate total cost
        total_cost = sum(item["total_cost"] for item in st.session_state.boq)
        
        # Display total
        st.subheader("Summary")
        st.markdown(f"### Total Estimated Cost: **{total_cost:,.2f} {currency}**")
        
        # Export options
        st.subheader("Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export
            csv = pd.DataFrame(st.session_state.boq).to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="boq_export.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # In a real app, you would implement Excel export
            st.button("Export to Excel")
        
        with col3:
            # In a real app, you would implement PDF export
            st.button("Export to PDF")