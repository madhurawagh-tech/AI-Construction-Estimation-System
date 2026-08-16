import streamlit as st
import pandas as pd
import joblib

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ESTIM AI",
    page_icon="🏗️",
    layout="centered"
)

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    building_model = joblib.load(
        "construction_rf_model.pkl"
    )

    building_features = joblib.load(
        "construction_feature_columns.pkl"
    )

    road_model = joblib.load(
        "road_rf_model.pkl"
    )

    road_features = joblib.load(
        "road_feature_columns.pkl"
    )

    road_y_scaler = joblib.load(
        "road_y_scaler.pkl"
    )

    return (
        building_model,
        building_features,
        road_model,
        road_features,
        road_y_scaler
    )


try:

    (
        building_model,
        building_features,
        road_model,
        road_features,
        road_y_scaler
    ) = load_models()

except FileNotFoundError as e:

    st.error("❌ Required model file not found.")
    st.code(str(e))
    st.stop()


# =========================================================
# PDF GENERATION FUNCTION
# =========================================================

def generate_pdf(
    project_type,
    input_data,
    result_data
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    story = []

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "ESTIM AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Powered Construction Estimation Report",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            f"<b>Project Type:</b> {project_type}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # -----------------------------------------------------
    # INPUT PARAMETERS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Project Parameters",
            styles["Heading2"]
        )
    )

    input_table_data = [
        ["Parameter", "Value"]
    ]

    for key, value in input_data.items():

        input_table_data.append(
            [
                str(key),
                str(value)
            ]
        )

    input_table = Table(
        input_table_data,
        colWidths=[230, 230]
    )

    input_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1F4E78")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(input_table)

    story.append(
        Spacer(1, 20)
    )

    # -----------------------------------------------------
    # PREDICTED RESULTS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Predicted Estimation Results",
            styles["Heading2"]
        )
    )

    result_table_data = [
        ["Output", "Predicted Value"]
    ]

    for key, value in result_data.items():

        result_table_data.append(
            [
                str(key),
                str(value)
            ]
        )

    result_table = Table(
        result_table_data,
        colWidths=[230, 230]
    )

    result_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#008080")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(result_table)

    story.append(
        Spacer(1, 25)
    )

    story.append(
        Paragraph(
            "Generated by ESTIM AI - AI-Powered Construction Estimation System",
            styles["Normal"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer


# =========================================================
# TITLE
# =========================================================

st.title("🏗️ ESTIM AI")

st.subheader(
    "AI-Powered Construction Estimation System"
)

st.write(
    "Predict material requirements and construction costs "
    "for Building and Road Construction using Machine Learning."
)

st.divider()


# =========================================================
# CONSTRUCTION TYPE
# =========================================================

construction_type = st.selectbox(
    "Select Construction Type",
    ["Building", "Road"]
)


# =========================================================
# BUILDING CONSTRUCTION
# =========================================================

if construction_type == "Building":

    st.header("🏠 Building Construction")

    area_type = st.selectbox(
        "Area Type",
        ["Urban", "Semi-Urban", "Rural"]
    )

    built_up_area = st.number_input(
        "Built-Up Area (sq.ft)",
        min_value=100.0,
        value=2000.0
    )

    floors = st.number_input(
        "Number of Floors",
        min_value=1,
        max_value=20,
        value=2
    )

    wall_thickness = st.selectbox(
        "Wall Thickness (inches)",
        [6, 9, 12]
    )

    roof_type = st.selectbox(
        "Roof Type",
        [
            "RCC Flat Roof",
            "RCC Sloped Roof",
            "Slab Roof",
            "Metal Sheet Roof"
        ]
    )

    soil_type = st.selectbox(
        "Soil Type",
        [
            "Hard",
            "Medium",
            "Soft"
        ]
    )

    paint_type = st.selectbox(
        "Paint Type",
        [
            "Distemper",
            "Plastic Emulsion",
            "Standard Acrylic",
            "Premium Acrylic",
            "Royal Emulsion",
            "Lustre Paint",
            "Apex Weather Proof"
        ]
    )

    flooring = st.selectbox(
        "Flooring",
        [
            "Ceramic Tiles",
            "Vitrified Tiles",
            "Granite",
            "Mosaic",
            "Imported Vitrified Tiles",
            "Italian Marble"
        ]
    )

    construction_quality = st.selectbox(
        "Construction Quality",
        [
            "Basic",
            "Standard",
            "Premium"
        ]
    )

    if st.button(
        "🔮 Predict Building Estimate",
        use_container_width=True
    ):

        building_input = pd.DataFrame(
            [[
                area_type,
                built_up_area,
                floors,
                wall_thickness,
                roof_type,
                soil_type,
                paint_type,
                flooring,
                construction_quality
            ]],
            columns=[
                "Area_Type",
                "Built_Up_Area_sqft",
                "Floors",
                "Wall_Thickness_inches",
                "Roof_Type",
                "Soil_Type",
                "Paint_Type",
                "Flooring",
                "Construction_Quality"
            ]
        )

        building_input_encoded = pd.get_dummies(
            building_input
        )

        building_input_encoded = building_input_encoded.reindex(
            columns=building_features,
            fill_value=0
        )

        prediction = building_model.predict(
            building_input_encoded
        )[0]

        st.divider()

        st.header("📊 Building Estimation Results")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🧱 Cement Required",
                f"{prediction[0]:,.2f} bags"
            )

            st.metric(
                "🏖️ Sand Required",
                f"{prediction[1]:,.2f} m³"
            )

            st.metric(
                "🪨 Aggregate Required",
                f"{prediction[2]:,.2f} m³"
            )

        with col2:

            st.metric(
                "🔩 Steel Required",
                f"{prediction[3]:,.2f} kg"
            )

            st.metric(
                "🧱 Bricks Required",
                f"{prediction[4]:,.0f} units"
            )

            st.metric(
                "🎨 Paint Required",
                f"{prediction[5]:,.2f} litres"
            )

        st.subheader("💰 Cost Estimation")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Material Cost",
                f"₹{prediction[6]:,.2f}"
            )

        with col2:

            st.metric(
                "Labour Cost",
                f"₹{prediction[7]:,.2f}"
            )

        with col3:

            st.metric(
                "Total Cost",
                f"₹{prediction[8]:,.2f}"
            )

        # -------------------------------------------------
        # PDF DATA
        # -------------------------------------------------

        building_input_data = {
            "Area Type": area_type,
            "Built-Up Area": f"{built_up_area} sq.ft",
            "Number of Floors": floors,
            "Wall Thickness": f"{wall_thickness} inches",
            "Roof Type": roof_type,
            "Soil Type": soil_type,
            "Paint Type": paint_type,
            "Flooring": flooring,
            "Construction Quality": construction_quality
        }

        building_result_data = {
            "Cement Required": f"{prediction[0]:,.2f} bags",
            "Sand Required": f"{prediction[1]:,.2f} m³",
            "Aggregate Required": f"{prediction[2]:,.2f} m³",
            "Steel Required": f"{prediction[3]:,.2f} kg",
            "Bricks Required": f"{prediction[4]:,.0f} units",
            "Paint Required": f"{prediction[5]:,.2f} litres",
            "Material Cost": f"₹{prediction[6]:,.2f}",
            "Labour Cost": f"₹{prediction[7]:,.2f}",
            "Total Construction Cost": f"₹{prediction[8]:,.2f}"
        }

        pdf_file = generate_pdf(
            "Building Construction",
            building_input_data,
            building_result_data
        )

        st.download_button(
            label="📄 Download Building Estimation PDF",
            data=pdf_file,
            file_name="building_estimation_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# =========================================================
# ROAD CONSTRUCTION
# =========================================================

else:

    st.header("🛣️ Road Construction")

    road_type = st.selectbox(
        "Road Type",
        [
            "Asphalt",
            "Concrete"
        ]
    )

    road_length = st.number_input(
        "Road Length (km)",
        min_value=0.1,
        value=1.0,
        step=0.1
    )

    road_width = st.number_input(
        "Road Width (m)",
        min_value=1.0,
        value=7.0,
        step=0.5
    )

    road_thickness = st.number_input(
        "Road Thickness (m)",
        min_value=0.01,
        value=0.20,
        step=0.01,
        format="%.2f"
    )

    number_of_lanes = st.number_input(
        "Number of Lanes",
        min_value=1,
        max_value=8,
        value=2
    )

    soil_type = st.selectbox(
        "Soil Type",
        [
            "Weak",
            "Medium",
            "Strong"
        ]
    )

    construction_quality = st.selectbox(
        "Construction Quality",
        [
            "Standard",
            "Premium"
        ]
    )

    if st.button(
        "🔮 Predict Road Estimate",
        use_container_width=True
    ):

        road_input = pd.DataFrame(
            [[
                road_type,
                road_length,
                road_width,
                road_thickness,
                number_of_lanes,
                soil_type,
                construction_quality
            ]],
            columns=[
                "Road_Type",
                "Road_Length_km",
                "Road_Width_m",
                "Road_Thickness_m",
                "Number_of_Lanes",
                "Soil_Type",
                "Construction_Quality"
            ]
        )

        road_input_encoded = pd.get_dummies(
            road_input
        )

        road_input_encoded = road_input_encoded.reindex(
            columns=road_features,
            fill_value=0
        )

        scaled_prediction = road_model.predict(
            road_input_encoded
        )

        road_prediction = road_y_scaler.inverse_transform(
            scaled_prediction
        )[0]

        st.divider()

        st.header("📊 Road Estimation Results")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🪨 Aggregate Required",
                f"{road_prediction[0]:,.2f} m³"
            )

            st.metric(
                "🏖️ Sand Required",
                f"{road_prediction[1]:,.2f} m³"
            )

            st.metric(
                "🧱 Cement Required",
                f"{road_prediction[2]:,.2f} tonnes"
            )

        with col2:

            st.metric(
                "🛢️ Bitumen Required",
                f"{road_prediction[3]:,.2f} tonnes"
            )

            st.metric(
                "🔩 Steel Required",
                f"{road_prediction[4]:,.2f} tonnes"
            )

        st.subheader("💰 Cost Estimation")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Material Cost",
                f"₹{road_prediction[5]:,.2f}"
            )

        with col2:

            st.metric(
                "Labour Cost",
                f"₹{road_prediction[6]:,.2f}"
            )

        with col3:

            st.metric(
                "Total Road Cost",
                f"₹{road_prediction[7]:,.2f}"
            )

        # -------------------------------------------------
        # PDF DATA
        # -------------------------------------------------

        road_input_data = {
            "Road Type": road_type,
            "Road Length": f"{road_length} km",
            "Road Width": f"{road_width} m",
            "Road Thickness": f"{road_thickness} m",
            "Number of Lanes": number_of_lanes,
            "Soil Type": soil_type,
            "Construction Quality": construction_quality
        }

        road_result_data = {
            "Aggregate Required": f"{road_prediction[0]:,.2f} m³",
            "Sand Required": f"{road_prediction[1]:,.2f} m³",
            "Cement Required": f"{road_prediction[2]:,.2f} tonnes",
            "Bitumen Required": f"{road_prediction[3]:,.2f} tonnes",
            "Steel Required": f"{road_prediction[4]:,.2f} tonnes",
            "Material Cost": f"₹{road_prediction[5]:,.2f}",
            "Labour Cost": f"₹{road_prediction[6]:,.2f}",
            "Total Construction Cost": f"₹{road_prediction[7]:,.2f}"
        }

        pdf_file = generate_pdf(
            "Road Construction",
            road_input_data,
            road_result_data
        )

        st.download_button(
            label="📄 Download Road Estimation PDF",
            data=pdf_file,
            file_name="road_estimation_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.success(
            "✅ Road construction estimation completed successfully!"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "ESTIM AI | AI-Powered Construction Estimation System"
)
