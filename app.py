import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ESTIM AI",
    page_icon="🏗️",
    layout="centered"
)

# =========================================================
# LOAD MODELS AND FEATURE COLUMNS
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
# SELECT CONSTRUCTION TYPE
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

    st.write(
        "Enter the building construction parameters."
    )

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    area_type = st.selectbox(
        "Area Type",
        [
            "Urban",
            "Semi-Urban",
            "Rural"
        ]
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

    # -----------------------------------------------------
    # BUILDING PREDICTION
    # -----------------------------------------------------

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

        # One-Hot Encoding
        building_input = pd.get_dummies(
            building_input
        )

        # Match exact training features
        building_input = building_input.reindex(
            columns=building_features,
            fill_value=0
        )

        # Prediction
        prediction = building_model.predict(
            building_input
        )[0]

        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------

        st.divider()

        st.header(
            "📊 Building Estimation Results"
        )

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


# =========================================================
# ROAD CONSTRUCTION
# =========================================================

else:

    st.header("🛣️ Road Construction")

    st.write(
        "Enter the road construction parameters."
    )

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # ROAD PREDICTION
    # -----------------------------------------------------

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

        # One-Hot Encoding
        road_input = pd.get_dummies(
            road_input
        )

        # IMPORTANT:
        # Match only the Road model features
        road_input = road_input.reindex(
            columns=road_features,
            fill_value=0
        )

        # -------------------------------------------------
        # SCALED PREDICTION
        # -------------------------------------------------

        scaled_prediction = road_model.predict(
            road_input
        )

        # Convert scaled output back to original values
        road_prediction = road_y_scaler.inverse_transform(
            scaled_prediction
        )[0]

        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------

        st.divider()

        st.header(
            "📊 Road Estimation Results"
        )

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
