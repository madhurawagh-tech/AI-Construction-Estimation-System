
import streamlit as st
import pandas as pd
import joblib

# Load models
building_model = joblib.load("construction_model.pkl")
road_model = joblib.load("road_model.pkl")

# Load building column names
model_columns = joblib.load("model_columns.pkl")

# Page configuration
st.set_page_config(
    page_title="AI Construction Estimator",
    page_icon="🏗️",
    layout="centered"
)

# Title
st.title("🏗️ AI-Based Parametric Construction Estimation System")
st.write("Predict raw material requirements and construction cost using AI.")

# Select construction type
construction_type = st.selectbox(
    "Select Construction Type",
    ["Building", "Road"]
)

# =========================================================
# BUILDING
# =========================================================

if construction_type == "Building":

    st.header("🏠 Building Construction")
    st.write("Enter Building Construction Details")

    area_type = st.selectbox(
        "Area Type",
        ["Urban", "Semi-Urban", "Rural"]
    )

    built_up_area = st.number_input(
        "Built-Up Area (sq.ft)",
        min_value=100,
        value=2000
    )

    floors = st.number_input(
        "Number of Floors",
        min_value=1,
        max_value=10,
        value=2
    )

    wall_thickness = st.selectbox(
        "Wall Thickness (inches)",
        [6, 9, 12]
    )

    roof_type = st.selectbox(
        "Roof Type",
        ["RCC Flat Roof", "Sloped RCC Roof", "Metal Sheet Roof"]
    )

    soil_type = st.selectbox(
        "Soil Type",
        ["Good", "Medium", "Weak"]
    )

    paint_type = st.selectbox(
        "Paint Type",
        ["Distemper", "Acrylic Emulsion", "Premium Emulsion"]
    )

    flooring = st.selectbox(
        "Flooring",
        ["Ceramic Tiles", "Vitrified Tiles", "Granite"]
    )

    construction_quality = st.selectbox(
        "Construction Quality",
        ["Economy", "Standard", "Premium"]
    )

    # Prediction button
    if st.button("🔮 Predict Building Estimate"):

        new_proj = pd.DataFrame([[
            area_type,
            built_up_area,
            floors,
            wall_thickness,
            roof_type,
            soil_type,
            paint_type,
            flooring,
            construction_quality
        ]], columns=[
            "Area_Type",
            "Built_Up_Area_sqft",
            "Floors",
            "Wall_Thickness_inches",
            "Roof_Type",
            "Soil_Type",
            "Paint_Type",
            "Flooring",
            "Construction_Quality"
        ])

        # One-hot encoding
        new_proj = pd.get_dummies(new_proj)

        # Match columns used during training
        new_proj = new_proj.reindex(
            columns=model_columns,
            fill_value=0
        )

        # Prediction
        prediction = building_model.predict(new_proj)[0]

        # Display results
        st.header("📊 Building Estimation Results")

        st.write("🧱 Cement Required:", round(prediction[0], 2), "bags")
        st.write("🏖️ Sand Required:", round(prediction[1], 2), "m³")
        st.write("🪨 Aggregate Required:", round(prediction[2], 2), "m³")
        st.write("🔩 Steel Required:", round(prediction[3], 2), "kg")
        st.write("🧱 Bricks Required:", round(prediction[4], 2), "units")
        st.write("🎨 Paint Required:", round(prediction[5], 2), "litres")

        st.subheader("💰 Cost Estimation")

        st.write("Total Material Cost: ₹", round(prediction[6], 2))
        st.write("Labour Cost: ₹", round(prediction[7], 2))

        st.success(
            f"🏗️ Total Construction Cost: ₹{round(prediction[8], 2)}"
        )


# =========================================================
# ROAD
# =========================================================

else:

    st.header("🛣️ Road Construction")
    st.write("Enter Road Construction Details")

    road_type = st.selectbox(
        "Road Type",
        ["Asphalt", "Concrete"]
    )

    road_length = st.number_input(
        "Road Length (km)",
        min_value=0.1,
        value=5.0
    )

    road_width = st.number_input(
        "Road Width (m)",
        min_value=1.0,
        value=7.0
    )

    road_thickness = st.number_input(
        "Road Thickness (m)",
        min_value=0.05,
        value=0.20
    )

    number_of_lanes = st.number_input(
        "Number of Lanes",
        min_value=1,
        max_value=8,
        value=2
    )

    soil_type = st.selectbox(
        "Soil Type",
        ["Weak", "Medium", "Strong"]
    )

    construction_quality = st.selectbox(
        "Construction Quality",
        ["Standard", "Premium"]
    )

    # Prediction button
    if st.button("🔮 Predict Road Estimate"):

        road_input = pd.DataFrame([[
            road_type,
            road_length,
            road_width,
            road_thickness,
            number_of_lanes,
            soil_type,
            construction_quality
        ]], columns=[
            "Road_Type",
            "Road_Length_km",
            "Road_Width_m",
            "Road_Thickness_m",
            "Number_of_Lanes",
            "Soil_Type",
            "Construction_Quality"
        ])

        # One-hot encoding
        road_input = pd.get_dummies(road_input)

        # Match columns used during road model training
        road_input = road_input.reindex(
            columns=road_model.feature_names_in_,
            fill_value=0
        )

        # Prediction
        road_prediction = road_model.predict(road_input)[0]

        # Display results
        st.header("📊 Road Estimation Results")

        st.write(
            "🪨 Aggregate Required:",
            round(road_prediction[0], 2),
            "m³"
        )

        st.write(
            "🏖️ Sand Required:",
            round(road_prediction[1], 2),
            "m³"
        )

        st.write(
            "🧱 Cement Required:",
            round(road_prediction[2], 2),
            "tonnes"
        )

        st.write(
            "🛢️ Bitumen Required:",
            round(road_prediction[3], 2),
            "tonnes"
        )

        st.write(
            "🔩 Steel Required:",
            round(road_prediction[4], 2),
            "tonnes"
        )

        st.subheader("💰 Cost Estimation")

        st.write(
            "Total Material Cost: ₹",
            round(road_prediction[5], 2)
        )

        st.write(
            "Labour Cost: ₹",
            round(road_prediction[6], 2)
        )

        st.success(
            f"🛣️ Total Road Construction Cost: ₹{round(road_prediction[7], 2)}"
        )
