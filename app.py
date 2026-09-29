import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="AI Flight Data Forensics",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ AI Flight Data Forensics")
st.write("AI-based flight telemetry anomaly detection and forensic analysis.")

uploaded_file = st.file_uploader(
    "Upload flight telemetry CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Flight data loaded successfully!")

    st.subheader("Flight Telemetry")
    st.dataframe(df.head(20), use_container_width=True)

    features = [
        "Altitude_ft",
        "Airspeed_kts",
        "Vertical_Speed_fpm",
        "Pitch_deg",
        "Roll_deg",
        "Engine_Temperature_C",
        "Fuel_Flow_kg_h"
    ]

    if all(feature in df.columns for feature in features):

        model = IsolationForest(
            n_estimators=100,
            contamination=0.10,
            random_state=42
        )

        model.fit(df[features])

        df["Anomaly_Score"] = model.decision_function(df[features])
        df["AI_Anomaly"] = model.predict(df[features])

        df["AI_Anomaly"] = df["AI_Anomaly"].map({
            1: "Normal",
            -1: "Anomaly"
        })

        st.subheader("🚨 Anomaly Detection")

        anomaly_count = (df["AI_Anomaly"] == "Anomaly").sum()

        col1, col2 = st.columns(2)

        col1.metric("Total Records", len(df))
        col2.metric("Detected Anomalies", anomaly_count)

        st.dataframe(
            df[df["AI_Anomaly"] == "Anomaly"],
            use_container_width=True
        )

        st.subheader("📊 Airspeed Over Time")

        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            st.line_chart(
                df.set_index("Timestamp")["Airspeed_kts"]
            )

        st.subheader("🌡️ Engine Temperature")

        if "Timestamp" in df.columns:
            st.line_chart(
                df.set_index("Timestamp")["Engine_Temperature_C"]
            )

        st.subheader("📋 Analysis Summary")

        st.write(
            "The AI model analyzes aircraft telemetry and "
            "identifies records that behave differently from "
            "the normal flight pattern."
        )

    else:
        st.error("The uploaded CSV does not contain all required telemetry columns.")
