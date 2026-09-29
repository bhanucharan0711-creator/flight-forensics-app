import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    st.subheader("📋 Flight Telemetry")
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

    required_columns = features + ["Event_Label"]

    if all(column in df.columns for column in required_columns):

        # -----------------------------
        # 1. ANOMALY DETECTION
        # -----------------------------

        model = IsolationForest(
            n_estimators=100,
            contamination=0.10,
            random_state=42
        )

        model.fit(df[features])

        df["Anomaly_Score"] = model.decision_function(
            df[features]
        )

        df["AI_Anomaly"] = model.predict(
            df[features]
        )

        df["AI_Anomaly"] = df["AI_Anomaly"].map({
            1: "Normal",
            -1: "Anomaly"
        })

        st.subheader("🚨 Anomaly Detection")

        anomaly_count = (
            df["AI_Anomaly"] == "Anomaly"
        ).sum()

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Records",
            len(df)
        )

        col2.metric(
            "Detected Anomalies",
            anomaly_count
        )

        st.dataframe(
            df[df["AI_Anomaly"] == "Anomaly"],
            use_container_width=True
        )

        # -----------------------------
        # 2. EVENT CLASSIFICATION
        # -----------------------------

        st.subheader("🧠 Event Classification")

        abnormal_df = df[
            df["Event_Label"] != "Normal"
        ].copy()

        if len(abnormal_df) > 10:

            X_cls = abnormal_df[features]
            y_cls = abnormal_df["Event_Label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X_cls,
                y_cls,
                test_size=0.25,
                random_state=42,
                stratify=y_cls
            )

            classifier = RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )

            classifier.fit(
                X_train,
                y_train
            )

            y_pred = classifier.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            st.metric(
                "Classification Accuracy",
                f"{accuracy * 100:.2f}%"
            )

            # Predict event for detected anomalies
            df["Predicted_Event"] = "Normal"

            anomaly_mask = (
                df["AI_Anomaly"] == "Anomaly"
            )

            if anomaly_mask.sum() > 0:

                df.loc[
                    anomaly_mask,
                    "Predicted_Event"
                ] = classifier.predict(
                    df.loc[
                        anomaly_mask,
                        features
                    ]
                )

            st.write("Detected abnormal events:")

            st.dataframe(
                df[
                    [
                        "Timestamp",
                        "AI_Anomaly",
                        "Predicted_Event",
                        "Anomaly_Score"
                    ]
                ][
                    df["AI_Anomaly"] == "Anomaly"
                ],
                use_container_width=True
            )

        else:

            st.warning(
                "Not enough abnormal records for event classification."
            )

        # -----------------------------
        # 3. AIRSPEED GRAPH
        # -----------------------------

        if "Timestamp" in df.columns:

            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"]
            )

            st.subheader("📈 Airspeed Over Time")

            st.line_chart(
                df.set_index("Timestamp")[
                    "Airspeed_kts"
                ]
            )

        # -----------------------------
        # 4. ENGINE TEMPERATURE
        # -----------------------------

        if "Timestamp" in df.columns:

            st.subheader("🌡️ Engine Temperature")

            st.line_chart(
                df.set_index("Timestamp")[
                    "Engine_Temperature_C"
                ]
            )

        # -----------------------------
        # 5. SUMMARY
        # -----------------------------

        st.subheader("📋 Analysis Summary")

        st.write(
            "The AI system detects unusual flight telemetry "
            "patterns and classifies detected abnormal records "
            "into different event types."
        )

    else:

        st.error(
            "The uploaded CSV does not contain all required columns."
        )
