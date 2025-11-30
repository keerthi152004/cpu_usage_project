import streamlit as st
import pandas as pd
import joblib

# Load model + scaler
model = joblib.load("outputs/model_rf.joblib")
scaler = joblib.load("outputs/scaler.joblib")

st.title("CPU Usage Prediction App")
st.write("Fill in the values below to predict CPU usage.")

# User inputs for all features
cpu_request = st.number_input("CPU Request", value=0.0)
mem_request = st.number_input("Memory Request", value=0.0)
cpu_limit = st.number_input("CPU Limit", value=0.0)
mem_limit = st.number_input("Memory Limit", value=0.0)
runtime_minutes = st.number_input("Runtime (Minutes)", value=0.0)

# Categorical Kubernetes controller one-hot indicators
ck_DaemonSet = st.selectbox("Is DaemonSet?", [0, 1])
ck_Job = st.selectbox("Is Job?", [0, 1])
ck_ReplicaSet = st.selectbox("Is ReplicaSet?", [0, 1])
ck_ReplicationController = st.selectbox("Is ReplicationController?", [0, 1])
ck_StatefulSet = st.selectbox("Is StatefulSet?", [0, 1])
ck_UNKNOWN = st.selectbox("Is UNKNOWN Controller?", [0, 1])

# Build a dataframe
input_df = pd.DataFrame([[
    cpu_request, mem_request, cpu_limit, mem_limit, runtime_minutes,
    ck_DaemonSet, ck_Job, ck_ReplicaSet, ck_ReplicationController,
    ck_StatefulSet, ck_UNKNOWN
]], columns=[
    "cpu_request", "mem_request", "cpu_limit", "mem_limit", "runtime_minutes",
    "ck_DaemonSet", "ck_Job", "ck_ReplicaSet", "ck_ReplicationController",
    "ck_StatefulSet", "ck_UNKNOWN"
])

# Scale the input
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict CPU Usage"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"Predicted CPU Usage: {prediction:.2f}%")
