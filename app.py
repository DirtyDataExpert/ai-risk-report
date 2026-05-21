import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Risk Report", layout="wide")

st.title("AI Risk Report")
st.write("Upload a CSV file and assess AI data risk")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    missing_pct = round(
        (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
        2
    )

    st.subheader("AI Risk Indicators")
    st.metric("Missing Data %", f"{missing_pct}%")

    if missing_pct > 20:
        st.error("High AI Risk")
    elif missing_pct > 10:
        st.warning("Moderate AI Risk")
    else:
        st.success("Low AI Risk")
