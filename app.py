import streamlit as st
import pandas as pd
import plotly.express as px
import requests, zipfile
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import math

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config("College Academic Dashboard", layout="wide")

# ----------------------------------
# LOAD DATA
# ----------------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    r = requests.get(url)
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        return pd.read_csv(z.open("student-mat.csv"), sep=";")

df = load_data()

# ----------------------------------
# TABS
# ----------------------------------
tab1, tab2, tab3 = st.tabs(["🎓 Academic Dashboard", "🤖 Model Evaluation", "📄 Reports"])

# =====================================================
# TAB 1: COLLEGE DASHBOARD
# =====================================================
with tab1:
    st.subheader("🎓 Student Academic Performance")

    fig_grades = px.histogram(
        df, x="G3", nbins=20,
        title="Final Grade Distribution"
    )
    st.plotly_chart(fig_grades, use_container_width=True)

    fig_attendance = px.scatter(
        df, x="absences", y="G3",
        color="sex",
        title="Attendance vs Performance"
    )
    st.plotly_chart(fig_attendance, use_container_width=True)

# =====================================================
# TAB 2: MODEL COMPARISON (FACULTY VIEW)
# =====================================================
with tab2:
    st.subheader("🤖 Predictive Model Evaluation")

    model_df = pd.get_dummies(df, drop_first=True)
    X = model_df.drop("G3", axis=1)
    y = model_df["G3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    rf_rmse = math.sqrt(mean_squared_error(y_test, rf.predict(X_test)))

    st.metric("Random Forest RMSE", f"{rf_rmse:.2f}")

    if xgb_available:
        xgb = XGBRegressor(n_estimators=150, learning_rate=0.05)
        xgb.fit(X_train, y_train)
        xgb_rmse = math.sqrt(mean_squared_error(y_test, xgb.predict(X_test)))
        st.metric("XGBoost RMSE", f"{xgb_rmse:.2f}")
    else:
        st.warning("XGBoost not installed")

# =====================================================
# TAB 3: PDF REPORT (ADMIN)
# =====================================================
with tab3:
    st.subheader("📄 Academic Performance Report")

    def generate_pdf():
        pdf = "college_report.pdf"
        doc = SimpleDocTemplate(pdf)
        styles = getSampleStyleSheet()

        content = [
            Paragraph("College Academic Performance Report", styles["Title"]),
            Paragraph(f"Random Forest RMSE: {rf_rmse:.2f}", styles["Normal"]),
        ]

        doc.build(content)
        return pdf

    if st.button("Generate PDF Report"):
        file = generate_pdf()
        with open(file, "rb") as f:
            st.download_button(
                "Download Report",
                f,
                file_name="college_report.pdf",
                mime="application/pdf"
            )
