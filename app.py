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
import time

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="College Academic Dashboard",
    layout="wide"
)

# ----------------------------------
# GLOBAL STATE
# ----------------------------------
rf_rmse = None

# ----------------------------------
# LOADING SKELETON HELPERS
# ----------------------------------
def show_skeleton(container, lines=5):
    for _ in range(lines):
        container.markdown("▮▮▮▮▮▮▮▮▮▮")

# ----------------------------------
# DATA LOADING (CACHED + SAFE)
# ----------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    r = requests.get(url, timeout=10)
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        return pd.read_csv(z.open("student-mat.csv"), sep=";")

# ----------------------------------
# FEATURE ENGINEERING (CACHED)
# ----------------------------------
@st.cache_data(show_spinner=False)
def prepare_features(df):
    model_df = pd.get_dummies(df, drop_first=True)
    X = model_df.drop("G3", axis=1)
    y = model_df["G3"]
    return X, y

# ----------------------------------
# MODEL TRAINING (CACHED RESOURCE)
# ----------------------------------
@st.cache_resource(show_spinner=False)
def train_rf(X_train, y_train, progress):
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    for i in range(1, 6):
        time.sleep(0.3)      # simulate training steps
        progress.progress(i * 20)
    model.fit(X_train, y_train)
    return model

# ----------------------------------
# MAIN UI
# ----------------------------------
st.title("🎓 College Academic Performance Dashboard")

# -----------------------
# ASYNC DATA LOAD UI
# -----------------------
data_placeholder = st.empty()

with st.spinner("Loading academic dataset..."):
    show_skeleton(data_placeholder, 6)
    df = load_data()
    data_placeholder.empty()

st.success("Dataset loaded successfully ✅")

# ----------------------------------
# TABS
# ----------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Academic Dashboard", "🤖 Model Evaluation", "📄 Reports"]
)

# =====================================================
# TAB 1: DASHBOARD
# =====================================================
with tab1:
    st.subheader("📊 Student Academic Overview")

    chart_placeholder = st.empty()
    show_skeleton(chart_placeholder, 8)

    time.sleep(0.5)

    chart_placeholder.plotly_chart(
        px.histogram(df, x="G3", nbins=20,
                     title="Final Grade Distribution"),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(df, x="absences", y="G3",
                   color="sex",
                   title="Attendance vs Performance"),
        use_container_width=True
    )

# =====================================================
# TAB 2: MODEL EVALUATION (LAZY LOAD)
# =====================================================
with tab2:
    st.subheader("🤖 Faculty – Predictive Model Evaluation")

    if st.button("▶ Run Model Evaluation"):
        progress = st.progress(0)
        status = st.empty()

        status.info("Preparing features...")
        X, y = prepare_features(df)

        time.sleep(0.5)
        status.info("Splitting dataset...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        status.info("Training Random Forest model...")
        rf = train_rf(X_train, y_train, progress)

        rf_rmse = math.sqrt(
            mean_squared_error(y_test, rf.predict(X_test))
        )

        progress.empty()
        status.success("Model training completed ✅")

        st.metric("Random Forest RMSE", f"{rf_rmse:.2f}")

    else:
        st.info("Click the button to run model evaluation")

# =====================================================
# TAB 3: REPORT GENERATION
# =====================================================
with tab3:
    st.subheader("📄 Academic Performance Report")

    if rf_rmse is None:
        st.warning("Run Model Evaluation before generating report.")
    else:
        def generate_pdf():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                doc = SimpleDocTemplate(tmp.name)
                styles = getSampleStyleSheet()

                content = [
                    Paragraph("College Academic Performance Report", styles["Title"]),
                    Paragraph(f"Random Forest RMSE: {rf_rmse:.2f}", styles["Normal"]),
                    Paragraph("This report summarizes student performance analysis.",
                              styles["Normal"])
                ]

                doc.build(content)
                return tmp.name

        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating report..."):
                pdf_path = generate_pdf()
                time.sleep(1)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇ Download Report",
                    f,
                    file_name="college_report.pdf",
                    mime="application/pdf"
                )
