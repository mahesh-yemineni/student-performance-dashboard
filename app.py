import streamlit as st
import pandas as pd
import plotly.express as px
import requests, zipfile
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="College Academic Portal",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "data" not in st.session_state:
    st.session_state.data = None
if "model" not in st.session_state:
    st.session_state.model = None
if "rmse" not in st.session_state:
    st.session_state.rmse = None

# --------------------------------------------------
# LOGIN UI
# --------------------------------------------------
def login_ui():
    st.title("🔐 College Portal Login")
    role = st.selectbox("Login as", ["Student", "Faculty"])
    if st.button("Login"):
        st.session_state.logged_in = True
        st.session_state.role = role
        st.experimental_rerun()

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# --------------------------------------------------
# LOGOUT
# --------------------------------------------------
with st.sidebar:
    st.success(f"Logged in as: {st.session_state.role}")
    if st.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

# --------------------------------------------------
# DATA LOADING (SAFE + CACHED)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    r = requests.get(url, timeout=10)
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        return pd.read_csv(z.open("student-mat.csv"), sep=";")

if st.session_state.data is None:
    with st.spinner("Loading student academic data..."):
        st.session_state.data = load_data()

df = st.session_state.data

# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Dashboard", "🤖 Faculty ML", "📈 Insights"]
)

# ==================================================
# TAB 1: DASHBOARD (STUDENT + FACULTY)
# ==================================================
with tab1:
    st.header("📊 Student Academic Dashboard")

    st.plotly_chart(
        px.histogram(df, x="G3", nbins=20, title="Final Grade Distribution"),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(
            df,
            x="absences",
            y="G3",
            color="sex",
            title="Attendance vs Final Grade"
        ),
        use_container_width=True
    )

    pass_fail = df["G3"].apply(lambda x: "Pass" if x >= 10 else "Fail")
    pf_counts = pass_fail.value_counts().reset_index()
    pf_counts.columns = ["Result", "Count"]

    st.plotly_chart(
        px.pie(
            pf_counts,
            names="Result",
            values="Count",
            title="Pass vs Fail Ratio"
        ),
        use_container_width=True
    )

# ==================================================
# TAB 2: FACULTY ML ONLY
# ==================================================
with tab2:
    st.header("🤖 Faculty – Predictive Model")

    if st.session_state.role != "Faculty":
        st.warning("Faculty access only")
        st.stop()

    if st.button("Train & Evaluate Model"):
        model_df = pd.get_dummies(df, drop_first=True)
        X = model_df.drop("G3", axis=1)
        y = model_df["G3"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        with st.spinner("Training Random Forest model..."):
            model = RandomForestRegressor(
                n_estimators=150,
                random_state=42
            )
            model.fit(X_train, y_train)

        rmse = math.sqrt(
            mean_squared_error(y_test, model.predict(X_test))
        )

        st.session_state.model = model
        st.session_state.rmse = rmse

        st.success("Model trained successfully ✅")
        st.metric("RMSE", f"{rmse:.2f}")

# ==================================================
# TAB 3: INSIGHTS
# ==================================================
with tab3:
    st.header("📈 Academic Insights")

    st.plotly_chart(
        px.box(
            df,
            x="sex",
            y="G3",
            title="Gender vs Performance"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(
            df,
            x="studytime",
            y="G3",
            trendline="ols",
            title="Study Time vs Final Grade"
        ),
        use_container_width=True
    )

    corr = df.corr(numeric_only=True)
    st.plotly_chart(
        px.imshow(
            corr,
            text_auto=True,
            title="Correlation Heatmap"
        ),
        use_container_width=True
    )
