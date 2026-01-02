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
for key, val in {
    "logged_in": False,
    "role": None,
    "data": None,
    "model": None,
    "rmse": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --------------------------------------------------
# LOGIN UI
# --------------------------------------------------
def login_ui():
    st.title("🔐 College Portal Login")
    role = st.selectbox("Login as", ["Student", "Faculty"])
    if st.button("Login"):
        st.session_state.logged_in = True
        st.session_state.role = role
        st.rerun()

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
        st.rerun()

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    r = requests.get(url, timeout=10)
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        return pd.read_csv(z.open("student-mat.csv"), sep=";")

if st.session_state.data is None:
    with st.spinner("Loading academic data..."):
        st.session_state.data = load_data()

df = st.session_state.data

# --------------------------------------------------
# DERIVED COLUMNS (ANALYTICS READY)
# --------------------------------------------------
df["Result"] = df["G3"].apply(lambda x: "Pass" if x >= 10 else "Fail")
df["Attendance_Level"] = pd.cut(
    df["absences"],
    bins=[-1, 5, 15, 100],
    labels=["High", "Medium", "Low"]
)

# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Student Dashboard", "👨‍🏫 Faculty Analytics", "📈 Insights"]
)

# ==================================================
# TAB 1: STUDENT DASHBOARD (INFORMATIVE)
# ==================================================
with tab1:
    st.header("📊 Student Academic Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Final Grade", f"{df['G3'].mean():.2f}")
    col2.metric("Pass Percentage", f"{(df['Result']=='Pass').mean()*100:.1f}%")
    col3.metric("Avg Absences", f"{df['absences'].mean():.1f}")

    st.plotly_chart(
        px.histogram(df, x="G3", nbins=20,
                     title="Final Grade Distribution"),
        use_container_width=True
    )

    st.plotly_chart(
        px.bar(
            df.groupby("Attendance_Level")["G3"].mean().reset_index(),
            x="Attendance_Level",
            y="G3",
            title="Average Grade by Attendance Level"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.pie(
            df["Result"].value_counts().reset_index(),
            names="index",
            values="Result",
            title="Pass vs Fail Ratio"
        ),
        use_container_width=True
    )

# ==================================================
# TAB 2: FACULTY ANALYTICS + ML
# ==================================================
with tab2:
    st.header("👨‍🏫 Faculty Analytics & Predictive Tools")

    if st.session_state.role != "Faculty":
        st.warning("Faculty access only")
        st.stop()

    st.subheader("📌 Academic Performance Patterns")

    st.plotly_chart(
        px.box(df, x="sex", y="G3",
               title="Gender vs Academic Performance"),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(df, x="studytime", y="G3",
                   trendline="ols",
                   title="Study Time vs Final Grade"),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(df, x="absences", y="G3",
                   trendline="ols",
                   title="Absences Impact on Performance"),
        use_container_width=True
    )

    st.divider()
    st.subheader("🤖 Predictive Model (At-Risk Students)")

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

        st.success(f"Model RMSE: {rmse:.2f}")

        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)

        st.subheader("🔍 Top Factors Affecting Performance")
        st.plotly_chart(
            px.bar(
                feature_importance,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 10 Important Features"
            ),
            use_container_width=True
        )

# ==================================================
# TAB 3: INSTITUTIONAL INSIGHTS
# ==================================================
with tab3:
    st.header("📈 Institutional Insights")

    corr = df.corr(numeric_only=True)
    st.plotly_chart(
        px.imshow(
            corr,
            text_auto=True,
            title="Correlation Heatmap (Academic Factors)"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.violin(
            df,
            x="schoolsup",
            y="G3",
            box=True,
            title="School Support vs Final Grade"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.violin(
            df,
            x="famsup",
            y="G3",
            box=True,
            title="Family Support vs Final Grade"
        ),
        use_container_width=True
    )
