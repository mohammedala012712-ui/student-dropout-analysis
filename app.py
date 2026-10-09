
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(
    page_title="Student Behavior Analysis",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Dropout Behavior Analysis")
st.caption(
    "Explore student data, discover groups, and analyze academic performance."
)

# =========================
# LOAD DATA
# =========================
st.sidebar.header("Data")
data_path = Path(__file__).parent / "student_dropout_behavior_dataset.csv"

uploaded_file = st.sidebar.file_uploader(
    "Upload your student CSV file",
    type=["csv"]
)

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif data_path.exists():
        df = pd.read_csv(data_path)
    else:
        st.error(
            "Dataset not found. Put student_dropout_behavior_dataset.csv "
            "in the same folder as app.py, or upload it from the sidebar."
        )
        st.stop()
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()

# =========================
# DATA OVERVIEW
# =========================
st.header("1. Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Students", df.shape[0])
c2.metric("Total Columns", df.shape[1])
c3.metric("Columns With Missing Values", int(df.isnull().any().sum()))

with st.expander("Preview Dataset"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Missing Values"):
    missing = df.isnull().sum().to_frame("Missing Values")
    st.dataframe(missing[missing["Missing Values"] > 0])

with st.expander("Statistical Summary"):
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

# =========================
# FEATURE SELECTION
# =========================
st.header("2. Choose Features")

default_features = [
    "quiz1_marks",
    "quiz2_marks",
    "quiz3_marks",
    "total_assignments",
    "midterm_marks",
    "final_marks",
    "previous_gpa",
    "lectures_attended",
    "labs_attended"
]

features = [
    col for col in default_features
    if col in df.columns
    and pd.api.types.is_numeric_dtype(df[col])
    and df[col].notna().any()
]

if len(features) < 2:
    st.error("The dataset needs at least two usable numeric features.")
    st.stop()

selected_features = st.multiselect(
    "Select the features used for clustering",
    options=features,
    default=features
)

if len(selected_features) < 2:
    st.warning("Select at least two features.")
    st.stop()

X = df[selected_features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

# Keep rows with at least one valid selected value
valid_rows = X.notna().any(axis=1)
X = X.loc[valid_rows].copy()
result_base = df.loc[valid_rows].copy()

# Remove entirely empty features, then fill remaining missing values
X = X.dropna(axis=1, how="all")

if X.shape[1] < 2:
    st.error("At least two usable features are required.")
    st.stop()

X = X.fillna(X.median())

if len(X) < 3:
    st.error("Not enough valid rows for clustering.")
    st.stop()

# =========================
# CLUSTERING OPTIONS
# =========================
st.header("3. Run K-Means Analysis")

max_k = min(10, len(X) - 1)
k = st.slider(
    "Number of clusters (K)",
    min_value=2,
    max_value=max_k,
    value=min(5, max_k)
)

if st.button("Run Student Analysis", type="primary"):
    with st.spinner("Analyzing the data..."):

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compare K values
        score_rows = []
        inertia_rows = []

        for candidate_k in range(2, min(6, max_k) + 1):
            test_model = KMeans(
                n_clusters=candidate_k,
                random_state=42,
                n_init=10
            )
            test_labels = test_model.fit_predict(X_scaled)

            score_rows.append({
                "K": candidate_k,
                "Silhouette Score": silhouette_score(
                    X_scaled, test_labels
                )
            })

            inertia_rows.append({
                "K": candidate_k,
                "Inertia": test_model.inertia_
            })

        # Final selected model
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(X_scaled)

        result = result_base.copy()
        result["Cluster"] = labels

        # Average of available academic score columns
        score_columns = [
            col for col in [
                "quiz1_marks",
                "quiz2_marks",
                "quiz3_marks",
                "midterm_marks",
                "final_marks"
            ]
            if col in result.columns
        ]

        if score_columns:
            result["Average_Performance"] = result[score_columns].mean(axis=1)

        summary_columns = list(X.columns)
        summary = result.groupby("Cluster")[summary_columns].mean()

        if "Average_Performance" in result.columns:
            summary["Average_Performance"] = (
                result.groupby("Cluster")["Average_Performance"].mean()
            )

        # PCA visualization
        pca = PCA(n_components=2)
        pca_values = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            "PCA 1": pca_values[:, 0],
            "PCA 2": pca_values[:, 1],
            "Cluster": labels.astype(str)
        })

        # Store results between Streamlit reruns
        st.session_state["analysis"] = {
            "result": result,
            "summary": summary,
            "scores": pd.DataFrame(score_rows),
            "inertia": pd.DataFrame(inertia_rows),
            "pca": pca_df,
            "silhouette": silhouette_score(X_scaled, labels),
            "variance": pca.explained_variance_ratio_
        }

# =========================
# DISPLAY RESULTS
# =========================
if "analysis" in st.session_state:
    analysis = st.session_state["analysis"]

    result = analysis["result"]
    summary = analysis["summary"]
    scores = analysis["scores"]
    inertia = analysis["inertia"]
    pca_df = analysis["pca"]

    st.success("Analysis completed successfully!")

    a, b, c = st.columns(3)
    a.metric("Students Analyzed", len(result))
    b.metric("Number of Clusters", result["Cluster"].nunique())
    c.metric("Silhouette Score", f'{analysis["silhouette"]:.4f}')

    # Cluster sizes
    st.header("4. Student Group Distribution")

    counts = result["Cluster"].value_counts().sort_index()
    st.bar_chart(counts)

    st.subheader("PCA Visualization")

    fig, ax = plt.subplots(figsize=(9, 5))
    for cluster in sorted(pca_df["Cluster"].unique()):
        part = pca_df[pca_df["Cluster"] == cluster]
        ax.scatter(
            part["PCA 1"],
            part["PCA 2"],
            label=f"Cluster {cluster}",
            alpha=0.7
        )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Student Groups")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The first two PCA components explain "
        f'{analysis["variance"].sum() * 100:.2f}% of the variance.'
    )

    # Evaluation charts
    st.header("5. Model Evaluation")

    left, right = st.columns(2)

    with left:
        st.subheader("Silhouette Scores")
        st.dataframe(scores, use_container_width=True)
        st.bar_chart(scores.set_index("K")["Silhouette Score"])

    with right:
        st.subheader("Elbow Method")
        st.line_chart(inertia.set_index("K")["Inertia"])

    # Summary table
    st.header("6. Cluster Summary")
    st.dataframe(summary.round(2), use_container_width=True)

    # Full student results
    st.header("7. Student Results")
    st.dataframe(result, use_container_width=True)

    # Downloads
    st.header("8. Download Results")

    st.download_button(
        "Download Student Results CSV",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="student_dropout_clusters.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Cluster Summary CSV",
        data=summary.to_csv().encode("utf-8-sig"),
        file_name="cluster_summary.csv",
        mime="text/csv"
    )

    st.info(
        "This application groups students with similar characteristics. "
        "It does not directly predict dropout, because the dataset does "
        "not contain a verified dropout outcome."
    )
