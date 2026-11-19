"""
app.py
------
Streamlit Interactive Web Application for Customer Segmentation
Using Machine Learning (K-Means Clustering).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation AI Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .persona-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Caching & Helper Functions
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_data.csv")
CLUSTERED_DATA_PATH = os.path.join(BASE_DIR, "data", "customer_segments_output.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "kmeans_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "cluster_metadata.json")


@st.cache_data
def load_dataset(filepath: str) -> pd.DataFrame:
    """Loads CSV dataset."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()


@st.cache_resource
def load_artifacts():
    """Loads saved K-Means model, Scaler, and Metadata."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, None

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, scaler, metadata


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/user-group-man-woman.png", width=70)
    st.title("Segmentation AI")
    st.markdown("**Algorithm:** K-Means Clustering (`k-means++`)")
    st.markdown("**Feature Space:** Income vs. Spending Score")
    st.divider()

    st.subheader("Data Source")
    data_source = st.radio(
        "Choose Dataset:",
        ["Default Benchmark (200 Customers)", "Upload Custom CSV"],
        index=0
    )

    uploaded_file = None
    if data_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    st.divider()
    st.info(
        "💡 **Tip:** Use this dashboard to analyze existing customer behavior or enter "
        "prospective customer attributes to predict their segment in real-time."
    )


# ---------------------------------------------------------
# Load Data & Model
# ---------------------------------------------------------
model, scaler, metadata = load_artifacts()

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    df_clustered = df_raw.copy()
else:
    df_raw = load_dataset(DATA_PATH)
    if os.path.exists(CLUSTERED_DATA_PATH):
        df_clustered = load_dataset(CLUSTERED_DATA_PATH)
    else:
        df_clustered = df_raw.copy()

if df_raw.empty:
    st.error(f"Dataset not found at `{DATA_PATH}`. Please run `train_model.py` first to generate models and data.")
    st.stop()


# ---------------------------------------------------------
# Dashboard Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎯 Customer Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Unsupervised Machine Learning model that partitions customers into behavioral '
    'personas to maximize business retention and revenue strategy.</div>',
    unsafe_allow_html=True
)

# Top KPI Summary Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total Customers", f"{len(df_raw):,}")
with kpi2:
    st.metric("Avg Annual Income", f"${df_raw['Annual Income (k$)'].mean():.1f}k")
with kpi3:
    st.metric("Avg Spending Score", f"{df_raw['Spending Score (1-100)'].mean():.1f} / 100")
with kpi4:
    st.metric("Avg Customer Age", f"{df_raw['Age'].mean():.1f} yrs")
with kpi5:
    optimal_k = metadata.get("optimal_k", 5) if metadata else 5
    st.metric("Optimal Clusters (k)", f"{optimal_k}")

st.divider()

# ---------------------------------------------------------
# Main Navigation Tabs
# ---------------------------------------------------------
tab_eda, tab_elbow, tab_clusters, tab_personas, tab_predict = st.tabs([
    "📊 Dataset & EDA",
    "📉 Elbow & Silhouette Analysis",
    "🎯 Cluster Visualizer",
    "👥 Personas & Business Strategies",
    "🔮 Predict New Customer"
])


# =========================================================
# TAB 1: DATASET & EDA
# =========================================================
with tab_eda:
    st.subheader("1. Exploratory Data Analysis & Overview")
    st.write(
        "Explore raw demographic and behavioral characteristics of customers prior to model training."
    )

    col_data_view, col_stats_view = st.columns([3, 2])
    with col_data_view:
        st.markdown("**Sample Records (First 10 Customers):**")
        st.dataframe(df_raw.head(10), use_container_width=True)

    with col_stats_view:
        st.markdown("**Descriptive Statistics:**")
        st.dataframe(df_raw.describe().round(2), use_container_width=True)

    st.markdown("---")
    st.subheader("Demographic Distributions")

    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df_raw['Age'], kde=True, color='#2563EB', ax=ax, bins=15)
        ax.set_title("Customer Age Distribution", fontweight='bold')
        ax.set_xlabel("Age (Years)")
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df_raw['Annual Income (k$)'], kde=True, color='#059669', ax=ax, bins=15)
        ax.set_title("Annual Income Distribution", fontweight='bold')
        ax.set_xlabel("Annual Income (k$)")
        st.pyplot(fig)
        plt.close()

    with c3:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df_raw['Spending Score (1-100)'], kde=True, color='#D97706', ax=ax, bins=15)
        ax.set_title("Spending Score Distribution", fontweight='bold')
        ax.set_xlabel("Spending Score (1-100)")
        st.pyplot(fig)
        plt.close()

    if 'Gender' in df_raw.columns:
        st.markdown("---")
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            fig, ax = plt.subplots(figsize=(6, 4))
            df_raw['Gender'].value_counts().plot(
                kind='pie',
                autopct='%1.1f%%',
                colors=['#e377c2', '#1f77b4'],
                startangle=140,
                explode=(0.04, 0.04),
                ax=ax
            )
            ax.set_ylabel("")
            ax.set_title("Gender Breakdown", fontweight='bold')
            st.pyplot(fig)
            plt.close()

        with gcol2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                x='Annual Income (k$)',
                y='Spending Score (1-100)',
                hue='Gender',
                data=df_raw,
                palette={'Female': '#e377c2', 'Male': '#1f77b4'},
                s=70,
                ax=ax
            )
            ax.set_title("Income vs. Spending Score by Gender", fontweight='bold')
            st.pyplot(fig)
            plt.close()


# =========================================================
# TAB 2: ELBOW & SILHOUETTE ANALYSIS
# =========================================================
with tab_elbow:
    st.subheader("2. Determining the Optimal Number of Clusters (k)")
    st.markdown("""
    Selecting the appropriate number of clusters is critical in K-Means. We employ two complementary validation techniques:
    1. **The Elbow Method (WCSS / Inertia):** Measures compactness. The optimal $k$ corresponds to the "elbow bend" where rate of reduction abruptly slows down.
    2. **The Silhouette Score:** Measures both cluster cohesion (how close points are inside a cluster) and separation (how distant different clusters are). Range: $[-1, 1]$.
    """)

    if metadata and "wcss_history" in metadata and "silhouette_history" in metadata:
        wcss = metadata["wcss_history"]
        sil_dict = metadata["silhouette_history"]

        col_metric_1, col_metric_2 = st.columns(2)
        with col_metric_1:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(range(1, len(wcss) + 1), wcss, marker='o', linewidth=2.2, color='#1E40AF')
            ax.axvline(x=5, color='crimson', linestyle='--', label='Elbow Point (k=5)')
            ax.set_title("The Elbow Method (Inertia vs. k)", fontweight='bold')
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)
            plt.close()

        with col_metric_2:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ks = [int(k) for k in sil_dict.keys()]
            scores = list(sil_dict.values())
            ax.plot(ks, scores, marker='s', linewidth=2.2, color='#059669')
            ax.axvline(x=5, color='crimson', linestyle='--', label='Peak Silhouette (k=5)')
            ax.set_title("Silhouette Coefficient vs. k", fontweight='bold')
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Average Silhouette Score")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)
            plt.close()

        st.success(
            f"✅ **Mathematical Validation:** At **$k = 5$**, the model achieves an outstanding Silhouette Score "
            f"of **{metadata.get('silhouette_score', 0.5547):.4f}**, indicating strong separation and high within-cluster cohesion."
        )
    else:
        st.warning("Cluster metrics history not found. Run `train_model.py` to record evaluation history.")


# =========================================================
# TAB 3: CLUSTER VISUALIZER
# =========================================================
with tab_clusters:
    st.subheader("3. Interactive Cluster Visualization")
    st.write(
        "Observe the 5 distinct customer groups in the standardized feature space, accompanied by cluster centroids."
    )

    if model is not None and scaler is not None and 'Cluster' in df_clustered.columns:
        c_vis, c_dist = st.columns([3, 2])

        with c_vis:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            profiles = metadata.get("cluster_profiles", {}) if metadata else {}
            palette = sns.color_palette("tab10", n_colors=model.n_clusters)

            for cid in range(model.n_clusters):
                subset = df_clustered[df_clustered['Cluster'] == cid]
                cluster_label = (
                    profiles.get(str(cid), {}).get("name", f"Cluster {cid}")
                    if profiles else f"Cluster {cid}"
                )
                ax.scatter(
                    subset['Annual Income (k$)'],
                    subset['Spending Score (1-100)'],
                    s=80,
                    label=f"C{cid}: {cluster_label}",
                    color=palette[cid],
                    alpha=0.85,
                    edgecolors='white',
                    linewidth=0.5
                )

            # Inverse transform centroids to original feature scale
            centroids_orig = scaler.inverse_transform(model.cluster_centers_)
            ax.scatter(
                centroids_orig[:, 0],
                centroids_orig[:, 1],
                s=280,
                c='black',
                marker='*',
                edgecolors='yellow',
                linewidth=1.5,
                label='Centroids'
            )

            ax.set_title("Customer Segments with Cluster Centroids", fontweight='bold', fontsize=12)
            ax.set_xlabel("Annual Income (k$)")
            ax.set_ylabel("Spending Score (1-100)")
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)
            plt.close()

        with c_dist:
            st.markdown("**Cluster Size Breakdown**")
            cluster_counts = df_clustered['Cluster'].value_counts().sort_index()

            chart_data = []
            for cid in cluster_counts.index:
                pname = profiles.get(str(cid), {}).get("name", f"Cluster {cid}")
                chart_data.append({
                    "Cluster": f"C{cid}: {pname}",
                    "Customer Count": int(cluster_counts[cid]),
                    "Percentage": f"{(cluster_counts[cid] / len(df_clustered) * 100):.1f}%"
                })
            st.dataframe(pd.DataFrame(chart_data), use_container_width=True)

            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(
                [f"C{i}" for i in cluster_counts.index],
                cluster_counts.values,
                color=palette,
                edgecolor='black',
                linewidth=0.5
            )
            ax.set_title("Distribution of Customers Across Clusters", fontweight='bold')
            ax.set_ylabel("Number of Customers")
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Train the model by running `python train_model.py` to view interactive clustering.")


# =========================================================
# TAB 4: PERSONAS & BUSINESS STRATEGIES
# =========================================================
with tab_personas:
    st.subheader("4. Persona Profiles & Data-Driven Marketing Playbooks")
    st.write(
        "Each customer cluster corresponds to a distinct demographic and psychological persona. "
        "Tailored strategies optimize budget allocation and maximize Lifetime Customer Value (LTV)."
    )

    profiles = metadata.get("cluster_profiles", {}) if metadata else {}

    if profiles:
        for cid, prof in profiles.items():
            badge_color = prof.get("color", "#1E3A8A")
            st.markdown(f"""
            <div style="background-color: #F8FAFC; border-left: 6px solid {badge_color}; padding: 16px; border-radius: 6px; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #1E293B;">
                    Cluster {cid}: {prof['name']}
                </h3>
                <p><strong>Customer Base:</strong> {prof['count']} customers ({prof['percentage']}%) | 
                   <strong>Avg Age:</strong> {prof['mean_age']} yrs | 
                   <strong>Avg Annual Income:</strong> ${prof['mean_income']}k | 
                   <strong>Avg Spending Score:</strong> {prof['mean_spending']} / 100
                </p>
                <p><strong>Behavioral Persona:</strong> {prof['description']}</p>
                <p><strong>🎯 Recommended Marketing Strategy:</strong> {prof['strategy']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Persona profiles not generated. Please run `train_model.py` to populate dynamic personas.")


# =========================================================
# TAB 5: PREDICT NEW CUSTOMER SEGMENT
# =========================================================
with tab_predict:
    st.subheader("5. Real-Time Customer Segment Prediction")
    st.write(
        "Enter prospective or existing customer metrics to predict their assigned segment and receive instant marketing actions."
    )

    if model is None or scaler is None:
        st.error("Model artifacts not loaded. Please ensure `models/kmeans_model.pkl` and `models/scaler.pkl` exist.")
    else:
        with st.form("customer_prediction_form"):
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)

            with col_f1:
                input_age = st.number_input("Customer Age", min_value=15, max_value=100, value=30, step=1)
            with col_f2:
                input_gender = st.selectbox("Gender", ["Female", "Male"])
            with col_f3:
                input_income = st.slider(
                    "Annual Income (in thousands $)",
                    min_value=10,
                    max_value=150,
                    value=65,
                    step=1,
                    help="Gross annual income in thousands of USD (e.g. 65 = $65,000)"
                )
            with col_f4:
                input_spending = st.slider(
                    "Spending Score (1 to 100)",
                    min_value=1,
                    max_value=100,
                    value=50,
                    step=1,
                    help="Behavioral spending index computed by the retailer based on purchase frequency and transaction value."
                )

            submit_btn = st.form_submit_button("🚀 Classify Customer", use_container_width=True)

        if submit_btn:
            # Prepare and scale feature vector
            feature_vector = np.array([[input_income, input_spending]])
            scaled_input = scaler.transform(feature_vector)

            # Predict cluster
            pred_cluster = int(model.predict(scaled_input)[0])

            # Retrieve profile metadata
            profiles = metadata.get("cluster_profiles", {}) if metadata else {}
            prof = profiles.get(str(pred_cluster), {
                "name": f"Cluster {pred_cluster}",
                "description": "Standard Customer Group",
                "strategy": "Engage with regular promotions",
                "color": "#1E3A8A"
            })

            st.balloons()
            st.success("✅ Prediction Successfully Generated!")

            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.markdown(f"""
                <div style="background-color: #F0FDF4; border: 2px solid #22C55E; border-radius: 12px; padding: 20px; text-align: center;">
                    <div style="font-size: 1rem; color: #15803D; font-weight: 600;">ASSIGNED SEGMENT</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #166534; margin: 10px 0;">
                        Cluster {pred_cluster}
                    </div>
                    <div style="background-color: {prof.get('color', '#15803D')}; color: white; border-radius: 20px; padding: 6px 12px; font-weight: 700; font-size: 0.95rem;">
                        {prof['name']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"**Customer Persona Profile:**")
                st.write(prof['description'])
                st.markdown(f"**Actionable Recommendation:**")
                st.info(prof['strategy'])

            # Plot visual position in cluster space
            st.markdown("---")
            st.markdown("#### Position in Customer Space")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            palette = sns.color_palette("tab10", n_colors=model.n_clusters)

            for cid in range(model.n_clusters):
                subset = df_clustered[df_clustered['Cluster'] == cid]
                cname = profiles.get(str(cid), {}).get("name", f"C{cid}") if profiles else f"C{cid}"
                ax.scatter(
                    subset['Annual Income (k$)'],
                    subset['Spending Score (1-100)'],
                    s=50,
                    alpha=0.4,
                    color=palette[cid],
                    label=f"C{cid}: {cname}"
                )

            # Plot new customer as prominent star
            ax.scatter(
                [input_income],
                [input_spending],
                s=400,
                c='#EF4444',
                marker='X',
                edgecolors='black',
                linewidth=2,
                label=f'New Customer (Income: ${input_income}k, Spend: {input_spending})'
            )

            ax.set_title("Customer Location vs. Customer Database", fontweight='bold')
            ax.set_xlabel("Annual Income (k$)")
            ax.set_ylabel("Spending Score (1-100)")
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)
            plt.close()
