"""
train_model.py
--------------
Trains the K-Means clustering model for Customer Segmentation.
Computes Elbow Method (WCSS) and Silhouette Scores, fits the model,
dynamically derives persona labels from cluster characteristics,
saves trained artifacts, and exports publication-ready visualizations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preprocessing import (
    load_data,
    clean_data,
    select_and_scale_features,
    save_scaler
)


def compute_clustering_metrics(scaled_data: np.ndarray, max_k: int = 10) -> tuple:
    """
    Computes Within-Cluster Sum of Squares (WCSS / Inertia) and Silhouette Scores
    for k in range 1 to max_k.

    Parameters:
        scaled_data (np.ndarray): Scaled feature matrix.
        max_k (int): Maximum number of clusters to evaluate.

    Returns:
        tuple: (wcss_list: list, silhouette_scores: dict)
    """
    wcss = []
    silhouette_dict = {}

    print("\n--- Evaluating Number of Clusters (k) ---")
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

        if k >= 2:
            score = silhouette_score(scaled_data, kmeans.labels_)
            silhouette_dict[k] = float(score)
            print(f"k = {k:2d} | WCSS (Inertia): {kmeans.inertia_:10.2f} | Silhouette Score: {score:.4f}")
        else:
            print(f"k = {k:2d} | WCSS (Inertia): {kmeans.inertia_:10.2f} | Silhouette Score: N/A")

    return wcss, silhouette_dict


def plot_and_save_visualizations(
    df: pd.DataFrame,
    scaled_data: np.ndarray,
    wcss: list,
    silhouette_dict: dict,
    kmeans_model: KMeans,
    feature_cols: list,
    cluster_profiles: dict,
    output_dir: str
) -> None:
    """
    Generates and saves exploratory and clustering visual charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Exploratory Distribution Charts
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(df['Age'], kde=True, color='teal', bins=15)
    plt.title("Customer Age Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Age")

    plt.subplot(1, 3, 2)
    sns.histplot(df['Annual Income (k$)'], kde=True, color='crimson', bins=15)
    plt.title("Annual Income Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Annual Income (k$)")

    plt.subplot(1, 3, 3)
    sns.histplot(df['Spending Score (1-100)'], kde=True, color='darkorange', bins=15)
    plt.title("Spending Score Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Spending Score (1-100)")

    plt.tight_layout()
    dist_path = os.path.join(output_dir, "distribution_charts.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {dist_path}")

    # 2. Raw Scatter: Income vs Spending Score
    plt.figure(figsize=(8, 6))
    if 'Gender' in df.columns:
        sns.scatterplot(
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            hue='Gender',
            data=df,
            palette={'Male': '#1f77b4', 'Female': '#e377c2'},
            s=80,
            alpha=0.85
        )
    else:
        plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], color='steelblue', s=70)
    plt.title("Annual Income vs Spending Score (Raw Data)", fontsize=13, fontweight='bold')
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    raw_path = os.path.join(output_dir, "income_vs_spending_raw.png")
    plt.savefig(raw_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {raw_path}")

    # 3. Elbow Method & Silhouette Score
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, len(wcss) + 1), wcss, marker='o', linewidth=2.5, color='#2b5c8f')
    ax1.set_title("The Elbow Method (WCSS vs k)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax1.axvline(x=5, color='red', linestyle='--', label='Optimal k=5 (Elbow Point)')
    ax1.legend()

    ks = list(silhouette_dict.keys())
    scores = list(silhouette_dict.values())
    ax2.plot(ks, scores, marker='s', linewidth=2.5, color='#27ae60')
    ax2.set_title("Silhouette Score vs k", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Average Silhouette Score")
    ax2.axvline(x=5, color='red', linestyle='--', label='Peak k=5 (Best Separation)')
    ax2.legend()

    plt.tight_layout()
    eval_path = os.path.join(output_dir, "elbow_silhouette_analysis.png")
    plt.savefig(eval_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {eval_path}")

    # 4. Final Customer Clusters Scatter with Centroids
    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab10", n_colors=kmeans_model.n_clusters)

    for cluster_id in range(kmeans_model.n_clusters):
        subset = df[df['Cluster'] == cluster_id]
        label = cluster_profiles[cluster_id]['name']
        plt.scatter(
            subset[feature_cols[0]],
            subset[feature_cols[1]],
            s=80,
            label=f"Cluster {cluster_id}: {label}",
            color=palette[cluster_id],
            edgecolor='black',
            linewidth=0.5
        )

    # Inverse transform centroids to original feature scale for intuitive visualization
    scaler = joblib.load(os.path.join(os.path.dirname(output_dir), "models", "scaler.pkl"))
    centroids_orig = scaler.inverse_transform(kmeans_model.cluster_centers_)

    plt.scatter(
        centroids_orig[:, 0],
        centroids_orig[:, 1],
        s=300,
        c='black',
        marker='*',
        edgecolor='yellow',
        linewidth=1.5,
        label='Centroids'
    )

    plt.title("Customer Segments Identified by K-Means", fontsize=14, fontweight='bold')
    plt.xlabel(f"{feature_cols[0]}")
    plt.ylabel(f"{feature_cols[1]}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    cluster_path = os.path.join(output_dir, "customer_clusters.png")
    plt.savefig(cluster_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {cluster_path}")


def dynamic_persona_tagger(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Dynamically identifies customer personas for each cluster based on
    actual statistical characteristics (mean income, mean spending score, and mean age),
    instead of assuming fixed cluster indices.

    Criteria:
    - High Income & High Spending Score -> High-Value / Premium Customers
    - Low Income & High Spending Score -> Potential / Aspirational Spenders
    - High Income & Low Spending Score -> Cautious / Frugal Savers
    - Low Income & Low Spending Score -> Budget / Low-Spending Customers
    - Moderate Income & Moderate Spending Score -> Average / Balanced Customers
    """
    overall_inc_median = df[feature_cols[0]].median()
    overall_spd_median = df[feature_cols[1]].median()

    cluster_profiles = {}
    grouped = df.groupby('Cluster')

    for cluster_id, group in grouped:
        mean_inc = float(group[feature_cols[0]].mean())
        mean_spd = float(group[feature_cols[1]].mean())
        mean_age = float(group['Age'].mean()) if 'Age' in group.columns else 0.0
        count = int(len(group))
        percentage = round((count / len(df)) * 100, 2)

        is_high_income = mean_inc > (overall_inc_median * 1.1)
        is_low_income = mean_inc < (overall_inc_median * 0.9)
        is_high_spend = mean_spd > (overall_spd_median * 1.1)
        is_low_spend = mean_spd < (overall_spd_median * 0.9)

        if is_high_income and is_high_spend:
            persona_name = "High-Value / Premium Customers"
            description = (
                "High income earners who actively spend. Loyal, brand-conscious, and value premium quality over discounts."
            )
            strategy = (
                "Target with VIP loyalty rewards, exclusive early access to product launches, personalized concierge service, and premium experiences."
            )
            color = "#2ca02c"
        elif is_low_income and is_high_spend:
            persona_name = "Potential / Aspirational Spenders"
            description = (
                "Lower to moderate income earners with very high shopping enthusiasm. Tend to be trend-driven and impulsive buyers."
            )
            strategy = (
                "Offer flexible installment/EMI plans (BNPL), trendy micro-deals, engaging flash sales, and rewards programs to build sustainable loyalty."
            )
            color = "#1f77b4"
        elif is_high_income and is_low_spend:
            persona_name = "Cautious Savers / Frugal Affluent"
            description = (
                "Substantial income but very disciplined expenditure. Cautious decision makers who calculate ROI before buying."
            )
            strategy = (
                "Send value-focused promotional campaigns, emphasize durability and warranties, provide member-only high-tier cashbacks, and highlight investment value."
            )
            color = "#ff7f0e"
        elif is_low_income and is_low_spend:
            persona_name = "Budget / Low-Spending Customers"
            description = (
                "Modest financial capacity with strictly restricted purchasing habits. Highly price-sensitive."
            )
            strategy = (
                "Promote clearance discounts, bundle bargains, coupon codes, and essential everyday items with lowest price guarantees."
            )
            color = "#d62728"
        else:
            persona_name = "Average / Balanced Customers"
            description = (
                "Middle-class income bracket with moderate, steady purchasing behavior. Represents the bulk of everyday steady revenue."
            )
            strategy = (
                "Retain with seasonal promotions, cross-selling popular bestsellers, standard membership perks, and personalized recommendations."
            )
            color = "#9467bd"

        cluster_profiles[cluster_id] = {
            "cluster_id": cluster_id,
            "name": persona_name,
            "count": count,
            "percentage": percentage,
            "mean_income": round(mean_inc, 2),
            "mean_spending": round(mean_spd, 2),
            "mean_age": round(mean_age, 1),
            "description": description,
            "strategy": strategy,
            "color": color
        }

    return cluster_profiles


def train_and_save_model(
    data_path: str,
    model_output_dir: str,
    viz_output_dir: str,
    optimal_k: int = 5
) -> dict:
    """
    Main training workflow:
    - Loads & cleans dataset
    - Scales features
    - Computes metrics (Elbow + Silhouette)
    - Fits K-Means with optimal k
    - Dynamically generates cluster personas
    - Saves model artifacts and visual charts
    """
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(viz_output_dir, exist_ok=True)

    # 1. Preprocess
    df = load_data(data_path)
    df_clean = clean_data(df)
    feature_cols = ['Annual Income (k$)', 'Spending Score (1-100)']

    scaler_path = os.path.join(model_output_dir, "scaler.pkl")
    scaled_data, scaler, _ = select_and_scale_features(df_clean, feature_cols=feature_cols)
    save_scaler(scaler, scaler_path)

    # 2. Evaluate metrics
    wcss, silhouette_dict = compute_clustering_metrics(scaled_data, max_k=10)

    # 3. Train K-Means
    print(f"\n[INFO] Training K-Means model with optimal k={optimal_k}...")
    kmeans = KMeans(
        n_clusters=optimal_k,
        init='k-means++',
        n_init=15,
        max_iter=300,
        random_state=42
    )
    cluster_labels = kmeans.fit_predict(scaled_data)
    df_clean['Cluster'] = cluster_labels

    final_silhouette = float(silhouette_score(scaled_data, cluster_labels))
    print(f"[SUCCESS] Model fitted. Overall Silhouette Score: {final_silhouette:.4f}")

    # 4. Save trained K-Means model
    model_path = os.path.join(model_output_dir, "kmeans_model.pkl")
    joblib.dump(kmeans, model_path)
    print(f"[INFO] KMeans model saved to: {model_path}")

    # 5. Dynamic Segment Profiling
    cluster_profiles = dynamic_persona_tagger(df_clean, feature_cols)

    # Add persona label column to dataframe and export processed dataset
    df_clean['Segment_Name'] = df_clean['Cluster'].map(
        lambda c: cluster_profiles[c]['name']
    )
    processed_csv_path = os.path.join(viz_output_dir, "customer_segments_output.csv")
    df_clean.to_csv(processed_csv_path, index=False)
    print(f"[INFO] Clustered dataset saved to: {processed_csv_path}")

    # 6. Save metadata for Streamlit app
    metadata = {
        "features": feature_cols,
        "optimal_k": optimal_k,
        "silhouette_score": final_silhouette,
        "wcss_inertia": float(kmeans.inertia_),
        "wcss_history": wcss,
        "silhouette_history": silhouette_dict,
        "cluster_profiles": cluster_profiles
    }
    metadata_path = os.path.join(model_output_dir, "cluster_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"[INFO] Cluster metadata saved to: {metadata_path}")

    # 7. Generate and export charts
    plot_and_save_visualizations(
        df=df_clean,
        scaled_data=scaled_data,
        wcss=wcss,
        silhouette_dict=silhouette_dict,
        kmeans_model=kmeans,
        feature_cols=feature_cols,
        cluster_profiles=cluster_profiles,
        output_dir=viz_output_dir
    )

    print("\n================== CLUSTERING SUMMARY ==================")
    for cid, prof in cluster_profiles.items():
        print(f"\n[Cluster {cid}] - {prof['name']}")
        print(f"  Count: {prof['count']} ({prof['percentage']}%) | Avg Age: {prof['mean_age']}")
        print(f"  Avg Annual Income: ${prof['mean_income']}k | Avg Spending Score: {prof['mean_spending']}/100")
        print(f"  Business Strategy: {prof['strategy']}")
    print("========================================================\n")

    return metadata


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_csv = os.path.join(base_dir, "data", "customer_data.csv")
    models_dir = os.path.join(base_dir, "models")
    viz_dir = os.path.join(base_dir, "data")

    train_and_save_model(
        data_path=data_csv,
        model_output_dir=models_dir,
        viz_output_dir=viz_dir,
        optimal_k=5
    )
