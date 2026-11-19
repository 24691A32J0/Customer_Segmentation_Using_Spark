# Customer Segmentation Using Machine Learning with K-Means Clustering

An intermediate-level Machine Learning and Data Science project that groups retail customers into actionable behavioral segments based on annual income, spending scores, age, and demographics using **K-Means Clustering** and an interactive **Streamlit Dashboard**.

---

## 📌 Project Overview

Customer segmentation is the commercial practice of dividing a customer base into distinct sub-groups of individuals that share similar purchasing characteristics. By applying unsupervised machine learning:
- Marketing teams transition from generic mass-market broadcasting to targeted, high-conversion campaigns.
- E-commerce stores and retail chains optimize discounts, loyalty rewards, and product recommendations.
- Customer Lifetime Value (LTV) and customer retention rates are substantially elevated.

This project delivers an end-to-end Machine Learning pipeline:
1. **Data Ingestion & Cleaning:** Validation, duplicate removal, and missing-value imputation.
2. **Exploratory Data Analysis (EDA):** Statistical distributions and demographic correlations.
3. **Feature Scaling:** Standardization using `StandardScaler` to satisfy Euclidean distance requirements.
4. **Optimal Cluster Selection:** Quantitative evaluation using the **Elbow Method (WCSS)** and **Silhouette Score Analysis**.
5. **Dynamic Behavioral Persona Profiling:** Algorithmic segment classification that labels clusters based on actual feature coordinates rather than hardcoded IDs.
6. **Interactive Streamlit Web Dashboard:** Interactive exploration, cluster inspection, and real-time customer segment prediction with instant marketing recommendations.

---

## 🏗️ Project Architecture & Directory Structure

```text
customer_segmentation_kmeans/
│
├── data/
│   ├── customer_data.csv              # Raw customer dataset (200 records)
│   ├── customer_segments_output.csv   # Processed dataset with assigned cluster & persona labels
│   ├── distribution_charts.png        # Exploratory histogram & KDE plots
│   ├── income_vs_spending_raw.png     # Raw scatter plot of Income vs Spending Score
│   ├── elbow_silhouette_analysis.png  # Elbow curve and Silhouette coefficient plots
│   └── customer_clusters.png          # Final 2D clustering plot with cluster centroids
│
├── models/
│   ├── kmeans_model.pkl               # Serialized trained Scikit-Learn KMeans model
│   ├── scaler.pkl                     # Serialized StandardScaler object
│   └── cluster_metadata.json          # Cluster centroids, statistics, and marketing strategies
│
├── preprocessing.py                   # Data ingestion, cleaning, EDA, and scaling utilities
├── train_model.py                     # Model evaluation, training, dynamic profiling & chart generation
├── app.py                             # Interactive Streamlit dashboard & real-time prediction engine
├── requirements.txt                   # Project Python dependencies
└── README.md                          # Comprehensive project documentation
```

---

## 📊 Dataset Schema

The dataset represents mall retail shoppers and includes the following features:

| Column Name | Data Type | Description | Range / Values |
| :--- | :--- | :--- | :--- |
| `CustomerID` | Integer | Unique identifier for each customer | 1 – 200 |
| `Gender` | String | Customer biological gender | Male / Female |
| `Age` | Integer | Customer age in years | 18 – 70 |
| `Annual Income (k$)` | Integer | Annual income in thousands of USD ($k) | 15 – 137 |
| `Spending Score (1-100)` | Integer | Score assigned based on customer behavior and purchase data | 1 – 99 |

---

## 🧮 Mathematical Foundations

### 1. K-Means Clustering Algorithm
K-Means is an iterative unsupervised partitioning algorithm that divides $N$ observations into $K$ disjoint clusters. Each cluster is parameterized by its mean vector (centroid) $\mu_k$:

$$\min_{S} \sum_{k=1}^{K} \sum_{x \in S_k} || x - \mu_k ||^2$$

- **Initialization (`k-means++`):** Selects initial centroids with probability proportional to the squared distance from the closest existing centroid, preventing poor local minima convergence.
- **Assignment Step:** Assigns each observation $x_i$ to the nearest cluster centroid using Euclidean distance.
- **Update Step:** Recomputes each centroid as the arithmetic mean of all points assigned to that cluster.

### 2. The Elbow Method (Inertia / WCSS)
Within-Cluster Sum of Squares (WCSS) quantifies how compact the clusters are:

$$WCSS = \sum_{k=1}^{K} \sum_{i \in C_k} \text{dist}(x_i, \mu_k)^2$$

As $k$ increases, WCSS monotonically decreases. The optimal $k$ is found at the **elbow bend**, where adding further clusters yields diminishing reductions in variance.

### 3. Silhouette Score
The Silhouette Coefficient measures both cluster cohesion and cluster separation for sample $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

- $a(i)$: Mean intra-cluster distance between sample $i$ and all other points in the same cluster.
- $b(i)$: Mean nearest-cluster distance between sample $i$ and points in the closest neighboring cluster.
- **Result:** $k = 5$ achieves the global maximum score of **0.5547**, confirming distinct, well-isolated clusters.

---

## 👥 Dynamic Customer Persona Profiles

Unlike naive implementations that hardcode cluster numbers (which can change between training runs), our system dynamically derives persona labels from the statistical coordinates of each cluster's centroid relative to the population median:

| Cluster ID | Segment Name | Base Size (%) | Avg Income | Avg Spend | Strategic Action |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **C1** | **High-Value / Premium Customers** | 19.5% | ~$86.5k | ~82 / 100 | VIP loyalty perks, early product drops, personalized concierge. |
| **C2** | **Potential / Aspirational Spenders** | 11.0% | ~$25.7k | ~79 / 100 | Buy Now Pay Later (BNPL), flash sales, trend-driven micro-rewards. |
| **C3** | **Cautious Savers / Frugal Affluent** | 17.5% | ~$88.2k | ~17 / 100 | Value-focused campaigns, warranty guarantees, high-tier cashbacks. |
| **C0** | **Average / Balanced Customers** | 40.5% | ~$55.3k | ~50 / 100 | Seasonal campaigns, cross-selling bestsellers, standard rewards. |
| **C4** | **Budget / Low-Spending Customers** | 11.5% | ~$26.3k | ~21 / 100 | Deep clearance discounts, bundle bargains, essential staples. |

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13 installed.
- VS Code (recommended editor).

### Step 1: Open the Project in VS Code
1. Launch **Visual Studio Code**.
2. Select **File > Open Folder...** and navigate to:
   `C:\Users\lokes\.gemini\antigravity\scratch\customer_segmentation_kmeans`
3. Open the integrated terminal (`Ctrl + ~` or **Terminal > New Terminal**).

### Step 2: Create and Activate a Virtual Environment
In your VS Code terminal (PowerShell or Bash):

```bash
# Create virtual environment named .venv
python -m venv .venv

# Activate on Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# (If PowerShell displays an Execution_Policy error, run):
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# Then rerun: .\.venv\Scripts\Activate.ps1

# Activate on macOS/Linux:
# source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Train the Model & Generate Visualizations
Run the model training pipeline:
```bash
python train_model.py
```
**Output Highlights:**
- Evaluates $k=1$ to $10$ with WCSS and Silhouette Scores.
- Trains `KMeans(n_clusters=5)` and serializes `models/kmeans_model.pkl` and `models/scaler.pkl`.
- Exports high-resolution diagnostic charts in the `data/` folder.
- Generates `models/cluster_metadata.json` for live dashboard consumption.

### Step 5: Launch the Streamlit Web Application
Launch the interactive web application:
```bash
streamlit run app.py
```
Streamlit will automatically open your default browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Dashboard Capabilities

The dashboard consists of 5 dedicated interactive modules:
1. **Dataset & EDA:** Interactive table view, five-number summary statistics, and demographic distribution plots.
2. **Elbow & Silhouette Analysis:** Graphical representation of the Elbow curve and Silhouette score curve demonstrating why $k=5$ was selected.
3. **Cluster Visualizer:** 2D scatter plot of customer clusters with centroid star overlays and cluster population breakdown.
4. **Personas & Business Strategies:** Strategic playbooks outlining target offers, marketing channels, and engagement techniques for every segment.
5. **Real-Time Customer Prediction:** Input Age, Gender, Annual Income, and Spending Score to classify a new customer instantly, visualize their location on the cluster map, and receive customized marketing actions.

---

## 🛠️ Troubleshooting & Common FAQs

1. **`streamlit: command not found`**
   - Ensure you activated your virtual environment (`.\.venv\Scripts\Activate.ps1`).
   - Run `python -m streamlit run app.py` instead.

2. **`Script execution is disabled on this system` (PowerShell)**
   - Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   - Then activate the environment again.

3. **Port 8501 is already in use**
   - Run Streamlit on a different port:
     ```bash
     streamlit run app.py --server.port 8502
     ```

4. **Matplotlib Font Warning on Windows**
   - Warnings like `Matplotlib is building the font cache` occur only once on the initial run and resolve automatically.
