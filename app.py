
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ----------------------------------
# Page configuration
# ----------------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# ----------------------------------
# Custom blush-pink CSS
# ----------------------------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #FFF7F8, #FCE8ED);
    color: #49383D;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
h1 {
    color: #9F5268 !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #A85D73 !important;
    font-weight: 700 !important;
}

/* General text */
p, label, .stMarkdown {
    color: #49383D;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #F7DFE5;
    border-right: 1px solid #ECC5D0;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #91445D !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #F0D0D9;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(159, 82, 104, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #916C79 !important;
}

[data-testid="stMetricValue"] {
    color: #A84F6C !important;
    font-weight: 800;
}

/* Dataframe border */
[data-testid="stDataFrame"] {
    border: 1px solid #F0D0D9;
    border-radius: 12px;
    overflow: hidden;
}

/* Buttons */
.stDownloadButton button,
.stButton button {
    background: #BD7188;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.2rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stDownloadButton button:hover,
.stButton button:hover {
    background: #984C66;
    color: white;
    transform: translateY(-2px);
}

/* Input fields */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: #FFFAFB;
    border-color: #E8BFCB;
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #EFD3DC !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Download button spacing */
.stDownloadButton {
    margin-top: 1rem;
}

/* Hide default footer */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------
# Load dataset
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Mall_Customers.csv was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()


# ----------------------------------
# Dashboard header
# ----------------------------------
st.title("🛍️ Customer Segmentation")
st.markdown(
    "### Understand your customers. Discover meaningful groups."
)
st.write(
    "An interactive machine learning dashboard "
    "powered by K-Means Clustering."
)

st.divider()


# ----------------------------------
# Dataset preview
# ----------------------------------
st.subheader("📊 Customer Dataset")

with st.expander("View customer records"):
    st.dataframe(df, use_container_width=True)


# ----------------------------------
# Key statistics
# ----------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Average Age", f"{df['Age'].mean():.1f} years")

with col3:
    st.metric(
        "Average Spending Score",
        f"{df['Spending Score (1-100)'].mean():.1f} / 100"
    )

st.divider()


# ----------------------------------
# Sidebar controls
# ----------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Customise your clustering model.")

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

selected_features = st.sidebar.multiselect(
    "Select at least 2 features",
    options=features,
    default=[
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
)

k = st.sidebar.slider(
    "Number of clusters (K)",
    min_value=2,
    max_value=10,
    value=5
)


# ----------------------------------
# Validate selected features
# ----------------------------------
if len(selected_features) < 2:
    st.warning("Please select at least two features.")
    st.stop()


# ----------------------------------
# Prepare and scale data
# ----------------------------------
X = df[selected_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ----------------------------------
# Train K-Means model
# ----------------------------------
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X_scaled)


# ----------------------------------
# Model evaluation
# ----------------------------------
score = silhouette_score(X_scaled, df["Cluster"])

st.subheader("🎯 Clustering Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Customer Segments", k)

with col2:
    st.metric("Silhouette Score", f"{score:.3f}")

st.caption(
    "Silhouette scores range from -1 to 1. "
    "Higher values generally indicate better-separated clusters. "
    "Use this score alongside the graph and business context."
)

st.divider()


# ----------------------------------
# Visualisation
# ----------------------------------
st.subheader("📈 Customer Cluster Visualisation")

x_axis = st.selectbox(
    "Horizontal axis",
    options=selected_features,
    index=0
)

y_axis = st.selectbox(
    "Vertical axis",
    options=selected_features,
    index=1
)

# Pinterest blush-pink cluster colours
cluster_colors = [
    "#BD7188",
    "#E8A6B7",
    "#A85D73",
    "#D9A7A0",
    "#84616D",
    "#F0C4CF",
    "#C9828D",
    "#B58B9A",
    "#E2BFAF",
    "#9F5268"
]

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#FFF7F8")
ax.set_facecolor("#FFFAFB")

for cluster_id in sorted(df["Cluster"].unique()):
    cluster_data = df[df["Cluster"] == cluster_id]

    ax.scatter(
        cluster_data[x_axis],
        cluster_data[y_axis],
        color=cluster_colors[int(cluster_id) % len(cluster_colors)],
        label=f"Cluster {cluster_id + 1}",
        s=85,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.7
    )

# Plot cluster centres in original units
x_index = selected_features.index(x_axis)
y_index = selected_features.index(y_axis)

centres = scaler.inverse_transform(model.cluster_centers_)

ax.scatter(
    centres[:, x_index],
    centres[:, y_index],
    marker="X",
    s=240,
    color="#49383D",
    edgecolors="white",
    linewidths=1.2,
    label="Cluster Centres"
)

ax.set_xlabel(x_axis, color="#49383D", fontsize=11)
ax.set_ylabel(y_axis, color="#49383D", fontsize=11)
ax.set_title(
    "Customer Groups Using K-Means",
    color="#9F5268",
    fontsize=15,
    fontweight="bold",
    pad=15
)

ax.tick_params(colors="#49383D")
ax.grid(alpha=0.15)
ax.legend(frameon=False)

for spine in ax.spines.values():
    spine.set_color("#EFD3DC")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ----------------------------------
# Customer segment summary
# ----------------------------------
st.subheader("👥 Customer Segment Summary")

summary = df.groupby("Cluster").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending_Score=("Spending Score (1-100)", "mean")
).round(2)

summary.index = [f"Cluster {i + 1}" for i in summary.index]

st.dataframe(summary, use_container_width=True)

st.caption(
    "Cluster numbers are arbitrary labels. "
    "Interpret each group using its average income, age, "
    "and spending score."
)


# ----------------------------------
# Download results
# ----------------------------------
st.subheader("📥 Export Results")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Customer Segments CSV",
    data=csv,
    file_name="customer_segments.csv",
    mime="text/csv"
)


# ----------------------------------
# Footer
# ----------------------------------
st.divider()

st.markdown(
    "<div style='text-align:center; color:#916C79;'>"
    "Customer Segmentation using K-Means Clustering"
    "<br>Machine Learning Project"
    "</div>",
    unsafe_allow_html=True
)
