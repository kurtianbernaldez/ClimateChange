import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Custom styling
st.set_page_config(
    page_title="Climate Insights Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        color: #212529;
    }
    .big-font {
        font-size: 20px !important;
        color: #4a4a4a;
    }
    .stTitle {
        color: #2c3e50;
        text-align: center;
    }
    .stHeader {
        color: #34495e;
    }
    .stSubheader {
        color: #7f8c8d;
    }
    .css-1aumxhk { /* Sidebar background */
        background-color: #f4f4f4;
    }
    .stButton>button {
        color: white;
        background-color: #3498db;
        border: none;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    file_path = "climate_change.csv"
    return pd.read_csv(file_path)


def analyze_clusters(data, clusters, clustering_columns):
    """
    Provide insights for each cluster
    """
    cluster_insights = {}
    for i in range(len(set(clusters))):
        cluster_data = data[data['Cluster'] == i]

        # Compute mean values for each feature in the cluster
        cluster_means = cluster_data[clustering_columns].mean()

        # Generate an insight based on the cluster characteristics
        insight = f"**Cluster {i} Characteristics:**\n"
        for col, mean_val in cluster_means.items():
            insight += f"- Average {col}: {mean_val:.2f}\n"

        cluster_insights[i] = insight

    return cluster_insights


def main():
    st.title("🌍 Climate Data Insights Dashboard")
    st.markdown("*Advanced Analytics for Climate Change Data*", unsafe_allow_html=True)

    # Dataset Preview
    data = load_data()

    st.sidebar.header("🔍 Data Exploration")

    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Data Overview", "Clustering Analysis", "Regression Analysis"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(data.head(10), use_container_width=True)

        with col2:
            st.subheader("Descriptive Statistics")
            st.dataframe(data.describe(), use_container_width=True)

    with tab2:
        st.header("🧩 Clustering Analysis")

        # Clustering Section
        n_clusters = st.slider("Select number of clusters (k):", 2, 10, 3)

        # Update to use numeric columns
        numeric_columns = [
            col for col in data.select_dtypes(include=['float64', 'int64']).columns
            if col.lower() not in ['year', 'month']
        ]

        clustering_columns = st.multiselect(
            "Select features for clustering:",
            numeric_columns
        )

        # Check if at least one column is selected
        if len(clustering_columns) > 0:
            # Prepare data for clustering
            cluster_data = data[clustering_columns]

            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)

            # Perform PCA to reduce to 2D for visualization
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)

            # Add clusters to dataframe
            data["Cluster"] = clusters

            # Cluster visualization
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Cluster Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    pca_data[:, 0], pca_data[:, 1],
                    c=clusters, cmap='viridis', alpha=0.7
                )
                ax.set_title(f"K-means Clustering (PCA)")
                ax.set_xlabel("First Principal Component")
                ax.set_ylabel("Second Principal Component")
                plt.colorbar(scatter, ax=ax, label='Cluster')
                st.pyplot(fig)

            with col2:
                st.subheader("Cluster Insights")
                # Generate and display cluster insights
                cluster_insights = analyze_clusters(data, clusters, clustering_columns)
                for cluster, insight in cluster_insights.items():
                    st.markdown(insight)

            # Show explained variance
            st.subheader("PCA Explained Variance")
            st.write(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
        else:
            st.warning("Please select at least one numeric column for clustering")

    with tab3:
        st.header("📈 Regression Analysis")

        # Regression Section
        target = st.selectbox("Select target variable:", ["Temp"])

        # Filter potential features (excluding target and non-numeric columns)
        potential_features = [
            col for col in data.columns
            if col != target and data[col].dtype in ['int64', 'float64'] and col.lower() not in ['year', 'month']
        ]
        feature = st.selectbox("Select predictor variable:", potential_features)

        X = data[[feature]]
        y = data[target]

        # Train the model
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # Regression results and visualization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Regression Plot")
            fig, ax = plt.subplots()
            ax.scatter(X, y, color="blue", label="Actual")
            ax.plot(X, predictions, color="red", label="Predicted")
            ax.set_title(f"Regression: {feature} vs. {target}")
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Regression Metrics")
            st.metric("Mean Squared Error", f"{mean_squared_error(y, predictions):.4f}")
            st.metric("R-squared", f"{r2_score(y, predictions):.4f}")

        # Regression Insights
        st.subheader("Regression Insights")
        coefficient = model.coef_[0]
        intercept = model.intercept_

        insight_text = f"""
        **Regression Analysis Insights:**
        - **Linear Relationship**: The linear regression model suggests a {'positive' if coefficient > 0 else 'negative'} relationship between {feature} and {target}.
        - **Coefficient**: For each unit increase in {feature}, {target} is expected to {'increase' if coefficient > 0 else 'decrease'} by {abs(coefficient):.4f}.
        - **Intercept**: When {feature} is zero, the predicted {target} is {intercept:.4f}.
        - **Model Fit**: The R-squared value of {r2_score(y, predictions):.4f} indicates the proportion of variance explained by the model.
        """
        st.markdown(insight_text)

    # Conclusion Section
    st.write("## 🔬 Overall Conclusions")
    st.markdown("""
    - **Clustering Insights**: Revealed distinct groups in climate data based on selected features.
    - **Regression Analysis**: Explored relationships between climate variables.
    - **Recommendations**: Conduct further in-depth analysis to understand complex climate patterns.
    """)


if __name__ == "__main__":
    main()