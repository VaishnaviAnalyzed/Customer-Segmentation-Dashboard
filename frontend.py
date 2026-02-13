import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Page Setup
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍️ Mall Customer Segmentation")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("User Input & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Mall_Customers.csv", type="csv")

algo_choice = st.sidebar.selectbox(
    "Select Clustering Algorithm",
    ("K-Means", "Agglomerative Clustering")
)

n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

# --- MAIN LOGIC ---
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    # Using columns 3 and 4 as per your provided script 
    X = dataset.iloc[:, [3, 4]].values 
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Dataset Preview")
        st.write(dataset.head())

    # Algorithm Execution
    if algo_choice == "K-Means":
        # Elbow Method Logic from your K-Means script
        if st.sidebar.button("Show Elbow Method"):
            st.subheader("The Elbow Method")
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init="k-means++", random_state=0)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
            fig_elbow, ax_elbow = plt.subplots()
            ax_elbow.plot(range(1, 11), wcss)
            ax_elbow.set_title("The Elbow Method")
            ax_elbow.set_xlabel('Number of clusters')
            ax_elbow.set_ylabel('WCSS')
            st.pyplot(fig_elbow)

        # Training
        model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0)
        y_pred = model.fit_predict(X)
        centroids = model.cluster_centers_

    else:
        # Dendrogram Logic from your Hierarchical script
        if st.sidebar.button("Show Dendrogram"):
            st.subheader("Dendrogram")
            fig_den, ax_den = plt.subplots()
            dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
            ax_den.set_title('Dendrogram')
            ax_den.set_xlabel('Customers')
            ax_den.set_ylabel('Euclidean distances')
            st.pyplot(fig_den)

        # Training
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        y_pred = model.fit_predict(X)
        centroids = None

    # --- VISUALIZATION ---
    with col2:
        st.subheader(f"Clusters via {algo_choice}")
        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i in range(n_clusters):
            ax.scatter(X[y_pred == i, 0], X[y_pred == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
        
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1], s=300, c='yellow', label='Centroids')
            
        ax.set_title('Clusters of customers')
        ax.set_xlabel('Annual Income (k$)')
        ax.set_ylabel('Spending Score (1-100)')
        ax.legend()
        st.pyplot(fig)

    # Add cluster results to dataframe
    dataset['Cluster'] = y_pred
    st.divider()
    st.subheader("Processed Data with Cluster Scores")
    st.dataframe(dataset)

else:
    st.info("Please upload the 'Mall_Customers.csv' file in the sidebar to begin.")