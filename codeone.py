import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

rdata = pd.read_csv("cleaner.csv")

feats = rdata.select_dtypes(include=[np.number])

imputer = SimpleImputer(strategy='mean')
imputed_rdata = pd.DataFrame(imputer.fit_transform(feats), columns=feats.columns)

scaler = StandardScaler()
scaled_rdata = scaler.fit_transform(imputed_rdata)

k_values = [2, 5, 10]

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    rdata[f'cluster_{k}'] = kmeans.fit_predict(scaled_rdata)

    for cluster_id in range(k):
        samples_in_cluster = rdata[rdata[f'cluster_{k}'] == cluster_id].head(20)

        print(f"Cluster {cluster_id} (k={k}):")
        print(samples_in_cluster)

        plt.scatter(scaled_rdata[rdata[f'cluster_{k}'] == cluster_id, 0], scaled_rdata[rdata[f'cluster_{k}'] == cluster_id, 1], label=f'Cluster {cluster_id}')
    
    plt.title(f'K-Means Clustering (k={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    top_feats = imputed_rdata.columns[np.argsort(np.abs(kmeans.cluster_centers_[cluster_id]))[::-1][:5]]

    for cluster_id in range(k):
        plt.bar(top_feats, kmeans.cluster_centers_[cluster_id, np.argsort(np.abs(kmeans.cluster_centers_[cluster_id]))[::-1][:5]])
        plt.title(f'Top 5 Features in Cluster {cluster_id} (k={k})')
        plt.xlabel('Features')
        plt.ylabel('Values')
        plt.show()

