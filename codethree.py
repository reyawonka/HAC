import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


cleaned_data = pd.read_csv("cleaner.csv")


selected_features = list(cleaned_data.columns[:5])  

for i in range(len(selected_features)):
    for j in range(i + 1, len(selected_features)):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=selected_features[i], y=selected_features[j], data=cleaned_data)
        plt.title(f"Scatter Plot: {selected_features[i]} vs {selected_features[j]}")
        plt.show()

for k in [2, 5, 10]:
    clustering = AgglomerativeClustering(n_clusters=k)
    labels = clustering.fit_predict(cleaned_data[selected_features])

    plt.figure(figsize=(12, 6))
    dendrogram(linkage(cleaned_data[selected_features], method='ward'), leaf_rotation=90, leaf_font_size=8)
    plt.title(f'Dendrogram for k={k}')
    plt.show()

    print(f'Cluster Labels for k={k}:')
    print(labels)


