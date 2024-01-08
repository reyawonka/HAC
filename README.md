# HAC
Clustering using Hierarchical Agglomerative Clustering (HAC)


<!DOCTYPE html>
<html>
<head>

</head>
<body>

<h1>Lab on Hierarchical Agglomerative Clustering</h1>
<p>This repository contains the code and documentation for a lab exercise focused on Hierarchical Agglomerative Clustering, an important technique in unsupervised machine learning.</p>

<h2>Overview of Scripts</h2>

<h3>1. DataPreparationAndKMeans.py </h3>
<p>Prepares the dataset by imputing missing values, scaling, and then applying KMeans clustering to understand the data structure.</p>
<pre class="code">
# Sample Code Snippet
imputer = SimpleImputer(strategy='mean')
#...
kmeans = KMeans(n_clusters=k)
#...
</pre>

<h3>2. InitialDataCleaning.py </h3>
<p>Focuses on reading and cleaning the initial dataset, setting up the foundation for further analysis.</p>
<pre class="code">
# Sample Code Snippet
feats = pd.read_csv("NUSW-NB15_features.csv")
#...
readdata.columns = col
#...
</pre>

<h3>3. CorrelationMatrixHeatmap.py </h3>
<p>Generates a heatmap for the correlation matrix of the dataset, aiding in feature selection and data understanding.</p>
<pre class="code">
# Sample Code Snippet
correlation_matrix = rdata.corr()
sns.heatmap(correlation_matrix, annot=True)
#...
</pre>

<h3>4. AgglomerativeClusteringVisualization.py </h3>
<p>Implements Hierarchical Agglomerative Clustering and visualizes data points to analyze clustering patterns.</p>
<pre class="code">
# Sample Code Snippet
AgglomerativeClustering()
#...
dendrogram(linkage(cleaned_data, method='ward'))
#...
</pre>

<h2>Conclusion</h2>
<p>This lab explores the application of Hierarchical Agglomerative Clustering, emphasizing data preparation, analysis, and visualization techniques in unsupervised learning.</p>

</body>
</html>
