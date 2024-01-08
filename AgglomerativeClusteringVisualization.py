import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'cleaner.csv'  
rdata = pd.read_csv(file_path)

correlation_matrix = rdata.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

num_top_features = 5
top_features = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(num_top_features)

print("Top Features:")
print(top_features)
