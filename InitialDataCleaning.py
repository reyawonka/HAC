import pandas as pd

feats = pd.read_csv("NUSW-NB15_features.csv", delimiter=',', skiprows=1, names=['No.', 'Name', 'Type', 'Description'], encoding='ISO-8859-1')

readdata = pd.read_csv("UNSW-NB15_4.csv", delimiter=',', header=None)


col = feats['Name'].tolist()
readdata.columns = col

print("Columns in the readdataset before processing:")
print(readdata.head())


colDrop = ['dur', 'Sload', 'Dload', 'Sjit', 'Djit', 
           'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 
           'synack', 'ackdat', 'Label']
readdata = readdata.drop(columns=colDrop, axis=1)
readdata = readdata.apply(lambda col: pd.Categorical(col).codes if col.dtype == 'object' else col)

print("\nColumns in the readdataset after processing:")
print(readdata.head())


readdata.to_csv("cleaner.csv", index=False)
cleaned_readdata = pd.read_csv("cleaner.csv")

