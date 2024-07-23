import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score  
import pickle as pkl

# Load the dataset
dataframe1 = pd.read_csv('/Users/joshuawambugu/Desktop/DeployML/iris.csv')

# Map the target column to numeric values for model training
dataframe1['variety'] = dataframe1['variety'].map({'Setosa': 1, 'Versicolor': 2, 'Virginica': 3})

# Check the mapping result
print(dataframe1.head())

# Define features and target variable
X = dataframe1.iloc[:, :-1]    
y = dataframe1.iloc[:, -1]    

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  

# Initialize and train the KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can set n_neighbors to any appropriate value
knn_classifier.fit(X_train, y_train)  

# Make predictions
y_pred = knn_classifier.predict(X_test)  

# Calculate accuracy
score = accuracy_score(y_test, y_pred)  
print("Prediction Accuracy: ", score)

# Save the model to a file
pickle_out1 = open("classifier2.pkl", "wb")    
pkl.dump(knn_classifier, pickle_out1)    
pickle_out1.close()
