# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries and load the Mall_Customers.csv dataset.
2.Select Annual Income and Spending Score as the features for customer segmentation.
3.Apply the K-Means clustering algorithm with a suitable number of clusters.
4.Train the model and assign each customer to a cluster.
5.Visualize the customer segments using a scatter plot with different cluster labels.

## Program:

```python

# Program to implement the K Means Clustering for Customer Segmentation.
# Developed by: THARUN S
# Register Number:  212225240174


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Select features
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Create K-Means model
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# Fit the model
data["Cluster"] = kmeans.fit_predict(X)

# Display cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Display first 10 customers with cluster labels
print("\nCustomer Segments:")
print(data[["CustomerID", "Annual Income (k$)",
            "Spending Score (1-100)", "Cluster"]].head(10))

# Plot the clusters
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=data["Cluster"]
)

# Plot cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.show()
```

## Output:

<img width="1263" height="530" alt="image" src="https://github.com/user-attachments/assets/97fdfc0d-80d5-450b-8932-05c9e322f166" />
<img width="1135" height="588" alt="image" src="https://github.com/user-attachments/assets/3ffe1169-f6f9-4a9c-8fef-a82f1dbc724c" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
