# knn_classification.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap

#  1.Choose a classification dataset and normalize features
# a. Load the dataset
df = pd.read_csv("iris.csv")

# b. Separate features and target
X = df.iloc[:, 0:2].values   # Use only first 2 features for visualization
y = df.iloc[:, -1].values    # Assuming last column is the target

# c. Encode class labels if not numeric
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# d. Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# e. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 2.Use KNeighborsClassifier from sklearn.
#  Train and evaluate for multiple k
for k in [1, 3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(f"\n🔍 K={k}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
#  3.Experiment with different values of K.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#  4.Evaluate model using accuracy, confusion matrix.
# Create meshgrid
h = 0.02
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#  5.Visualize decision boundaries
# Plot decision boundary
plt.figure(figsize=(8, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.contourf(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_bold, edgecolor='k')
plt.title("KNN Decision Boundary (k=5)")
plt.xlabel("Feature 1 (normalized)")
plt.ylabel("Feature 2 (normalized)")
plt.grid(True)
plt.tight_layout()
plt.savefig("decision_boundary.png")
plt.show()

