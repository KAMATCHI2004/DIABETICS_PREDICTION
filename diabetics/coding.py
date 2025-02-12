import pandas

data = pandas.read_csv("diabetes.csv")
data['Outcome'].unique()
data.info()
data.describe()
data.head(5)
data.tail(5)

data.isnull().sum()
data.dropna(inplace=True)
data.isnull().sum()


data.duplicated().sum()
data.drop_duplicates(inplace=True)

from sklearn.preprocessing import LabelEncoder
columns_to_encode=["Outcome"]
label_encoder=LabelEncoder()
for col in columns_to_encode:
    data[col]=label_encoder.fit_transform(data[col])
data.info()

X=data.drop(columns="Outcome")
print(X)
Y=data[["Outcome"]]
#from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=10,test_size=0.20)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
data.shape


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

# List of classifiers to compare
models = [
    ("Decision Tree", DecisionTreeClassifier()),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("Support Vector Classifier", SVC()),
    ("Random Forest", RandomForestClassifier()),
    ("Logistic Regression", LogisticRegression())
]

# Initialize storage for results
results = []
confusion_matrices = {}

# Iterate through each model
for name, model in models:
    # Train the model
    model.fit(x_train, y_train)
    
    # Make predictions
    pred = model.predict(x_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, pred) * 100
    report = classification_report(y_test, pred, output_dict=True)
    cm = confusion_matrix(y_test, pred)
    
    # Store metrics
    results.append({
        "Model": name,
        "Accuracy (%)": accuracy,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"],
    })
    
    # Store confusion matrix for visualization
    confusion_matrices[name] = cm

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print the table of metrics
print("Performance Metrics:")
print(results_df)

# Plot accuracy comparison chart
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy (%)", data=results_df, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45)
plt.show()

# Display confusion matrices
for name, cm in confusion_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()



import pickle as pk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

# Define the parameter grid for Logistic Regression
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga', 'lbfgs'],
    'max_iter': [100, 200, 500]
}

# Create Logistic Regression model
log_reg = LogisticRegression()

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    verbose=2,
    n_jobs=-1  # Use all available CPU cores
)

# Fit the grid search on the training data
grid_search.fit(x_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print(f"Best Accuracy: {grid_search.best_score_ * 100:.2f}%")

# Get the best model
best_model = grid_search.best_estimator_

# Save the best model to a file
with open("model.pkl", "wb") as file:
    pk.dump(best_model, file)

# Evaluate the tuned model on the test data
y_pred = best_model.predict(x_test)

# Test set performance
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
  

with open("model.pkl", "rb") as file:
    model = pk.load(file)
