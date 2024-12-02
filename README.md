# Python_Project
from sklearn.datasets import load_wine

# Load the Wine dataset
data = load_wine()

# Print dataset description
print(data.DESCR)

# View features and target names
print("Feature names:")
print(data.feature_names)

print("\nTarget names:")
print(data.target_names)





import pandas as pd
from sklearn.datasets import load_wine

# Load the Wine dataset
data = load_wine()

# Create a DataFrame for the dataset
wine_df = pd.DataFrame(data.data, columns=data.feature_names)

# Add a column for the target values
wine_df['target'] = data.target

# Display the entire dataset
print(wine_df)
print(wine_df.head())  # Display the first few rows of the dataset
print(wine_df.tail())  # Display the last few rows of the dataset


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Wine dataset
data = load_wine()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a BaggingClassifier using DecisionTreeClassifier as base estimator
base_model = DecisionTreeClassifier(random_state=42)
bagging_model = BaggingClassifier(base_model, n_estimators=100, random_state=42)

# Fit the BaggingClassifier on the training data
bagging_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = bagging_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



