from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from scipy.stats import norm, shapiro, gaussian_kde
import numpy as np
import matplotlib.pyplot as plot

# check which datasets can be imported
#list_available_datasets()

# Fetch the dataset from OpenML
heart_disease = fetch_ucirepo(id=45)   # Dataset ID for Heart Disease
#Assign the data to a dataframe
df = heart_disease.data.original


print(df.head())
print(df.info())

categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
target_column = 'num' 

# Handle missing values (example: fill missing with median) and adjusting data
for col in continuous_features:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_features:
    df[col] = df[col].fillna(df[col].mode()[0])

df['num'] = (df['num'] > 0).astype(int)
#saving to csv to examine data
df.to_csv('test.csv', index=False)

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df[categorical_features + continuous_features]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate categorical and continuous data for training and testing
X_train_cat = X_train[categorical_features]
X_train_cont = X_train[continuous_features]
X_test_cat = X_test[categorical_features]
X_test_cont = X_test[continuous_features]

cat_nb = CategoricalNB()
cat_nb.fit(X_train_cat, y_train)

classes = np.unique(y_train)
class_kdes = {c: [] for c in classes}

for c in classes:
    for feature_idx in continuous_features:
        # Filter rows for class 'c' and get data for the specific feature
        feature_data = X_train[y_train == c][feature_idx].values
        
        # Fit Kernel Density Estimator
        kde = gaussian_kde(feature_data)
        class_kdes[c].append(kde)


#CategoricalNB predictions
cat_pred = cat_nb.predict(X_test[categorical_features])
cat_accuracy = accuracy_score(y_test, cat_pred)
print(f"CategoricalNB Accuracy: {cat_accuracy:.2f}")

#KDE Naive Bayes predictions
def predict_kde(X_cont_test):
    log_probs = []
    for c in classes:
        class_probs = 0
        for feature_idx in continuous_features:
            # Use the index of the feature in the list of continuous columns
            feature_kde = class_kdes[c][continuous_features.index(feature_idx)]
            class_probs += np.log(feature_kde.evaluate(X_cont_test[feature_idx]) + 1e-9)
        log_probs.append(class_probs)
    return np.array(log_probs).T

def predict_combined(X_cat_test, X_cont_test):
    # Log probabilities from CategoricalNB for categorical features
    cat_log_prob = cat_nb.predict_log_proba(X_cat_test)
    kde_log_prob = predict_kde(X_cont_test)

    # Combine log probabilities
    combined_log_prob = cat_log_prob + kde_log_prob
    return combined_log_prob

kde_log_prob = predict_kde(X_test[continuous_features])
kde_pred = np.argmax(kde_log_prob, axis=1)
kde_accuracy = accuracy_score(y_test, kde_pred)
print(f"KDE Naive Bayes Accuracy (Continuous Features): {kde_accuracy:.2f}")

#Combined predictions
combined_log_prob = predict_combined(X_test[categorical_features], X_test[continuous_features])
combined_pred = np.argmax(combined_log_prob, axis=1)
combined_accuracy = accuracy_score(y_test, combined_pred)
print(f"Combined Model Accuracy: {combined_accuracy:.2f}")

models = ['CategoricalNB', 'KdeNB', 'Combined']
accuracies = [cat_accuracy, kde_accuracy, combined_accuracy]

# Plotting
plot.figure(figsize=(8, 5))
bars = plot.bar(models, accuracies, color=['blue', 'red', 'purple'], alpha=0.7)

# Add accuracy values on top of bars
for bar, acc in zip(bars, accuracies):
    plot.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{acc:.2f}", ha='center', va='bottom', fontsize=10)


# Customize the plot
plot.ylim(0, 1)
plot.xlabel('Models', fontsize=12)
plot.ylabel('Accuracy', fontsize=12)
plot.title('Model Accuracies Comparison', fontsize=14)
plot.grid(axis='y', linestyle='--', alpha=0.7)
plot.tight_layout()

# Show the plot
print("Plot show?")
plot.show()
