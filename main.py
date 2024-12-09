from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import norm, shapiro, gaussian_kde
import numpy as np
import matplotlib.pyplot as plot

#KDE Naive Bayes predictions (for continuous features)
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

def predict_combined(X_cat_test, X_cont_test, alpha, beta):
    # Log probabilities from CategoricalNB for categorical features
    cat_log_prob = cat_nb.predict_log_proba(X_cat_test)
    kde_log_prob = predict_kde(X_cont_test)

    # Combine log probabilities
    combined_log_prob = cat_log_prob * alpha + kde_log_prob * beta
    return combined_log_prob

# check which datasets can be imported
#list_available_datasets()

# Fetch the dataset from OpenML
heart_disease = fetch_ucirepo(id=45)   # Dataset ID for Heart Disease
#Assign the data to a dataframe
df = heart_disease.data.original
#Assign the data to either categorical, continous, and target features
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
target_column = 'num' 

# Handle missing values continous use the median and categorical use the most common
for col in continuous_features:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_features:
    df[col] = df[col].fillna(df[col].mode()[0])
# Convert the target to a binary true false. 
df['num'] = (df['num'] > 0).astype(int)
#saving to csv to examine data
#df.to_csv('test.csv', index=False)

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    # Fit the encoder on the combined dataset (training + testing)
    df[col] = le.fit_transform(df[col])  # Fit on the whole data (train + test)
    label_encoders[col] = le


# Separate features and target
X = df[categorical_features + continuous_features]
y = df[target_column]

train_sizes = [0.15, 0.25, 0.33, 0.48, 0.66, 0.8]
accuracy_results_list = []
for train_size in train_sizes:
    #standard random state data selection
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1 - train_size, random_state=4)

    #Test if there's enough data to fill up the training categories
    #for col in categorical_features:
    #    print(f"Categories in {col}: {X_train[col].nunique()} in train and {X_test[col].nunique()} in test")


    #assign kde classes to the two possible results 0 and 1
    classes = [0,1]
    class_kdes = {c: [] for c in classes}

    for c in classes:
        for feature_index in continuous_features:
            # Filter rows for class 'c' and get data for the specific feature
            feature_data = X_train[y_train == c][feature_index].values
            
            # Fit Kernel Density Estimator
            kde = gaussian_kde(feature_data)
            class_kdes[c].append(kde)

    #Categorical initialization and fit
    cat_nb = CategoricalNB()
    cat_nb.fit(X_train[categorical_features], y_train)

    # CategoricalNB predictions
    cat_pred = cat_nb.predict(X_test[categorical_features])
    cat_accuracy = accuracy_score(y_test, cat_pred)
    print(f"CategoricalNB Accuracy: {cat_accuracy:.2f}")

    #KDE predictions
    kde_log_prob = predict_kde(X_test[continuous_features])
    kde_pred = np.argmax(kde_log_prob, axis=1)
    kde_accuracy = accuracy_score(y_test, kde_pred)
    print(f"KDE Naive Bayes Accuracy (Continuous Features): {kde_accuracy:.2f}")

    #Combined predictions
    combined_log_prob = predict_combined(X_test[categorical_features], X_test[continuous_features], 0.5, 0.5)
    combined_pred = np.argmax(combined_log_prob, axis=1)
    combined_accuracy = accuracy_score(y_test, combined_pred)
    print(f"Combined Model Accuracy: {combined_accuracy:.2f}")

    #Adding data to graph list
    accuracy_results_list.append([cat_accuracy, kde_accuracy, combined_accuracy])

# Convert the list to a NumPy array for easier manipulation
accuracy_results_list = np.array(accuracy_results_list)

# Extract the columns (each represents one line)
line1 = accuracy_results_list[:, 0]  # First element in each array
line2 = accuracy_results_list[:, 1]  # Second element in each array
line3 = accuracy_results_list[:, 2]  # Third element in each array

# Plot the lines. Maintaining color standards
plot.plot(train_sizes, line1, label='Categorical', marker='o', color='blue')
plot.plot(train_sizes, line2, label='KDE', marker='x', color='red')
plot.plot(train_sizes, line3, label='Combined', marker='s', color='purple')

# Add labels, legend, and title
plot.xlabel('Training Size')
plot.ylabel('Accuracy')
plot.title('Accuracy with increases in training data')
plot.legend()

# Show the plot
plot.grid(True)
plot.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state=4)

# Combine log probabilities
#alphas = [0.2, 0.3, 0.4, 0.42, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.7, 0.8, 0.9]  # Weights for CategoricalNB
alphas = np.arange(0.2, 0.7 + 0.01, 0.01).tolist()
alpha_combined_accuracies = []
for alpha in alphas:
    combined_log_prob = predict_combined(X_test[categorical_features], X_test[continuous_features], alpha, 1 - alpha)
    combined_pred = np.argmax(combined_log_prob, axis=1)
    combined_accuracy = accuracy_score(y_test, combined_pred)
    alpha_combined_accuracies.append(combined_accuracy)

#alpha_combined_accuracies = np.array(alpha_combined_accuracies)
plot.plot(alphas, alpha_combined_accuracies, label='Combined', marker='o', color='purple')
plot.xlabel('Alpha/Categorical Weight')
plot.ylabel('Accuracy')
plot.title('Accuracy with varied Alphas')
plot.legend()

# Show the plot
plot.grid(True)
plot.show()


#Final Bar plot
models = ['CategoricalNB', 'GaussianNB', 'Combined']
accuracies = [accuracy_results_list[4][0], accuracy_results_list[4][1], accuracy_results_list[4][2]]

plot.figure(figsize=(8, 5))
bars = plot.bar(models, accuracies, color=['blue', 'red', 'purple'], alpha=0.7)

# Add accuracy values on top of bars
for bar, acc in zip(bars, accuracies):
    plot.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{acc * 100:.2f}%", ha='center', va='bottom', fontsize=10)


# Customize the plot
plot.ylim(0, 1)  # Set y-axis from 0 to 1 (since accuracy is a percentage)
plot.xlabel('Models', fontsize=12)
plot.ylabel('Accuracy', fontsize=12)
plot.title('Final Model Accuracy Comparison with an alpha of 0.5 and training size of 50%', fontsize=14)
plot.grid(axis='y', linestyle='--', alpha=0.7)
plot.tight_layout()

# Show the plot
print("Plot show?")
plot.show()