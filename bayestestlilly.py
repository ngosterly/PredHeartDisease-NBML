from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from scipy.stats import norm, shapiro, gaussian_kde, chi2_contingency
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

#fetch the dataset from ucimlrepo
heart_disease = fetch_ucirepo(id=45)   #UCI Repository id#45
#data to a dataframe
df = heart_disease.data.original


print(df.head())
print(df.info())

categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
target_column = 'num' 

#handle the missing values in 'ca"
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

# separate features and target
X = df[categorical_features + continuous_features]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# separate categorical and continuous data for training and testing
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
        feature_data = X_train[y_train == c][feature_idx].values
        
        # fit KDE
        kde = gaussian_kde(feature_data)
        class_kdes[c].append(kde)

#C-NB predictions
cat_pred = cat_nb.predict(X_test[categorical_features])
cat_accuracy = accuracy_score(y_test, cat_pred)
print(f"\nCategoricalNB Accuracy: {cat_accuracy:.2f}")

#KDE NB predictions
def predict_kde(X_cont_test):
    log_probs = []
    for c in classes:
        class_probs = 0
        for feature_idx in continuous_features:

            feature_kde = class_kdes[c][continuous_features.index(feature_idx)]
            class_probs += np.log(feature_kde.evaluate(X_cont_test[feature_idx]) + 1e-9)
        log_probs.append(class_probs)
    return np.array(log_probs).T

def predict_combined(X_cat_test, X_cont_test):
    # log probabilities from C-NB for categorical features
    cat_log_prob = cat_nb.predict_log_proba(X_cat_test)
    kde_log_prob = predict_kde(X_cont_test)

    # combine log probabilities
    combined_log_prob = cat_log_prob + kde_log_prob
    return combined_log_prob

kde_log_prob = predict_kde(X_test[continuous_features])
kde_pred = np.argmax(kde_log_prob, axis=1)
kde_accuracy = accuracy_score(y_test, kde_pred)
print(f"KDE Naive Bayes Accuracy (Continuous Features): {kde_accuracy:.2f}")

#combined predictions
combined_log_prob = predict_combined(X_test[categorical_features], X_test[continuous_features])
combined_pred = np.argmax(combined_log_prob, axis=1)
combined_accuracy = accuracy_score(y_test, combined_pred)
print(f"Combined Model Accuracy: {combined_accuracy:.2f}\n")

models = ['CategoricalNB', 'KdeNB', 'Combined']
accuracies = [cat_accuracy, kde_accuracy, combined_accuracy]

# plotting
plot.figure(figsize=(8, 5))
bars = plot.bar(models, accuracies, color=['blue', 'red', 'purple'], alpha=0.7)
for bar, acc in zip(bars, accuracies):
    plot.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{acc:.2f}", ha='center', va='bottom', fontsize=10)

plot.ylim(0, 1)
plot.xlabel('Models', fontsize=12)
plot.ylabel('Accuracy', fontsize=12)
plot.title('Model Accuracies Comparison', fontsize=14)
plot.grid(axis='y', linestyle='--', alpha=0.7)
plot.tight_layout()
print("Original (Unweighted) Plot Show") #showing the original plot
plot.show()

#cp & exang / fbs & exang / restecg & slope
#age & chol / trestbps & thalach 

# EXPANDED CORRELATION MATRIX FOR CONTINUOUS FEATURES
correlation_matrix = df[continuous_features].corr(method='pearson') #measures the linear relationship between two variables
print("\nContinuous Features Correlation Matrix")
print(correlation_matrix)

# visualize correlations
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm") #each cell's color represents the strength of the correlation between two variables
plot.title("Continuous Features Correlation Heatmap")
plot.show()

continuous_pairs = [ #chosen from pearson correlations of weights > |0.20|
    ('age', 'thalach'),
    ('oldpeak', 'thalach'),
    ('ca', 'thalach'),
    ('age', 'ca'),
    ('age', 'chol')
]

continuous_weights = {}
for feature1, feature2 in continuous_pairs:
    continuous_weights[(feature1, feature2)] = abs(correlation_matrix.loc[feature1, feature2])
total_continuous_weight = sum(continuous_weights.values())
normalized_continuous_weights = {pair: weight / total_continuous_weight for pair, weight in continuous_weights.items()}

print("Normalized Continuous Feature Weights:\n", normalized_continuous_weights)

#using cramer's v to give weights & strength of association between two CATEGORICAL variables
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    
    chi2, _, _, _ = chi2_contingency(contingency_table)
    
    #compute cramer's v
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2_corr / min(k_corr-1, r_corr-1))

#chosen categorical pairs based 
categorical_pairs = [('cp', 'exang'), ('fbs', 'exang'), ('restecg', 'slope')]

#give weights for categorical pairs
categorical_weights = {}
for feature1, feature2 in categorical_pairs:
    categorical_weights[(feature1, feature2)] = cramers_v(df[feature1], df[feature2])

#proportionalize the categorical weights
total_categorical_weight = sum(categorical_weights.values())
normalized_categorical_weights = {pair: weight / total_categorical_weight for pair, weight in categorical_weights.items()}

print("\nNormalized Categorical Feature Weights:\n", normalized_categorical_weights)

#use the weights and predict again
def predict_kde_weighted(X_cont_test):
    log_probs = []
    for c in classes:
        class_probs = np.zeros(X_cont_test.shape[0])
        for feature_idx in range(len(continuous_features)):
            feature_name = continuous_features[feature_idx]
            feature_values = X_cont_test[feature_name].values
            feature_kde = class_kdes[c][feature_idx]
            weight_sum = sum(
                weight for pair, weight in normalized_continuous_weights.items() if feature_name in pair
            )
            class_probs += weight_sum * np.log(feature_kde.evaluate(feature_values) + 1e-9)
        log_probs.append(class_probs)
    return np.array(log_probs).T

def predict_combined_weighted(X_cat_test, X_cont_test):
    cat_log_prob = cat_nb.predict_log_proba(X_cat_test)
    weighted_cat_log_prob = np.zeros_like(cat_log_prob)
    for feature1, feature2 in categorical_pairs:
        weight = normalized_categorical_weights.get((feature1, feature2), 0)
        weighted_cat_log_prob += weight * cat_log_prob
    kde_log_prob = predict_kde_weighted(X_cont_test)
    return weighted_cat_log_prob + kde_log_prob

cat_accuracy = accuracy_score(y_test, cat_nb.predict(X_test_cat))
kde_accuracy = accuracy_score(y_test, np.argmax(predict_kde(X_test_cont), axis=1))
combined_accuracy = accuracy_score(y_test, np.argmax(predict_combined(X_test_cat, X_test_cont), axis=1))
kde_accuracy_weighted = accuracy_score(y_test, np.argmax(predict_kde_weighted(X_test_cont), axis=1))
combined_accuracy_weighted = accuracy_score(y_test, np.argmax(predict_combined_weighted(X_test_cat, X_test_cont), axis=1))

#print plot again
models = ['CategoricalNB', 'KDE', 'Combined', 'KDE (Weighted)', 'Combined (Weighted)']
accuracies = [cat_accuracy, kde_accuracy, combined_accuracy, kde_accuracy_weighted, combined_accuracy_weighted]
plot.figure(figsize=(10, 6))
bars = plot.bar(models, accuracies, color=['blue', 'red', 'purple', 'orange', 'green'], alpha=0.7)
for bar, acc in zip(bars, accuracies):
    plot.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{acc:.2f}", ha='center', va='bottom', fontsize=10)
plot.ylim(0, 1)
plot.xlabel('Models')
plot.ylabel('Accuracy')
plot.title('Model Accuracies Comparison')
plot.grid(axis='y', linestyle='--', alpha=0.7)
plot.tight_layout()
plot.show()


def plot_kde_with_details(feature_data, bandwidth=None, feature_name="Feature", class_label=None):
    """
    Enhanced KDE visualization for interpretability and aesthetics.
    """
    #fit the KDE
    kde = gaussian_kde(feature_data, bw_method=bandwidth)
    
    #generate x values for the KDE plot
    x_min, x_max = np.percentile(feature_data, [5, 95])  # Focus on main data range
    x_values = np.linspace(x_min, x_max, 1000)
    kde_values = kde.evaluate(x_values)
    
    #plot the KDE
    sns.set(style="whitegrid")
    plot.figure(figsize=(10, 6))
    plot.plot(x_values, kde_values, label='KDE (Density)', color='blue', linewidth=2)
    plot.fill_between(x_values, kde_values, color='blue', alpha=0.3, label='Density Area')
    
    #plot a subset of individual kernel contributions
    for i, point in enumerate(feature_data):
        if i % 10 == 0:  # Plot every 10th kernel to avoid clutter
            kernel = norm(loc=point, scale=kde.factor).pdf(x_values)
            plot.plot(x_values, kernel, linestyle='--', color='lightgray', alpha=0.15)
    
    #highlight key statistics
    mean_value = np.mean(feature_data)
    median_value = np.median(feature_data)
    plot.axvline(mean_value, color='red', linestyle='--', label=f'Mean ({mean_value:.2f})')
    plot.axvline(median_value, color='green', linestyle='-.', label=f'Median ({median_value:.2f})')
    
    #add labels and title
    plot.title(f'KDE of {feature_name} for Class {class_label} (Mean: {mean_value:.2f}, Median: {median_value:.2f})', fontsize=14)
    plot.xlabel(feature_name, fontsize=12)
    plot.ylabel('Density', fontsize=12)
    plot.legend()
    plot.tight_layout()
    plot.ylim(0, max(kde_values) * 1.1)  # Adjust y-axis scaling
    plot.show()


# categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
# continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
 # select a specific feature and class label to visualize
feature_name = 'trestbps'  
class_label = 1  
feature_data = X_train[y_train == class_label][feature_name].values  

# feature_name = 'thalach'  
# class_label = 1  
# feature_data = X_train[y_train == class_label][feature_name].values 

# feature_name = 'cp'  
# class_label = 1 
# feature_data = X_train[y_train == class_label][feature_name].values 

# Call the plot_kde_with_details function
plot_kde_with_details(
    feature_data=feature_data,
    feature_name=feature_name,
    class_label=class_label
)
