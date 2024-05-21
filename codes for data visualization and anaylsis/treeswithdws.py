import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load activity data: sit, std, wlk
sit = pd.read_csv('sit.csv')
std = pd.read_csv('std.csv')
wlk = pd.read_csv('wlk.csv')

# Load ups data: ups1, ups2, ups3
ups1 = pd.read_csv('ups4.csv')
ups2 = pd.read_csv('ups5.csv')
ups3 = pd.read_csv('ups6.csv')

dws1 = pd.read_csv('dws1.csv')
dws2 = pd.read_csv('dws2.csv')
dws3 = pd.read_csv('dws3.csv')

# Concatenate ups data into one DataFrame
ups = pd.concat([ups1, ups2,ups3])
dws=pd.concat([dws1,dws2,dws3])

# Assign activity labels to each dataset
sit['act'] = 0
std['act'] = 1
wlk['act'] = 2
ups['act'] = 3
dws['act']=4

# Calculate the size of each part to ensure nearly equal numbers of rows
sit_part_size = -(-sit.shape[0] // 90)  # Ceiling division to handle odd numbers
std_part_size = -(-std.shape[0] // 90)  # Ceiling division to handle odd numbers
wlk_part_size = -(-wlk.shape[0] // 90)  # Ceiling division to handle odd numbers
ups_part_size = -(-ups.shape[0] // 90)  # Ceiling division to handle odd numbers
dws_part_size = -(-dws.shape[0] // 90)  # Ceiling division to handle odd numbers

# Add trial column to each dataset
sit['trial'] = [i // sit_part_size + 1 for i in range(sit.shape[0])]
std['trial'] = [i // std_part_size + 1 for i in range(std.shape[0])]
wlk['trial'] = [i // wlk_part_size + 1 for i in range(wlk.shape[0])]
ups['trial'] = [i // ups_part_size + 1 for i in range(ups.shape[0])]
dws['trial'] = [i // ups_part_size + 1 for i in range(dws.shape[0])]

# Concatenate all datasets into one final dataset
final_data = pd.concat([sit, std, wlk, ups,dws])

# Optional: Reset the index
final_data.reset_index(drop=True, inplace=True)

# Define the number of trials to select for training
num_train_trials = 10
# Initialize lists to store accuracy scores
accuracy_scores = []
feature_importances_list = []
confusion_matrices = []

# Initialize lists to store accuracy scores for each activity
accuracy_by_activity = [[] for _ in range(5)]  # 5 activities
activities = ["sit", "std", "wlk","ups","dws"]

# Define number of iterations
num_iterations = 5

for i in range(num_iterations):
    # Split data into training and testing sets
    train_indices = []
    test_indices = []
    
    for activity_id in final_data['act'].unique():
        train_trials = np.random.choice(np.arange(1, 91), size=num_train_trials, replace=False)
        train_indices.extend(final_data[(final_data['act'] == activity_id) &
                                        (final_data['trial'].isin(train_trials))].index)
        test_indices.extend(final_data[(final_data['act'] == activity_id) & 
                                       (~final_data['trial'].isin(train_trials))].index)
    
    train_data = final_data.loc[train_indices]
    test_data = final_data.loc[test_indices]
    
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    
    train_data.drop('Time', axis=1, inplace=True)
    test_data.drop('Time', axis=1, inplace=True)
    
    # Group the training data by both 'trial' and 'act' columns and aggregate the features
    X_train_grouped = train_data.groupby(['trial','act']).agg(['mean', 'std', 'min', 'max', 'median', 'sum',  'var', 'skew'])
    X_train_grouped.reset_index(inplace=True)
    y_train_grouped = X_train_grouped['act']
    
    # Group the test data by both 'trial' and 'act' columns and aggregate the features
    X_test_grouped = test_data.groupby(['trial','act']).agg(['mean', 'std', 'min', 'max', 'median', 'sum',  'var', 'skew'])
    X_test_grouped.reset_index(inplace=True)
    y_test_grouped = X_test_grouped['act']
    
    # Exclude 'trial' and 'act' columns from feature scaling and fitting
    X_train_grouped_features = X_train_grouped.drop(['trial', 'act'], axis=1)
    X_test_grouped_features = X_test_grouped.drop(['trial', 'act'], axis=1)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_grouped_scaled = scaler.fit_transform(X_train_grouped_features)
    X_test_grouped_scaled = scaler.transform(X_test_grouped_features)
    
    # Initialize the decision tree classifier
    clf = DecisionTreeClassifier()
    
    
    
    
    clf.fit(X_train_grouped_scaled, y_train_grouped)
    
    # Predictions for each trial in the test set
    y_pred_grouped = clf.predict(X_test_grouped_scaled)
    
    # Evaluate the model
    accuracy_grouped = accuracy_score(y_test_grouped, y_pred_grouped)
    print("Accuracy:", accuracy_grouped)
    accuracy_scores.append(accuracy_grouped)
    
    for idx, activity_name in enumerate(activities):
        accuracy_by_activity[idx].append(accuracy_score(y_test_grouped[y_test_grouped == idx], 
                                                        y_pred_grouped[y_test_grouped == idx]))
    for idx, activity_name in enumerate(activities):
        print(f"Iteration {i+1}, Activity {activity_name} Accuracy: {accuracy_by_activity[idx][-1]}")

    # Calculate confusion matrix and classification report
    confusion_matrices.append(confusion_matrix(y_test_grouped, y_pred_grouped))
    
    
    
    # Append feature importances to the list
    feature_importances_list.append(clf.feature_importances_)
# Perform PCA separately for training and testing data
pca_test = PCA(n_components=2)
pca_test_result = pca_test.fit_transform(X_test_grouped_scaled)

# Create DataFrame for PCA results for testing data
pca_test_df = pd.DataFrame(data=pca_test_result, columns=['PC1', 'PC2'])
pca_test_df = pd.concat([pca_test_df, test_data[['act', 'trial']]], axis=1)

# Plot PCA visualization for testing data with true labels
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_test_result[:, 0], y=pca_test_result[:, 1], hue=y_test_grouped,palette='Set1',s=50)
plt.title('PCA Visualization of Testing Data (True Labels)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Activity', loc='best')

plt.show()

# Plot PCA visualization for testing data with predicted labels
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_test_result[:, 0], y=pca_test_result[:, 1], hue=y_pred_grouped,palette='Set1',s=50)
plt.title('PCA Visualization of Testing Data (Predicted Labels)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Activity')
plt.show()    
# Plotting the average feature importances
avg_feature_importances = np.mean(feature_importances_list, axis=0)
column_names = ['_'.join(col) for col in X_train_grouped_features.columns]
plt.bar(column_names, avg_feature_importances)
plt.title('Average Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()    

# Calculate overall accuracy
overall_accuracy = np.mean(accuracy_scores)
print("Overall Accuracy (Trial Level):", overall_accuracy)

# Plot average accuracy for each activity
avg_accuracy_by_activity = [np.mean(activity_accuracies) for activity_accuracies in accuracy_by_activity]
plt.bar(activities, avg_accuracy_by_activity)
plt.title('Average Accuracy by Activity')
plt.xlabel('Activity')
plt.ylabel('Average Accuracy')
plt.ylim(0, 1)
plt.show()

# Plot overall confusion matrix
overall_confusion_matrix = np.sum(confusion_matrices, axis=0)
# Plot overall confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(overall_confusion_matrix, fmt='d', cmap='Blues', xticklabels=activities, yticklabels=activities)
plt.title('Overall Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)  # Adjust rotation as needed
plt.show()


# Plot Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X_train_grouped_features.columns, class_names=activities)
plt.title('Decision Tree')
plt.show()
