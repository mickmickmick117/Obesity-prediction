# -*- coding: utf-8 -*-
"""
Created on Thu May  8 22:53:01 2025

@author: miroz
"""

#       Obesity Prediction Analysis 

# # Obesity Prediction Analysis

# Obesity is a medical condition, considered by multiple organizations to be a
# disease,] in which excess body fat has accumulated to such an extent that it
# can potentially have negative effects on health. People are classified as
# obese when their body mass index (BMI)—a person's weight divided by the
# square of the person's height—is over 30 kg/m2; the range 25–30 kg/m2 is
# defined as overweight.Some East Asian countries use lower values to calculate
# obesity. Obesity is a major cause of disability and is correlated with various
# diseases and conditions, particularly cardiovascular diseases, type 2 diabetes,
# obstructive sleep apnea, certain types of cancer, and osteoarthritis.

# ## About the Dataset

# This dataset helps estimate obesity levels based on eating habits, family
# history and physical condition. It includes data from individuals in Mexico,
# Peru, and Colombia, covering 16 lifestyle and health-related features with
# 2111 records. The labels classify obesity levels, ranging from underweight
# to different obesity types.

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For enhanced data visualization
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For data preprocessing
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier  # Ensemble learning methods
from sklearn.tree import DecisionTreeClassifier  # Decision tree classifier
from sklearn.linear_model import LogisticRegression  # Logistic regression classifier
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.metrics import accuracy_score, classification_report  # Model evaluation metrics
import warnings  # For handling warnings
import logging  # For logging information

# Configure warnings and logging
warnings.filterwarnings('ignore')  # Ignore warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Set up logging configuration

# Load the dataset from CSV file
df = pd.read_csv(r"C:\Users\miroz\OneDrive\Documents\Miro\Miro\Python\ML & AI\ML & AI Project\Obesity Prediction Analysis\ObesityDataSet_raw_and_data_sinthetic.csv")



# Handle outliers in specified features
features_to_clean = ['Age', 'NCP']  # Features to clean
for feature in features_to_clean:
    Q1 = df[feature].quantile(0.25)  # Calculate first quartile
    Q3 = df[feature].quantile(0.75)  # Calculate third quartile
    IQR = Q3 - Q1  # Calculate interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Calculate lower bound
    upper_bound = Q3 + 1.5 * IQR  # Calculate upper bound
    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]  # Filter outliers

# --- BMI Calculation ---
logging.info("\n--- BMI Calculation ---")  # Log BMI calculation section
df['BMI'] = df['Weight'] / (df['Height']**2)  # Calculate BMI (weight/height^2)
logging.info(df[['Height', 'Weight', 'BMI']].head().to_string())  # Log first few rows of height, weight and BMI

# --- Add BMI to the dataset ---
logging.info("\n--- Dataset with BMI ---")  # Log dataset with BMI section
logging.info(df.head().to_string())  # Log first few rows of the dataset

# Define function to categorize BMI according to WHO standards
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal weight'
    elif 25 <= bmi < 30:
        return 'Overweight'
    elif 30 <= bmi < 35:
        return 'Obesity I'
    elif 35 <= bmi < 40:
        return 'Obesity II'
    else:
        return 'Obesity III'

# Apply BMI categorization to the dataset
df['Obesity_Class_BMI'] = df['BMI'].apply(categorize_bmi)
logging.info("\n--- Dataset with BMI and Obesity Class (BMI-based) ---")  # Log dataset with BMI classes
logging.info(df[['BMI', 'NObeyesdad', 'Obesity_Class_BMI']].head().to_string())  # Log relevant columns

# --- Plot BMI distribution across original obesity classes ---
logging.info("\n--- Plotting BMI Distribution across Original Obesity Classes ---")  # Log plot section
plt.figure(figsize=(12, 7))  # Set figure size
sns.histplot(data=df, x='BMI', hue='NObeyesdad', kde=True, multiple="stack")  # Create stacked histogram with KDE
plt.title('Distribution of BMI Across Original Obesity Levels')  # Set title
plt.xlabel('BMI')  # Set x-axis label
plt.ylabel('Frequency')  # Set y-axis label
plt.legend(title='Obesity Level')  # Add legend with title
plt.tight_layout()  # Adjust layout
plt.show()  # Display plot

# --- Plot boxplot of BMI by original obesity level ---
logging.info("\n--- Plotting Boxplot of BMI by Original Obesity Level (Reordered) ---")  # Log boxplot section
original_order = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 
                 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']  # Define order of categories
plt.figure(figsize=(12, 7))  # Set figure size
sns.boxplot(x='NObeyesdad', y='BMI', data=df, order=original_order, palette='viridis')  # Create boxplot with viridis palette
plt.title('Box Plot of BMI by Obesity Level')  # Set title
plt.xlabel('Obesity Level')  # Set x-axis label
plt.ylabel('BMI')  # Set y-axis label
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.tight_layout()  # Adjust layout
plt.show()  # Display plot

# --- Calculate and log BMI statistics per original obesity class ---
logging.info("\n--- BMI Statistics per Original Obesity Class ---")  # Log statistics section
bmi_stats_original = df.groupby('NObeyesdad')['BMI'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])  # Calculate stats
logging.info("\n" + bmi_stats_original.to_string())  # Log statistics

# --- Plot BMI distribution across BMI-based classes ---
logging.info("\n--- Plotting BMI Distribution across BMI-based Obesity Classes ---")  # Log plot section
plt.figure(figsize=(12, 7))  # Set figure size
sns.histplot(data=df, x='BMI', hue='Obesity_Class_BMI', kde=True, multiple="stack")  # Create stacked histogram
plt.title('Distribution of BMI Across BMI-based Obesity Classes')  # Set title
plt.xlabel('BMI')  # Set x-axis label
plt.ylabel('Frequency')  # Set y-axis label
plt.legend(title='Obesity Class (BMI)')  # Add legend with title
plt.tight_layout()  # Adjust layout
plt.show()  # Display plot

# --- Plot boxplot of BMI by BMI-based classes ---
logging.info("\n--- Plotting Boxplot of BMI by BMI-based Obesity Class ---")  # Log boxplot section
bmi_order = ['Underweight', 'Normal weight', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III']  # Define order
plt.figure(figsize=(10, 7))  # Set figure size
sns.boxplot(x='Obesity_Class_BMI', y='BMI', data=df, order=bmi_order)  # Create boxplot
plt.title('Box Plot of BMI by BMI-based Obesity Class')  # Set title
plt.xlabel('Obesity Class (BMI)')  # Set x-axis label
plt.ylabel('BMI')  # Set y-axis label
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.tight_layout()  # Adjust layout
plt.show()  # Display plot

# --- Calculate and log BMI statistics per BMI-based class ---
logging.info("\n--- BMI Statistics per BMI-based Obesity Class ---")  # Log statistics section
bmi_stats_bmi_based = df.groupby('Obesity_Class_BMI')['BMI'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])  # Calculate stats
logging.info("\n" + bmi_stats_bmi_based.to_string())  # Log statistics

# --- Data Exploration and Preprocessing ---
logging.info("\n--- Data Exploration and Preprocessing ---")  # Log data exploration section
logging.info(df.head().to_string())  # Log first few rows
logging.info("\n" + df.describe().to_string())  # Log descriptive statistics
logging.info("\nDataset Info:")  # Log dataset info
df.info(verbose=False)  # Get concise dataset info
logging.info(f"\nMissing Values:\n{df.isnull().sum().to_string()}")  # Log missing values

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()  # Get numeric columns
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()  # Get categorical columns

logging.info(f"\nNumeric Columns: {numeric_cols}")  # Log numeric columns
logging.info(f"Categorical Columns: {categorical_cols}")  # Log categorical columns

# Create bar plots for each categorical column
for col in categorical_cols:
    plt.figure(figsize=(8, 6))  # Set figure size
    sns.countplot(x=col, data=df, hue='NObeyesdad')  # Create count plot
    plt.title(f"Chart for {col} by Obesity Level")  # Set title
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.legend(title='Obesity Level')  # Add legend
    plt.tight_layout()  # Adjust layout
    plt.show()  # Display plot

# Create histograms for each numeric column
for col in numeric_cols:
    plt.figure(figsize=(10, 6))  # Set figure size
    sns.histplot(data=df, x=col, hue='NObeyesdad', kde=True, multiple="stack")  # Create histogram with KDE
    plt.title(f"Distribution of {col} by Obesity Level")  # Set title
    plt.xlabel(col)  # Set x-axis label
    plt.ylabel("Frequency")  # Set y-axis label
    plt.legend(title='Obesity Level')  # Add legend
    plt.tight_layout()  # Adjust layout
    plt.show()  # Display plot

# Create box plots for numeric features by obesity level
for col in numeric_cols:
    plt.figure(figsize=(8, 6))  # Set figure size
    order_list = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 
                 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']  # Define order
    if col == 'BMI':
        sns.boxplot(x='NObeyesdad', y=col, data=df, order=order_list, hue='Gender')  # Create boxplot with gender hue for BMI
    else:
        sns.boxplot(x='NObeyesdad', y=col, data=df, order=order_list)  # Create boxplot for other features
    plt.title(f"Box Plot of {col} by Obesity Level")  # Set title
    plt.xlabel("Obesity Level")  # Set x-axis label
    plt.ylabel(col)  # Set y-axis label
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.tight_layout()  # Adjust layout
    plt.show()  # Display plot



# Encode the target variable
le = LabelEncoder()  # Initialize label encoder
df['NObeyesdad_Encoded'] = le.fit_transform(df['NObeyesdad'])  # Encode target variable
target_classes = le.classes_  # Get class names

# Prepare features and target for modeling
X = pd.get_dummies(df.drop(columns=['NObeyesdad', 'NObeyesdad_Encoded', 'Obesity_Class_BMI', 'Weight', 'Height']))  # Create feature matrix
X = pd.concat([X, df['BMI']], axis=1)  # Add BMI back to features
y_encoded = df['NObeyesdad_Encoded']  # Set target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Standardize numerical features
scaler = StandardScaler()  # Initialize scaler
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)  # Transform test data
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)  # Create DataFrame for scaled training data
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)  # Create DataFrame for scaled test data

# --- Feature Importance Analysis ---
logging.info("\n--- Feature Importance (RandomForest) ---")  # Log feature importance section

# Split data by gender for separate analysis
male_df = df[df['Gender'] == 'Male']  # Male subset
female_df = df[df['Gender'] == 'Female']  # Female subset

# Prepare male data for feature importance analysis
X_male = pd.get_dummies(male_df.drop(columns=['NObeyesdad', 'NObeyesdad_Encoded', 'Obesity_Class_BMI', 'Weight', 'Height', 'Gender']))
X_male = pd.concat([X_male, male_df['BMI']], axis=1)  # Add BMI
y_male = male_df['NObeyesdad_Encoded']  # Male target

# Prepare female data for feature importance analysis
X_female = pd.get_dummies(female_df.drop(columns=['NObeyesdad', 'NObeyesdad_Encoded', 'Obesity_Class_BMI', 'Weight', 'Height', 'Gender']))
X_female = pd.concat([X_female, female_df['BMI']], axis=1)  # Add BMI
y_female = female_df['NObeyesdad_Encoded']  # Female target

# Function to plot feature importance
def plot_feature_importance(X, y, title):
    """
    Calculates and plots feature importance using RandomForestClassifier.
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        title (str): The title of the plot.
    """
    # Check class distribution
    class_counts = y.value_counts()
    if any(class_counts < 2):
        logging.warning(f"Warning: Some classes in '{title}' have fewer than 2 samples.")
        min_class = class_counts.idxmin()
        min_class_label = target_classes[min_class]
        logging.warning(f"The class with the fewest samples is: {min_class_label}")
        
        # Handle small classes by merging with closest class
        label_to_num = {label: i for i, label in enumerate(target_classes)}
        y_original = df[df['Gender'] == title.split()[-2]]['NObeyesdad'].map(label_to_num)
        
        closest_class = None
        min_distance = float('inf')
        for i, label in enumerate(target_classes):
            if i != min_class:
                distance = abs(i - min_class)
                if distance < min_distance:
                    min_distance = distance
                    closest_class = i
        closest_class_label = target_classes[closest_class]
        logging.warning(f"Merging this class with: {closest_class_label}")
        
        y = y.replace(min_class, closest_class)
        df.loc[df['Gender'] == title.split()[-2], 'NObeyesdad_Encoded'] = y
        y = df[df['Gender'] == title.split()[-2]]['NObeyesdad_Encoded']
        logging.warning(f"New class distribution: \n{y.value_counts()}")

    # Split data and train RandomForest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rf_model_importance = RandomForestClassifier(random_state=42)
    rf_model_importance.fit(X_train, y_train)
    
    # Get and sort feature importances
    feature_importances = rf_model_importance.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Log and plot results
    logging.info(f"\nFeature Importance (DataFrame) for {title}:\n" + feature_importance_df.to_string())
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Plot feature importance for males and females
plot_feature_importance(X_male, y_male, "Feature Importance for Male Participants")
plot_feature_importance(X_female, y_female, "Feature Importance for Female Participants")

# --- Analyzing Feature Distributions per Class ---
logging.info("\n--- Analyzing Feature Statistics per Obesity Level ---")
for col in numeric_cols:
    stats_df = df.groupby('NObeyesdad')[col].agg(['mean', 'median', 'std', 'min', 'max'])
    logging.info(f"\n--- Statistics for {col} by Obesity Level ---\n{stats_df.to_string()}")

logging.info("\n--- Analyzing Value Counts for Categorical Features per Obesity Level ---")
for col in categorical_cols:
    cross_tab = pd.crosstab(df[col], df['NObeyesdad'])
    logging.info(f"\n--- Value Counts for {col} by Obesity Level ---\n{cross_tab.to_string()}")
    cross_tab_norm = pd.crosstab(df[col], df['NObeyesdad'], normalize='columns')
    logging.info(f"\n--- Proportions for {col} by Obesity Level ---\n{cross_tab_norm.to_string()}")

# --- AdaBoost with Different Weak Learners ---
logging.info("\n--- AdaBoost with Different Weak Learners ---")
results_adaboost = {}  # Store AdaBoost results
y_pred_dict = {}  # Store predictions

def train_and_evaluate_adaboost(weak_learner, X_train, y_train, X_test, y_test, target_names, name, n_estimators=50, learning_rate=1.0):
    """
    Train and evaluate AdaBoost classifier with specified weak learner.
    Args:
        weak_learner: The base estimator
        X_train, y_train: Training data
        X_test, y_test: Test data
        target_names: Names of target classes
        name: Name for the model
        n_estimators: Number of estimators
        learning_rate: Learning rate
    """
    ada_model = AdaBoostClassifier(estimator=weak_learner, n_estimators=n_estimators, 
                                 learning_rate=learning_rate, random_state=42)
    ada_model.fit(X_train, y_train)
    y_pred = ada_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Raw Accuracy for {name}: {accuracy} ---")
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    results_adaboost[name] = {'accuracy': float(accuracy), 'report': report}
    y_pred_dict[name] = y_pred
    logging.info(f"\n--- AdaBoost with {name} ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=target_names)}")
    return accuracy

# Test different weak learners
train_and_evaluate_adaboost(DecisionTreeClassifier(max_depth=1), X_train_scaled_df, y_train, 
                           X_test_scaled_df, y_test, target_classes, name='Decision Tree (depth 1)')
train_and_evaluate_adaboost(DecisionTreeClassifier(max_depth=3), X_train_scaled_df, y_train, 
                           X_test_scaled_df, y_test, target_classes, name='Decision Tree (depth 3)')
train_and_evaluate_adaboost(DecisionTreeClassifier(max_depth=5), X_train_scaled_df, y_train, 
                           X_test_scaled_df, y_test, target_classes, name='Decision Tree (depth 5)')
train_and_evaluate_adaboost(LogisticRegression(solver='liblinear', random_state=42), X_train_scaled_df, y_train, 
                           X_test_scaled_df, y_test, target_classes, name='Logistic Regression')
train_and_evaluate_adaboost(SVC(probability=True, random_state=42, kernel='linear'), X_train_scaled_df, y_train, 
                           X_test_scaled_df, y_test, target_classes, name='SVC (linear, n_estimators=20)', n_estimators=20)

# --- Plot AdaBoost Results ---
model_names_adaboost = list(results_adaboost.keys())
accuracies_adaboost = [results_adaboost[name]['accuracy'] for name in model_names_adaboost]

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names_adaboost, y=accuracies_adaboost, palette='viridis')
plt.title('AdaBoost Performance with Different Weak Learners')
plt.xlabel('Weak Learner')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Save AdaBoost Results ---
results_df_adaboost = pd.DataFrame.from_dict(results_adaboost, orient='index')
logging.info("\n--- AdaBoost Results (DataFrame) ---")
logging.info("\n" + results_df_adaboost.to_string())
results_df_adaboost.to_csv('adaboost_results.csv', index_label='Model')
logging.info("\nAdaBoost results saved to adaboost_results.csv")

# Create simplified results DataFrame
results_data = {}
for model_name, data in results_adaboost.items():
    results_data[model_name] = {'accuracy': data['accuracy']}

results_df_adaboost = pd.DataFrame.from_dict(results_data, orient='index')
results_df_adaboost.index.name = 'Model'
logging.info("\n--- AdaBoost Results (DataFrame) ---")
logging.info("\n" + results_df_adaboost.to_string(float_format='%.2f'))
results_df_adaboost.to_csv('adaboost_results.csv', float_format='%.2f')
logging.info("\nAdaBoost results (accuracy) saved to adaboost_results.csv")

# Create detailed classification reports
detailed_results_list = []
for model_name, report_data in results_adaboost.items():
    report = report_data['report']
    flat_report = {'model': model_name}
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            flat_report[f'{class_name}_precision'] = metrics['precision']
            flat_report[f'{class_name}_recall'] = metrics['recall']
            flat_report[f'{class_name}_f1-score'] = metrics['f1-score']
            flat_report[f'{class_name}_support'] = metrics['support']
        elif class_name in ['accuracy', 'macro avg', 'weighted avg']:
            flat_report[class_name] = metrics
    detailed_results_list.append(flat_report)

detailed_df_adaboost_corrected = pd.DataFrame(detailed_results_list)
logging.info("\n--- Detailed AdaBoost Classification Reports (DataFrame - Corrected) ---")
logging.info("\n" + detailed_df_adaboost_corrected.to_string(float_format='%.2f'))
detailed_df_adaboost_corrected.to_csv('adaboost_detailed_results.csv', index=False, float_format='%.2f')
logging.info("\nDetailed AdaBoost results saved to adaboost_detailed_results.csv")

# --- Plot Logistic Regression Coefficients from AdaBoost ---
logging.info("\n--- Coefficients of Logistic Regression (as weak learner in AdaBoost) ---")
ada_lr = AdaBoostClassifier(estimator=LogisticRegression(solver='liblinear', random_state=42), 
                           n_estimators=50, random_state=42)
ada_lr.fit(X_train_scaled_df, y_train)

if hasattr(ada_lr.estimator_, 'coef_'):
    num_classes = len(target_classes)
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 7))
    if num_classes == 1:
        axes = [axes]

    for i, class_name in enumerate(target_classes):
        coefficients = ada_lr.estimator_.coef_[i]
        importance_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': coefficients})
        importance_df['Abs_Coefficient'] = np.abs(importance_df['Coefficient'])
        importance_df = importance_df.sort_values(by='Abs_Coefficient', ascending=False).head(10)
        importance_df_sorted = importance_df.sort_values(by='Coefficient', ascending=False)
        sns.barplot(x='Coefficient', y='Feature', data=importance_df_sorted, ax=axes[i], palette='coolwarm_r')
        axes[i].set_title(f'Top Features (LR in AdaBoost) for: {class_name}')
        axes[i].set_xlabel('Coefficient Value')
        axes[i].set_ylabel('Feature')

    plt.tight_layout()
    plt.show()
else:
    logging.info("The weak learner (Logistic Regression) does not have 'coef_' attribute after being used in AdaBoost.")