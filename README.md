Depression and Anxiety Analysis
Overview
This project analyzes data related to depression and anxiety using machine learning techniques. It aims to provide insights and visualizations that help understand mental health patterns.
# DEPRESSION AND ANXIETY ANALYSIS


# Importing Libraries and Dataset

#, Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, r2_score

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
np.random.seed(42)

print("All libraries imported successfully!")
## Dataset Exploration
# Generate Synthetic Mental Health Dataset
def generate_mental_health_data(n_patients=1000):
    "Generate realistic synthetic mental health treatment data"
    
    np.random.seed(42)
    
    # Patient demographics
    age = np.random.normal(35, 12, n_patients).astype(int)
    age = np.clip(age, 18, 70)
    
    gender = np.random.choice(['Male', 'Female', 'Other'], n_patients, p=[0.4, 0.55, 0.05])
    
    # Socioeconomic factors
    income_level = np.random.choice(['Low', 'Medium', 'High'], n_patients, p=[0.3, 0.5, 0.2])
    education = np.random.choice(['High School', 'Bachelor', 'Graduate'], n_patients, p=[0.4, 0.4, 0.2])
    
    # Clinical characteristics
    primary_diagnosis = np.random.choice(['Depression', 'Anxiety', 'Mixed'], n_patients, p=[0.4, 0.35, 0.25])
    
    # Baseline severity scores (PHQ-9 for depression: 0-27, GAD-7 for anxiety: 0-21)
    baseline_phq9 = np.random.normal(15, 4, n_patients)
    baseline_phq9 = np.clip(baseline_phq9, 5, 27)
    
    baseline_gad7 = np.random.normal(12, 3, n_patients)
    baseline_gad7 = np.clip(baseline_gad7, 3, 21)
    
    # Treatment assignments
    treatment_type = np.random.choice(['Medication', 'Therapy', 'Combined'], n_patients, p=[0.3, 0.4, 0.3])
    
    # Treatment duration (weeks)
    treatment_duration = np.random.normal(12, 4, n_patients)
    treatment_duration = np.clip(treatment_duration, 4, 24)
    
    # Social support (1-10 scale)
    social_support = np.random.normal(6, 2, n_patients)
    social_support = np.clip(social_support, 1, 10)
    
    # Generate outcomes based on realistic relationships
    treatment_effect = np.where(treatment_type == 'Combined', 1.5,
                               np.where(treatment_type == 'Therapy', 1.2, 1.0))
    
    improvement_factor = (
        treatment_effect * 
        (social_support / 10) * 
        (treatment_duration / 12) * 
        np.random.normal(0.7, 0.2, n_patients)
    )
    
    followup_16w_phq9 = baseline_phq9 * (1 - improvement_factor * 0.6)
    followup_16w_phq9 = np.clip(followup_16w_phq9, 0, 27)
    
    followup_16w_gad7 = baseline_gad7 * (1 - improvement_factor * 0.55)
    followup_16w_gad7 = np.clip(followup_16w_gad7, 0, 21)
    
    # Treatment adherence and completion
    adherence_prob = 0.7 + (social_support - 5) * 0.05
    treatment_adherence = np.random.binomial(1, adherence_prob, n_patients)
    
    side_effects = np.random.choice(['None', 'Mild', 'Moderate', 'Severe'], 
                                  n_patients, p=[0.4, 0.35, 0.2, 0.05])
    
    completion_prob = 0.8 * treatment_adherence + 0.1
    treatment_completed = np.random.binomial(1, completion_prob, n_patients)
    
    # Create DataFrame
    data = pd.DataFrame({
        'patient_id': range(1, n_patients + 1),
        'age': age,
        'gender': gender,
        'income_level': income_level,
        'education': education,
        'primary_diagnosis': primary_diagnosis,
        'baseline_phq9': baseline_phq9.round(1),
        'baseline_gad7': baseline_gad7.round(1),
        'treatment_type': treatment_type,
        'treatment_duration_weeks': treatment_duration.round(1),
        'social_support_score': social_support.round(1),
        'followup_16w_phq9': followup_16w_phq9.round(1),
        'followup_16w_gad7': followup_16w_gad7.round(1),
        'treatment_adherence': treatment_adherence,
        'side_effects': side_effects,
        'treatment_completed': treatment_completed
    })
    
    return data

# Generate dataset
df = generate_mental_health_data(1000)
print(f"Dataset created with {len(df)} patients")
print(f"Dataset shape: {df.shape}")
# Dataset Overview
print("First 5 rows of the dataset:")
print(df.head())
print("\nBasic Statistics:")
print(df.describe())
## Calculate Treatment Outcomes
# Calculate improvement scores
df['phq9_improvement'] = df['baseline_phq9'] - df['followup_16w_phq9'] 
df['gad7_improvement'] = df['baseline_gad7'] - df['followup_16w_gad7']

# Calculate response rates (≥50% improvement)
df['phq9_response'] = (df['phq9_improvement'] / df['baseline_phq9']) >= 0.5
df['gad7_response'] = (df['gad7_improvement'] / df['baseline_gad7']) >= 0.5

# Calculate remission rates (PHQ-9 < 5, GAD-7 < 5)
df['phq9_remission'] = df['followup_16w_phq9'] < 5
df['gad7_remission'] = df['followup_16w_gad7'] < 5

print("Treatment outcome measures calculated!")
print(f"Average PHQ-9 improvement: {df['phq9_improvement'].mean():.2f} points")
print(f"PHQ-9 response rate: {df['phq9_response'].mean():.1%}")
print(f"PHQ-9 remission rate: {df['phq9_remission'].mean():.1%}")
# CELL 5: Treatment Effectiveness Analysis
print("Treatment Effectiveness by Type:")
effectiveness = df.groupby('treatment_type').agg({
    'phq9_improvement': 'mean',
    'gad7_improvement': 'mean', 
    'phq9_response': 'mean',
    'gad7_response': 'mean',
    'treatment_completed': 'mean'
}).round(3)

print(effectiveness)
# Data Visualization - Treatment Outcomes
plt.figure(figsize=(15, 10))
#PHQ-9 improvement by treatment type
plt.subplot(2, 3, 1)
sns.boxplot(data=df, x='treatment_type', y='phq9_improvement')
plt.title('PHQ-9 Improvement by Treatment Type')
plt.ylabel('Improvement Score')
plt.tight_layout()
plt.show()
### Response rates by treatment type
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 2)
response_data = df.groupby('treatment_type')[['phq9_response', 'gad7_response']].mean()
response_data.plot(kind='bar', ax=plt.gca())
plt.title('Response Rates by Treatment Type')
plt.ylabel('Response Rate')
plt.legend(['PHQ-9 Response', 'GAD-7 Response'])
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
### Treatment completion rates
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 3)
completion_rates = df.groupby('treatment_type')['treatment_completed'].mean()
completion_rates.plot(kind='bar', color='green', ax=plt.gca())
plt.title('Treatment Completion Rates')
plt.ylabel('Completion Rate')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
### Age distribution
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 4)
plt.hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
### Baseline vs Follow-up PHQ-9
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 5)
plt.scatter(df['baseline_phq9'], df['followup_16w_phq9'], alpha=0.6)
plt.plot([0, 27], [0, 27], 'r--', label='No Change')
plt.xlabel('Baseline PHQ-9')
plt.ylabel('16-week Follow-up PHQ-9')
plt.title('Baseline vs Follow-up PHQ-9')
plt.legend()
plt.tight_layout()
plt.show()

###  Social support impact
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 6)
plt.scatter(df['social_support_score'], df['phq9_improvement'], alpha=0.6, color='orange')
plt.xlabel('Social Support Score')
plt.ylabel('PHQ-9 Improvement')
plt.title('Social Support vs Improvement')

plt.tight_layout()
plt.show()
## Statistical Testing
# ANOVA for treatment effectiveness
print("ANOVA: Treatment Type vs PHQ-9 Improvement")
model = ols('phq9_improvement ~ C(treatment_type)', data=df).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

print("\n Pairwise T-tests for PHQ-9 Improvement:")
treatment_groups = df['treatment_type'].unique()
for i, group1 in enumerate(treatment_groups):
    for group2 in treatment_groups[i+1:]:
        data1 = df[df['treatment_type'] == group1]['phq9_improvement']
        data2 = df[df['treatment_type'] == group2]['phq9_improvement']
        t_stat, p_value = ttest_ind(data1, data2)
        print(f"{group1} vs {group2}: t={t_stat:.3f}, p={p_value:.3f}")
##  Correlation Analysis
# Select numeric variables for correlation
numeric_vars = ['age', 'baseline_phq9', 'baseline_gad7', 'treatment_duration_weeks', 
                'social_support_score', 'phq9_improvement', 'gad7_improvement']

correlation_matrix = df[numeric_vars].corr()
print(" Correlation Matrix:")
print(correlation_matrix.round(3))
### Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.3f')
plt.title('Correlation Matrix of Key Variables')
plt.tight_layout()
plt.show()
## Predictive Modeling Preparation
# Prepare features for modeling
features = ['age', 'baseline_phq9', 'baseline_gad7', 'treatment_duration_weeks', 
           'social_support_score']

# Encode categorical variables
le_gender = LabelEncoder()
le_treatment = LabelEncoder()
le_income = LabelEncoder()

df['gender_encoded'] = le_gender.fit_transform(df['gender'])
df['treatment_encoded'] = le_treatment.fit_transform(df['treatment_type'])
df['income_encoded'] = le_income.fit_transform(df['income_level'])

features.extend(['gender_encoded', 'treatment_encoded', 'income_encoded'])

X = df[features]
y_response = df['phq9_response'].astype(int)
y_improvement = df['phq9_improvement']

print(f"Features prepared for modeling: {len(features)} features")
print(f"Feature names: {features}")
#  Machine Learning - Treatment Response Prediction

