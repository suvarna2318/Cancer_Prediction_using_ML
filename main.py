import streamlit as st
import pandas as pd
import warnings

# Filter out warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Define your preprocessing pipeline ---
def create_preprocessing_pipeline():
    # Define numerical and categorical feature names based on your dataset
    numerical_features = ['Age', 'Tumor_Size']
    categorical_features = ['Gender', 'Tumor_Grade', 'Symptoms_Severity',
                            'Family_History', 'Smoking_History', 
                            'Alcohol_Consumption', 'Exercise_Frequency']
    
    # Pipeline for numerical features: impute missing values and scale
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for categorical features: impute missing values and encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )
    return preprocessor

# --- Load dataset using st.cache_data ---
@st.cache_data
def load_data():
    data = pd.read_csv(r'cancer_prediction_data .csv')
    return data

data = load_data()

st.title("🩺Cancer Prediction App🎗️")
#st.write("This app uses an SVM model to predict the presence of cancer based on patient data.")

#if st.checkbox("Show Data Preview"):
    #st.write(data.head())

# Assume the target column is 'Cancer_Present'
target_col = 'Cancer_Present'
if target_col not in data.columns:
    st.error(f"Target column '{target_col}' not found in data!")
    st.stop()

# Split features and target
X = data.drop(columns=[target_col])
y = data[target_col]

# Create preprocessing pipeline
preprocess = create_preprocessing_pipeline()

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Create and train the SVM model pipeline ---
svm_pipeline = Pipeline([
    ('preprocessing', preprocess),
    ('svm', SVC())
])

svm_pipeline.fit(X_train, y_train)
svm_accuracy = svm_pipeline.score(X_test, y_test)
#st.write(f"**Model Training Complete!** SVM Accuracy on Test Data: {svm_accuracy * 100:.2f}%")

# --- Sidebar for user input ---
st.sidebar.header("Enter Patient Data")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=50)
    tumor_size = st.sidebar.number_input("Tumor Size", min_value=0.0, max_value=100.0, value=5.0)
    
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    tumor_grade = st.sidebar.selectbox("Tumor Grade", options=["Low", "Medium", "High"])
    symptoms_severity = st.sidebar.selectbox("Symptoms Severity", options=["Mild", "Moderate", "Severe"])
    family_history = st.sidebar.selectbox("Family History", options=["Yes", "No"])
    smoking_history = st.sidebar.selectbox("Smoking History", options=["Current Smoker", "Non-Smoker"])
    alcohol_consumption = st.sidebar.selectbox("Alcohol Consumption", options=["Low", "Moderate", "High"])
    exercise_frequency = st.sidebar.selectbox("Exercise Frequency", options=["Never", "Rarely", "Occasionally", "Often"])
    
    # Create a dictionary of features. Keys must match dataset columns.
    data_dict = {
        'Age': age,
        'Tumor_Size': tumor_size,
        'Gender': gender,
        'Tumor_Grade': tumor_grade,
        'Symptoms_Severity': symptoms_severity,
        'Family_History': family_history,
        'Smoking_History': smoking_history,
        'Alcohol_Consumption': alcohol_consumption,
        'Exercise_Frequency': exercise_frequency
    }
    return pd.DataFrame(data_dict, index=[0])

st.sidebar.markdown("### Patient Data Input")
input_df = user_input_features()

st.subheader("User Input Data")
st.write(input_df)

# --- Make Prediction ---
if st.button("Predict Cancer Presence"):
    prediction = svm_pipeline.predict(input_df)
    result = "🛑 Cancer Detected" if prediction[0] == 1 else "✅No Cancer Detected"
    st.write(f"### Prediction: {result}")
