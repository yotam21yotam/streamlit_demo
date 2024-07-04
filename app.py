import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, roc_curve

# Title of the app
st.title("Iris Dataset Machine Learning App")

# Load the Iris dataset
@st.cache
def load_data():
    data = sns.load_dataset('iris')
    return data

data = load_data()

# Display the dataset
st.subheader("Iris Dataset")
st.write(data.head())

# Data wrangling
st.subheader("Data Wrangling")
st.write("Checking for missing values:")
st.write(data.isnull().sum())

# Correlation analysis
st.subheader("Correlation Analysis")
numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
corr = numeric_data.corr(method='pearson')
st.write(corr)

# Correlation heatmap
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
plt.title('Correlation Heatmap')
st.pyplot(fig)

# Splitting data
X = data.drop(columns=['species'])
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar for model selection
st.sidebar.header("Model Selection")

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[model_choice]

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

# Evaluation metrics
st.subheader(f'{model_choice} Evaluation Metrics')

if y_prob is not None:
    y_prob_multiclass = pd.get_dummies(y_test)
    auc = roc_auc_score(y_prob_multiclass, y_prob, multi_class='ovr')
    st.write(f"AUC: {auc:.2f}")

cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")

# Confusion matrix
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot(fig)

# ROC Curve
if y_prob is not None:
    fpr = {}
    tpr = {}
    for i, label in enumerate(y_prob_multiclass.columns):
        fpr[label], tpr[label], _ = roc_curve(y_prob_multiclass.iloc[:, i], y_prob[:, i])
        plt.plot(fpr[label], tpr[label], label=f'ROC curve (area = {auc:.2f}) for label {label}')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# User input for prediction
st.sidebar.header("User Input for Prediction")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', float(data.sepal_length.min()), float(data.sepal_length.max()), float(data.sepal_length.mean()))
    sepal_width = st.sidebar.slider('Sepal width', float(data.sepal_width.min()), float(data.sepal_width.max()), float(data.sepal_width.mean()))
    petal_length = st.sidebar.slider('Petal length', float(data.petal_length.min()), float(data.petal_length.max()), float(data.petal_length.mean()))
    petal_width = st.sidebar.slider('Petal width', float(data.petal_width.min()), float(data.petal_width.max()), float(data.petal_width.mean()))
    input_data = {'sepal_length': sepal_length,
                  'sepal_width': sepal_width,
                  'petal_length': petal_length,
                  'petal_width': petal_width}
    features = pd.DataFrame(input_data, index=[0])
    return features

input_df = user_input_features()

# Prediction
st.subheader('User Input for Prediction')
st.write(input_df)

prediction = model.predict(input_df)
st.write(f"The predicted species is: {prediction[0]}")

# About
st.sidebar.header("About")
st.sidebar.text("Created by Yotam")
