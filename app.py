import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title("Interactive Machine Learning and Distribution App by Yotam")

# Header
st.header("Explore Different Models, Visualizations, and Distributions")

# Sidebar input
st.sidebar.header("User Input Features")

def user_input_features():
    st.sidebar.markdown("**Input your data below:**")
    feature1 = st.sidebar.slider('Feature 1', 0.0, 100.0, 50.0)
    feature2 = st.sidebar.slider('Feature 2', 0.0, 100.0, 50.0)
    feature3 = st.sidebar.slider('Feature 3', 0.0, 100.0, 50.0)
    data = {'Feature 1': feature1,
            'Feature 2': feature2,
            'Feature 3': feature3}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main panel
st.subheader('User Input Features')
st.write(input_df)

# Create a sample dataset
np.random.seed(42)
X = np.random.rand(100, 3) * 100
y = 3*X[:, 0] + 2*X[:, 1] + X[:, 2] + np.random.randn(100) * 10

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[model_choice]

# Hyperparameters for Decision Tree
if model_choice == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
    model.set_params(max_depth=max_depth, min_samples_split=min_samples_split)

# Hyperparameters for Random Forest
if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100, step=10)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
    model.set_params(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

# Fit the model
model.fit(X_train, y_train)

# Predict using the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model metrics
st.subheader(f'{model_choice} Metrics')
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared: {r2:.2f}")

# Predict on user input
prediction = model.predict(input_df)

st.subheader('Prediction')
st.write(f"The predicted value is {prediction[0]:.2f}")

# Visualize the data
st.subheader('Data Visualization')
fig, ax = plt.subplots()
sns.scatterplot(x=X[:, 0], y=y, ax=ax, label="Data")
sns.lineplot(x=X_test[:, 0], y=y_pred, ax=ax, color="red", label="Prediction")
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.title('Feature 1 vs Target')
st.pyplot(fig)

# Bias-Variance Trade-off Demonstration
st.subheader("Bias-Variance Trade-off")

# Function to calculate and plot bias-variance trade-off
def plot_bias_variance_tradeoff(model, X, y, param_name, param_range):
    train_scores, test_scores = [], []
    for param in param_range:
        model.set_params(**{param_name: param})
        train_score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
        test_score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
        train_scores.append(-train_score)
        test_scores.append(-test_score)
    
    fig, ax = plt.subplots()
    ax.plot(param_range, train_scores, label="Training Error")
    ax.plot(param_range, test_scores, label="Testing Error")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Mean Squared Error")
    ax.set_title(f"Bias-Variance Trade-off ({param_name})")
    ax.legend()
    st.pyplot(fig)

# Bias-Variance Trade-off for Decision Tree
if model_choice == "Decision Tree":
    param_name = "max_depth"
    param_range = range(1, 21)
    plot_bias_variance_tradeoff(model, X, y, param_name, param_range)

# Bias-Variance Trade-off for Random Forest
if model_choice == "Random Forest":
    param_name = "n_estimators"
    param_range = range(10, 201, 10)
    plot_bias_variance_tradeoff(model, X, y, param_name, param_range)

# Standard Normal Distribution
st.sidebar.header("Distributions")

# Choose distribution
distribution_choice = st.sidebar.selectbox(
    "Choose Distribution",
    ["Normal", "Bernoulli", "Binomial", "Poisson", "Exponential"]
)

if distribution_choice == "Normal":
    st.sidebar.subheader("Normal Distribution")
    mean = st.sidebar.number_input("Mean", value=0.0)
    std = st.sidebar.number_input("Standard Deviation", value=1.0)
    size = st.sidebar.number_input("Number of Samples", value=1000, step=100)
    data = np.random.normal(mean, std, size)
    fig, ax = plt.subplots()
    sns.histplot(data, kde=True, ax=ax)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Normal Distribution (mean={mean}, std={std})')
    st.pyplot(fig)

elif distribution_choice == "Bernoulli":
    st.sidebar.subheader("Bernoulli Distribution")
    p = st.sidebar.number_input("Probability of Success", min_value=0.0, max_value=1.0, value=0.5)
    n = st.sidebar.number_input("Number of Experiments", value=100, step=10)
    data = np.random.binomial(1, p, n)
    fig, ax = plt.subplots()
    sns.histplot(data, kde=False, discrete=True, ax=ax)
    plt.xlabel('Outcome')
    plt.ylabel('Frequency')
    plt.title(f'Bernoulli Distribution (p={p}, n={n})')
    st.pyplot(fig)

elif distribution_choice == "Binomial":
    st.sidebar.subheader("Binomial Distribution")
    p_binom = st.sidebar.number_input("Probability of Success", min_value=0.0, max_value=1.0, value=0.5)
    n_binom = st.sidebar.number_input("Number of Trials", value=100, step=10)
    size_binom = st.sidebar.number_input("Number of Samples", value=1000, step=100)
    data = np.random.binomial(n_binom, p_binom, size_binom)
    fig, ax = plt.subplots()
    sns.histplot(data, kde=False, ax=ax)
    plt.xlabel('Number of Successes')
    plt.ylabel('Frequency')
    plt.title(f'Binomial Distribution (n={n_binom}, p={p_binom})')
    st.pyplot(fig)

elif distribution_choice == "Poisson":
    st.sidebar.subheader("Poisson Distribution")
    lambda_poisson = st.sidebar.number_input("Lambda", value=1.0)
    size_poisson = st.sidebar.number_input("Number of Samples", value=1000, step=100)
    data = np.random.poisson(lambda_poisson, size_poisson)
    fig, ax = plt.subplots()
    sns.histplot(data, kde=False, ax=ax)
    plt.xlabel('Number of Events')
    plt.ylabel('Frequency')
    plt.title(f'Poisson Distribution (lambda={lambda_poisson})')
    st.pyplot(fig)

elif distribution_choice == "Exponential":
    st.sidebar.subheader("Exponential Distribution")
    lambda_exp = st.sidebar.number_input("Rate (lambda)", value=1.0)
    size_exp = st.sidebar.number_input("Number of Samples", value=1000, step=100)
    data = np.random.exponential(1/lambda_exp, size_exp)
    fig, ax = plt.subplots()
    sns.histplot(data, kde=True, ax=ax)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Exponential Distribution (lambda={lambda_exp})')
    st.pyplot(fig)

# Add a checkbox for raw data display
if st.checkbox("Show raw data"):
    st.subheader('Raw Data')
   
