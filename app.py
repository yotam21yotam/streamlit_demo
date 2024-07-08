import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

# Title of the app
st.title("Interactive Machine Learning App")

# Sidebar for data generation
st.sidebar.header("Generate Data")

distribution_choice = st.sidebar.selectbox(
    "Choose Distribution",
    ["Normal", "Bernoulli", "Binomial", "Poisson", "Exponential"]
)

if distribution_choice == "Normal":
    mean = st.sidebar.number_input("Mean", value=0.0)
    std = st.sidebar.number_input("Standard Deviation", value=1.0)
    size = st.sidebar.number_input("Number of Samples", value=100, step=10)
    data = np.random.normal(mean, std, size)
elif distribution_choice == "Bernoulli":
    p = st.sidebar.number_input("Probability of Success", min_value=0.0, max_value=1.0, value=0.5)
    size = st.sidebar.number_input("Number of Samples", value=100, step=10)
    data = np.random.binomial(1, p, size)
elif distribution_choice == "Binomial":
    p_binom = st.sidebar.number_input("Probability of Success", min_value=0.0, max_value=1.0, value=0.5)
    n_binom = st.sidebar.number_input("Number of Trials", value=10, step=1)
    size = st.sidebar.number_input("Number of Samples", value=100, step=10)
    data = np.random.binomial(n_binom, p_binom, size)
elif distribution_choice == "Poisson":
    lambda_poisson = st.sidebar.number_input("Lambda", value=1.0)
    size = st.sidebar.number_input("Number of Samples", value=100, step=10)
    data = np.random.poisson(lambda_poisson, size)
elif distribution_choice == "Exponential":
    lambda_exp = st.sidebar.number_input("Rate (lambda)", value=1.0)
    size = st.sidebar.number_input("Number of Samples", value=100, step=10)
    data = np.random.exponential(1/lambda_exp, size)

# Plot the distribution
st.subheader(f'{distribution_choice} Distribution')
fig, ax = plt.subplots()
sns.histplot(data, kde=True, ax=ax)
plt.xlabel('Value')
plt.ylabel('Frequency')
st.pyplot(fig)

# Show sample data
st.subheader('Sample Data')
st.write(pd.DataFrame(data, columns=['Sample Data']))

# Generate sample points for regression and clustering
X = np.linspace(0, 100, size)
y = 3 * X + np.random.randn(size) * 10  # Linear relationship with noise

# Split the dataset for regression
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)

# Sidebar for model selection and hyperparameters
st.sidebar.header("Choose Model for Regression")

model_choice = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Decision Tree", "Random Forest"])

if model_choice == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
elif model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100, step=10)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
else:
    model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot regression line
st.subheader('Regression Line')
fig, ax = plt.subplots()
sns.scatterplot(x=X_test.flatten(), y=y_test, ax=ax, label="Test Data")
sns.lineplot(x=X_test.flatten(), y=y_pred, ax=ax, color="red", label="Prediction")
plt.xlabel('X')
plt.ylabel('y')
st.pyplot(fig)

# Display model metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.subheader(f'{model_choice} Metrics')
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared: {r2:.2f}")

# Bias-Variance Trade-off
st.subheader("Bias-Variance Trade-off")

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

if model_choice == "Decision Tree":
    param_name = "max_depth"
    param_range = range(1, 21)
    plot_bias_variance_tradeoff(model, X_train, y_train, param_name, param_range)
elif model_choice == "Random Forest":
    param_name = "n_estimators"
    param_range = range(10, 201, 10)
    plot_bias_variance_tradeoff(model, X_train, y_train, param_name, param_range)

# Unsupervised learning: Clustering
st.subheader("Clustering")

# Generate sample points for clustering
X_cluster = np.vstack((X_train, y_train)).T

n_clusters = st.sidebar.slider("Number of Clusters", 1, 10, 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_cluster)
labels = kmeans.labels_

# Plot clustering
fig, ax = plt.subplots()
scatter = ax.scatter(X_cluster[:, 0], X_cluster[:, 1], c=labels, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Visualization')
st.pyplot(fig)

# Add a checkbox for raw data display
if st.checkbox("Show raw data"):
    st.subheader('Raw Data')
    st.write(pd.DataFrame(X_cluster, columns=['Feature 1', 'Feature 2']))
