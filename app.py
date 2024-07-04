import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title("Simple Machine Learning App")

# Header
st.header("Linear Regression with User Data")

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
X = np.random.rand(100, 3) * 100
y = 3*X[:, 0] + 2*X[:, 1] + X[:, 2] + np.random.randn(100) * 10

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model metrics
st.subheader('Model Metrics')
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared: {r2:.2f}")

# Predict on user input
prediction = model.predict(input_df)

st.subheader('Prediction')
st.write(f"The predicted value is {prediction[0]:.2f}")

# Add success, warning, info, and error messages
st.success("The model ran successfully!")
st.warning("This is a simple example.")
st.info("You can add more features and data.")
st.error("Make sure your input data is correct.")

# Checkbox example
if st.checkbox("Show raw data"):
    st.subheader('Raw Data')
    st.write(pd.DataFrame(X, columns=['Feature 1', 'Feature 2', 'Feature 3']))

# Radio button example
state = st.radio("What is your favorite Machine Learning model?", 
                 ("Linear Regression", "Decision Tree", "Random Forest"))

if state == 'Linear Regression':
    st.success("Linear Regression is a great choice!")

occupation = st.selectbox("What is your role?", ["Student", "Data Scientist", "Engineer"])
st.text(f"Selected option is {occupation}")

# Button example
if st.button("Example Button"):
    st.error("You clicked the button!")

st.sidebar.header("About")
st.sidebar.text("Created by Yotam")
