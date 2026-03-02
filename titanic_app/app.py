import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering

# -----------------------------
# Title
# -----------------------------
st.title("🚢 Titanic Survival Prediction App")

# -----------------------------
# Load Dataset (LOCAL FILE)
# -----------------------------
df = pd.read_csv("train.csv")

# Keep required columns
df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]
df.dropna(inplace=True)

# Convert categorical values
df['Sex'] = df['Sex'].map({'male':0,'female':1})
df['Embarked'] = df['Embarked'].map({'S':0,'C':1,'Q':2})

# Split features & target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (only for supervised)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train ALL 4 Models
# -----------------------------

# Supervised
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Unsupervised
kmeans_model = KMeans(n_clusters=2, random_state=42)
kmeans_model.fit(X_scaled)

hier_model = AgglomerativeClustering(n_clusters=2)
hier_clusters = hier_model.fit_predict(X_scaled)

# -----------------------------
# Algorithm Selection
# -----------------------------
algorithm = st.selectbox(
    "Choose Algorithm",
    ["KNN", "Logistic Regression", "K-Means", "Hierarchical Clustering"]
)

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Sex", ["male","female"])
age = st.number_input("Age", min_value=0)
sibsp = st.number_input("Siblings/Spouse", min_value=0)
parch = st.number_input("Parents/Children", min_value=0)
fare = st.number_input("Fare", min_value=0.0)
embarked = st.selectbox("Embarked", ["S","C","Q"])

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):

    # Convert input to numeric
    sex = 0 if sex == "male" else 1
    embarked = {"S":0,"C":1,"Q":2}[embarked]

    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    input_scaled = scaler.transform(input_data)

    # Supervised Predictions
    if algorithm == "KNN":
        prediction = knn_model.predict(input_scaled)
        if prediction[0] == 1:
            st.success("🎉 Passenger would SURVIVE")
        else:
            st.error("❌ Passenger would NOT survive")

    elif algorithm == "Logistic Regression":
        prediction = log_model.predict(input_scaled)
        if prediction[0] == 1:
            st.success("🎉 Passenger would SURVIVE")
        else:
            st.error("❌ Passenger would NOT survive")

    # Unsupervised Predictions
    elif algorithm == "K-Means":
        cluster = kmeans_model.predict(input_scaled)
        st.info(f"🌀 Assigned to Cluster: {cluster[0]}")

    elif algorithm == "Hierarchical Clustering":
        # Hierarchical doesn't have predict(), so we assign manually
        distances = np.linalg.norm(X_scaled - input_scaled, axis=1)
        closest_index = np.argmin(distances)
        cluster_label = hier_clusters[closest_index]
        st.info(f"🌀 Assigned to Cluster: {cluster_label}")