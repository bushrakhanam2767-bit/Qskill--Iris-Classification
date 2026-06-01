import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Force page configuration to wide mode
st.set_page_config(page_title="Iris Classifier Dashboard", layout="wide")

# 1. Dashboard Header (Standard English)
st.title("🌸 Iris Flower Classification & Analysis Dashboard")
st.markdown("An advanced Machine Learning application designed to classify Iris flower species based on morphological features.")
st.write("---")

# Directly loading reliable Wikimedia URLs for stable image rendering
flower_images = {
    'setosa': 'https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg',
    'versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
    'virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
}

# 2. Dataset Setup & Model Training
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

X = df.drop(['species', 'species_name'], axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 3. Sidebar Features
st.sidebar.header("🔧 User Input Features")
st.sidebar.write("Adjust the sliders to change feature measurements:")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df.iloc[:,0].min()), float(df.iloc[:,0].max()), float(df.iloc[:,0].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df.iloc[:,1].min()), float(df.iloc[:,1].max()), float(df.iloc[:,1].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df.iloc[:,2].min()), float(df.iloc[:,2].max()), float(df.iloc[:,2].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df.iloc[:,3].min()), float(df.iloc[:,3].max()), float(df.iloc[:,3].mean()))

# 4. Central Main Dashboard Actions (The Predict Button)
st.subheader("🔮 Model Execution Trigger")
predict_btn = st.button("🌸 CLASSIFY CURRENT SAMPLE", type="primary", use_container_width=True)
st.write("---")

# Create Two Columns for Output Layout
col1, col2 = st.columns([1, 1])

if predict_btn:
    # Logic executed ONLY when the user clicks the center button
    user_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)
    
    # Model Predictions
    prediction = model.predict(user_data)
    predicted_species = iris.target_names[prediction[0]] 
    prediction_proba = model.predict_proba(user_data)[0]
    
    with col1:
        st.subheader("🎯 Live Prediction Report")
        st.success(f"**Predicted Target Species:** {predicted_species.upper()}")
        st.metric(label="Maximum Identification Confidence", value=f"{max(prediction_proba) * 100:.2f}%")
        
        # Performance/Probability Table
        st.write("#### Confidence Breakdown Reference:")
        prob_df = pd.DataFrame([prediction_proba], columns=iris.target_names, index=["Probability Score"])
        st.dataframe(prob_df.style.format("{:.2%}"))
        
        # Dynamic Visual Representation Generation
        st.write("#### 📸 Flower Visual Verification:")
        st.image(flower_images[predicted_species], caption=f"Field Sample: Iris {predicted_species.capitalize()}", use_container_width=True)

    with col2:
        st.subheader("📊 Interactive Data Space Visualizer")
        fig, ax = plt.subplots(figsize=(6, 5.2))
        sns.scatterplot(data=df, x=iris.feature_names[2], y=iris.feature_names[3], hue='species_name', ax=ax, palette='Set1', s=60)
        
        # Overlay User Parameter Coordinates
        ax.scatter(petal_length, petal_width, color='black', marker='*', s=350, label='Your Sample Coordinates')
        ax.set_title("Feature Plot: Petal Length vs Width")
        ax.set_xlabel("Petal Length (cm)")
        ax.set_ylabel("Petal Width (cm)")
        plt.legend(loc='upper left')
        st.pyplot(fig)

else:
    # Default State before submission
    with col1:
        st.info("System Ready. Please modify the sidebar settings and click the **'CLASSIFY CURRENT SAMPLE'** button to compute predictions.")
    with col2:
        st.subheader("📊 General Historical Trend")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.scatterplot(data=df, x=iris.feature_names[2], y=iris.feature_names[3], hue='species_name', ax=ax, palette='Set1')
        ax.set_title("Feature Plot: Petal Length vs Width")
        st.pyplot(fig)

st.write("---")
if st.sidebar.checkbox("Display System Raw Dataset"):
    st.write("### Historical Dataset Reference Matrix")
    st.dataframe(df)