import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.metrics import accuracy_score

# Cargar los datos
@st.cache_resource
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    column_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
    data = pd.read_csv(url, header=None, names=column_names)
    return data

raw_data = load_data()

# Preprocesar los datos
le_dict = {}
data_encoded = raw_data.copy()

for column in raw_data.columns:
    le = LabelEncoder()
    data_encoded[column] = le.fit_transform(raw_data[column])
    le_dict[column] = le

X_encoded = data_encoded.drop('class', axis=1)
y_encoded = data_encoded['class']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)

# Entrenar el modelo CategoricalNB
model_cnb = CategoricalNB()
model_cnb.fit(X_train, y_train)

# Entrenar el modelo GaussianNB
model_gnb = GaussianNB()
model_gnb.fit(X_train, y_train)

# Interfaz de usuario con Streamlit
st.title("Naive Bayes Fungal Classifier")
st.header("Miguel Angel Quintero Villegas 187684")

accuracy_cnb = accuracy_score(y_test, model_cnb.predict(X_test))
accuracy_gnb = accuracy_score(y_test, model_gnb.predict(X_test))

st.subheader(f"CategoricalNB Accuracy: {accuracy_cnb:.3f}")
st.subheader(f"GaussianNB Accuracy: {accuracy_gnb:.3f}")

# Selector para elegir el modelo
model_choice = st.selectbox("Seleccione el modelo de clasificación", ["CategoricalNB", "GaussianNB"])

# Crear selectores para las características de los hongos
features = {}
for feature in raw_data.columns[1:]:  # Excluye la columna 'class'
    unique_values = raw_data[feature].unique()
    selected_value = st.selectbox(f"Select {feature}", unique_values)
    features[feature] = selected_value

# Predecir y mostrar resultados
if st.button('Clasificar hongo'):
    # Encode selected features
    features_encoded = {feat: le_dict[feat].transform([val])[0] for feat, val in features.items()}
    input_data = pd.DataFrame([features_encoded])
    
    # Choose model based on user selection
    if model_choice == "CategoricalNB":
        prediction = model_cnb.predict(input_data)[0]
    else:
        prediction = model_gnb.predict(input_data)[0]

    # Show the result
    if prediction == 0:
        st.success("The mushroom with the selected characteristics is edible.")
    else:
        st.error("The mushroom with the selected characteristics is poisonous.")
