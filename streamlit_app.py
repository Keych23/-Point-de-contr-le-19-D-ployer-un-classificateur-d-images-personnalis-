import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Classificateur Iris – GOMYCODE",
    page_icon="🌸",
    layout="wide"
)

CLASS_NAMES = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
COLORS      = ['#378ADD', '#1D9E75', '#D85A30']
EMOJIS      = ['🌸', '🌺', '🌷']

@st.cache_resource
def load_model():
    model_path = "iris_classifier.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f), None
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model, acc

model, train_acc = load_model()

st.title("🌸 Classificateur de Fleurs Iris")
st.markdown(
    "Application de **déploiement de modèle ML** – GOMYCODE Bootcamp Data Scientist  \n"
    "Modèle : **Random Forest** · Dataset : **Iris** (scikit-learn)"
)
if train_acc:
    st.success(f"✓ Modèle entraîné avec une précision de {train_acc*100:.1f}%")
st.divider()

with st.sidebar:
    st.header("À propos")
    st.markdown("""
**Modèle :** Random Forest Classifier  
**Dataset :** Iris (150 échantillons, 3 classes)  
**Librairie :** scikit-learn + pickle  

---
**Cours GOMYCODE :**  
*Model Serving with Streamlit*

**Étapes :**
1. Entraîner le modèle
2. Sauvegarder avec pickle
3. Charger dans Streamlit
4. Prédire depuis les sliders
5. Afficher résultats et graphique
    """)
    st.header("Paramètres")
    show_probs = st.checkbox("Afficher les probabilités", value=True)
    show_chart = st.checkbox("Afficher le graphique", value=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Mesures de la fleur")
    sepal_length = st.slider("Longueur sépale (cm)", 4.0, 8.0, 5.4, 0.1)
    sepal_width  = st.slider("Largeur sépale (cm)",  2.0, 5.0, 3.4, 0.1)
    petal_length = st.slider("Longueur pétale (cm)", 1.0, 7.0, 4.7, 0.1)
    petal_width  = st.slider("Largeur pétale (cm)",  0.1, 2.5, 1.5, 0.1)
    st.markdown(f"""
---
**Valeurs saisies :**
| Mesure | Valeur |
|---|---|
| Longueur sépale | {sepal_length} cm |
| Largeur sépale  | {sepal_width} cm |
| Longueur pétale | {petal_length} cm |
| Largeur pétale  | {petal_width} cm |
    """)

with col2:
    st.subheader("Résultat de la prédiction")
    features   = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    probs      = model.predict_proba(features)[0]
    confidence = float(probs[prediction]) * 100

    st.success(f"**{EMOJIS[prediction]} Espèce prédite : {CLASS_NAMES[prediction]}**")
    st.metric("Score de confiance", f"{confidence:.1f}%")

    if show_probs:
        st.markdown("**Probabilités par classe :**")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
            st.progress(int(prob * 100), text=f"{EMOJIS[i]} {name} — {prob*100:.1f}%")

    if show_chart:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        bars = ax.barh(CLASS_NAMES, probs * 100, color=COLORS, edgecolor='none')
        ax.set_xlabel("Probabilité (%)")
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{prob*100:.1f}%", va='center', fontsize=9)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(left=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with st.expander("Informations techniques"):
    st.markdown(f"""
| Paramètre | Valeur |
|---|---|
| Modèle | Random Forest (100 arbres) |
| Features | 4 mesures de sépale et pétale |
| Classes | 3 espèces d'Iris |
| Sérialisation | pickle (.pkl) |
| Classe prédite | {CLASS_NAMES[prediction]} (index {prediction}) |
| Confiance | {confidence:.2f}% |
    """)

st.divider()
st.markdown(
    "<div style='text-align:center;color:grey;font-size:12px'>"
    "GOMYCODE Bootcamp Data Scientist – Model Serving with Streamlit · "
    "Cheikh Ahmed Tidiane Diop · 2026"
    "</div>",
    unsafe_allow_html=True
)
