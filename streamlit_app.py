# ==============================================================================
#  CHECKPOINT : Model Serving with Streamlit
#  Application de classification d'images personnalisée
#  Modèle : VGG16 pré-entraîné (Transfer Learning – TF Flowers 5 classes)
#  Basé sur le cours GOMYCODE – Déploiement Web / Application
# ==============================================================================
#
#  INSTALLATION (terminal) :
#  pip install streamlit tensorflow pillow numpy matplotlib
#
#  LANCEMENT :
#  streamlit run streamlit_app.py
#
# ==============================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import io
import os

# ─────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Classificateur de Fleurs – GOMYCODE",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
CLASS_NAMES  = ['🌼 Daisy', '🌻 Dandelion', '🌹 Rose', '🌻 Sunflower', '🌷 Tulip']
CLASS_LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
IMG_SIZE     = (224, 224)

COLORS = ['#378ADD', '#1D9E75', '#D85A30', '#EF9F27', '#D4537E']

# ─────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# Selon le cours : on charge le modèle sauvegardé
# en .keras ou on reconstruit avec VGG16 si absent
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Charge le modèle sauvegardé.
    Si le fichier n'existe pas, reconstruit et entraîne
    un modèle VGG16 léger sur TF Flowers (demo rapide).
    """
    model_path = "flowers_classifier.keras"

    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        return model, "Modèle chargé depuis le fichier sauvegardé"

    # ── Construction du modèle VGG16 (Feature Extraction) ──
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras import layers

    base = VGG16(weights='imagenet', include_top=False,
                 input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    inputs  = keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.5)(x)
    outputs = layers.Dense(5, activation='softmax')(x)
    model   = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # ── Entraînement rapide (5 époques) sur TF Flowers ──
    import tensorflow_datasets as tfds

    def preprocess(img, label):
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        img = keras.applications.vgg16.preprocess_input(img)
        return img, label

    ds, _ = tfds.load('tf_flowers', split='train', as_supervised=True,
                      with_info=True)
    ds = ds.map(preprocess).shuffle(500).batch(32).prefetch(tf.data.AUTOTUNE)

    model.fit(ds, epochs=5, verbose=0)
    model.save(model_path)

    return model, "Modèle VGG16 reconstruit et entraîné (5 époques)"


# ─────────────────────────────────────────────
# PRÉTRAITEMENT DE L'IMAGE UPLOADÉE
# Selon le cours : préparer l'image pour predict()
# ─────────────────────────────────────────────
def preprocess_image(uploaded_file):
    """Charge, redimensionne et normalise l'image pour VGG16."""
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array  = np.array(img_resized, dtype=np.float32)
    img_array  = keras.applications.vgg16.preprocess_input(img_array)
    img_batch  = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)
    return img, img_batch


def make_bar_chart(probs):
    """Génère le graphique de probabilités (st.bar_chart natif + matplotlib)."""
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(CLASS_LABELS, probs * 100, color=COLORS, edgecolor='none')
    ax.set_xlabel('Probabilité (%)')
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{prob*100:.1f}%', va='center', fontsize=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# INTERFACE STREAMLIT
# ─────────────────────────────────────────────

# ── En-tête ────────────────────────────────
st.title("🌸 Classificateur de Fleurs")
st.markdown(
    "Application de **déploiement de modèle ML** – GOMYCODE Bootcamp Data Scientist  \n"
    "Modèle : **VGG16** (Transfer Learning) · 5 classes de fleurs"
)
st.divider()

# ── Sidebar ────────────────────────────────
with st.sidebar:
    st.header("À propos")
    st.markdown("""
    **Modèle :** VGG16 pré-entraîné (ImageNet)  
    **Technique :** Feature Extraction  
    **Dataset :** TF Flowers  
    **Classes :** Daisy · Dandelion · Rose · Sunflower · Tulip  
    
    ---
    **Cours GOMYCODE :**  
    *Model Serving with Streamlit*
    
    **Étapes du déploiement :**
    1. Charger le modèle (.keras)
    2. Uploader une image
    3. Prétraiter (resize 224×224, normalisation ImageNet)
    4. Prédire avec `model.predict()`
    5. Afficher la classe et les probabilités
    """)

    st.header("Paramètres")
    show_probs    = st.checkbox("Afficher les probabilités", value=True)
    show_raw      = st.checkbox("Afficher les valeurs brutes", value=False)
    top_k         = st.slider("Top-K prédictions", 1, 5, 3)
    confidence_thr = st.slider("Seuil de confiance (%)", 0, 100, 50)

# ── Chargement du modèle ───────────────────
with st.spinner("Chargement du modèle..."):
    model, model_status = load_model()

st.success(f"✓ {model_status}")

# ── Upload de l'image ──────────────────────
st.subheader("1. Téléverser une image de fleur")
uploaded_file = st.file_uploader(
    "Choisir une image (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"],
    help="Uploadez une photo de fleur : daisy, dandelion, rose, sunflower ou tulip"
)

# ── Traitement et prédiction ───────────────
if uploaded_file is not None:

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("2. Aperçu de l'image")
        original_img, img_batch = preprocess_image(uploaded_file)

        # Aperçu avec st.image (selon le cours)
        st.image(original_img, caption=f"Image uploadée : {uploaded_file.name}",
                 use_column_width=True)
        st.caption(f"Taille originale : {original_img.size[0]}×{original_img.size[1]} px  "
                   f"→ Redimensionnée : 224×224 px pour VGG16")

    with col2:
        st.subheader("3. Prédiction du modèle")

        # ── Prédiction ───────────────────────
        with st.spinner("Analyse en cours..."):
            probs      = model.predict(img_batch, verbose=0)[0]
            pred_idx   = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx]) * 100

        # ── Résultat principal ───────────────
        if confidence >= confidence_thr:
            st.success(f"**Espèce détectée : {pred_class}**")
            st.metric(label="Score de confiance", value=f"{confidence:.1f}%")
        else:
            st.warning(
                f"Prédiction : **{pred_class}** ({confidence:.1f}%)  \n"
                f"⚠️ Confiance inférieure au seuil ({confidence_thr}%)"
            )

        # ── Top-K prédictions ────────────────
        st.markdown(f"**Top {top_k} prédictions :**")
        top_indices = np.argsort(probs)[::-1][:top_k]
        for rank, idx in enumerate(top_indices):
            bar_val = int(probs[idx] * 100)
            st.progress(bar_val,
                        text=f"{rank+1}. {CLASS_NAMES[idx]}  —  {probs[idx]*100:.1f}%")

        # ── Graphique de probabilités ─────────
        if show_probs:
            st.subheader("4. Graphique des probabilités")
            fig = make_bar_chart(probs)
            st.pyplot(fig)
            plt.close(fig)

        # ── Valeurs brutes ───────────────────
        if show_raw:
            st.subheader("Valeurs brutes (softmax)")
            raw_data = {CLASS_LABELS[i]: float(f"{probs[i]:.4f}")
                        for i in range(len(CLASS_LABELS))}
            st.json(raw_data)

    # ── Informations techniques ───────────────
    with st.expander("Informations techniques – Prétraitement"):
        st.markdown(f"""
        | Étape | Détail |
        |---|---|
        | Image originale | {original_img.size[0]}×{original_img.size[1]} px, RGB |
        | Redimensionnement | 224×224 px (taille attendue par VGG16) |
        | Normalisation | `vgg16.preprocess_input()` – centrage par moyenne ImageNet |
        | Format modèle | Batch shape : `(1, 224, 224, 3)` |
        | Sortie | Vecteur softmax de dimension 5 |
        | Classe prédite | `{CLASS_LABELS[pred_idx]}` (index {pred_idx}) |
        | Confiance | {confidence:.2f}% |
        """)

else:
    # ── Message d'accueil si aucun fichier ───
    st.info(
        "Uploadez une image de fleur ci-dessus pour lancer la classification.  \n"
        "Fleurs reconnues : **Daisy · Dandelion · Rose · Sunflower · Tulip**"
    )

    # Exemple de classes avec couleurs
    st.subheader("Classes disponibles")
    cols = st.columns(5)
    emojis = ['🌼', '🌻', '🌹', '🌻', '🌷']
    for i, (col, name, emoji) in enumerate(zip(cols, CLASS_LABELS, emojis)):
        with col:
            st.markdown(
                f"<div style='text-align:center;padding:12px;"
                f"border-radius:8px;border:1px solid #ddd'>"
                f"<div style='font-size:2rem'>{emoji}</div>"
                f"<div style='font-weight:500;margin-top:6px'>{name.capitalize()}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

# ── Footer ─────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:grey;font-size:12px'>"
    "GOMYCODE Bootcamp Data Scientist – Model Serving with Streamlit · "
    "Cheikh Ahmed Tidiane Diop · 2026"
    "</div>",
    unsafe_allow_html=True
)
