import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Maladie de Parkinson",
    page_icon="🧠",
    layout="wide"
)

# Titre principal de l'application
st.title("🧠 Système d'Aide au Diagnostic : Maladie de Parkinson")
st.markdown("""
Cette interface utilise un modèle de Machine Learning (**XGBoost**) pour évaluer la probabilité qu'un patient soit atteint de la maladie de Parkinson à partir de ses données cliniques et fonctionnelles.
""")

# 1. Chargement du modèle et du scaler sauvegardés
@st.cache_resource # Permet de charger le modèle une seule fois en mémoire
def load_model_artifacts():
    with open('parkinson_xgboost_model.pkl', 'rb') as file:
        artifacts = pickle.load(file)
    return artifacts

try:
    artifacts = load_model_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    features_names = artifacts['features']
except FileNotFoundError:
    st.error("❌ Le fichier 'parkinson_xgboost_model.pkl' est introuvable. Veuillez d'abord exécuter le script d'entraînement.")
    st.stop()

# 2. Création des formulaires de saisie dans la barre latérale (Sidebar)
st.sidebar.header("📋 Données du Patient")

# Section : Informations Démographiques et Générales
st.sidebar.subheader("Informations Générales")
age = st.sidebar.slider("Âge", 18, 100, 65)
gender = st.sidebar.selectbox("Genre", options=[0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
ethnicity = st.sidebar.slider("Ethnicité (Code)", 0, 4, 0)
education_level = st.sidebar.slider("Niveau d'éducation (Code)", 0, 4, 1)
bmi = st.sidebar.slider("IMC (BMI)", 10.0, 50.0, 24.5)

# Section : Habitudes de vie & Antécédents
st.sidebar.subheader("Habitudes & Clinique")
smoking = st.sidebar.selectbox("Fumeur", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
alcohol = st.sidebar.slider("Consommation d'alcool", 0.0, 20.0, 5.0)
physical_activity = st.sidebar.slider("Activité physique (heures/semaine)", 0.0, 10.0, 3.0)
diet_quality = st.sidebar.slider("Qualité de l'alimentation (Score)", 0.0, 10.0, 5.0)
sleep_quality = st.sidebar.slider("Qualité du sommeil (Score)", 0.0, 10.0, 6.0)
family_history = st.sidebar.selectbox("Antécédents familiaux de Parkinson", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
tbi = st.sidebar.selectbox("Traumatisme crânien (TBI)", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")

# Section : Constantes Médicales
st.sidebar.subheader("Constantes Médicales")
hypertension = st.sidebar.selectbox("Hypertension", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
diabetes = st.sidebar.selectbox("Diabète", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
depression = st.sidebar.selectbox("Dépression", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
stroke = st.sidebar.selectbox("AVC (Stroke)", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
systolic_bp = st.sidebar.slider("Pression Systolique (mmHg)", 90, 200, 130)
diastolic_bp = st.sidebar.slider("Pression Diastolique (mmHg)", 60, 120, 80)
chol_total = st.sidebar.slider("Cholestérol Total", 100.0, 400.0, 200.0)
chol_ldl = st.sidebar.slider("Cholestérol LDL", 50.0, 250.0, 120.0)
chol_hdl = st.sidebar.slider("Cholestérol HDL", 20.0, 100.0, 50.0)
chol_trig = st.sidebar.slider("Triglycérides", 50.0, 400.0, 150.0)

# Section : Évaluations Spécifiques de Parkinson
st.sidebar.subheader("Scores & Symptômes Parkinson")
updrs = st.sidebar.slider("Score UPDRS", 0.0, 260.0, 45.0)
moca = st.sidebar.slider("Score MoCA", 0.0, 30.0, 24.0)
functional_assess = st.sidebar.slider("Évaluation Fonctionnelle", 0.0, 10.0, 5.0)

tremor = st.sidebar.selectbox("Tremblements (Tremor)", [0, 1], format_func=lambda x: "Absent" if x == 0 else "Présent")
rigidity = st.sidebar.selectbox("Rigidité", [0, 1], format_func=lambda x: "Absente" if x == 0 else "Présente")
bradykinesia = st.sidebar.selectbox("Bradykinésie", [0, 1], format_func=lambda x: "Absente" if x == 0 else "Présente")
postural_instability = st.sidebar.selectbox("Instabilité Posturale", [0, 1], format_func=lambda x: "Absente" if x == 0 else "Présente")
speech_problems = st.sidebar.selectbox("Troubles de la parole", [0, 1], format_func=lambda x: "Absent" if x == 0 else "Présent")
sleep_disorders = st.sidebar.selectbox("Troubles du sommeil spécifiques", [0, 1], format_func=lambda x: "Absent" if x == 0 else "Présent")
constipation = st.sidebar.selectbox("Constipation", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")

# 3. Organisation des données saisies dans l'ordre exact attendu par le modèle
input_data = {
    'Age': age, 'Gender': gender, 'Ethnicity': ethnicity, 'EducationLevel': education_level, 'BMI': bmi,
    'Smoking': smoking, 'AlcoholConsumption': alcohol, 'PhysicalActivity': physical_activity, 'DietQuality': diet_quality, 'SleepQuality': sleep_quality,
    'FamilyHistoryParkinsons': family_history, 'TraumaticBrainInjury': tbi, 'Hypertension': hypertension, 'Diabetes': diabetes, 'Depression': depression, 'Stroke': stroke,
    'SystolicBP': systolic_bp, 'DiastolicBP': diastolic_bp, 'CholesterolTotal': chol_total, 'CholesterolLDL': chol_ldl, 'CholesterolHDL': chol_hdl, 'CholesterolTriglycerides': chol_trig,
    'UPDRS': updrs, 'MoCA': moca, 'FunctionalAssessment': functional_assess,
    'Tremor': tremor, 'Rigidity': rigidity, 'Bradykinesia': bradykinesia, 'PosturalInstability': postural_instability, 'SpeechProblems': speech_problems,
    'SleepDisorders': sleep_disorders, 'Constipation': constipation
}

input_df = pd.DataFrame([input_data])

# Réaligner les colonnes pour être sûr du même ordre
input_df = input_df[features_names]

# 4. Affichage et Prédiction
st.subheader("📊 Analyse en Temps Réel")

col1, col2 = st.columns(2)

with col1:
    st.write("### Aperçu des données transmises au modèle :")
    st.dataframe(input_df.T.rename(columns={0: "Valeurs Saisies"}))

with col2:
    st.write("### Résultat du Diagnostic :")
    
    # Bouton pour lancer la prédiction
    if st.button("🔴 Lancer la Prédiction", type="primary"):
        # Application de la normalisation (scaler)
        input_scaled = scaler.transform(input_df)
        
        # Prédiction de la classe et de la probabilité
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        st.write("---")
        if prediction == 1:
            st.error(f"⚠️ **Résultat : Parkinson Détecté (Classe 1)**")
            st.metric(label="Indice de confiance clinique (Probabilité)", value=f"{probability * 100:.2f}%")
            st.warning("Recommandation : Ce résultat est une aide algorithmique. Une consultation neurologique approfondie et un examen clinique complet sont nécessaires.")
        else:
            st.success(f"✅ **Résultat : Sujet Sain / Non Détecté (Classe 0)**")
            st.metric(label="Indice de certitude (Sain)", value=f"{(1 - probability) * 100:.2f}%")
            st.info("Le profil de données actuel ne présente pas les caractéristiques prédictives fortes de la maladie de Parkinson selon le modèle.")
