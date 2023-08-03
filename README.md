# OC - Projet 7 - Parcours Data Scientist

### 🚀 Objectif

Je suis data scientist dans un établissement bancaire où de nombreux clients souhaitent contracter un prêt.
Voici les missions du projet :

- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
- Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.
- Mettre en production le modèle de scoring de prédiction à l’aide d’une API, ainsi que le dashboard interactif qui appelle l’API pour les prédictions.

### 💾 Dataset

Le jeu de données utilisé provient d'une compétition Kaggle : [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)
Un kernel Kaggle a été utilisé pour les premières étapes d'analyse.

### 📁 Découpage des dossiers
------------

    ├── .github/workflows   <- GitHub action
    │
    ├── api                 <- Flask API
    │
    ├── dashboard           <- Dashboard Streamlit
    │
    ├── data                <- Fichiers CSV
    │
    ├── documentation       <- Note méthodologique + support de présentation
    │
    ├── mlruns              <- Fichiers générés par MLFlow
    │
    ├── models              <- Modèles/Objets exportés via Pickle
    │
    ├── notebooks           <- Jupyter notebooks
    │
    ├── tests               <- Fichiers de test
    |
    ├── .gitignore
    |
    ├── README.md
    |
    ├── requirements.txt    <- Librairies et dépendances utilisées
    

### 🧰 Outils

- Python 3.9
- pip
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- LightGBM
- SHAP
- Evidently
- MLFlow
- Pickle
- Pytest
- Flask
- Streamlit
- Streamlit community cloud
- Pythonanywhere
- Subprocess
- GitPython
- GitHub Actions

### 🔗 Liens

- Dashboard : https://kiliandatadev-p7-dashboard.streamlit.app/
- API : https://kiliandatadev.pythonanywhere.com/
