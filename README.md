# Airline Customer Satisfaction Analysis

## Description
Ce projet analyse les facteurs qui influencent la satisfaction des clients dans le secteur aérien. Il comprend un modèle de prédiction de la satisfaction client basé sur différents aspects de l'expérience de vol, ainsi qu'un dashboard interactif pour explorer les données et les tendances.

## Fonctionnalités

- **Analyse exploratoire des données** : Visualisation des tendances et facteurs influençant la satisfaction client
- **Modèle prédictif** : Classification binaire pour prédire si un passager sera satisfait ou non
- **API REST** : Endpoints pour obtenir des insights et faire des prédictions
- **Dashboard interactif** : Interface utilisateur pour explorer les données et simuler des scénarios

## Structure du projet

```
airline-satisfaction-analysis/
├── data/                       # Dossier contenant les données
│   └── airline_satisfaction.csv
├── notebooks/                  # Notebooks Jupyter pour l'exploration et la modélisation
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/                        # Code source
│   ├── api.py                  # API Flask
│   ├── preprocessing.py        # Fonctions de prétraitement des données
│   ├── model.py                # Fonctions pour l'entraînement et l'évaluation du modèle
│   └── train_model.py          # Script pour entraîner le modèle
├── app/                        # Application frontend
│   └── dashboard.py            # Dashboard Streamlit
├── models/                     # Modèles entraînés
│   └── satisfaction_model.pkl
├── Dockerfile                  # Pour containeriser l'application
├── entrypoint.sh               # Script d'entrée pour Docker
├── requirements.txt            # Dépendances
└── README.md                   # Documentation
```

## Installation

### Prérequis
- Python 3.9+
- pip

### Installation locale

1. Cloner le dépôt
```bash
git clone https://github.com/YsnHdn/airline-satisfaction-analysis.git
cd airline-satisfaction-analysis
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
```

3. Télécharger les données
```bash
mkdir -p data
# Utiliser Kaggle API pour télécharger les données
kaggle datasets download -d teejmahal20/airline-passenger-satisfaction -p data/
unzip data/airline-passenger-satisfaction.zip -d data/
```

4. Entraîner le modèle
```bash
python src/train_model.py
```

5. Lancer l'API
```bash
python src/api.py
```

6. Lancer le dashboard (dans un autre terminal)
```bash
streamlit run app/dashboard.py
```

### Utilisation avec Docker

1. Construire l'image Docker
```bash
docker build -t airline-satisfaction .
```

2. Lancer le conteneur
```bash
docker run -p 5000:5000 -p 8080:8080 airline-satisfaction
```

3. Accéder à l'application
   - Dashboard: http://localhost:8080
   - API: http://localhost:5000

## API Endpoints

- `GET /api/health` - Vérifier l'état de l'API
- `GET /api/insights` - Obtenir les facteurs clés influençant la satisfaction
- `GET /api/stats` - Obtenir des statistiques descriptives sur les données
- `POST /api/predict` - Prédire la satisfaction client basée sur les paramètres fournis

## Démonstration

[Insérer une capture d'écran du dashboard]

## Déploiement

Ce projet peut être facilement déployé sur diverses plateformes cloud:

### Heroku
```bash
heroku container:push web -a airline-satisfaction
heroku container:release web -a airline-satisfaction
```

### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/your-project-id/airline-satisfaction
gcloud run deploy --image gcr.io/your-project-id/airline-satisfaction --platform managed
```

## Technologies utilisées

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **API**: Flask
- **Visualisation**: Plotly, Streamlit
- **Containerisation**: Docker

## Améliorations possibles

- Implémenter des modèles plus avancés (XGBoost, Deep Learning)
- Ajouter une fonctionnalité de mise à jour en temps réel des données
- Intégrer une analyse de sentiment des commentaires clients
- Développer une version mobile du dashboard

## Auteur

HANDANE Yassine - y.handane@gmail.com
