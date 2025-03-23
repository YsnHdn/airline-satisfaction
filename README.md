# Airline Customer Satisfaction Analysis

## Aperçu
Ce projet est une plateforme d'analyse de la satisfaction client dans le secteur aérien. Il combine un modèle de machine learning prédictif avec un dashboard interactif, permettant d'identifier les facteurs clés qui influencent la satisfaction des passagers et de prédire la satisfaction future en fonction de différents paramètres.

## Fonctionnalités

- **Analyse exploratoire des données** : Visualisation des tendances et statistiques sur la satisfaction client
- **Identification des facteurs clés d'influence** : Découverte des services ayant le plus d'impact sur la satisfaction
- **Prédiction de satisfaction** : Modèle machine learning pour prédire si un passager sera satisfait
- **Dashboard interactif** : Interface utilisateur intuitive pour explorer les données
- **API REST** : Endpoints pour accéder aux prédictions et aux informations

## Démonstration

[Lien vers une démo en ligne - À venir]

## Technologies utilisées

- **Backend** : Python, Flask
- **Frontend** : Streamlit
- **Data Processing** : Pandas, NumPy
- **Machine Learning** : Scikit-learn
- **Visualisation** : Plotly
- **Déploiement** : Docker

## Structure du projet

```
airline-satisfaction-analysis/
├── app/                         # Application frontend Streamlit
│   └── dashboard.py             # Dashboard interactif
├── data/                        # Données d'entraînement et de test
│   ├── train.csv
│   └── test.csv
├── models/                      # Dossier pour les modèles entraînés
├── notebooks/                   # Notebooks d'analyse
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/                         # Code source du backend
│   ├── api.py                   # API Flask
│   ├── model.py                 # Fonctions pour l'entraînement et la prédiction
│   ├── preprocessing.py         # Fonctions de traitement des données
│   └── train_model.py           # Script d'entraînement du modèle
├── Dockerfile                   # Configuration Docker
├── entrypoint.sh                # Script de démarrage Docker
├── requirements.txt             # Dépendances Python
└── README.md                    # Documentation
```

## Installation et utilisation

### Prérequis
- Python 3.8+
- pip

### Installation locale

1. Clonez ce dépôt
```bash
git clone https://github.com/YsnHdn/airline-satisfaction.git
cd airline-satisfaction
```

2. Créez un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. Installez les dépendances
```bash
pip install -r requirements.txt
```

4. Téléchargez les données (si nécessaire)
```bash
mkdir -p data
# Téléchargez manuellement depuis https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction 
# et placez les fichiers dans le dossier data/
```

### Important : Génération du modèle

Ce dépôt ne contient pas le modèle préentraîné en raison de sa taille. Après avoir installé les dépendances et téléchargé les données, veuillez générer le modèle localement :

```bash
# Créer le dossier models s'il n'existe pas
mkdir -p models

# Entraîner le modèle
python src/train_model.py
```

Le modèle sera sauvegardé dans le dossier `models/` et l'application pourra l'utiliser automatiquement.

5. Lancez l'API
```bash
python src/api.py
```

6. Dans un autre terminal, lancez le dashboard
```bash
streamlit run app/dashboard.py
```

7. Accédez au dashboard à l'adresse http://localhost:8501

### Utilisation avec Docker

1. Construire l'image Docker
```bash
docker build -t airline-satisfaction .
```

2. Lancer le conteneur
```bash
docker run -p 5000:5000 -p 8501:8501 airline-satisfaction
```

3. Accéder à l'application
   - Dashboard: http://localhost:8501
   - API: http://localhost:5000

## API Endpoints

- `GET /api/health` - Vérifier l'état de l'API
- `GET /api/insights` - Obtenir les facteurs clés influençant la satisfaction
- `GET /api/stats` - Obtenir des statistiques descriptives sur les données
- `POST /api/predict` - Prédire la satisfaction client basée sur les paramètres fournis

## Dataset

Ce projet utilise le jeu de données "Airline Passenger Satisfaction" disponible sur Kaggle. Il contient plus de 100 000 réponses de passagers avec différentes caractéristiques comme la classe de voyage, la distance, les évaluations de services, et le niveau de satisfaction global.

[Télécharger sur Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

## Modèle

Le modèle de prédiction utilise un algorithme Random Forest avec une précision d'environ 85% pour prédire si un client sera satisfait ou non de son expérience de vol. Les facteurs les plus importants identifiés incluent la classe de voyage, le type de client, et plusieurs services à bord.

## Contribution

Les contributions sont les bienvenues! Pour contribuer:

1. Fork le projet
2. Créez votre branche de fonctionnalité (`git checkout -b feature/amazing-feature`)
3. Committez vos changements (`git commit -m 'Add some amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrez une Pull Request

## Développement futur

- Ajout de visualisations plus complexes
- Amélioration du modèle avec des algorithmes plus avancés
- Intégration de fonctionnalités de traitement du langage naturel pour analyser les commentaires
- Création d'une version mobile de l'application

## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Contact

Yassine HANDANE - y.handane@gmail.com

Lien du projet: [https://github.com/YsnHdn/airline-satisfaction](https://github.com/YsnHdn/airline-satisfaction)