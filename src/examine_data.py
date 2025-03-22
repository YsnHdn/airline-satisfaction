# Créez un fichier examine_data.py dans le dossier src/
import pandas as pd

# Charger le fichier
df = pd.read_csv('data/train.csv')

# Afficher les informations de base
print("Aperçu des premières lignes :")
print(df.head())

print("\nNoms des colonnes :")
print(df.columns.tolist())

print("\nInformations sur les types de données :")
print(df.info())

print("\nStatistiques descriptives :")
print(df.describe())

# Vérifier s'il y a une colonne de satisfaction
satisfaction_cols = [col for col in df.columns if 'satisf' in col.lower()]
print("\nColonnes liées à la satisfaction :", satisfaction_cols)