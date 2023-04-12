import numpy as np

def estimatePrice(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

# Chargement des paramètres du modèle depuis le fichier
theta0, theta1 = np.loadtxt('model.txt', delimiter=',')

# Demande de saisie utilisateur pour le kilométrage
mileage = float(input("Entrez le kilométrage de la voiture : "))

# Prédiction du prix en utilisant le modèle entraîné
estimatedPrice = estimatePrice(mileage, theta0, theta1)

# Affichage du prix estimé
print("Le prix estimé pour un kilométrage de {} km est de {} euros".format(mileage, estimatedPrice))