import numpy as np
import pandas as pd 
import os 

def estimatePrice(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage) 

# Check if file exists and is a good file to read 
# if not, exit program 
if not os.path.isfile("thetas.csv") :
    print("File thetas.csv does not exist")
    exit()
try :
    thetas = pd.read_csv("thetas.csv")
except :
    print("File thetas.csv is not a valid csv file")
    exit()

# Chargement des paramètres du modèle depuis le fichier
theta0, theta1 = np.loadtxt('thetas.csv', delimiter=',')

# Demande de saisie utilisateur pour le kilométrage
mileage = float(input("Entrez le kilométrage de la voiture : "))

# Prédiction du prix en utilisant le modèle entraîné
estimatedPrice = estimatePrice(mileage, theta0, theta1)

# Affichage du prix estimé arrondi à l'euro près
print("Le prix estimé de la voiture est de", round(estimatedPrice), "euros")