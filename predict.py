import numpy as np
import pandas as pd 
import os, sys
RED   = "\033[1;31m"  
BLUE  = "\033[1;34m"
REVERSE = "\033[;7m"



sys.stdout.write(RED)


def estimatePrice(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage) 


if not os.path.isfile("thetas.csv") :
    print("File thetas.csv does not exist")
    exit()
try :
    thetas = pd.read_csv("thetas.csv")
except :
    print("File thetas.csv is not a valid csv file")
    exit()

theta0, theta1 = np.loadtxt('thetas.csv', delimiter=',')
sys.stdout.write(BLUE)
mileage = float(input("Entrez le kilométrage de la voiture : "))

estimatedPrice = estimatePrice(mileage, theta0, theta1)

sys.stdout.write(REVERSE)
print("Le prix estimé de la voiture est de", round(estimatedPrice), "euros")