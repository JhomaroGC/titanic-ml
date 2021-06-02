import pandas as pd
import numpy as np
from pandas.core.base import DataError
import csv

class ProcessTitanic():
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
    
    def exploration_data(self):
        df = _get_data(self.train_path)
        print('Resumen del conjunto de datos de entrenamiento: ')
        print(f"1. Tamaño conjunto de datos entrenamiento: {len(df)} registros.")
        #Calcular el porcentaje de mujeres vs hombres a bordo
        men = round(df['Sex'].value_counts()[0]/len(df)*100, 2)
        women = round(df['Sex'].value_counts()[1]/len(df)*100, 2)
        print(f"2. Del total de pasajeros en el conjunto de entrenamiento, un {men}% son hombres y un {women}% son mujeres.")
        #Create a new feature for separating childs and adultos
        df['Child'] = np.where((df['Age'] <18), 1, 0)
        child = round(df['Child'].value_counts()[1]/len(df)*100, 2)
        adult = round(df['Child'].value_counts()[0]/len(df)*100, 2)
        print(f"3. Del total de pasajeros en el conjunto de entrenamiento, un {child}% son niños, y un {adult}% son adultos.")
        # Calculate percent of deaths
        deaths = round(df['Survived'].value_counts()[0]/len(df)*100, 2)
        survived = round(df['Survived'].value_counts()[1]/len(df)*100, 2)
        print(f"4. Del total de pasajeros en el conjunto de entrenamiento, un {deaths}% murió, y un {survived}% sobrevivió.")
        #Calculate passenger for each Class
        _1st = round(df['Pclass'].value_counts()[1]/len(df)*100, 2)
        _2nd = round(df['Pclass'].value_counts()[2]/len(df)*100, 2)
        _3rd = round(df['Pclass'].value_counts()[3]/len(df)*100, 2)        
        print(f"5. Del total de pasajeros en el conjunto de entrenamiento, un {_1st}% viajó en primera clase\
,un {_2nd}% en segunda clase y un {_3rd}% de tercerca clase.")

    def process_titanic(self, train_data):
        train_data = train_data
        test_data = _get_data(self.test_path)
        X_train, X_test, y_train = _select_var(train_data, test_data)
        return X_train, X_test, y_train  
    
#Exploration of dataset train

#Get data train y test
def _get_data(path):
    data = pd.read_csv(path)
    return data

#Select variables for ML process
def _select_var(train_data, test_data):
    with open('.\outputs\\train_data_info.txt', 'w') as f:
        train_data.info(buf=f)
    with open('.\outputs\\test_data_info.txt', 'w') as f:
        test_data.info(buf=f)
        print("Se ha guardado exitosamente la descripcion básica del dataset titanic")
    features = ['Pclass', 'Sex', 'SibSp', 'Parch']
    y = train_data['Survived']
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])
    return X, X_test, y

def _changeDtype(data):
    return data
