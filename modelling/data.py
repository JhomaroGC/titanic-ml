import pandas as pd
import numpy as np
from pandas.core.base import DataError
from imblearn.over_sampling import SMOTE

class ProcessTitanic():
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
    
    def exploration_data(self):
        test_df = _get_data(self.test_path)
        df = _get_data(self.train_path)
        print('+----------------------Reporte estadístico de conjunto de entrenamiento------------------------+')
        print(f"1. Tamaño conjunto de datos entrenamiento: {len(df)} registros.")
        #Calcular el porcentaje de mujeres vs hombres a bordo
        men = round(df['Sex'].value_counts()[0]/len(df)*100, 2)
        women = round(df['Sex'].value_counts()[1]/len(df)*100, 2)
        print(f"2. Del total de pasajeros en el conjunto de entrenamiento, un {men}% son hombres y un {women}% son mujeres.")
        #Calculate passenger for each Class
        _1st = round(df['Pclass'].value_counts()[1]/len(df)*100, 2)
        _2nd = round(df['Pclass'].value_counts()[2]/len(df)*100, 2)
        _3rd = round(df['Pclass'].value_counts()[3]/len(df)*100, 2)        
        print(f"3. Del total de pasajeros en el conjunto de entrenamiento, un {_1st}% viajó en primera clase\
,un {_2nd}% en segunda clase y un {_3rd}% de tercerca clase.")
# Calculate percent of deaths
        deaths = round(df['Survived'].value_counts()[0]/len(df)*100, 2)
        survived = round(df['Survived'].value_counts()[1]/len(df)*100, 2)
        print(f"4. Del total de pasajeros en el conjunto de entrenamiento, un {deaths}% murió, y un {survived}% sobrevivió.")
        #Calculate women survivors vs men survivors
        male_survivors = round(df.loc[df.Survived == 1]['Sex'].value_counts()[1] / df.loc[df.Survived == 1]['Sex'].count()*100, 2)
        female_survivors = round(df.loc[df.Survived == 1]['Sex'].value_counts()[0] / df.loc[df.Survived == 1]['Sex'].count()*100, 2)
        print(f"5. De un total de {df.loc[df.Survived == 1]['Sex'].count()} sobrevivientes en el conjunto de entrenamiento, un {male_survivors}% \
fueron hombres y un {female_survivors}% mujeres.")
        #Extracting title of each passenger for set a group of age on train_data
        personal_title = df['Name'].str.split(r".", expand= True)
        pt = personal_title[0].str.split(r", ", expand = True)
        df = pd.concat([df, pt[1]], axis = 1)
        df = df.replace({1: {"Dr":"Mr", "Capt": "Mr", "Major": "Mr", "Jonkheer": 'Mr'\
            , "Ms": "Miss", "Lady": 'Miss', "Don": "Mr", "Sir": "Mr", "the Countess": "Miss", "Mlle": "Miss",\
                "Mme":"Mrs", "Col": "Mr","Rev": "Mr" }})
        #Calculate survivors for each personal title
        miss_survivors = round(df.loc[df.Survived == 1][1].value_counts()['Miss'] / df.loc[df.Survived == 1][1].count()*100, 2)
        mrs_survivors = round(df.loc[df.Survived == 1][1].value_counts()['Mrs'] / df.loc[df.Survived == 1][1].count()*100, 2)
        mr_survivors = round(df.loc[df.Survived == 1][1].value_counts()['Mr'] / df.loc[df.Survived == 1][1].count()*100, 2)
        master_survivors = round(df.loc[df.Survived == 1][1].value_counts()['Master'] / df.loc[df.Survived == 1][1].count()*100, 2)
    
        print(f"6. De un total de {df.loc[df.Survived == 1][1].count()} sobrevivientes en el conjunto de entrenamiento, un {miss_survivors}% \
fueron niñas o mujeres jovenes, un {mrs_survivors}% fueron mujeres casadas, un {mr_survivors}% hombres y un {master_survivors} fueron niños y jovenes.")
#         [1] / df.loc[df.Survived == 1]['Child'].count()*100, 2)
#         print(f"7. De un total de {df.loc[df.Survived == 1]['Sex'].count()} sobrevivientes en el conjunto de entrenamiento, un {children_survivors}% \
# fueron niños y un {100-children_survivors}% adultos.")
        #Calculate survivors for each Pclass
        _1st_survivors = round(df.loc[df.Survived == 1]['Pclass'].value_counts()[1] / df.loc[df.Survived == 1]['Pclass'].count()*100,2)
        _2nd_survivors = round(df.loc[df.Survived == 1]['Pclass'].value_counts()[2] / df.loc[df.Survived == 1]['Pclass'].count()*100,2)
        _3nd_survivors = round(df.loc[df.Survived == 1]['Pclass'].value_counts()[2] / df.loc[df.Survived == 1]['Pclass'].count()*100,2)
        print(f"7. De un total de {df.loc[df.Survived == 1]['Sex'].count()} sobrevivientes en el conjunto de entrenamiento, un {_1st_survivors}% \
fueron de primera clase, un {_2nd_survivors}% de segunda clase y un {_3nd_survivors}% de tercera clase.")
        print("+----------------------------------------Fin del Reporte--------------------------------------+")
        return df, test_df

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
    train_data["personal_title"] = train_data[1]
    del train_data[1]
    # with open('.\outputs\\train_data_info.txt', 'w') as f:
    #     train_data.info(buf=f)
    # with open('.\outputs\\test_data_info.txt', 'w') as f:
    #     test_data.info(buf=f)
    #     print("Se ha guardado exitosamente la descripcion básica del dataset titanic")

    #Extracting title of each passenger for set a group of age on test_data
    personal_title = test_data['Name'].str.split(r".", expand= True)
    print(f"1. Extracción titulo personal de cada pasajero (Miss, Mrs, Mr, Master) para train_data y test_data")
    pt = personal_title[0].str.split(r", ", expand = True)
    test_data = pd.concat([test_data, pt[1]], axis = 1)
    test_data = test_data.replace({1: {"Dr":"Mr", "Capt": "Mr", "Major": "Mr", "Jonkheer": 'Mr'\
        , "Ms": "Miss", "Lady": 'Miss',"Dona": "Miss", "Don": "Mr", "Sir": "Mr", "the Countess": "Miss", "Mlle": "Miss",\
            "Mme":"Mrs", "Col": "Mr","Rev": "Mr" }})
    test_data['personal_title'] = test_data[1]
    features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'personal_title']    
    y = train_data['Survived']
    print("2. Separación de la variable objetivo")
    X = pd.get_dummies(train_data[features])
    print(f"3. Selección de variables para proceso de ML: {features} y codificación mediante la generación de dummies.")
    oversample = SMOTE()
    X, y = oversample.fit_resample(X,y)
    print(f"4. Balanceo de clases mediante metodología SMOTE, y = (0: {y.value_counts()[0]}, 1: {y.value_counts()[1]})")
    X_test = pd.get_dummies(test_data[features])
    print("5. Partición de los datos en entrenamiento y prueba")
    print(f"    -Forma de los datos de entrenamiento, X:  {X.shape}, y :{y.shape}")
    print(f"    -Forma de los datos de prueba, X_prueba:  {X_test.shape}")
    return X, X_test, y

