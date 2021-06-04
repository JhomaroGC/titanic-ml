from os import X_OK
from data import ProcessTitanic
from model import train_model, predict_model
import pandas as pd

#Get and split data train, and test
titanic = ProcessTitanic(train_path = "data\\train.csv" , test_path = "data\\test.csv")
#Exploration train_data
print("\nEtapa Exploración de los datos de entrenamiento: ")
train_data, test_data = titanic.exploration_data()
#Preprocessing train_data and test_data
print("\nEtapa Preprocesamiento de los datos de entrenamiento y prueba: ")
X_train, X_test, y_train = titanic.process_titanic(train_data = train_data)


#Train and evaluation
print("\nEtapa de Entrenamiento del modelo...")
trained_model = train_model(X_train,y_train)
#Evaluation
print("\nEtapa de predicción del modelo...")
predictions = predict_model(trained_model, X_test)
print(f"Algunas predicciones...: {predictions[:5]}")

# #Save my_submission.csv
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('.\outputs\\my_submission.csv', index=False)
print("Your submission was successfully saved!")


