from os import X_OK
from data import ProcessTitanic

#Get and split data train, and test
titanic = ProcessTitanic(train_path = "data\\train.csv" , test_path = "data\\test.csv")
train_data = titanic.exploration_data()
print(train_data)

# X_train, X_test, y_train = titanic.process_titanic(train_data)
# print(X_train)
# print(X_train.shape, X_test.shape, y_train.shape)
# # Split train-test data

#Train

#Evaluation

#Save my_submission.csv


