from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#Build parameters
def train_model(X_train, y_train):
    n_estimators_ = [60,80,120,150,180]
    cv = 5
    parameters = {'n_estimators': n_estimators_, 'max_depth':[4,6,8,10,12,14]}
    #Model
    rfc = RandomForestClassifier()
    clf = GridSearchCV(estimator = rfc, param_grid = parameters, cv = cv, scoring = 'accuracy', return_train_score=True)
    trained_model = clf.fit(X_train, y_train)
    print(f"Proceso de entrenamiento finalizado, puntaje {round(clf.score(X_train, y_train)*100, 2)}%")
    return trained_model
    
def predict_model(trained_model, X_test):
    predictions = trained_model.predict(X_test)
    print("Proceso de predicción finalizado...")
    return predictions

