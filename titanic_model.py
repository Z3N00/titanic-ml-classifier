import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

filepath = 'train.csv'

def titanic_model(filepath):
    df = pd.read_csv(filepath)
    
    print(df.shape)
    print("-------------------------------------------------------------------")

    # Clean Data
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df.drop(columns=['Cabin'], inplace=True)

    X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']].copy()
    y = df['Survived']
   
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
    }

    # Train and evaluate all three models
    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"{name}: {accuracy * 100:.2f}%")
    print("-------------------------------------------------------------------")
    
    # Cross validate all three models
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5)
        print(f"{name}: Mean={scores.mean()*100:.2f}% Std={scores.std()*100:.2f}%")

    # Hyperparameter tuning for XGBoost
    param_grid = {
    'n_estimators': [50, 70, 90],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.12, 0.15, 0.18]
    }

    grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("-------------------------------------------------------------------")

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Accuracy: {grid_search.best_score_*100:.2f}%")
    print("-------------------------------------------------------------------")
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    print(f"Final Tuned Model Accuracy: {accuracy * 100:.2f}%")
    print("-------------------------------------------------------------------")

    feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
    feature_importance.sort_values(ascending=False).plot(kind='bar')
    plt.title('Feature Importance')
    plt.ylabel('Importance Score')
    plt.show()
titanic_model(filepath)