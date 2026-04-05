import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


def load_and_prepare(path):
    df = pd.read_csv(path, sep=';')  # UCI bank-full uses ';'. If your file uses commas, use sep=','
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df['y'].value_counts())

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'y' in cat_cols:
        cat_cols.remove('y')

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    y = LabelEncoder().fit_transform(df['y'])
    X = df.drop(columns=['y'])

    return X, y


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 4, 5, 6, 8, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8]
    }

    search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    print("Best params:", search.best_params_)
    print("Best CV accuracy:", search.best_score_)

    y_pred = best.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred, digits=4))

    print("\nDecision tree rules:\n")
    print(export_text(best, feature_names=list(X.columns)))


if __name__ == '__main__':
    data_path = r"c:\Users\Reema Johnson\Desktop\prodigy 3a\bank.csv"  # or bank-full.csv
    X, y = load_and_prepare(data_path)
    train_and_evaluate(X, y)
