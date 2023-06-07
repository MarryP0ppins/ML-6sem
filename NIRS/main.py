# Импорт библиотек

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_text

# Загрузка датасета

dataset = pd.read_csv('NIRS\Churn_Modelling.csv')

# Обработка дубликатов, пропусков, кодирование категориальных данных

dataset.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)
dataset['Geography'] = dataset['Geography'].astype('category')
dataset['Gender'] = dataset['Gender'].astype('category')
dataset.drop_duplicates(inplace=True)
scaler = StandardScaler()
dataset[['CreditScore', 'Balance','EstimatedSalary']] = scaler.fit_transform(dataset[['CreditScore', 'Balance','EstimatedSalary']])

category_columns = dataset.select_dtypes(include=['category']).columns
ohe = OneHotEncoder()
encoded_columns = ohe.fit_transform(dataset[category_columns])
dataset[np.concatenate(ohe.categories_)] = encoded_columns.toarray()
dataset.drop(category_columns, axis=1, inplace=True)
dataset = dataset[[x for x in dataset.columns if x != 'Exited']+['Exited']]

dataset.columns = dataset.columns.astype(str)
X = dataset.drop(columns=['Exited'])
y = dataset.Exited

# Функция для обучения модели, принимает на вход размер тестовой выборки и параметр модели максимальная глкбина дерева
def train_model(max_depth, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return model, recall, precision

# Функция, которая выводит интерфейс приложения и вызывает функцию train_model для обучения модели при изменении гиперпараметров
def app():
    st.title('Decision Tree Classifier')

    max_depth = st.slider('Max depth', min_value=1, max_value=10, value=3, step=1)
    test_size = st.slider('Test size', min_value=0.1, max_value=0.5, value=0.2, step=0.1)

    model, recall, precision = train_model(max_depth, test_size)

    st.write('Recall:', recall)
    st.write('Precision:', precision)
    st.write('Visualize tree:')
    tree_rules = export_text(model, feature_names=X.columns.tolist(), show_weights=True)
    st.code(np.array2string(np.array(tree_rules.split("\n")).reshape(-1, 1), separator=",").replace("[[", "").replace("]]", ""), language='python')



if __name__ == '__main__':
    app()