import pandas as pd
import numpy as np
import csv
import os
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import re
from imblearn.under_sampling import NearMiss 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
#Ejecutar esta linea de abajo solo una vez para tener las stopwords en español
#nltk.download('stopwords')
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup
from imblearn.under_sampling import RandomUnderSampler

i = 1
contador = 0
while i < 29:
    URL = 'https://votainteligente.cl/propuestas/?page='+str(i)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.find(id='posts')
    job_elems = results.find_all('div', class_='post')
    print(job_elems)
    for job_elem in job_elems:
        try:
            category = job_elem.find('a')['href']
            category = re.search("clasification=(.*)", category)
            category = category.group(1)
            title = job_elem.find('h4')
            title = title.text
            title = re.sub("/", "-", title)
            leer_mas = job_elem.find('a', class_='btn btn-blue pull-right')['href']
            URL = 'https://votainteligente.cl'+leer_mas
            page = requests.get(URL)
            soup = BeautifulSoup(page.content, 'html.parser')
            cuerpo = soup.find('div', class_='col-md-12')
            cuerpo = cuerpo.find('p')
            cuerpo = cuerpo.text
            
            politica = ['politica', 'justicia', 'transparencia','democracia','institucionespublicas']
            medioambiente = ['medioambiente','medio_ambiente','sustentabilidad','recursosnaturales','emergencia']
            social = ['proteccion_y_familia','genero','proteccionsocial',
                     'infancia','terceraedad']
            sanidad = ['salud']
            culturayciencia = ['cultura','deporte','tecnologia','ciencias','ocio']
            interior = ['transporte','educacion','trabajo','empleo','economia']

            if category in politica:
                if not os.path.exists('./dataset/'+'politica'):
                    os.makedirs('./dataset/'+'politica')
                f= open('./dataset/'+'politica'+'/'+title,"w+")
                f.write(cuerpo)
                contador+=1
                f.close()
            if category in medioambiente:
                if not os.path.exists('./dataset/'+'medioambiente'):
                    os.makedirs('./dataset/'+'medioambiente')
                f= open('./dataset/'+'medioambiente'+'/'+title,"w+")
                f.write(cuerpo)
                contador+=1
                f.close()
            if category in social:
                if not os.path.exists('./dataset/'+'social'):
                    os.makedirs('./dataset/'+'social')
                f= open('./dataset/'+'social'+'/'+title,"w+")
                f.write(cuerpo)
                contador+=1
                f.close()
            if category in sanidad:
                if not os.path.exists('./dataset/'+'sanidad'):
                    os.makedirs('./dataset/'+'sanidad')
                f= open('./dataset/'+'sanidad'+'/'+title,"w+")
                f.write(cuerpo)
                contador+=1
                f.close()
            if category in culturayciencia:
                if not os.path.exists('./dataset/'+'culturayciencia'):
                    os.makedirs('./dataset/'+'culturayciencia')
                f= open('./dataset/'+'culturayciencia'+'/'+title,"w+")
                f.write(cuerpo)
                contador+=1
                f.close()
            if category in interior:
                if not os.path.exists('./dataset/'+'interior'):
                    os.makedirs('./dataset/'+'interior')
                f= open('./dataset/'+'interior'+'/'+title,"w+")
                f.write(cuerpo)
                contador+=1
                f.close()
        except:
            continue
    print('Pagina '+ str(i) + ' scrapeada')
    i += 1
print(f'Numero total de textos = {contador}')
col_names =  ['category', 'title', 'content'] 
df  = pd.DataFrame(columns = col_names) 
path = r"dataset\culturayciencia"
categoria= "culturayciencia"
directories = os.listdir( path )

for file in directories:
    print("El nombre de archivo es: "+ file)
    print("La categoria del archivo es: "+ categoria)
    objText = open(path+"\\"+file,"r",encoding='ANSI')
    lstLines = objText.read()
    titulo = file
    cuerpo = re.findall(r".*",lstLines)
    print("El titulo es: "+ titulo)
    print("El cuerpo es: "+ cuerpo[0])
    print(df)
    new_row = {'category':categoria, 'title':titulo, 'content':cuerpo[0]}
    print(new_row)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

path = r"dataset\interior"
categoria= "interior"
directories = os.listdir( path )
 

for file in directories:
    print("El nombre de archivo es: "+ file)
    print("La categoria del archivo es: "+ categoria)
    objText = open(path+"\\"+file,"r",encoding='ANSI')
    lstLines = objText.read()
    titulo = file
    cuerpo = re.findall(r".*",lstLines)
    print("El titulo es: "+ titulo)
    print("El cuerpo es: "+ cuerpo[0])
    new_row = {'category':categoria, 'title':titulo, 'content':cuerpo[0]}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print()

path = r"dataset\medioambiente"
categoria= "medioambiente"
directories = os.listdir( path )
 

for file in directories:
    print("El nombre de archivo es: "+ file)
    print("La categoria del archivo es: "+ categoria)
    objText = open(path+"\\"+file,"r",encoding='ANSI')
    lstLines = objText.read()
    titulo = file
    cuerpo = re.findall(r".*",lstLines)
    print("El titulo es: "+ titulo)
    print("El cuerpo es: "+ cuerpo[0])
    new_row = {'category':categoria, 'title':titulo, 'content':cuerpo[0]}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print()

path = r"dataset\politica"
categoria= "politica"
directories = os.listdir( path )
 

for file in directories:
    print("El nombre de archivo es: "+ file)
    print("La categoria del archivo es: "+ categoria)
    objText = open(path+"\\"+file,"r",encoding='ANSI')
    lstLines = objText.read()
    titulo = file
    cuerpo = re.findall(r".*",lstLines)
    print("El titulo es: "+ titulo)
    print("El cuerpo es: "+ cuerpo[0])
    new_row = {'category':categoria, 'title':titulo, 'content':cuerpo[0]}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print()

path = r"dataset\sanidad"
categoria= "sanidad"
directories = os.listdir( path )
 

for file in directories:
    print("El nombre de archivo es: "+ file)
    print("La categoria del archivo es: "+ categoria)
    objText = open(path+"\\"+file,"r",encoding='ANSI')
    lstLines = objText.read()
    titulo = file
    cuerpo = re.findall(r".*",lstLines)
    print("El titulo es: "+ titulo)
    print("El cuerpo es: "+ cuerpo[0])
    new_row = {'category':categoria, 'title':titulo, 'content':cuerpo[0]}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print()

path = r"dataset\social"
categoria= "social"
directories = os.listdir( path )
 

for file in directories:
    print("El nombre de archivo es: "+ file)
    print("La categoria del archivo es: "+ categoria)
    objText = open(path+"\\"+file,"r",encoding='ANSI')
    lstLines = objText.read()
    titulo = file
    cuerpo = re.findall(r".*",lstLines)
    print("El titulo es: "+ titulo)
    print("El cuerpo es: "+ cuerpo[0])
    new_row = {'category':categoria, 'title':titulo, 'content':cuerpo[0]}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print()
        
df['category_id'] = df['category'].factorize()[0]
df
category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

fig = plt.figure(figsize=(6,4))
df.groupby('category').content.count().plot.bar(ylim=0)
plt.show()

nltk.download('stopwords')
stop_es=stopwords.words('spanish')
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=stop_es)
features = tfidf.fit_transform(df.content).toarray()
labels = df.category_id
features.shape

N = 3
for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names_out())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(category))
  print("  . Unigramas mas representativos:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Bigramas mas representativos:\n. {}".format('\n. '.join(bigrams[-N:])))

SAMPLE_SIZE = int(len(features) * 0.2)
np.random.seed(0)
indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])
colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey', 'yellow']
for category, category_id in sorted(category_to_id.items()):
    points = projected_features[(labels[indices] == category_id).values]
    plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category)
plt.title("vector de características tf-idf para cada artículo, proyectado en 2 dimensiones.",
          fontdict=dict(fontsize=15))
plt.legend()

df[df.title.str.contains('mayores')]
df[df.content.str.lower().str.contains('pobreza')].category.value_counts()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.2, random_state=42)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
rus = RandomUnderSampler(random_state=42)
X_train, y_train = rus.fit_resample(X_train, y_train)

# MultinomialNB
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],  # Parámetro de suavizado de Laplace
    'fit_prior': [True, False]            
}

grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Mejores hiperparámetros: ", grid_search.best_params_)

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print('Precision: %.3f' % precision_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=0))
print('Recall: %.3f' % recall_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=0))
print('F1 Score: %.3f' % f1_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=0))

# LinearSVC
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],         
    'loss': ['hinge', 'squared_hinge'],   
    'max_iter': [1000, 5000, 10000]        
}

grid_search = GridSearchCV(LinearSVC(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Mejores hiperparámetros: ", grid_search.best_params_)

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print('Precision: %.3f' % precision_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=0))
print('Recall: %.3f' % recall_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=0))
print('F1 Score: %.3f' % f1_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=0))

#DecisionTreeClassifier
param_grid = {
    'max_depth': [3, 5, 10, 20],              
    'min_samples_split': [2, 5, 10],               
    'min_samples_leaf': [1, 2, 4],                  
    'criterion': ['gini', 'entropy']                
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)
print("Mejores hiperparámetros: ", grid_search.best_params_)

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print('Precision: %.3f' % precision_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=0))
print('Recall: %.3f' % recall_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=0))
print('F1 Score: %.3f' % f1_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=0))

#LogisticRegression
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Parámetro de regularización
    'penalty': ['l1', 'l2'],              # Penalización l1 o l2
    'solver': ['liblinear', 'saga']        # Solvers compatibles con l1 y l2
}

grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)
print("Mejores hiperparámetros: ", grid_search.best_params_)

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', 
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print('Precision: %.3f' % precision_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=0))
print('Recall: %.3f' % recall_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=0))
print('F1 Score: %.3f' % f1_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=0))

#En este bloque testeamos el modelo que se ha decidido usar al final
texts = ["Nos gustaria mejorar las calles de nuestro barrio.",
         "Faltan hospitales en mi pueblo.",
         "Hay que fomentar la innovacion y el emprendimiento",
         "Reduccion de jornadas laborales para todos los empleados",
         "Ayudas para los desempleados de larga duración"]
text_features = tfidf.transform(texts)
predictions = best_model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Se predice como: '{}'".format(id_to_category[predicted]))
  print("")





