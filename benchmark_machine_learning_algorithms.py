# coding: utf-8

## Challenge data posos

# Le but du projet est de battre les 16% d'erreurs de Posos sur la classification des intentions derrière les questions médicales
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier,BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
#nltk.download()

### Upload des donnees

df_input = pd.read_csv('DATA/input_train.csv', sep=";", index_col=0)
df_output = pd.read_csv('DATA/output_train.csv', sep=";", index_col=0)

features = df_input.columns
y = [x[0] for x in df_output.values]
X = [x[0] for x in df_input.values]

### TFID vectorize
tfid_vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word',
                            ngram_range =(1,3), stop_words={'french'})

X_preprocessed = tfid_vectorizer.fit_transform(X)

X_to_split, X_validation, y_to_split, y_validation = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_to_split, y_to_split, test_size=0.2, random_state=42)

X_train.shape

### Use sklearn to find a correct classifier
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
### KNN 
scores_knn = []
for i in range(2,20):
    knn = KNeighborsClassifier(n_neighbors=i, n_jobs=3)
    knn.fit(X_train,y_train)
    scores_knn.append(knn.score(X_test,y_test))
print(scores_knn)
#9 est le nombre de voisins optimal mais l'on n'obtient pas mieux que 52,8%

knn = KNeighborsClassifier(n_neighbors=9, n_jobs=3)
knn.fit(X_train,y_train)
print('score : {}'.format(knn.score(X_test,y_test)))

classes = knn.classes_
y_pred = knn.predict(X_test)
cnf_matrix = confusion_matrix(y_test,y_pred)
print(cnf_matrix)

### MLP classifier

scores_mlp = []
for i in [10]:
    print('iteration number {}'.format(i))
    mlp = MLPClassifier(hidden_layer_sizes=(150,i), solver='lbfgs',activation='logistic', early_stopping = True, max_iter=100)
    mlp.fit(X_train,y_train)
    score = mlp.score(X_test,y_test)
    print(score)
    scores_mlp.append(score)
print(scores_mlp)


classes = mlp.classes_

### D'autres classifiers potentiels
n_estimators = 30
svc = SVC(gamma = 0.1)
nb_gaussian = GaussianNB()
bagging = BaggingClassifier(n_estimators=n_estimators, n_jobs=3)
grad_boost = GradientBoostingClassifier(n_estimators=n_estimators, verbose=1)
xtrees = ExtraTreesClassifier(n_estimators=n_estimators)
ada = AdaBoostClassifier(n_estimators=n_estimators)
