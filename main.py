from objets import *
from journals import *
from authors import *
import nltk
import csv
import pandas as pd
import keras
from sklearn.utils import shuffle
from math import sqrt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from nlp import *
import seaborn as sns
import networkit as net
from sklearn.decomposition import PCA


node_info = pd.read_csv("data/node_information.csv", names = ["Id","Pubyear","Title","Authors","Journal","Abstract"])
node_edges = pd.read_csv("data/training_set.csv", delimiter=" ", names = ["Source","Target","Type"])

node_edges = shuffle(node_edges)



testing_set = node_edges[:30000]
training_set = node_edges[30000:]


n_topics_title = 20
n_topics_abstract = 20
n_topics_lda_abstract = 20

#creation du modèle LSI

nv = node_info[["Id","Abstract"]].values
ev = training_set.values
texts = []
hach = {}

print("Loop words asbstracts...")
for i in range(len(nv)):
    element = nv[i]
    title = element[1]
    Id = element[0]
    hach[Id] = i
    texts.append(title)
print("...Done")

print("Model abstract...")
lsimodel_abstract, corpus_abstract = LSI_topicExtraction(texts, n_topics_abstract)
print("...Done")


nv = node_info[["Id","Title"]].values

texts = []
hach = {}

print("Loop words titles...")
for i in range(len(nv)):
    element = nv[i]
    title = element[1]
    Id = element[0]
    hach[Id] = i
    texts.append(title)

print("...Done")

print("Model title...")
lsimodel_title, corpus_title = LSI_topicExtraction(texts, n_topics_title)
print("...Done")  

print("Abstract LDA...")

nv = node_info[["Id","Abstract"]].values
texts = []

for i in range(len(nv)):
    element=nv[i]
    abstract = element[1]
    texts.append(abstract)
    
ldamodel_abstract,corpus_lda_abstract = LDA_topicExtraction(texts, n_topics_lda_abstract)


#Matrice des journeaux

print("Journals matrix...")
journalMatrix,jindices = getJournalMatrix(node_info,training_set)
print("...Done")
#Matrice es auteurs
print("Authors matrix...")
authorMatrix,aindices = getAuthorMatrix(node_info,training_set)
print("...Done")
print("Mean Year Gap...")
mean_gap_year = node_info["Pubyear"].dropna().std()
print("...Done")



print("Paper neighbours...")
paperNeighbours = {}

for i in node_info["Id"].values:
    paperNeighbours[i] = []
    
for edge in ev:
    a = edge
    if edge[2] == 1:
        paperNeighbours[edge[0]].append(edge[1])
        paperNeighbours[edge[1]].append(edge[0])

print("...Done")


print("Paper Graph...")

paperGraph = net.Graph(n=len(node_info),weighted=False,directed=False)

for edge in ev:
    if edge[2] == 1:
        paperGraph.addEdge(hach[edge[0]],hach[edge[1]])
        
predictors = [net.linkprediction.AdamicAdarIndex(paperGraph),
              net.linkprediction.AdjustedRandIndex(paperGraph),
              net.linkprediction.CommonNeighborsIndex(paperGraph),
              net.linkprediction.JaccardIndex(paperGraph),
              net.linkprediction.NeighborhoodDistanceIndex(paperGraph),
              net.linkprediction.NeighborsMeasureIndex(paperGraph)
              ]
              
centralities = [net.centrality.PageRank(paperGraph),
                net.centrality.ApproxBetweenness(paperGraph),
                net.centrality.EigenvectorCentrality(paperGraph),
            ]       



def getTitleCosine(k_x,k_y):
    id1 = hach[k_x]
    id2 = hach[k_y]
    q1 = np.zeros(lsimodel_title.projection.u.shape[0])
    for t in corpus_title[id1]:
        q1[t[0]] = t[1]
    q2 = np.zeros(lsimodel_title.projection.u.shape[0])
    for t in corpus_title[id2]:
        q2[t[0]] = t[1]
        
    Lk = np.diag(lsimodel_title.projection.s)
    
    q1n = np.dot(np.dot(q1, lsimodel_title.projection.u), Lk)
    q2n = np.dot(np.dot(q2, lsimodel_title.projection.u), Lk)
    q = np.concatenate((q1n, q2n), axis = None)
    q = cosine(q1n,q2n)
    
    
    return q      

def getAbstractCosine(k_x,k_y):
    id1 = hach[k_x]
    id2 = hach[k_y]
    q1 = np.zeros(lsimodel_abstract.projection.u.shape[0])
    for t in corpus_abstract[id1]:
        q1[t[0]] = t[1]
    q2 = np.zeros(lsimodel_abstract.projection.u.shape[0])
    for t in corpus_abstract[id2]:
        q2[t[0]] = t[1]
        
    Lk = np.diag(lsimodel_abstract.projection.s)
    
    q1n = np.dot(np.dot(q1, lsimodel_abstract.projection.u), Lk)
    q2n = np.dot(np.dot(q2, lsimodel_abstract.projection.u), Lk)
    q = np.concatenate((q1n, q2n), axis = None)
    q = cosine(q1n,q2n)
    
    
    return q       



def getAbstractLdaCosine(k_x,k_y):
    id1 = hach[k_x]
    id2 = hach[k_y]
    cx = corpus_lda_abstract[id1]
    cy = corpus_lda_abstract[id2]
    
    
    
    q1n = ldamodel_abstract[cx]
    q2n = ldamodel_abstract[cy]
    
    qx = np.zeros(n_topics_lda_abstract)
    qy = np.zeros(n_topics_lda_abstract)
    for k in q1n:
        qx[k[0]] = k[1]
    for k in q2n:
        qy[k[0]] = k[1]
    q = cosine(qx,qy)
    
    
    return q   



def getAuthorSimilarity(k_x,k_y):
    a_x = getAuthorsFromList(node_info["Authors"].values[hach[k_x]])
    a_y = getAuthorsFromList(node_info["Authors"].values[hach[k_y]])
    
    res = 0
    
    for x in a_x:
        for y in a_y:
            res += authorMatrix[aindices[x]][aindices[y]]
            
    return res / (len(a_x)*len(a_y))


def getAuthorSimilarityBis(k_x,k_y):
    a_x = getAuthorsFromList(node_info["Authors"].values[hach[k_x]])
    a_y = getAuthorsFromList(node_info["Authors"].values[hach[k_y]])
    
    res = 0
    
    for x in a_x:
        for y in a_y:
            res += authorMatrix[aindices[x]][aindices[y]]
            
    return res

def getJournalSimilarity(k_x,k_y):
    j_x = node_info["Journal"].values[hach[k_x]]
    j_y = node_info["Journal"].values[hach[k_y]]
    if type(j_x)==float:
        j_x = "NO_JOURNAL"
    if type(j_y)==float:
        j_y = "NO_JOURNAL"
    
    return journalMatrix[jindices[j_x]][jindices[j_y]]

def getYearDifference(k_x, k_y):
    j_x = node_info["Pubyear"].values[hach[k_x]]
    j_y = node_info["Pubyear"].values[hach[k_y]]
    if np.isnan(j_x) or np.isnan(j_y):
        return mean_gap_year
    return abs(j_x - j_y)



def getNeighboursJIndex(k_x,k_y):
    lx = paperNeighbours[k_x][:]
    ly = paperNeighbours[k_y][:]

    
    K = 0
    for i in lx:
        if i in ly:
            K += 1
    union = lx+ly
    if len(union) == 0:
        return 0,0
    n = len(pd.DataFrame(union)[0].unique())
    if k_x in ly:
        K -= 2
        n -= 2
        
    if n <= 0:
        return 0,0

    return K/n,K



def get2NeighboursJIndex(k_x,k_y):
    lx = paperNeighbours[k_x]
    ly = paperNeighbours[k_y]
    
    neighbors_2_kx = []
    neighbors_2_ky = []
    
    for i in lx:
        if i != k_y:
            neighbors_2_kx += paperNeighbours[i]
    for i in ly:
        if i != k_x:
            neighbors_2_ky += paperNeighbours[i]
    
    K = 0
    nx = 0
    ny = 0
    
    for neighbor2 in neighbors_2_kx:
        if neighbor2 != k_x:
            nx+=1
            if neighbor2 in ly:
                K+=1
              
    for neighbor2 in neighbors_2_ky:
        if neighbor2 != k_y:
            ny+=1
            
    if nx*ny==0:
        return 0
    
    return 2*K/(nx+ny)

def getDegree(k_x,k_y):
    lx = paperNeighbours[k_x]
    ly = paperNeighbours[k_y]
    nx=len(lx)
    ny=len(ly)
    
    if k_y in lx:
        nx -= 1
        ny -= 1
        
    return nx, ny

print("Components...")
components = net.components.ConnectedComponents(paperGraph)
components.run()
print("...Done")
print("Louvain...")
community = net.community.PLM(paperGraph)
community.run()
louvain = community.getPartition()
print("...Done")
print("PLP...")
community = net.community.PLP(paperGraph)
community.run()
PLP = community.getPartition()
print("...Done")

for c in centralities:
    c.run()
    
def getCentralityFeatures(k_x,k_y):
    id1 = hach[k_x]
    id2 = hach[k_y]
    
    featur = []
    
    for c in centralities:
        featur.append(c.score(id1))
        featur.append(c.score(id2))
    
    return featur


def getClusteringFeatures(k_x,k_y):
    id1 = hach[k_x]
    id2 = hach[k_y]
    
    if louvain.inSameSubset(id1,id2):
        louv = 1
    else:
        louv = 0
        
    if PLP.inSameSubset(id1,id2):
        plp = 1
    else:
        plp = 0
        
    if components.componentOfNode(id1) == components.componentOfNode(id2):
        compo = 1
    else:
        compo = 0
        
    
    
    return [louv,plp,compo]

    
def get_features(k_x,k_y):
    
    authorSimilarity = getAuthorSimilarityBis(k_x, k_y)
    #journalSimilarity = getJournalSimilarity(k_x, k_y)
    abstractCosine = getAbstractCosine(k_x,k_y)
    ldaabstractCosine = getAbstractLdaCosine(k_x,k_y)
    titleCosine = getTitleCosine(k_x,k_y)  
    yearDifference = getYearDifference(k_x,k_y)
    #neighboursJIndex,neighboursJIndexBis = getNeighboursJIndex(k_x,k_y)
    #k_x_DegreeIndex, k_y_DegreeIndex = getDegree(k_x,k_y)
    return [authorSimilarity, abstractCosine,titleCosine,ldaabstractCosine, yearDifference] + [p.run(hach[k_x],hach[k_y]) for p in predictors]




    


"""
    return [authorSimilarity, abstractCosine, titleCosine, yearDifference,neighboursJIndex,neighboursJIndexBis]
    return [authorSimilarity, journalSimilarity, abstractCosine, titleCosine, yearDifference,neighboursJIndex,neighboursJIndexBis, k_x_DegreeIndex,k_y_DegreeIndex]
"""
ev = training_set.values

print("Compute X_train, Y_train...")

X_train=[]
Y_train=[]
n = len(ev)
for element in ev:
    X_train.append(get_features(element[0],element[1]))
    Y_train.append([element[2]])
    p = len(X_train)
    if(p % (n//100) == 0):
        print(p//(n//100))
print("...Done")


ev_test = testing_set.values

print("Compute X_test, Y_true...")
X_test=[]
Y_true=[]
n_test = len(ev_test)
for element in ev_test:
    X_test.append(get_features(element[0],element[1]))
    Y_true.append([element[2]])
    p = len(X_test)
    if(p % (n_test//100) == 0):
        print(p//(n_test//100))
print("...Done")

import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score

from sklearn.svm import LinearSVC

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import f1_score

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


Y_train = np.array(Y_train)


original_X_train = X_train[:]

df = pd.DataFrame(np.concatenate((X_train,Y_train.reshape(len(Y_train),1 )),axis=1))
df = df.dropna()
n = df.shape[1]
Y_train = df[n-1]
X_train = df.drop(n-1,axis=1)
X_test = np.array(X_test)
Y_true = np.array(Y_true)

X_train = X_train.reindex()
Y_train = Y_train.reindex()

"""
print("Computing PCA...")






pca = PCA(n_components=10).fit(X_train)
X_train_reduced = pca.transform(X_train)



print("...Done")
"""
def predict(classifier,X,reduced=False):
    Y = []
    t = 0
    for k in X:
    
        thereIsNan = False
        for i in k:
            if np.isnan(i):
                thereIsNan=True
        if thereIsNan:
            t+=1
            Y.append(0)
        else:
            if reduced:
                Y.append(classifier.predict(pca.transform([k]))[0])
            else:
                Y.append(int(classifier.predict([k])[0]))
    print("fefzefze : " + str(t) )
    return Y


def predict_proba(classifier,X,reduced=False):
    Y = []
    t = 0
    for k in X:
        print(X)
        thereIsNan = False
        for i in k:
            if np.isnan(i):
                thereIsNan=True
        if thereIsNan:
            t+=1
            Y.append(0.)
        else:
            if reduced:
                Y.append(classifier.predict_proba(pca.transform([k]))[0][1])
            else:
                Y.append(classifier.predict_proba([k])[0][1])
    print("fefzefze : " + str(t) )
    return Y

"""
classifiers = [
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(n_estimators=5, max_features=1, max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(),
    XGBClassifier()
    
]


n_estimators_for_random_forest = [50, 100, 200]
max_features_for_random_forest = [3]
max_depth_for_random_forest = [5, 10, None]

random_forest_classfiers = {}
random_forest_sc
scores = []
print("Compute f1 scores of classifiers...")
for classifier in classifiers:
    #cross_val_scores.append(cross_val_score(classifier,X=X_data, y=Y_data, scoring='f1', cv=10,verbose=1))
    
    print(str(type(classifier)) + " fitting...")
    classifier.fit(X_train, Y_train)
    print("...Fitting done")
    

    Y_pred = predict(classifier,X_test)
    score = f1_score(Y_true,Y_pred)
    scores.append(score)
    print(str(type(classifier)) + " : " + str(score) )
    #print("...cross_val_score computed...")
print("...All scores computed")

ores = {}


print("Compute random forest classifiers with various features...")
for n_estimators in n_estimators_for_random_forest:
    for max_features in max_features_for_random_forest:
        for max_depth in max_depth_for_random_forest:
            random_forest_classfiers[(n_estimators, max_features, max_depth)] = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features)

for key_classifier in random_forest_classfiers.keys():
    classifier = random_forest_classfiers[key_classifier]
    classifier.fit(X_train, Y_train)
    
    Y_pred = []
    for k in X_test:
        thereIsNan = False
        for i in k:
            if np.isnan(i):
                thereIsNan=True
        if thereIsNan:
            Y_pred.append(0)
        else:
            Y_pred.append(classifier.predict([k])[0])
    score = f1_score(Y_true,Y_pred)
    print(str(key_classifier) + " : " + str(score))

    random_forest_scores[key_classifier] = score

print("...All scores computed for random forests")
"""
"""
scores = []
print("Compute f1 scores of classifiers...")
for classifier in classifiers:
    #cross_val_scores.append(cross_val_score(classifier,X=X_data, y=Y_data, scoring='f1', cv=10,verbose=1))
    
    print(str(type(classifier)) + " fitting...")
    classifier.fit(X_train, Y_train)
    print("...Fitting done")
    

    Y_pred = predict(classifier,X_test)
    score = f1_score(Y_true,Y_pred)
    scores.append(score)
    print(str(type(classifier)) + " : " + str(score) )
    #print("...cross_val_score computed...")
print("...All scores computed")

"""
"""
model = keras.models.Sequential()
model.add(keras.layers.Dense(10, input_dim=21, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(10, kernel_initializer='normal', activation='relu'))

model.add(keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier = model
classifier.fit(X_train,Y_train,epochs=5)


classifiers.append(classifier)

X_train2 = np.delete(X_train.values,1,axis=1)
X_train2 = np.delete(X_train2,14,axis=1)
X_train2 = np.delete(X_train2,14,axis=1)

X_train2 = np.delete(X_train2,14,axis=1)

X_train2 = np.delete(X_train2,14,axis=1)

X_train2 = np.delete(X_train2,14,axis=1)
X_train2 = np.delete(X_train2,14,axis=1)

X_test2 = np.delete(X_test,1,axis=1)
X_test2 = np.delete(X_test2,14,axis=1)
X_test2 = np.delete(X_test2,14,axis=1)

X_test2 = np.delete(X_test2,14,axis=1)

X_test2 = np.delete(X_test2,14,axis=1)

X_test2 = np.delete(X_test2,14,axis=1)
X_test2 = np.delete(X_test2,14,axis=1)

"""
classifiers = []

classifiers.append(keras.models.Sequential())

classifiers[-1].add(keras.layers.Dense(10, input_dim=14, kernel_initializer='normal', activation='relu'))
classifiers[-1].add(keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
classifiers[-1].add(keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
classifiers[-1].add(keras.layers.Dense(10, kernel_initializer='normal', activation='relu'))

classifiers[-1].add(keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
classifiers[-1].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


classifiers[-1].fit(X_train2,Y_train,epochs=5)

classifiers.append(keras.models.Sequential())

classifiers[-1].add(keras.layers.Dense(20, input_dim=11, kernel_initializer='normal', activation='relu'))
classifiers[-1].add(keras.layers.Dropout(0.5))
classifiers[-1].add(keras.layers.Dense(20, kernel_initializer='normal', activation='relu'))

classifiers[-1].add(keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
classifiers[-1].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




scores = []
for i in range(100):  
    classifiers[-1].fit(X_train.values,Y_train,epochs=1,batch_size=100000)
    
    
    
    preds=[classifiers[-1].predict(X_test)]
    
    
    
    for y in preds:
        Y_eee = []
        for i in y:
            if i > 0.5:
                Y_eee.append(1)
            else:
                Y_eee.append(0)
        score = f1_score(Y_eee,Y_true)
        scores.append(score)
        print(score)
    



"""

new_X_train = []

print("Computing new X_train...")


n_train = len(original_X_train)
for i in range(len(ev)):
    element = ev[i]
    new_X_train.append(get_features2(element[0],element[1],i))
    p = len(new_X_train)
    if(p % (n_train//100) == 0):
        print(p//(n_train//100))
print("...Done")

new_X_test=[]
n_train = len(X_test)
for i in range(len(ev_test)):
    element = ev[i]
    new_X_test.append(get_features2(element[0],element[1],i))
    p = len(new_X_train)
    if(p % (n_train//100) == 0):
        print(p//(n_train//100))
print("...Done")


"""






"""
print("Stacking...")

xx = X_train[:]
for i in classifiers:
    xx = np.concatenate((xx,i.predict(X_train).reshape((len(xx),1))),axis=1)


print("...Done")

ccc = AdaBoostClassifier()
ccc.fit(xx,Y_train)
#classifiers = [classifiers[0],classifiers[2],classifiers[4],classifiers[5]]
xx_test = X_test[:]
for i in classifiers:
    xx_test = np.concatenate((xx_test,i.predict(X_test).reshape((len(xx_test),1))),axis=1)

y_pred = ccc.predict(xx_test)

print("Stacking score : " + str(f1_score(y_pred,Y_true)))

"""

"""
print("Fitting...")
classifier.fit(X_train, Y_train)
print("...Done")

ev_test = testing_set.values
"""





"""
print("Compute X_test, Y_true...")
X_test=[]
Y_true=[]
n_test = len(ev_test)
for element in ev_test:
    X_test.append(get_features(element[0],element[1]))
    Y_true.append([element[2]])
    p = len(X_train)
    if(p % (n_test//100) == 0):
        print(p)
print("...Done")


Y_pred = classifier.predict(X_test)

scores = precision_recall_fscore_support(Y_true, Y_pred)

"""



"""
node_edges_to_predict = pd.read_csv("data/testing_set.csv", delimiter=" ", names = ["Source","Target"])
ev_to_predict = node_edges_to_predict.values

print("Compute X_to_predict...")
X_to_predict=[]
n_to_predict = len(ev_to_predict)
for element in ev_to_predict:
    X_to_predict.append(get_features(element[0],element[1]))
    p = len(X_to_predict)
    if(p % (n_to_predict//100) == 0):
        print(p // (n_to_predict//100))
print("...Done")

Y_pred = predict(classifier,X_to_predict)

prediction = list(Y_pred)
predictions_SVM = zip(range(len(Y_pred)), prediction)

with open("data/improved_predictions3.csv","w") as pred1:
    csv_out = csv.writer(pred1)
    for row in predictions_SVM:
        csv_out.writerow(row)


"""
"""
for i in range(20):
    classifier = XGBClassifier(n_estimators=100*i,objective="binary:logistic")
    classifier.fit(X_train.values,Y_train.values)
    y_pred = classifier.predict(X_test)
    print(f1_score(Y_true,y_pred))

"""


"""
X_train,Y_train=e.train(10)
X_train = pd.DataFrame(X_train)
Y_train = pd.DataFrame(Y_train)
Y_train = Y_train.rename(columns={0:1})
df = pd.concat((X_train, Y_train), axis=1)
df = df.dropna()
df0 = df[df[1]==0]
df0 = df0.reindex()
df1 = df[df[1]==1]
df1 = df1.reindex()
sns.distplot(df0[0])
sns.distplot(df1[0])
"""
