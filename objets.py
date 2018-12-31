
import nltk
import csv
import pandas as pd
import keras
from sklearn.utils import shuffle
from math import sqrt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf























def distance(x,y):  
    n = min(len(x),len(y))
    A = 0
    for i in range(n):
        A += (x[i]-y[i])**2
    return sqrt(A)


def cosine(x,y):
    if np.linalg.norm(x)==0 or np.linalg.norm(y)==0:
        return 0
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

class Graph(object):
    def __init__(self):
        self.graph = igraph.Graph()
        self.nodes = {}
        self.edgesToCommit = []
        
    def addEdge(self,source,target):
        if source in self.nodes:
            sourceId = self.nodes[source]
        else:
            sourceId = len(self.nodes)
            self.nodes[source] = sourceId
            self.graph.add_vertices(1)
            
        if target in self.nodes:
            targetId = self.nodes[target]
        else:
            targetId = len(self.nodes)
            self.nodes[target] = targetId
            self.graph.add_vertices(1)
            
        self.edgesToCommit.append((sourceId,targetId))
        
    def commitEdges(self):
        self.graph.add_edges(self.edgesToCommit)
        self.edgesToCommit = []
        
        
    def clusterize(self):
        self.graph.es["width"] = 1
        self.graph.simplify(combine_edges={"width":"sum"})
        self.clustering = self.graph.community_fastgreedy().as_clustering()
        
    def getClusteringFeatures(self):
        dic = {}
        for k in self.nodes:
            dic[k] = self.clustering.membership[self.nodes[k]]
        return dic
        
class Network():


    def __init__(self, learning_rate=0.1,densenumber=400):
        ''' initialize the classifier with default (best) parameters '''
        # TODO
        self.alpha = learning_rate
        self.model = keras.models.Sequential([
            keras.layers.Dense(densenumber,input_dim=40,activation=tf.nn.relu),
            keras.layers.Dropout(0.5, noise_shape=None, seed=None),      
            keras.layers.Dense(20,activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer=keras.optimizers.RMSprop(), 
              loss="mean_squared_error",
              metrics=['accuracy'])



    def fit(self,X,Y,warm_start=True,n_epochs=210,batch_size=1000):
        ''' train the network, and if warm_start, then do not reinit. the network
            (if it has already been initialized)
        '''
        
        self.model.fit(X, Y, epochs=n_epochs,verbose=1,shuffle=True,batch_size=batch_size)

        # TODO
        return self

    def predict_proba(self,X):
        ''' return a matrix P where P[i,j] = P(Y[i,j]=1), 
        for all instances i, and labels j. '''
        # TODO
        return self.model.predict(X)

    def predict(self,X):
        ''' return a matrix of predictions for X '''
        return (self.predict_proba(X) >= 0.5).astype(int)        

       

class Estimator(object):
    def __init__(self,node_info,node_edges,LSI,hachs, corpus):
        self.node_info = node_info
        self.node_edges = node_edges
        self.authors = {}
        self.nodeFeatures = {}
        self.nv = self.node_info.values
        self.corpus = corpus
        self.LSI=LSI
        self.hachs = hachs
        
        
    def createGraphs(self):
        author_relation = []
        t = 0
        self.nv = self.node_info.values
        for i in range(len(self.node_info["Authors"])):
            try:
                author_relation.append(self.node_info["Authors"][i].split(", "))
            except AttributeError:
                t += 1
        print("{}/{} papers does not have authors".format(t,len(self.node_info["Authors"])))
        
        self.author_graph = Graph()
        for r in author_relation:
            for i in range(len(r)):
                for j in range(i+1,len(r)):
                    self.author_graph.addEdge(r[i],r[j])
                    self.author_graph.addEdge(r[j],r[i])

        self.author_graph.commitEdges()

        self.paper_graph = Graph()
        edges = self.node_edges[self.node_edges["Type"]==1]

        a = edges.values
        for i in range(len(a)):
            self.paper_graph.addEdge(a[i][0],a[i][1])
            self.paper_graph.addEdge(a[i][1],a[i][0])

        self.paper_graph.commitEdges()
        

    def train(self,n_epochs):
        """
        self.createGraphs()
        self.paper_graph.clusterize()
        #self.author_graph.clusterize()
        
        print("Clustering...")
        paper_cluster = self.paper_graph.getClusteringFeatures()
        """
        for element in self.nv:
            #self.nodeFeatures[element[0]] = {"pubyear":2000,"cluster":-1,"atfidf":[],"ttfidf":[]}
            self.nodeFeatures[element[0]] = {}
        """
        for k in paper_cluster:
            
            self.nodeFeatures[k]["cluster"] = paper_cluster[k]
        
        print("pubyear...")   
        for element in self.nv:
            self.nodeFeatures[element[0]]["pubyear"] = element[1]
        
        print("Title NLP ...")
        corpus = [element[3] for element in self.nv]
        for i in range(len(corpus)):
            if corpus[i] is np.nan:
                corpus[i] = " ".join(self.nv[i][5].split(" ")[:10])
        title_vectorizer = TfidfVectorizer(stop_words="english")
        titleFeatures = title_vectorizer.fit_transform(corpus)
        
        print("Abstract NLP...")
    Moui = np.zeros((len
        corpus = [element[5] for element in self.nv]
        abstract_vectorizer = TfidfVectorizer(stop_words="english")
        abstractFeatures = abstract_vectorizer.fit_transform(corpus)
        """
        print("Saving features...")
        """
        for i in range(len(self.nv)):
            self.nodeFeatures[self.nv[i][0]]["atfidf"] = abstractFeatures[i].data
            self.nodeFeatures[self.nv[i][0]]["ttfidf"] = titleFeatures[i].data
            
        """    
        print("Calculating features for every pair...")
        ev = self.node_edges.values
        
        
        
        X_train=[]
        Y_train=[]
        n = len(ev)
        for element in ev:
            
            X_train.append(self.get_features(element[0],element[1]))
            Y_train.append([element[2]])
            p = len(X_train)
            if(p % (n//100) == 0):
                print(p)
        
       # h = Network(densenumber=40)
        #print("Network training...")
       # h.fit(np.array(X_train),np.array(Y_train),batch_size=360,n_epochs=10)
        return X_train,Y_train
         

            
    def get_features(self,k_x,k_y):
        """
        f = [0,0,0,0,0]
        f[0] = abs(self.nodeFeatures[k_x]["pubyear"] - self.nodeFeatures[k_y]["pubyear"])
        f[1] = self.nodeFeatures[k_x]["cluster"]
        f[2] = self.nodeFeatures[k_y]["cluster"]
        self.test = self.nodeFeatures[k_x]["atfidf"]
        f[3] = distance(self.nodeFeatures[k_x]["atfidf"],self.nodeFeatures[k_y]["atfidf"])
        f[4] = distance(self.nodeFeatures[k_x]["ttfidf"],self.nodeFeatures[k_y]["ttfidf"])
        """
        
        id1 = self.hachs[k_x]
        id2 = self.hachs[k_y]
        q1 = np.zeros(self.LSI.projection.u.shape[0])
        for t in self.corpus[id1]:
            q1[t[0]] = t[1]
        q2 = np.zeros(self.LSI.projection.u.shape[0])
        for t in self.corpus[id2]:
            q2[t[0]] = t[1]
            
        Lk = np.diag(self.LSI.projection.s)
        
        q1n = np.dot(np.dot(q1, self.LSI.projection.u), Lk)
        q2n = np.dot(np.dot(q2, self.LSI.projection.u), Lk)
        q = np.concatenate((q1n, q2n), axis = None)
        q = cosine(q1n,q2n)
        
        
        return q          
    
        
    
    def predict(self,testing_set):
        return [1 for i in testing_set]
