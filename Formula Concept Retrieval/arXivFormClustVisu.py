import operator
from collections import Counter
import pickle
import numpy as np

from sklearn import preprocessing
import sklearn.cluster
import sklearn.mixture
import hdbscan

from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set file paths
#basePath = ""
#basePath = "F:\\SigMathLing_arXMLiv-08-2018\\"
basePath = "F:\\NTCIR-12_MathIR_arXiv_Corpus\\"

formulaPath = "formulae\\duplicates\\"
subjectClass = "astro-ph\\"

inputPath = basePath + formulaPath + subjectClass
outputPath = basePath + formulaPath + subjectClass

# initialize clustering
n_clusters = 10

clustering = sklearn.cluster.KMeans(n_clusters=n_clusters)#.fit(X)
#clustering = sklearn.cluster.AffinityPropagation()#.fit(X)
#clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters)#.fit(X)
#clustering = sklearn.cluster.MeanShift()#.fit(X)
#clustering = sklearn.mixture.GaussianMixture(n_components=n_clusters)#.fit(X).predict(X)
#clustering = hdbscan.HDBSCAN()#.fit(X)
##clustering = sklearn.cluster.DBSCAN(eps=3, min_samples=2)#.fit_predict(X)
##clustering = sklearn.cluster.SpectralClustering(n_clusters=n_clusters)#.fit(X)

# Load formula labels and math vectors

with open(outputPath + "formulaLabs.pkl",'rb') as f:
    formulaLabs = pickle.load(f)
with open(outputPath + "formulaDocs.pkl",'rb') as f:
    formulaDocs = pickle.load(f)
with open(outputPath + "surrTextData.pkl",'rb') as f:
    surrTextData = pickle.load(f)

with open(outputPath + "formulae_doc2vec.pkl",'rb') as f:
    formulae_doc2vec = pickle.load(f)
with open(outputPath + "formulae_tfidf.pkl",'rb') as f:
    formulae_tfidf = pickle.load(f)

with open(outputPath + "formulae_semantics_doc2vec.pkl",'rb') as f:
    formulae_semantics_doc2vec = pickle.load(f)
with open(outputPath + "formulae_semantics_tfidf.pkl",'rb') as f:
    formulae_semantics_tfidf = pickle.load(f)

# cluster function

def cluster(Vecs,clustering):

    # sparse tfidf matrix needs to be densified
    try:
        #clustering.fit(Vecs)
        #clustering = clustering.predict(Vecs)
        clustering = clustering.fit_predict(Vecs)
    except:
        docVecs = Vecs.toarray()
        #docVecs = TruncatedSVD(n_components=300).fit_transform(Vecs)
        #clustering.fit(Vecs)
        #clustering = clustering.predict(Vecs)
        clustering = clustering.fit_predict(Vecs)

    return clustering#.labels_

# SET ENCODING
encoding = formulae_doc2vec

# FIND K NEAREST NEIGHBORS
nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(encoding)
distances, indices = nbrs.kneighbors(encoding)

# FIND FORMULA DUPLICATES
formulaDuplicates_count = {}
formulaDuplicates_docs = {}
formulaDuplicates_text = {}
formulaDuplicates_kNN = {}
i = 0
for formulaLab in formulaLabs:

    try:
        formulaDuplicates_count[formulaLab] += 1
    except:
        formulaDuplicates_count[formulaLab] = 1

    try:
        for doc in formulaDocs[i]:
            formulaDuplicates_docs[formulaLab].append(doc)
    except:
        formulaDuplicates_docs[formulaLab] = formulaDocs[i]

    try:
        for word in surrTextData[i]:
            formulaDuplicates_text[formulaLab].append(word)
    except:
        formulaDuplicates_text[formulaLab] = surrTextData[i]

    for j in range(1, 4):
        try:
            formulaDuplicates_kNN[formulaLab].append(formulaLabs[indices[i,j]])
        except:
            formulaDuplicates_kNN[formulaLab] = []
            formulaDuplicates_kNN[formulaLab].append(formulaLabs[indices[i,j]])

    i += 1

# sort counts descending
formulaDuplicates_count = sorted(formulaDuplicates_count.items(), key=operator.itemgetter(1),reverse=True)
# count unique docs
formulaDuplicates_docs = [(tuple[0],len(set(formulaDuplicates_docs[tuple[0]]))) for tuple in formulaDuplicates_count]
# sort kNN as counts
formulaDuplicates_kNN = [(tuple[0],formulaDuplicates_kNN[tuple[0]]) for tuple in formulaDuplicates_count]

# (formulaLab, duplicateCount, uniqueDocs, kNN)
formulaDuplicates = {}
for i in range(0,len(formulaDuplicates_count)):
    formulaDuplicates[i] = (formulaDuplicates_count[i][0],formulaDuplicates_count[i][1],formulaDuplicates_docs[i][1],formulaDuplicates_kNN[i][1])

# CLUSTER FORMULAE

# get labels
# labels = cluster(encoding,clustering)
#
# # EXPORT CLUSTERED FORMULAE (LABELS)
# formula_cluster0 = {}
# for i in range(0, len(set(labels))):
#     # cluster i
#     formula_cluster0[i] = []
#     # all positions of specific cluster label i
#     for poss in np.where(labels == i):
#         # iterate positions to retrieve individual formulae
#         for pos in poss:
#             formula_cluster0[i].append(formulaLabs[int(pos)])
# # create sets to remove duplicates
# formula_cluster = {}
# for i in formula_cluster0:
#     formula_cluster[i] = set(formula_cluster0[i])
# del formula_cluster0

# DIMENSIONALITY REDUCTION

#red = PCA(n_components=2)
# red = TruncatedSVD(n_components=2)
# vectors_red = red.fit_transform(encoding)
#
# # plot reduced vectors
#
# fig = plt.figure()
# ax = plt.axes()
#
# ax.scatter(vectors_red[:,0], vectors_red[:,1],c=labels, cmap='rainbow')
#
# plt.show()

print("end")