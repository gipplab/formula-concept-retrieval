import arXivDocs2tfidf
import arXivDocs2Vec

from sklearn.preprocessing import LabelBinarizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

from sklearn.cluster import KMeans
from sklearn.neighbors._nearest_centroid import NearestCentroid

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy

from collections import Counter

tex_file = "diff_eqns_examples/three examples/diff_eqns_tex.csv"
content_file = "diff_eqns_examples/three examples/diff_eqns_content.csv"
qids_file = "diff_eqns_examples/three examples/diff_eqns_qids.csv"
labels_file = "diff_eqns_examples/three examples/diff_eqns_labels.csv"

# get equations tex

with open(tex_file,'r') as f:
    eqns_tex = f.readlines()

# get equations content

with open(content_file,'r') as f:
    lines = f.readlines()

eqns_cont = []

for line in lines:
    #eqns_cont.append(line.replace(",","").strip("\n"))
    tmp_cont = line.replace(",","").strip("\n")
    tmp_tmp_cont = ""
    for word in tmp_cont.split():
        tmp_tmp_cont += " math" + word
    eqns_cont.append(tmp_tmp_cont)

# get equations qids

with open(qids_file,'r') as f:
    lines = f.readlines()
eqns_qids = []
for line in lines:
    eqns_qids.append(line.strip("\n"))

# get equations labels

with open(labels_file,'r') as f:
     lines = f.readlines()
eqns_labs = []
for line in lines:
    eqns_labs.append(line.strip("\n"))

# ENCODING
# get encoding
#
# content
enc_str = 'cont_'
#
eqns_enc = arXivDocs2tfidf.docs2tfidf(eqns_cont)
enc_str += 'tfidf'
title_text = "Formula content space (TF-IDF)"
#
#eqns_enc = arXivDocs2Vec.docs2vec(eqns_cont,eqns_tex)
#enc_str += 'd2v'
#title_text = "Formula content space (Doc2Vec)"
#
# semantics
#enc_str = 'sem_'
#
#eqns_enc = arXivDocs2tfidf.docs2tfidf(eqns_qids)
#enc_str += 'tfidf'
#title_text = "Formula semantic space (TF-IDF)"
#
#eqns_enc = arXivDocs2Vec.docs2vec(eqns_qids,eqns_tex)
#enc_str += 'd2v'
#title_text = "Formula semantic space (Doc2Vec)"

# set vectors
X,y = eqns_enc,eqns_labs

# CLASSIFICATION

classifier = LinearSVC()#LogisticRegression()
classifier.fit(X,y)
prediction = list(classifier.predict(X))

# matches = 0
# total = len(prediction)
# for i in range(0,total):
#     if prediction[i] == eqns_labs[i]:
#         matches += 1
# accuracy = matches/total

accuracy = numpy.mean(cross_val_score(classifier, X, y, cv=3))

print('Classification accuracy: ' + str(accuracy))

# CLUSTERING

n_clusters = 3
clusterer = KMeans(n_clusters=n_clusters)

clustering = clusterer.fit_predict(X)

# calculate cluster purity
clusters = list(clustering)

# calculate purities
purities = []
ranges = [(0,10),(10,20),(20,30)]
for range in ranges:
    purities.append(max(Counter(clusters[range[0]:range[1]]).values())/10)
purity = numpy.mean(purities)

print('Cluster purity: ' + str(purity))

# DIMENSIONALITY REDUCTION

#labels = LabelBinarizer.fit_transform(prediction)
labels = list(clustering)
#labels = LabelBinarizer().fit_transform(eqns_labs)
# for i in range(0,len(eqns_labs)):
#     if eqns_labs[i] == "KGE":
#         eqns_labs[i] = 1
#     if eqns_labs[i] == "EFE":
#         eqns_labs[i] = 0
#     if eqns_labs[i] == "ME":
#         eqns_labs[i] = 2
# labels = eqns_labs

red = PCA(n_components=2)
red = TruncatedSVD(n_components=2)
vectors_red = red.fit_transform(X)

# calculate cluster centroid distances in reduced space
centroids = NearestCentroid()
centroids.fit(vectors_red,clustering)
c_vecs = []
for centroid in centroids.centroids_:
    c_vecs.append(centroid)

c_dists = []
for i in [0,1,2]:
    for j in [0,1,2]:
        if i != j:
            c_dists.append(numpy.linalg.norm(c_vecs[i]-c_vecs[j]))
c_mean_dist = numpy.mean(c_dists)

# Linear regression in 2D

# classifier = LogisticRegression()
# # select specific classes
# x_new = []
# y_new = []
# # class KGE
# x_new.extend(vectors_red[10:20])
# y_new.extend(y[10:20])
# # class ME
# x_new.extend(vectors_red[20:30])
# y_new.extend(y[20:30])
# # get linear equation
# classifier.fit(x_new,y_new)
# #classifier.fit(vectors_red,y)
# coef = classifier.coef_

# PLOT reduced vectors

fig = plt.figure()
ax = plt.axes()

# Plot a line from slope and intercept
# def plot_line(intercept, slope):
#     axes = plt.gca()
#     x_vals = numpy.array(axes.get_xlim())
#     y_vals = intercept + slope * x_vals
#     plt.plot(x_vals, y_vals, '--')
#
# intercept = coef[0][0]
# slope = coef[0][1]
# plot_line(intercept,slope)

# for i in range(0,len(set(y))):
#     intercept = coef[i][0]
#     slope = coef[i][1]
#     plot_line(intercept,slope)

# annotate

# label
for i, lab in enumerate(eqns_labs):
    ax.annotate(lab, (vectors_red[i,0],vectors_red[i,1]))
# annotate
#for i, text in enumerate(eqns_tex):
#    ax.annotate(text, (vectors_red[i,0],vectors_red[i,1]))
# text
text = 'Mean cluster purity: '
if enc_str == 'cont_tfidf':
    text += '0.50'
    x_pos, y_pos = 0.2, -0.3
elif enc_str == 'cont_d2v':
    text += '0.97'
    x_pos, y_pos = 0.2, -0.3
elif enc_str == 'sem_tfidf':
    text += '0.97'
    x_pos, y_pos = 0.2, -0.3
elif enc_str == 'sem_d2v':
    text += '0.97'
    x_pos, y_pos = 0.2, -0.3
plt.text(x_pos, y_pos, text)

plt.xlabel("PCA dimension 1")
plt.ylabel("PCA dimension 2")
plt.title(title_text)

ax.scatter(vectors_red[:,0], vectors_red[:,1],c=labels, cmap='rainbow')

plt.show()

print("end")