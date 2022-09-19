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

example_nr_prefix = ['three','ten'][1]
example_path = example_nr_prefix + " examples/"
full_path = "diff_eqns_examples/" + example_path

tex_file = full_path + "diff_eqns_tex.txt"
content_file = full_path + "diff_eqns_content.txt"
qids_file = full_path + "diff_eqns_qids.txt"
labels_file = full_path + "diff_eqns_labels.txt"

# parameters
#N_examples_included
N_classes = 10
N_examples_per_class = 10
N_examples = N_classes*N_examples_per_class

# get equations tex

with open(tex_file,'r') as f:
    eqns_tex = f.readlines()
# clean
eqns_tex[0] = eqns_tex[0].lstrip('ï»¿')
# cutoff
eqns_tex = eqns_tex[:N_examples]

# get equations content

with open(content_file,'r') as f:
    eqns_cont_tmp = f.readlines()
# clean
eqns_cont_tmp[0] = eqns_cont_tmp[0].lstrip('ï»¿')
#
eqns_cont = []
for eqn_cont in eqns_cont_tmp:
    #eqns_cont.append(eqn_cont.replace(",","").strip("\n"))
    tmp_cont = eqn_cont.replace(",","").strip("\n")
    tmp_tmp_cont = ""
    for word in tmp_cont.split():
        tmp_tmp_cont += " math" + word
    eqns_cont.append(tmp_tmp_cont)
del eqns_cont_tmp
# cutoff
eqns_cont = eqns_cont[:N_examples]

# get equations qids

with open(qids_file,'r') as f:
    eqns_qids_tmp = f.readlines()
eqns_qids = []
for eqn_qid in eqns_qids_tmp:
    eqns_qids.append(eqn_qid.strip("\n"))
del eqns_qids_tmp
# clean
eqns_qids[0] = eqns_qids[0].lstrip('ï»¿')
# cutoff
eqns_qids = eqns_qids[:N_examples]

# get equations labels

with open(labels_file,'r') as f:
     eqns_labs_tmp = f.readlines()
# clean
eqns_labs_tmp[0] = eqns_labs_tmp[0].lstrip('ï»¿')
#
eqns_labs = []
for eqn_lab in eqns_labs_tmp:
    eqns_labs.append(eqn_lab.strip("\n"))
eqns_labs = eqns_labs[:]
del eqns_labs_tmp
# cutoff
eqns_labs = eqns_labs[:N_examples]

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
#eqns_enc = arXivDocs2Vec.docs2vec(eqns_cont,eqns_tex)[1]
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
#eqns_enc = arXivDocs2Vec.docs2vec(eqns_qids,eqns_tex)[1]
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

accuracy = numpy.mean(cross_val_score(classifier, X, y, cv=N_classes))

print('Classification accuracy: ' + str(accuracy))

# CLUSTERING

n_clusters = N_classes
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
plt.text(0.2, -0.6, 'Mean classification accuracy: ' + str(round(accuracy,2)))
plt.text(0.2, -0.4, 'Mean cluster purity: ' + str(round(purity,2)))
# if enc_str == 'cont_tfidf':
#     text += '0.50'
#     x_pos, y_pos = 0.2, -0.3
# elif enc_str == 'cont_d2v':
#     text += '0.97'
#     x_pos, y_pos = 0.2, -0.3
# elif enc_str == 'sem_tfidf':
#     text += '0.97'
#     x_pos, y_pos = 0.2, -0.3
# elif enc_str == 'sem_d2v':
#     text += '0.97'
#     x_pos, y_pos = 0.2, -0.3
#plt.text(x_pos, y_pos, text)

plt.xlabel("PCA dimension 1")
plt.ylabel("PCA dimension 2")
plt.title(title_text)

ax.scatter(vectors_red[:,0], vectors_red[:,1], c=labels, cmap='rainbow')
#ax.legend()

#plt.show()

print("end")