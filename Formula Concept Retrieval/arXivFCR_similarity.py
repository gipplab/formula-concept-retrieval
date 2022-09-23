import pickle
import sklearn.metrics.pairwise

from fuzzywuzzy import fuzz

import numpy as np
import skimage.measure

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.neighbors._nearest_centroid import NearestCentroid
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

# PARAMETERS

# N_examples_included
N_classes = 100
N_examples_per_class = 1
N_examples = N_classes * N_examples_per_class
enc_str = ['cont_tfidf', 'cont_d2v', 'cont_fuzz'][0]

# LOADING

input_path = 'E:\\NTCIR-12_MathIR_arXiv_Corpus/formulae/clustering/'
arXiv_class = 'astro-ph'

with open(input_path + 'formulae_tfidf(' + arXiv_class + ').pkl', 'rb') as f:
	formulae_tfidf = pickle.load(f)
with open(input_path + 'formulae_doc2vec(' + arXiv_class + ').pkl', 'rb') as f:
	formulae_d2v = pickle.load(f)
with open(input_path + 'formulaLabs(' + arXiv_class + ').pkl', 'rb') as f:
	formulae_labs = pickle.load(f)


# SIMILARITY

# Formula similarity matrices

# 1) Cell for EACH EQUATION (full matrix)

# equation similarity function
def get_eqn_similarity(i, j):
	# vector product similarity
	if enc_str in ['cont_tfidf']:
		similarity = float(sklearn.metrics.pairwise.cosine_similarity(
			formulae_tfidf[i], formulae_tfidf[j]))
	elif enc_str in ['cont_d2v']:
		similarity = float(sklearn.metrics.pairwise.cosine_similarity(
			formulae_d2v[i].reshape(1, -1), formulae_d2v[j].reshape(1, -1)))
	# fuzzy string similarity
	elif enc_str in ['cont_fuzz']:
		similarity = fuzz.partial_ratio(formulae_labs[i], formulae_labs[j]) / 100

	if similarity > 0.8 and similarity < 1.0:
		print(formulae_labs[i])
		print(formulae_labs[j])
		print(str(similarity))

	return similarity


# get similarities to calculate separability
dim = N_examples
equation_similarities = np.zeros((dim, dim))
for i in range(0, dim):
	for j in range(0, dim):
		equation_similarities[i][j] = get_eqn_similarity(i, j)


# 2) Cell for EACH CLASS (sum matrix)

# class similarity function
def get_cls_similarity(i, j):
	# get subset of equation similarity matrix
	dim = N_examples_per_class
	equation_similarities_subset = equation_similarities[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim]

	# mean pooling for overall class similarity
	similarity = float(skimage.measure.block_reduce(image=equation_similarities_subset,
													block_size=1,
													func=np.mean))

	return similarity


# # get similarities to calculate separability
# dim = N_classes
# class_similarities = np.zeros((dim,dim))
# for i in range(0,dim):
#     for j in range(0,dim):
#         class_similarities[i][j] = get_cls_similarity(i,j)

# PLOT

title_text = {'cont_tfidf': "Formula content space (TF-IDF)",
			  'cont_d2v': "Formula content space (Doc2Vec)",
			  'cont_fuzz': "Formula content space (Fuzzy)"}[enc_str]

show_matrix = equation_similarities
plt.figure()
# plt.matshow(show_matrix)
# plt.colorbar()
plt.title(title_text + ' - random unlabeled')
# plt.title(title_text + ' - Equation similarities')
ax_labels = []
for i in range(N_examples):
	if i in [(j + 1 / 2) * N_examples_per_class for j in range(N_classes)]:
		ax_labels.append(formulae_labs[i])
	else:
		ax_labels.append('')
sns.heatmap(show_matrix,
			# xticklabels=ax_labels,
			# yticklabels=ax_labels,
			vmin=0, vmax=1, cmap='coolwarm')

# show_matrix = class_similarities
# plt.figure()
# # plt.matshow(show_matrix)
# # plt.colorbar()
# plt.title(title_text + ' - Class similarities')
# ax_labels = [formulae_labs[i*N_classes] for i in range(N_classes)]
# sns.heatmap(show_matrix,
#             #xticklabels=ax_labels,
#             #yticklabels=ax_labels,
#             vmin=0, vmax=1, cmap='coolwarm')

plt.show()

# # CLUSTERING
#
# X = formulae_tfidf
# #X = formulae_d2v
# y = formulae_labs
#
# clusterer = KMeans(n_clusters=int(N_classes/10))
#
# clustering = clusterer.fit_predict(X)
#
# # calculate cluster purity
# clusters = list(clustering)
#
# # calculate purities
# purities = []
#
# # three examples
# # ranges = [(0,10),(10,20),(20,30)]
# # ten examples
# ranges = []
# for N in range(N_classes):
#     ranges.append((N * N_examples_per_class, (N + 1) * N_examples_per_class))
#
# # get purity
# for rangee in ranges:
#     purities.append(max(Counter(clusters[rangee[0]:rangee[1]]).values()) / N_examples_per_class)
# purity = np.mean(purities)
#
# print('Cluster purity: ' + str(purity))
#
# # DIMENSIONALITY REDUCTION
#
# # labels = LabelBinarizer.fit_transform(prediction)
# labels = list(clustering)
#
# # red = PCA(n_components=2)
# red = TruncatedSVD(n_components=2) # 3 in 3D
# vectors_red = red.fit_transform(X)
#
# # calculate cluster centroid distances in reduced space
# centroids = NearestCentroid()
# centroids.fit(vectors_red, clustering)
# c_vecs = []
# for centroid in centroids.centroids_:
#     c_vecs.append(centroid)
#
# c_dists = []
# for i in [0, 1, 2]:
#     for j in [0, 1, 2]:
#         if i != j:
#             c_dists.append(np.linalg.norm(c_vecs[i] - c_vecs[j]))
# c_mean_dist = np.mean(c_dists)
#
# # PLOT
# # reduced vectors
#
# fig = plt.figure()
# # 2D
# ax = plt.axes()
# # 3D
# # ax = fig.add_subplot(projection='3d')
#
# # label
# for i, lab in enumerate(formulae_labs):
#     ax.annotate(i, (vectors_red[i, 0], vectors_red[i, 1])) # lab instead of i
#
# plt.xlabel("PCA dimension 1")
# plt.ylabel("PCA dimension 2")
# plt.title(title_text)
#
# # 2D
# ax.scatter(vectors_red[:, 0], vectors_red[:, 1],
#            c=labels, cmap='rainbow')
# # 3D
# #ax.scatter(vectors_red[:, 0], vectors_red[:, 1], vectors_red[:, 2],
# #           c=labels, cmap='rainbow')
# # ax.legend()
#
# plt.show()

print("end")
