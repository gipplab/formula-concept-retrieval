import sklearn.metrics.pairwise

import arXivDocs2tfidf
import arXivDocs2Vec

import random
from fuzzywuzzy import fuzz

import numpy as np
import skimage.measure

import matplotlib.pyplot as plt
import seaborn as sns

# PARAMETERS

# set paths
example_nr_prefix = ['two','three','ten'][2]
example_path = example_nr_prefix + " examples/"
full_path = "diff_eqns_examples/" + example_path

# set file names
tex_file = full_path + "diff_eqns_tex.txt"
content_file = full_path + "diff_eqns_content.txt"
qids_file = full_path + "diff_eqns_qids.txt"
labels_file = full_path + "diff_eqns_labels.txt"

# parameters
#N_examples_included
N_classes = 10
N_examples_per_class = 10
N_examples = N_classes*N_examples_per_class
enc_str = ['cont_tfidf','cont_d2v','cont_fuzz','sem_tfidf','sem_d2v','sem_fuzz'][0]

# LOADING

# get equations tex

with open(tex_file,'r') as f:
	eqns_tex = f.readlines()
# clean
eqns_tex[0] = eqns_tex[0].lstrip('ï»¿')
# cutoff
#eqns_tex = eqns_tex[:N_examples]

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
#eqns_cont = eqns_cont[:N_examples]

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
#eqns_qids = eqns_qids[:N_examples]

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
#eqns_labs = eqns_labs[:N_examples]

# ENCODING

# get encoding
#
# content
if enc_str == 'cont_tfidf':
	eqns_enc = arXivDocs2tfidf.docs2tfidf(eqns_cont)
	title_text = "Formula content space (TF-IDF)"
elif enc_str == 'cont_d2v':
	eqns_enc = arXivDocs2Vec.docs2vec(eqns_cont,eqns_tex)[1]
	title_text = "Formula content space (Doc2Vec)"
elif enc_str == 'cont_fuzz':
	eqns_enc = eqns_tex
	title_text = "Formula content space (Fuzzy)"
#
# semantics
elif enc_str == 'sem_tfidf':
	eqns_enc = arXivDocs2tfidf.docs2tfidf(eqns_qids)
	title_text = "Formula semantic space (TF-IDF)"
elif enc_str == 'sem_d2v':
	eqns_enc = arXivDocs2Vec.docs2vec(eqns_qids,eqns_tex)[1]
	title_text = "Formula semantic space (Doc2Vec)"
elif enc_str == 'sem_fuzz':
	eqns_enc = eqns_qids
	title_text = "Formula semantic space (Fuzzy)"

# SIMILARITY

# Formula similarity matrices

# 1) Cell for EACH EQUATION (full matrix)

# equation similarity function
def get_eqn_similarity(i,j):

	# vector product similarity
	if enc_str in ['cont_tfidf','sem_tfidf']:
		similarity = float(sklearn.metrics.pairwise.cosine_similarity(
				eqns_enc[i], eqns_enc[j]))
	elif enc_str in ['cont_d2v','sem_d2v']:
		similarity = float(sklearn.metrics.pairwise.cosine_similarity(
			eqns_enc[i].reshape(1,-1),eqns_enc[j].reshape(1,-1)))
	# fuzzy string similarity
	elif enc_str in ['cont_fuzz','sem_fuzz']:
		similarity = fuzz.partial_ratio(eqns_enc[i],eqns_enc[j])/100

	# detect very similar eqns
	if similarity > 0.8 and similarity < 1.0:
		print(eqns_labs[i])
		print(eqns_labs[j])
		print(str(similarity))

	return similarity

# get similarities to calculate separability
dim = N_examples
equation_similarities = np.zeros((dim,dim))
for i in range(0,dim):
	for j in range(0,dim):
		equation_similarities[i][j] = get_eqn_similarity(i,j)

# 2) Cell for EACH CLASS (sum matrix)

# class similarity function
def get_cls_similarity(i,j):

	# get subset of equation similarity matrix
	dim = N_examples_per_class
	equation_similarities_subset = equation_similarities[i*dim:(i+1)*dim,j*dim:(j+1)*dim]

	# mean pooling for overall class similarity
	similarity = float(skimage.measure.block_reduce(image=equation_similarities_subset,
											  block_size=10,
											  func=np.mean))

	return similarity

# get similarities to calculate separability
dim = N_classes
class_similarities = np.zeros((dim,dim))
for i in range(0,dim):
	for j in range(0,dim):
		class_similarities[i][j] = get_cls_similarity(i,j)

# PLOT

show_matrix = equation_similarities
plt.figure()
# plt.matshow(show_matrix)
# plt.colorbar()
plt.title(title_text + ' - labeled selection')
#plt.title(title_text + ' - formula similarities')
ax_labels = []
for i in range(N_examples):
	if i in [(j+1/2)*N_examples_per_class for j in range(N_classes)]:
		ax_labels.append(eqns_labs[i])
	else:
		ax_labels.append('')
sns.heatmap(show_matrix,
			xticklabels=ax_labels,yticklabels=ax_labels,
			vmin=0, vmax=1, cmap='coolwarm')

show_matrix = class_similarities
plt.figure()
# plt.matshow(show_matrix)
# plt.colorbar()
plt.title(title_text + ' - class similarities')
ax_labels = [eqns_labs[i*N_classes] for i in range(N_classes)]
sns.heatmap(show_matrix,
			xticklabels=ax_labels,yticklabels=ax_labels,
			vmin=0, vmax=1, cmap='coolwarm')

plt.show()

print("end")