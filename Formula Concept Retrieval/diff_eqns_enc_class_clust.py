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

import numpy as np

from collections import Counter
import itertools
import random

import sys
import json

# set paths
example_nr_prefix = ['three','ten'][1]
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
enc_str = ['cont_tfidf','cont_d2v','sem_tfidf','sem_d2v'][3]

# save protocol
protocol_path = full_path + 'results/' + enc_str + '_' + str(N_classes) + '.txt'
sys.stdout = open(protocol_path,'w')

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

# SELECTION

# default unshuffled
chosen_classes = list(range(N_classes))

# shuffled
# max possibilities:
# scipy.special.binom(n,k) = scipy.special.binom(10,5) = 252.0
# tot possibilities: sum(1,10,binom(n,k)) = 1275
#chosen_classes = random.sample(range(int(len(eqns_labs)/N_examples_per_class)),N_classes)
#chosen_classes = [7, 0, 9] # full performance

# combinations
potential_choices = list(itertools.combinations(sorted((range(10))),N_classes))
nr_choices = len(potential_choices)
# executed after main function definition

# DEF MAIN FUNCTION

def get_class_clust_plot(chosen_classes,
                         eqns_labs,eqns_tex,eqns_cont,eqns_qids,
                         toggle_plot):

    # SORTING

    # resort lists

    # init tmp
    eqns_labs_tmp = []
    eqns_tex_tmp = []
    eqns_cont_tmp = []
    eqns_qids_tmp = []
    # fill tmp
    for class_idx_init in chosen_classes:
        class_idx_init = class_idx_init*N_examples_per_class
        eqns_labs_tmp.extend(eqns_labs[class_idx_init:class_idx_init+N_examples_per_class])
        eqns_tex_tmp.extend(eqns_tex[class_idx_init:class_idx_init+N_examples_per_class])
        eqns_cont_tmp.extend(eqns_cont[class_idx_init:class_idx_init+N_examples_per_class])
        eqns_qids_tmp.extend(eqns_qids[class_idx_init:class_idx_init+N_examples_per_class])
    # final tmp
    eqns_labs = eqns_labs_tmp
    del eqns_labs_tmp
    eqns_tex = eqns_tex_tmp
    del eqns_tex_tmp
    eqns_cont = eqns_cont_tmp
    del eqns_cont_tmp
    eqns_qids = eqns_qids_tmp
    del eqns_qids_tmp
    print('Chosen: ' + str(set(eqns_labs)))

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
    #
    # semantics
    elif enc_str == 'sem_tfidf':
        eqns_enc = arXivDocs2tfidf.docs2tfidf(eqns_qids)
        title_text = "Formula semantic space (TF-IDF)"
    elif enc_str == 'sem_d2v':
        eqns_enc = arXivDocs2Vec.docs2vec(eqns_qids,eqns_tex)[1]
        title_text = "Formula semantic space (Doc2Vec)"

    # set vectors
    X,y = eqns_enc,eqns_labs

    # CLASSIFICATION

    classifier = LinearSVC()#LogisticRegression()
    classifier.fit(X,y)
    #prediction = list(classifier.predict(X))

    # matches = 0
    # total = len(prediction)
    # for i in range(0,total):
    #     if prediction[i] == eqns_labs[i]:
    #         matches += 1
    # accuracy = matches/total

    accuracy = np.mean(cross_val_score(classifier, X, y, cv=N_classes))

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
    purity = np.mean(purities)

    print('Cluster purity: ' + str(purity))

    # PLOT
    if toggle_plot == True:

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

        #red = PCA(n_components=2)
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
                    c_dists.append(np.linalg.norm(c_vecs[i]-c_vecs[j]))
        c_mean_dist = np.mean(c_dists)

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

        # PLOT
        # reduced vectors

        fig = plt.figure()
        ax = plt.axes()

        # Plot a line from slope and intercept
        # def plot_line(intercept, slope):
        #     axes = plt.gca()
        #     x_vals = np.array(axes.get_xlim())
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

        plt.show()

    return accuracy,purity

# RUN MAIN FUNCTION
accuracies = []
purities = []
choice_nr = 1
for chosen_classes in potential_choices:

    print('Combination Nr. ' + str(choice_nr))

    accuracy,purity =\
        get_class_clust_plot(chosen_classes,
                         eqns_labs, eqns_tex, eqns_cont, eqns_qids,
                         toggle_plot=False)
    accuracies.append(accuracy)
    purities.append(purity)

    print()
    choice_nr += 1

mean_accuracy = np.mean(accuracies)
mean_purity = np.mean(purities)
print('Mean classification accuracy: ' + str(mean_accuracy))
print('Mean cluster purity: ' + str(mean_purity))
print()

# REPORT

# save results to eval dict
eval_dict_path = full_path + 'results/' + 'eval_dict.json'
# load
with open(eval_dict_path,'r') as f:
    eval_dict = json.load(f)
# edit
eval_entry = {'nr_choices': nr_choices,
                'mean_accuracy': mean_accuracy, 'mean_purity': mean_purity}
try:
    eval_dict[enc_str][N_classes] = eval_entry
except:
    eval_dict[enc_str] = {N_classes: eval_entry}
# save
with open(eval_dict_path,'w') as f:
    json.dump(eval_dict,f)

print("end")