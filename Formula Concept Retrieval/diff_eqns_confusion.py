import arXivDocs2tfidf
import arXivDocs2Vec

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# PARAMETERS

# set paths
example_nr_prefix = ['two','three','ten'][1]
example_path = example_nr_prefix + " examples/"
full_path = "diff_eqns_examples/" + example_path

# set file names
tex_file = full_path + "diff_eqns_tex.txt"
content_file = full_path + "diff_eqns_content.txt"
qids_file = full_path + "diff_eqns_qids.txt"
labels_file = full_path + "diff_eqns_labels.txt"

# parameters
#N_examples_included
N_classes = 3
N_examples_per_class = 10
N_examples = N_classes*N_examples_per_class
enc_str = ['cont_tfidf','cont_d2v','sem_tfidf','sem_d2v'][2]

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
#
# semantics
elif enc_str == 'sem_tfidf':
    eqns_enc = arXivDocs2tfidf.docs2tfidf(eqns_qids)
    title_text = "Formula semantic space (TF-IDF)"
elif enc_str == 'sem_d2v':
    eqns_enc = arXivDocs2Vec.docs2vec(eqns_qids,eqns_tex)[1]
    title_text = "Formula semantic space (Doc2Vec)"

# CLASSIFICATION

# set vectors
X,y = eqns_enc,eqns_labs

# classifier
# fit
classifier = LinearSVC()#LogisticRegression()
classifier.fit(X,y)
# predict
eqns_pred = classifier.predict(X)
y_true = eqns_labs
y_pred = eqns_pred
# confusion
confusion_matrix = confusion_matrix(y_true,y_pred)

# PLOT

plt.matshow(confusion_matrix)
plt.colorbar()
plt.title('Classification confusion matrix (N = '
          + str(N_classes) + ')')
#sns.heatmap(show_matrix)

# annotate class squares
#ax = plt.axes()
#ax.annotate("KGE",(3,4),fontsize=10,weight='bold')
#ax.annotate("EFE",(12,14),fontsize=10,weight='bold')
#ax.annotate("ME",(21,22.25),fontsize=10,weight='bold')

plt.show()

print("end")