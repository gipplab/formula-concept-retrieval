import random
from fuzzywuzzy import fuzz

import numpy as np
import matplotlib.pyplot as plt

# define file names

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

# Formula similarity matrix

dim = len(eqns_labs)

matchings = np.zeros((dim,dim))

# random assignments
# assignments = []
# for i in range(0,30):
#     assignments.append(random.randrange(0, 3, 1))

# fuzzy string matching for LaTeX strings (formula content)
# get concept assignments for each formula to calculate separability
assignments = []
for i in range(0,dim):
    for j in range(0,dim):
        matchings[i][j] = fuzz.partial_ratio(eqns_qids[i],eqns_qids[j])
    # to which concept (KGE: 0, EFE: 1, ME: 2) is the formula assigned?
    sums = []
    sums.append(sum(matchings[i][0:10]))
    sums.append(sum(matchings[i][10:20]))
    sums.append(sum(matchings[i][20:30]))
    assignments.append(sums.index(max(sums)))

# calculate Formula Concept similarity matrix
# get concept assignments for each formula to calculate separability
# assignments = []
# for i in range(0,dim):
#     for j in range(0,dim):
#         matchings[i][j] = len(set(eqns_qids[i].split()).intersection(set(eqns_qids[j].split())))
#     # to which concept (KGE: 0, EFE: 1, ME: 2) is the formula assigned?
#     sums = []
#     sums.append(sum(matchings[i][0:10]))
#     sums.append(sum(matchings[i][10:20]))
#     sums.append(sum(matchings[i][20:30]))
#     assignments.append(sums.index(max(sums)))

# PLOT

plt.imshow(matchings)
plt.colorbar()

ax = plt.axes()
# annotate class squares
ax.annotate("KGE",(3,4),fontsize=10,weight='bold')
ax.annotate("EFE",(12,14),fontsize=10,weight='bold')
ax.annotate("ME",(21,22.25),fontsize=10,weight='bold')

plt.show()

print("end")