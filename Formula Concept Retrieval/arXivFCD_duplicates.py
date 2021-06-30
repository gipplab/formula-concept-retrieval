import operator
from collections import Counter

import pickle
import csv

from sklearn.neighbors import NearestNeighbors

# Set file paths
#basePath = ""
#basePath = "F:\\SigMathLing_arXMLiv-08-2018\\"
basePath = "F:\\NTCIR-12_MathIR_arXiv_Corpus\\"

subject_class = "hep-th"
# astro-ph
# gr-qc
# hep-th

inputPath = basePath + "formulae\\duplicates\\" + subject_class + "\\"
outputPath = basePath + "formulae\\duplicates\\" + subject_class + "\\"

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

# SET ENCODING
encoding = formulae_tfidf # formulae_doc2vec, formulae_tfidf, formulae_semantics, formulae_semantics_tfidf

# FIND K NEAREST NEIGHBORS
nbrs = NearestNeighbors(n_neighbors=13, algorithm='auto').fit(encoding)
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
        formulaDuplicates_docs[formulaLab].append(formulaDocs[i])
    except:
        formulaDuplicates_docs[formulaLab] = []
        formulaDuplicates_docs[formulaLab].append(formulaDocs[i])

    try:
        for word in surrTextData[i]:
            formulaDuplicates_text[formulaLab].append(word)
    except:
        formulaDuplicates_text[formulaLab] = surrTextData[i]

    for j in range(1, 11):
        try:
            formulaDuplicates_kNN[formulaLab].add(formulaLabs[indices[i,j]])
        except:
            formulaDuplicates_kNN[formulaLab] = set()
            #formulaDuplicates_kNN[formulaLab].add(formulaLabs[indices[i,j]])

    i += 1

# sort counts descending
formulaDuplicates_count = sorted(formulaDuplicates_count.items(), key=operator.itemgetter(1),reverse=True)
# count unique docs
formulaDuplicates_uniqdocs = [(tuple[0],len(set(formulaDuplicates_docs[tuple[0]]))) for tuple in formulaDuplicates_count]
# sort text as counts
formulaDuplicates_text = [(tuple[0],formulaDuplicates_text[tuple[0]]) for tuple in formulaDuplicates_count]
# sort kNN as counts
formulaDuplicates_kNN = [(tuple[0],formulaDuplicates_kNN[tuple[0]]) for tuple in formulaDuplicates_count]

# (sort doc names as counts)
formulaDuplicates_docs = [(tuple[0],formulaDuplicates_docs[tuple[0]]) for tuple in formulaDuplicates_count]

# FINAL RESULTS AS 5-TUPLE (formulaLab, duplicateCount, uniqueDocs, surrText, kNN)
formulaDuplicates = []
for i in range(0,len(formulaDuplicates_count)):
    # filter
    if len(formulaDuplicates_count[i][0]) < 30 and formulaDuplicates_count[i][1] > 1 and formulaDuplicates_uniqdocs[i][1] > 1:
        formulaDuplicates.append((formulaDuplicates_count[i][0],formulaDuplicates_count[i][1],formulaDuplicates_uniqdocs[i][1],Counter(formulaDuplicates_text[i][1]),Counter(formulaDuplicates_kNN[i][1])))

# save as table
# with open(outputPath + 'formulaeFCD.csv', 'w') as f:
#     fieldnames = ['Formula', 'Dupl', 'Docs']
#     writer = csv.DictWriter(f, fieldnames=fieldnames,delimiter=',',lineterminator='\n')
#     writer.writeheader()
#     for element in formulaDuplicates:
#         writer.writerow({'Formula': element[0].replace(',',''), 'Dupl': element[1], 'Docs': element[2]})

# kNN_candidates_doc2vec
# kNN_candidates_tfidf
# kNN_candidates_semantics_doc2vec
# kNN_candidates_semantics_tfidf

with open(outputPath + "kNN_candidates_doc2vec.pkl","wb") as f:
	pickle.dump(formulaDuplicates_kNN,f)

print("end")