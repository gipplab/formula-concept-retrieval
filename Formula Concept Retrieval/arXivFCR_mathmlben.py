import pickle
import operator
from collections import Counter

from FormulaRetrieval import getFormulae

from arXivDocs2Vec import docs2vec
from arXivDocs2tfidf import docs2tfidf

from fuzzywuzzy import fuzz

import numpy as np
from sklearn.neighbors import NearestNeighbors

# set file paths
#basePath = ""
#basePath = "F:\\SigMathLing\\"
basePath = "F:\\arXiv\\"
inputPath = basePath + "formulae\\mathmlben\\"
outputPath = basePath + "formulae\\mathmlben\\"

# load mmlben arXiv matches results
# with open(outputPath + "results_mmlben_arXiv.pkl", 'rb') as f:
#     matches = pickle.load(f)
with open(outputPath + "matches_EFE_arXiv.pkl", 'rb') as f:
    matches = pickle.load(f)

for formula in matches.items():
    print(formula[0])

# load formula catalog mmlben
# with open(outputPath + "formula_catalog_mmlben.pkl",'rb') as f:
#     formula_catalog_mmlben = pickle.load(f)
# load formula catalog arXiv all
with open(outputPath + "formula_catalog_all.pkl",'rb') as f:
    formula_catalog_all = pickle.load(f)

# formula catalog arXiv
# create and save
# with open(outputPath + "formula_catalog_backup.pkl",'rb') as f:
#     formula_catalog_backup = pickle.load(f)
# del formula_catalog_backup['operator_catalog']
# del formula_catalog_backup['identifier_catalog']
# formula_catalog_arXiv = {}
# for formula in formula_catalog_backup.items():
#     try:
#         print(formula[1]['TeX'])
#         formula_catalog_arXiv[formula[1]['TeX']] = {}
#         formula_catalog_arXiv[formula[1]['TeX']]['operators'] = []
#         formula_catalog_arXiv[formula[1]['TeX']]['identifiers'] = []
#         for operator in formula[1]['operators'].values():
#             formula_catalog_arXiv[formula[1]['TeX']]['operators'].append(operator)
#         for identifier in formula[1]['identifiers'].values():
#             formula_catalog_arXiv[formula[1]['TeX']]['identifiers'].append(identifier)
#     except:
#         pass
# with open(outputPath + "formula_catalog_arXiv.pkl",'wb') as f:
#     pickle.dump(formula_catalog_arXiv,f)
# load
# with open(outputPath + "formula_catalog_arXiv.pkl",'rb') as f:
#     formula_catalog_arXiv = pickle.load(f)

# formula catalog mmlben
# create and save
# with open(outputPath + "mathmlben.html", 'r', encoding="utf8") as f:
#     formula_catalog_mmlben = getFormulae(doc_str=f.read(),mode='math')
# with open(outputPath + "formula_catalog_mmlben.pkl",'wb') as f:
# #     pickle.dump(formula_catalog_mmlben,f)
# load
# with open(outputPath + "formula_catalog_mmlben.pkl",'rb') as f:
#     formula_catalog_mmlben = pickle.load(f)

# create formula labs and data list
#formulaLabs = []
#formulaData = []
# bring formula catalog operators and identifiers in data array
# def catalog_to_data(formula_catalog):
#     for formula in formula_catalog.items():
#         # data needs to be a string per formula
#         Data = ""
#         for operator in formula[1]['operators']:
#             Data += operator + " "
#         for identifier in formula[1]['identifiers']:
#             Data += identifier + " "
#         # cut off last whitespace at the end
#         formulaData.append(Data[:-1])
#         # tex string as label
#         formulaLabs.append(formula[0])
# insert formula catalog mmlben
#catalog_to_data(formula_catalog_mmlben)
# insert formula catalog arXiv
#catalog_to_data(formula_catalog_arXiv)

# save data and labels
# with open(outputPath + "formulaData.pkl","wb") as f:
#     pickle.dump(formulaData,f)
# with open(outputPath + "formulaLabs.pkl","wb") as f:
#     pickle.dump(formulaLabs,f)

# build and save doc2Vec vectors
# model,formulaVecs_doc2vec = docs2vec(formulaData,formulaLabs)
# with open(outputPath + "formulaVecs_doc2vec.pkl","wb") as f:
#     pickle.dump(formulaVecs_doc2vec,f)
# # build and save tf-idf vectors
# formulaVecs_tfidf = docs2tfidf(formulaData)
# with open(outputPath + "formulaVecs_tfidf.pkl","wb") as f:
#     pickle.dump(formulaVecs_tfidf,f)

# load data and labels
# with open(outputPath + "formulaData.pkl","rb") as f:
#     formulaData = pickle.load(f)
# with open(outputPath + "formulaLabs.pkl","rb") as f:
#     formulaLabs = pickle.load(f)
# load doc2vec and tf-idf vectors
# with open(outputPath + "formulaVecs_doc2vec.pkl","rb") as f:
#     formulaVecs_doc2vec = pickle.load(f)
# with open(outputPath + "formulaVecs_tfidf.pkl","rb") as f:
#     formulaVecs_tfidf = pickle.load(f)

# load FCR candidates
# with open(outputPath + "results_fuzzy.pkl","rb") as f:
#     results_fuzzy = pickle.load(f)
# with open(outputPath + "results_seq.pkl","rb") as f:
#     results_seq = pickle.load(f)
# with open(outputPath + "results_set.pkl","rb") as f:
#     results_set = pickle.load(f)
# with open(outputPath + "results_50.pkl","rb") as f:
#     results_50 = pickle.load(f)

# mmlben formulaLab nr.s neq GoldIDs
# selection = [24,219,275,285,294,295,300,301,302,303]
# formula_catalog_selection = {}
# for nr in selection:
#     formula_catalog_selection[formulaLabs[nr]] = {}
#     formula_catalog_selection[formulaLabs[nr]]['operators'] = set(formula_catalog_mmlben[formulaLabs[nr]]['operators'])
#     formula_catalog_selection[formulaLabs[nr]]['identifiers'] = set(formula_catalog_mmlben[formulaLabs[nr]]['identifiers'])
#selection = list(range(0,302))

# without encoding

# tex
# fuzzy formula tex string search
# results_fuzzy = {}
# for nr in selection:
#     formulaQuery = formulaLabs[nr]
#     #formulaQuery = "'\\frac {1}{c^2} \\frac{\\partial^2}{\\partial t^2} \\psi - \\nabla^2 \\psi + \\frac {m^2 c^2}{\\hbar^2} \\psi = 0'"
#     results = {}
#     i = 0
#     for formulaLab in formulaLabs:
#         print(str(round(np.multiply(np.divide(i,len(formulaLabs)),100),1)) + " %")
#         try:
#             fpr = fuzz.partial_ratio(formulaQuery,formulaLab)
#             if fpr > 80 and len(formulaLabs) > 3:
#                 results[formulaLab] = fpr
#         except:
#             pass
#         i += 1
#     results_fuzzy[formulaLabs[nr]] = sorted(results.items(),key=operator.itemgetter(1),reverse=True)

# mml
# fuzzy formula parts seq string search
# results_seq = {}
# for nr in selection:
#     formulaQuery = formulaLabs[nr]
#     #formulaQuery = "'\\frac {1}{c^2} \\frac{\\partial^2}{\\partial t^2} \\psi - \\nabla^2 \\psi + \\frac {m^2 c^2}{\\hbar^2} \\psi = 0'"
#     results = {}
#     i = 0
#     for formulaLab in formulaLabs:
#         print(str(round(np.multiply(np.divide(i,len(formulaLabs)),100),1)) + " %")
#         try:
#             fpr = fuzz.partial_ratio(formulaQuery,formulaData[i])
#             if fpr > 80 and len(formulaLabs) > 3:
#                 results[formulaLab] = fpr
#         except:
#             pass
#         i += 1
#     results_seq[formulaLabs[nr]] = sorted(results.items(),key=operator.itemgetter(1),reverse=True)

# search for mathmlben formulae by parts (op,id) set
#results_set = {}
# #for nr in selection:
# for nr in range(0,50):
#     print(nr)
#     formulaQuery = set(formulaData[nr].split())
#     #formulaQuery = set(['∂','ψ','t','V','∇','+','>','m','c','ℏ'])
#     #formulaQuery = set(['G','μ','ν','T','κ','π','c'])
#     results = {}
#     for i in range(0,len(formulaData)):
#         results[formulaLabs[i]] = len(formulaQuery.intersection(set(formulaData[i].split())))
#     results_set[formulaLabs[nr]] = sorted(results.items(), key=operator.itemgetter(1),reverse=True)

# FCR here

# results_set = {}
# for formula_query in formula_catalog_selection.items():
#     print("Query: " + formula_query[0])
#     results = {}
#     for formula_index in formula_catalog_all.items():
#         try:
#             overlap = len(formula_query[1]['identifiers'].intersection(formula_index[1]['identifiers']))# + len(formula_query[1]['operators'].intersection(formula_index[1]['operators']))
#             if overlap > 0:
#                 results[formula_index[0]] = overlap
#                 print(formula_index[0] + ": " + str(overlap))
#         except:
#             pass
#     results_set[formula_query[0]] = sorted(results.items(), key=operator.itemgetter(1),reverse=True)

# with encoding
# set encoding
#encoding = formulaVecs_doc2vec
#encoding = formulaVecs_tfidf.toarray()

# cosine similarity neighbors
# def cos_mat_mult (Vec, Mat):
#     Sim = []
#     for Vec2 in Mat:
#         Sim.append(float(np.inner(Vec, Vec2)))
#     return Sim

# euclidean distance neighbors
# def eucl_dist_vec (Vec, Mat):
#     Sim = []
#     for Vec2 in Mat:
#         Sim.append(float(np.linalg.norm(Vec-Vec2)))
#     return Sim

# search for mmlben formulae
# search_id = 1
# search_results = {}
# calculate cosine similarity with all vectors
#Sim = cos_mat_mult(encoding[idx],encoding)
# calculate euclidean distance to all vectors
#Sim = eucl_dist_vec(encoding[search_id],encoding)
# sort and label results
# for i in range(0,len(Sim)):
#     search_results[formulaLabs[i]] = Sim[i]
# search_results = sorted(search_results.items(),key=operator.itemgetter(1),reverse=True)

# k nearest neighbors
# nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(encoding)
# distances, indices = nbrs.kneighbors(encoding)
#
# with open(outputPath + "kNN_indices.pkl",'wb') as f:
#     pickle.dump(indices,f)

#Query equation by matching identifers

# Klein-Gordon
#formulaQuery = set(['∂','ψ','t','V','∇','+','>','m','c','ℏ'])
# Schrödinger
#formulaQuery = set(['ψ','t','m','c','ℏ'])
# Einstein
formulaQuery = set(['G', 'R', 'T', 'g', 'μ', 'ν', 'Λ', 'c', 'κ'])

matches = {}
arXiv_counter = 0
for formula_index in formula_catalog_all.items():
    arXiv_counter += 1
    print(arXiv_counter)
    # nr_identifiers = len(formula_index[1]['identifiers'])
    # same_identifiers = len(formula_index[1]['identifiers'].difference(formulaQuery)) == 0
    overlap = len(formulaQuery.intersection(formula_index[1]['identifiers']))
    if overlap > 5:
        print(formula_index[0] + ": " + str(formula_index[1]))
        matches[formula_index[0]] = str(formula_index[1])

with open(outputPath + "matches_EFE_arXiv.pkl", 'wb') as f:
    pickle.dump(matches, f)

# Query mathmlben formulae
# matches = {}
# mmlben_counter = 0
# arXiv_counter = 0
# for formula_query in formula_catalog_mmlben.items():
#     mmlben_counter += 1
#     print(formula_query[0] + ": " + str(formula_query[1]))
#     for formula_index in formula_catalog_all.items():
#         arXiv_counter += 1
#         print(str(mmlben_counter) + ": " + arXiv_counter)
#         nr_operators = len(formula_index[1]['operators'])
#         nr_identifiers = len(formula_index[1]['identifiers'])
#         same_operators = len(formula_index[1]['operators'].difference(formula_query[1]['operators'])) == 0
#         same_identifiers = len(formula_index[1]['identifiers'].difference(formula_query[1]['identifiers'])) == 0
#         if same_operators and same_identifiers and nr_operators > 0 and nr_identifiers > 1:
#             print(formula_index[0] + ": " + str(formula_index[1]))
#             matches[formula_index[0]] = str(formula_index[1])

# with open(outputPath + "results_id_mmlben_arXiv.pkl",'wb') as f:
#      pickle.dump(results_set,f)

print("end")