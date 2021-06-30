import pickle

from collections import Counter
import operator

# Set file paths
#basePath = ""
#basePath = "F:\\SigMathLing\\"
basePath = "F:\\NTCIR-12_MathIR_arXiv_Corpus\\"

formulaPath = "formulae\\duplicates\\"
subjectClass = "hep-th\\"

inputPath = basePath + formulaPath + subjectClass
outputPath = basePath + formulaPath + subjectClass

# Load formula labels and math vectors

with open(outputPath + "formulaLabs.pkl",'rb') as f:
    formulaLabs = pickle.load(f)
# with open(outputPath + "formulae_word2vec.pkl",'rb') as f:
#     formulae_word2vec = pickle.load(f)
# with open(outputPath + "formulae_tfidf.pkl",'rb') as f:
#     formulae_tfidf = pickle.load(f)

# find formula duplicates
duplicates = {}
for item, count in Counter(formulaLabs).items():
    if count > 1:
        duplicates[item] = count

# sort by decreasing counts
duplicates_sorted = sorted(duplicates.items(),key=operator.itemgetter(1),reverse=True)

print("end")