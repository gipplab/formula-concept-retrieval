import pickle

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set file paths
#basePath = ""
#basePath = "F:\\SigMathLing_arXMLiv-08-2018\\"
basePath = "F:\\NTCIR-12_MathIR_arXiv_Corpus\\"

formulaPath = "formulae\\duplicates\\"
subjectClass = "gr-qc\\"

inputPath = basePath + formulaPath + subjectClass
outputPath = basePath + formulaPath + subjectClass

# Load formula labels and math vectors

with open(outputPath + "formulaLabs.pkl",'rb') as f:
    formulaLabs = pickle.load(f)

with open(outputPath + "formulae_doc2vec.pkl",'rb') as f:
    formulae_doc2vec = pickle.load(f)
with open(outputPath + "formulae_tfidf.pkl",'rb') as f:
    formulae_tfidf = pickle.load(f)
with open(outputPath + "formulae_semantics_doc2vec.pkl",'rb') as f:
    formulae_semantics_doc2vec = pickle.load(f)
with open(outputPath + "formulae_semantics_tfidf.pkl",'rb') as f:
    formulae_semantics_tfidf = pickle.load(f)

encoding = formulae_semantics_doc2vec

# VISUALIZE FORMULA SPACE
# dimensionality reduction

red = PCA(n_components=2)
red = TruncatedSVD(n_components=2)
vectors_red = red.fit_transform(encoding)

# plot reduced vectors

# new plot figure
fig = plt.figure()
ax = plt.axes()

# scatter plot
ax.scatter(vectors_red[:,0], vectors_red[:,1],s=1)
#ax.scatter(vectors_red[:,0], vectors_red[:,1],c=formulaLabs, cmap='rainbow')

# formula labs as text
for i in range(0,len(formulaLabs)):
    plt.text(vectors_red[i,0],vectors_red[i,1],formulaLabs[i].replace("$", ""))
    #plt.text(vectors_red[i,0], vectors_red[i,1],"N/A")

plt.show()

print("end")