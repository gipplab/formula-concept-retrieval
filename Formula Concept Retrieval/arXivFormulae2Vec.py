from os import linesep
from os import listdir
import pickle

from bs4 import BeautifulSoup

from arXivDocs2Vec import docs2vec
from arXivDocs2tfidf import docs2tfidf

# Set file paths
#basePath = "F:\\SigMathLing\\"
basePath = "F:\\NTCIR-12_MathIR_arXiv_Corpus\\"

#datasetPath = basePath + "dataset-arXMLiv-08-2018\\no_problem\\"
datasetPath = basePath + "NTCIR12\\"
#valid_folder_prefix = ["0001"]
#valid_folder_prefix = ["00", "01", "02", "03", "04", "05", "06"]
#valid_folder_prefix = ["00", "01"]
valid_folder_prefix = [""]

outputPath = basePath + "formulae\\duplicates\\gr-qc\\"

# Define class limit and desired classes
classCounter = {}
# arXiv 2012
#classLimit = 250
#desired_classes = ['astro-ph', 'cond-mat', 'cs', 'gr-qc', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'math', 'nlin', 'quant-ph', 'physics', 'alg-geom', 'q-alg']
# arXiv 2018
#classLimit = 350
#desired_classes = ['hep-ph', 'astro-ph', 'quant-ph', 'physics', 'cond-mat', 'hep-ex', 'hep-lat', 'nucl-th', 'nucl-ex', 'hep-th', 'math', 'gr-qc', 'nlin', 'cs']

classLimit = 2158
# 680 in astro-ph
# 2158 in gr-qc
# 19867 in hep-th
desired_classes = ['gr-qc']
# astro-ph, gr-qc, hep-th

formulaData = []
formulaLabs = []

#extract TeX formula
import re
start = 'alttext="'
end = '" display='

def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i + 1)

# Fetch content and labels of documents
for Dir in listdir(datasetPath):
    for prefix in valid_folder_prefix:
        if Dir.startswith(prefix):
            for File in listdir(datasetPath + "\\" + Dir):
                if not File.startswith("1") and File.endswith(".tei"):
                    # fetch label from file prefix
                    if Dir.startswith("9"):
                        classLab = File.split("9")[0]
                    else:
                        classLab = File.split("0")[0]
                    # check if class is desired and limit is not exceeded
                    try:
                        classCounter[classLab] += 1
                    except:
                        classCounter[classLab] = 1
                    #if True: # switch off desired_classes / classLimit constraints
                    if classLab in desired_classes and classCounter[classLab] <= classLimit:
                        print(Dir + "\\" + File)

                        # retrieve math data (formulae) from document
                        with open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8") as f:
                            formulae = BeautifulSoup(f.read(), 'html.parser').find_all('formula')
                            #formulae = BeautifulSoup(f.read(), 'html.parser').find_all('formula')

                        for formula in formulae:

                            formulaString = str(formula.contents)

                            # # extract operators/identifiers from formulae
                            # Data = ""
                            #
                            # # retrieve operators/identifiers
                            # #for i in findall('</mo', formulaString):
                            # for i in findall('</m:mo', formulaString):
                            #     try:
                            #         tmp = formulaString[i - 5:i]
                            #         # character can be formula operator or identifier
                            #         character = tmp.split('>')[1]
                            #         #print(character)
                            #         Data += character + " "
                            #     except:
                            #         pass
                            # # retrieve identifiers
                            # # for i in findall('</mi', formulaString):
                            # for i in findall('</m:mi', formulaString):
                            #     try:
                            #         tmp = formulaString[i - 5:i]
                            #         # character can be formula operator or identifier
                            #         character = tmp.split('>')[1]
                            #         # print(character)
                            #         Data += character + " "
                            #     except:
                            #         pass
                            try:
                                # retrieve formula TeX string as label
                                s = formulaString
                                Lab = re.search('%s(.*)%s' % (start, end), s).group(1)
                            except:
                                Lab = "N/A"
                            # store if formula fullfills filter requirements (length, equation, no digits)
                            if len(Lab) > 10 and "=" in Lab and "{}" not in Lab and Lab != "N/A" and not any(c.isdigit() for c in Lab):
                                #### append formula data ###
                                # remove final whitespace at the end of each document
                                #formulaData.append(Data[:-1])
                                ### append formula label ###
                                formulaLabs.append(Lab)

# Save formulae

with open(outputPath + "formulae.txt","w") as f:
    f.writelines([x + linesep for x in formulaLabs])

# Build Doc2Vec math model

model,formulae_word2vec = docs2vec(formulaData,formulaLabs)

# # Save Doc2Vec math model
#
# try:
#     with open(outputPath + "doc2vecMath_op.model", 'wb') as f:
#         pickle.dump(model, f)
# except:
#     print("Failed to save model!")

# Save formula labels and math vectors

try:
    with open(outputPath + "formulaLabs.pkl",'wb') as f:
        pickle.dump(formulaLabs, f)
except:
    print("Failed to save labels!")

try:
    with open(outputPath + "formulae_word2vec.pkl",'wb') as f:
        pickle.dump(formulae_word2vec, f)
except:
    print("Failed to save math vectors!")

# Build and save tf_idf math vectors

formulae_tfidf = docs2tfidf(formulaData)

try:
    with open(outputPath + "formulae_tfidf.pkl",'wb') as f:
        pickle.dump(formulae_tfidf, f)
except:
    print("Failed to save math vectors!")

print("end")