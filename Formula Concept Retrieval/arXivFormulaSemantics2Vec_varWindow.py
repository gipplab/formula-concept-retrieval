from os import listdir
import pickle

from bs4 import BeautifulSoup
import re

from arXivDocs2Vec import docs2vec
from arXivDocs2tfidf import docs2tfidf

# Set file paths
basePath = "F:\\NTCIR-12_MathIR_arXiv_Corpus\\"

datasetPath = basePath + "NTCIR12\\"
#valid_folder_prefix = ["0001"]
#valid_folder_prefix = ["00", "01", "02", "03", "04", "05", "06"]
#valid_folder_prefix = ["00", "01"]
valid_folder_prefix = [""]

outputPath = basePath + "formulae\\duplicates\\"

# # Define class limit and desired classes
# classCounter = {} #14*250 = 3500
# # arXiv 2012
# # classLimit = 250
# # desired_classes = ['astro-ph', 'cond-mat', 'cs', 'gr-qc', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'math', 'nlin', 'quant-ph', 'physics', 'alg-geom', 'q-alg']
#
# classLimit = 680
# desired_classes = ['astro-ph']
#
# # Create lists for data and labels
#
# formulaLabs = []
# formulaDocs = []
# formulaData = []
# surrTextData = []
#
# #extract TeX formula
# import re
# start = 'alttext="'
# end = '" display='
#
# def findall(p, s):
#     '''Yields all the positions of
#     the pattern p in the string s.'''
#     i = s.find(p)
#     while i != -1:
#         yield i
#         i = s.find(p, i + 1)
#
# # exclude formulae, stopwords, html and letters from candidates
# excluded = [">", "<", "=", "~",'"', "_"]
# with open("stopwords.txt") as f:
#     stopwords = [line.strip() for line in f]
# #invalid = ["times"]
# with open("letters.txt") as f:
#     letters = [line.strip() for line in f]
#
# # Fetch content and labels of documents
# for Dir in listdir(datasetPath):
#     for prefix in valid_folder_prefix:
#         if Dir.startswith(prefix):
#             for File in listdir(datasetPath + "\\" + Dir):
#                 if not File.startswith("1") and File.endswith(".tei"):
#                     # fetch label from file prefix
#                     if Dir.startswith("9"):
#                         classLab = File.split("9")[0]
#                     else:
#                         classLab = File.split("0")[0]
#                     # check if class is desired and limit is not exceeded
#                     try:
#                         classCounter[classLab] += 1
#                     except:
#                         classCounter[classLab] = 1
#                     #if True: # switch off desired_classes / classLimit constraints
#                     if classLab in desired_classes and classCounter[classLab] <= classLimit:
#                         print(Dir + "\\" + File)
#
#                         # retrieve math data (formulae) from document
#                         with open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8") as f:
#                             filestring = f.read()
#                             formulae = BeautifulSoup(filestring, 'html.parser').find_all('formula')
#
#                         #extract operators/identifiers from formulae
#                         for formula in formulae:
#
#                             formulaString = str(formula.contents)
#
#                             # extract operators/identifiers from formulae
#                             MathData = []
#
#                             # retrieve operators/identifiers
#                             #for i in findall('</mo', formulaString):
#                             for i in findall('</m:mo', formulaString):
#                                 try:
#                                     tmp = formulaString[i - 5:i]
#                                     # character can be formula operator or identifier
#                                     character = tmp.split('>')[1]
#                                     #print(character)
#                                     MathData.append(character)
#                                 except:
#                                     pass
#                             # retrieve identifiers
#                             # for i in findall('</mi', formulaString):
#                             for i in findall('</m:mi', formulaString):
#                                 try:
#                                     tmp = formulaString[i - 5:i]
#                                     # character can be formula operator or identifier
#                                     character = tmp.split('>')[1]
#                                     # print(character)
#                                     MathData.append(character)
#                                 except:
#                                     pass
#
#                             try:
#                                 # retrieve formula TeX string as label
#                                 s = formulaString
#                                 Lab = re.search('%s(.*)%s' % (start, end), s).group(1)
#                             except:
#                                 Lab = "N/A"
#                             # store if formula fullfills filter requirements (length, equation, no digits)
#                             if len(Lab) > 10 and "=" in Lab and "{}" not in Lab and Lab != "N/A" and not any(c.isdigit() for c in Lab):
#
#                                 # extract surrounding tex
#                                 index = filestring.find('alttext="' + Lab + '" display=')
#                                 #window_size = 500
#                                 window_size = 100
#                                 surrounding_text_candidates = filestring[index - window_size:index + window_size]
#
#                                 TextData = []  # list of surrounding text words
#                                 for word in surrounding_text_candidates.split():
#                                     # lowercase and remove .:,-()
#                                     word = word.lower()
#                                     char_excl = [".", ":", ",", "-", "(", ")"]
#                                     for c in char_excl:
#                                         word = word.replace(c, "")
#                                     # not part of a formula environment
#                                     not_formula = not True in [ex in word for ex in excluded]
#                                     # not stopword
#                                     not_stopword = word not in stopwords
#                                     # not invalid html
#                                     #not_invalid = not True in [inv in word for inv in invalid]
#                                     # not a latin or greek letter
#                                     not_letter = word not in letters
#                                     if not_formula and not_stopword and not_letter: #and not_invalid
#                                         TextData.append(word)
#
#                                 if len(TextData) > 0:
#
#                                     ### append formula label ###
#                                     formulaLabs.append(Lab)
#                                     ### append formula document ###
#                                     formulaDocs.append(File)
#                                     #### append form math data ###
#                                     formulaData.append(MathData)
#                                     #### append surr text data ###
#                                     surrTextData.append(TextData)
#
# # Save text data and doc labels
#
# try:
#     with open(outputPath + "surrTextData_100.pkl",'wb') as f:
#         pickle.dump(surrTextData, f)
# except:
#     print("Failed to save text data!")

#DELETE

try:
    with open(outputPath + "surrTextData_500.pkl",'rb') as f:
        surrTextData = pickle.load(f)
except:
    print("Failed to load text data!")

# i = 0
# for formula in surrTextData:
#     for word in formula:
#         if "\\" in word:
#             surrTextData[i].remove(word)
#     i += 1

try:
    with open(outputPath + "formula_labs.pkl",'rb') as f:
        formulaLabs = pickle.load(f)
except:
    print("Failed to load labels!")

#DELETE

# try:
#     with open(outputPath + "formulaDocs.pkl",'wb') as f:
#         pickle.dump(formulaDocs, f)
# except:
#     print("Failed to save doc labels!")
#
# # Build Doc2Vec form math model
#
# model1,formulae_doc2vec = docs2vec(formulaData,formulaLabs)

# Build Doc2Vec surr text model

model2,formulae_semantics_doc2vec = docs2vec(surrTextData,formulaLabs)

# # Save formula labels and vectors
#
# try:
#     with open(outputPath + "formulaLabs.pkl",'wb') as f:
#         pickle.dump(formulaLabs, f)
# except:
#     print("Failed to save labels!")
#
# try:
#     with open(outputPath + "formulae_doc2vec.pkl",'wb') as f:
#         pickle.dump(formulae_doc2vec, f)
# except:
#     print("Failed to save vectors!")

try:
    with open(outputPath + "formulae_semantics_doc2vec.pkl",'wb') as f:
        pickle.dump(formulae_semantics_doc2vec, f)
except:
    print("Failed to save vectors!")

# # Build and save tf_idf math vectors
#
# # formula strings for tfidf
# formulaData_strings = []
# for form in formulaData:
#     formString = ""
#     for word in form:
#         formString += word + " "
#     # remove whitespace at the end
#     formulaData_strings.append(formString[:-1])
#
# formulae_tfidf = docs2tfidf(formulaData_strings)

# surr text strings for tfidf
surrTextData_strings = []
for form in surrTextData:
    textString = ""
    for word in form:
        textString += word + " "
    # remove whitespace at the end
    surrTextData_strings.append(textString[:-1])

formulae_semantics_tfidf = docs2tfidf(surrTextData_strings)

# try:
#     with open(outputPath + "formulae_tfidf.pkl",'wb') as f:
#         pickle.dump(formulae_tfidf, f)
# except:
#     print("Failed to save vectors!")

try:
    with open(outputPath + "formulae_semantics_tfidf.pkl",'wb') as f:
        pickle.dump(formulae_semantics_tfidf, f)
except:
    print("Failed to save vectors!")

print("end")