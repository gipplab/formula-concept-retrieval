from os import listdir
import pickle

from FormulaRetrieval import getFormulae

# Set file paths
#basePath = "F:\\SigMathLing\\"
basePath = "F:\\arXiv\\"

#datasetPath = basePath + "dataset-arXMLiv-08-2018\\no_problem\\"
datasetPath = basePath + "NTCIR12\\"
#valid_folder_prefix = ["0001"]
#valid_folder_prefix = ["00", "01", "02", "03", "04", "05", "06"]
#valid_folder_prefix = ["00", "01"]
valid_folder_prefix = [""]

outputPath = basePath + "formulae\\"

# Define class limit and desired classes
classCounter = {}
# arXiv 2012
#classLimit = 250
#desired_classes = ['astro-ph', 'cond-mat', 'cs', 'gr-qc', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'math', 'nlin', 'quant-ph', 'physics', 'alg-geom', 'q-alg']
# arXiv 2018
#classLimit = 350
#desired_classes = ['hep-ph', 'astro-ph', 'quant-ph', 'physics', 'cond-mat', 'hep-ex', 'hep-lat', 'nucl-th', 'nucl-ex', 'hep-th', 'math', 'gr-qc', 'nlin', 'cs']

#classLimit = 2
#desired_classes = ['astro-ph']

# create formula catalog
formula_catalog = {}

# Fetch content and labels of documents
for Dir in listdir(datasetPath):
    for prefix in valid_folder_prefix:
        if Dir.startswith(prefix):
            for File in listdir(datasetPath + "\\" + Dir):
                if File.endswith(".tei"):
                #if not File.startswith("1") and File.endswith(".tei"):
                    # fetch label from file prefix
                    # if Dir.startswith("9"):
                    #     classLab = File.split("9")[0]
                    # else:
                    #     classLab = File.split("0")[0]
                    # # check if class is desired and limit is not exceeded
                    # try:
                    #     classCounter[classLab] += 1
                    # except:
                    #     classCounter[classLab] = 1
                    if True: # switch off desired_classes / classLimit constraints
                    #if classLab in desired_classes and classCounter[classLab] <= classLimit:
                        print(Dir + "\\" + File)

                        # retrieve math data (formulae) from document
                        with open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8") as f:
                            formula_catalog.update(getFormulae(doc_str=f.read(), mode='formula'))
                        # save updated formula catalog to file
                        with open(outputPath + "formula_catalog_all.pkl", 'wb') as f:
                            pickle.dump(formula_catalog, f)

print("end")