import multiprocessing

from os import listdir
import pickle

from FormulaRetrieval import getFormulae

# set paths
inputPath = "F:\\arXiv\\NTCIR12"
outputPath = "F:\\arXiv\\formulae\\"

def process_files(dir):
    formula_catalog = {}
    onlyfiles = [f for f in listdir(dir)]
    for filename in onlyfiles:
        fullpath = dir + "\\" + filename
        print(fullpath)
        with open(fullpath, "r", encoding="utf8") as f:
            formula_catalog.update(getFormulae(doc_str=f.read(),mode='formula'))
    return formula_catalog

#################
# MULTIPROCESSING
#################

if __name__ == '__main__':
    formula_catalog = {}
    tmp_catalogs = {}

    # open data
    path = inputPath
    dir_list = []
    for dir in listdir(path):
        dir_list.append(path + "\\" + dir)

    with multiprocessing.Pool() as p:
        try:
            tmp_catalogs = p.map(process_files, [dir for dir in dir_list])
        except:
            pass
    for catalog in tmp_catalogs:
        for formula in catalog.items():
            try:
                formula_catalog[formula[0]] = formula[1]
                #formula_catalog[formula[0]].update(formula[1])
            except:
                pass
                #formula_catalog[formula[0]] = formula[1]

    with open(outputPath + "formula_catalog_all.pkl", "wb") as f:
        pickle.dump(formula_catalog, f)

print("end")