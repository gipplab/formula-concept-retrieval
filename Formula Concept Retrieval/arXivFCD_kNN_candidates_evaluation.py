import pickle

base_path = "F:\\NTCIR-12_MathIR_arXiv_Corpus/formulae/duplicates/"
subject_class = "hep-th/"
mode = "kNN_candidates_doc2vec.pkl"

path = base_path + subject_class + mode

with open(path,"rb") as f:
    kNN_candidates = pickle.load(f)

print("end")