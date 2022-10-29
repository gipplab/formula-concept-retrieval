import pandas as pd
import numpy as np
import scipy

example_nr_prefix = ['three','ten'][1]
file_path = example_nr_prefix + " examples/"

# input
semantics_file = file_path + "diff_eqns_semantics.csv"
# output
dict_file = file_path + "diff_eqns_semantics_dict.json"

semantics_table = pd.read_csv(semantics_file,delimiter=';')
eqns_semantics = semantics_table['eqns_semantics']

semantics_dict = {}
semantics_dict_counts = {}
for eqn in eqns_semantics:
    for semantics in eqn.split(', '):
        part,annotation = semantics.split(': ')
        try:
            semantics_dict[part].add(annotation)
        except:
            semantics_dict[part] = set([annotation])
            semantics_dict_counts[part] = {}

        try:
            semantics_dict_counts[part][annotation] += 1
        except:
            try:
                semantics_dict_counts[part][annotation] = 1
            except:
                semantics_dict_counts[part] = {annotation: 1}

# dict statistics
lens = []
ents = []
for part_annotation in semantics_dict.items():
   part,annotation = part_annotation[0],part_annotation[1]
   lens.append(len(annotation))
   counts = semantics_dict_counts[part]
   ents.append(scipy.stats.entropy(list(counts.values())))
print('Min length of annotations per formula part: ' + str(min(lens)))
print('Max length of annotations per formula part: ' + str(max(lens)))
print('Mean length of annotations per formula part: ' + str(np.mean(lens)))
print('Average entropy of part annotation frequencies: ' + str(np.mean(ents)))

print(semantics_dict)

print()