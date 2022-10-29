import re

example_nr_prefix = ['three','ten'][1]
file_path = example_nr_prefix + " examples/"

# input
semantics_file = file_path + "diff_eqns_semantics.txt"
# output
qids_file = file_path + "diff_eqns_qids.txt"

# open semantics file

with open(semantics_file,'r') as f:
    eqns_sem = f.readlines()

# save qids file

with open(qids_file,'w') as f:
    # convert semantics to qids

    def get_qids(line):
        qids = ""
        search_results = re.finditer(r'\(.*?\)', line)
        for item in search_results:
            qids += item.group(0).lstrip("(").rstrip(")") + " "
        return qids[:-1]

    for line in eqns_sem:
        qids = get_qids(line)
        f.write(qids + "\n")