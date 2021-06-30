import re

semantics_file = "diff_eqns_semantics.csv"
qids_file = "diff_eqns_qids.csv"

# open semantics file

with open(semantics_file,'r') as f:
    eqns_sem = f.readlines()

# save qids file

with open(qids_file,'w') as f:
    # convert semantics to qids

    for line in eqns_sem:
        qids = ""
        search_results = re.finditer(r'\(.*?\)', line)
        for item in search_results:
            qids += item.group(0).lstrip("(").rstrip(")") + " "
        qids = qids[:-1]
        f.write(qids + "\n")