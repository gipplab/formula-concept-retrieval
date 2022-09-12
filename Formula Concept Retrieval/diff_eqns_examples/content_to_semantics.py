import re

example_nr_prefix = ['three','ten'][1]
file_path = example_nr_prefix + " examples/"

semantics_file_tmp = file_path + "diff_eqns_semantics_tmp.txt"
semantics_file_old = file_path + "diff_eqns_semantics_old.txt"
semantics_file_new = file_path + "diff_eqns_semantics_new.txt"

# open template semantics file

with open(semantics_file_tmp,'r') as f:
    eqns_sem_tmp = f.readlines()

    # create template semantics dict
    tmp_sem_dict = {}

    def get_cont(line):
        formula_parts = line.split(",")
        for part in formula_parts:
            cont_sem = part.split(':')
            cont = cont_sem[0].strip().lstrip('ï»¿')
            sem = cont_sem[1].lstrip().rstrip('\n')
            tmp_sem_dict[cont] = sem

    def get_qids(line):
        qids = ""
        search_results = re.finditer(r'\(.*?\)', line)
        for item in search_results:
            qids += item.group(0).lstrip("(").rstrip(")") + " "
        return qids[:-1]

    for line in eqns_sem_tmp:
        cont = get_cont(line)
        qids = get_qids(line)

    print()

# open old semantics file

with open(semantics_file_old,'r') as f:
    eqns_sem_old = f.readlines()

# save new semantics file

with open(semantics_file_new,'w') as f:
    # convert semantics to qids

    for line in eqns_sem_old:
        new_line = ''
        for cont in line.split(','):
            cont = cont.strip().lstrip('ï»¿')
            try:
                new_line += cont + ': ' + tmp_sem_dict[cont] + ', '
            except:
                new_line += cont + ': ' + '"?" (Q?)' + ', '
        f.write(new_line.rstrip(', ')+'\n')