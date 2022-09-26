from FCRsearch_wikidata import *
from FCRsearch_arXiv_wikipedia import *

# Testset

#formula = 'E = m c^2'
#identifiers = ['E','m','c']

# set paths
example_nr_prefix = ['two','three','ten'][2]
example_path = example_nr_prefix + " examples/"
full_path = "diff_eqns_examples/" + example_path

# set file names
tex_file = full_path + "diff_eqns_tex.txt"
content_file = full_path + "diff_eqns_content.txt"

# parameters
#N_examples_included
N_classes = 10
N_examples_per_class = 10
N_examples = N_classes*N_examples_per_class

# get equations tex

with open(tex_file,'r') as f:
    eqns_tex = f.readlines()
# clean
eqns_tex[0] = eqns_tex[0].lstrip('ï»¿')
# cutoff
#eqns_tex = eqns_tex[:N_examples]

# get equations content

with open(content_file,'r') as f:
    eqns_cont = f.readlines()
# clean
eqns_cont[0] = eqns_cont[0].lstrip('ï»¿')
# cutoff
#eqns_cont = eqns_cont[:N_examples]

# select formula example
selected_example_idx = 0

formula = eqns_tex[selected_example_idx]
identifiers = [id.strip()
			   for id in eqns_cont[selected_example_idx].rstrip('\n').split(',')
			   if not '\\' in id]

result_limit = 10

# Wikidata

query_results_wikidata_formula = search_formulae_by_fuzzystring_wikidata(formula_input_string=formula,
																result_limit=result_limit)
query_results_wikidata_identifiers = search_formulae_by_identifiers_Wikidata(identifiers,
																			 result_limit=result_limit)

# arXiv

catalog = "NTCIR-12_arXiv"
query_results_arxiv_formula\
	= search_formulae_by_fuzzystring_arxivwikipedia(formula_input_string=formula,
									 result_limit=result_limit,catalog=catalog)

# arxiv,symbs,single if mode_number == 4
query_results_arxiv_identifiers\
	= search_formulae_by_identifiers(input=identifiers,
									 result_limit=result_limit,mode_number=4)

# Wikipedia

catalog = "NTCIR-12_Wikipedia"
query_results_wikipedia_formula\
	= search_formulae_by_fuzzystring_arxivwikipedia(formula_input_string=formula,
									 result_limit=result_limit,catalog=catalog)

# wikip,symbs,single if mode_number == 1
query_results_wikipedia_identifiers\
	= search_formulae_by_identifiers(input=identifiers,
									 result_limit=result_limit,mode_number=1)

print('end')