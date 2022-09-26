from FCRsearch_wikidata import *
from FCRsearch_arXiv_wikipedia import *
import pandas as pd

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
labels_file = full_path + "diff_eqns_labels.txt"

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

# get equations labels

with open(labels_file,'r') as f:
	 eqns_labs_tmp = f.readlines()
# clean
eqns_labs_tmp[0] = eqns_labs_tmp[0].lstrip('ï»¿')
#
eqns_labs = []
for eqn_lab in eqns_labs_tmp:
	eqns_labs.append(eqn_lab.strip("\n"))
eqns_labs = eqns_labs[:]
del eqns_labs_tmp
# cutoff
#eqns_labs = eqns_labs[:N_examples]

query_result_lines = []

# select formula example(s)
#selected_example_idx = 1
for selected_example_idx in range(1,101):

	print('Example nr. ' + str(selected_example_idx))

	formula = eqns_tex[selected_example_idx-1].rstrip('\n')
	identifiers = [id.strip()
				   for id in eqns_cont[selected_example_idx-1].rstrip('\n').split(',')
				   if not '\\' in id]
	label = eqns_labs[selected_example_idx-1]

	result_limit = 10

	# Wikidata

	query_results_wikidata_formula = search_formulae_by_fuzzystring_wikidata(formula_input_string=formula,
																	result_limit=result_limit)
	print('Wikidata formula')
	print('\n'.join([str(r) for r in query_results_wikidata_formula]))

	query_results_wikidata_identifiers = search_formulae_by_identifiers_Wikidata(identifiers,
																				 result_limit=result_limit)
	print('Wikidata identifiers')
	print('\n'.join([str(r) for r in query_results_wikidata_identifiers]))

	# arXiv

	catalog = "NTCIR-12_arXiv"
	query_results_arxiv_formula\
		= search_formulae_by_fuzzystring_arxivwikipedia(formula_input_string=formula,
										 result_limit=result_limit,catalog=catalog)
	print('arXiv formula')
	print('\n'.join([str(r) for r in query_results_arxiv_formula]))

	# arxiv,symbs,single if mode_number == 4
	query_results_arxiv_identifiers\
		= search_formulae_by_identifiers(input=identifiers,
										 result_limit=result_limit,mode_number=4)
	print('arXiv identifiers')
	print('\n'.join([str(r) for r in query_results_arxiv_identifiers]))

	# Wikipedia

	catalog = "NTCIR-12_Wikipedia"
	query_results_wikipedia_formula\
		= search_formulae_by_fuzzystring_arxivwikipedia(formula_input_string=formula,
										 result_limit=result_limit,catalog=catalog)
	print('Wikipedia formula')
	print('\n'.join([str(r) for r in query_results_wikipedia_formula]))

	# wikip,symbs,single if mode_number == 1
	query_results_wikipedia_identifiers\
		= search_formulae_by_identifiers(input=identifiers,
										 result_limit=result_limit,mode_number=1)
	print('Wikipedia identifiers')
	print('\n'.join([str(r) for r in query_results_wikipedia_identifiers]))

	# add query result lines
	new_result_line = {'formula index': selected_example_idx,
								'arXiv formula': query_results_arxiv_formula,
							   	'arxiv identifiers': query_results_arxiv_identifiers,
							   	'wikidata formula': query_results_wikidata_formula,
							   	'wikidata identifiers': query_results_wikidata_identifiers,
							   	'wikipedia formula': query_results_wikipedia_formula,
							   	'wikipedia identifiers': query_results_wikipedia_identifiers
							   }
	query_result_lines.append(new_result_line)
	#print(new_result_line)

result_table = pd.DataFrame(query_result_lines)
result_table.to_csv('fcr_search_results.csv')

print('end')