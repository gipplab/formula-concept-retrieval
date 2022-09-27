import json

import SPARQLWrapper
import pywikibot
import rapidfuzz
import re

# DEFINE

# get identifier qid from name using pywikibot
def get_identifier_qid(identifier_name):
	try:
		site = pywikibot.Site("en", "wikipedia")
		page = pywikibot.Page(site, identifier_name)
		item = pywikibot.ItemPage.fromPage(page)
		qid = item.id
	except:
		qid = None
	return qid

def get_formula_qid(formula_name):

	sparql_query_string = """SELECT distinct ?item ?itemLabel ?itemDescription ?formula WHERE{  
			?item ?label "%s"@en.
			?item wdt:P2534 ?formula.
			?article schema:about ?item.
			?article schema:inLanguage "en".
			?article schema:isPartOf <https://en.wikipedia.org/>. 
			SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }    
			}""" % formula_name

	sparql_results = get_sparql_results(sparql_query_string)
	first_hit = sparql_results['results']['bindings'][0]
	qid = first_hit['item']['value'].split("/")[-1]

	return qid

# get sparql query for Wikidata 'has part' or 'calculated from' properties
def get_sparql_string_identifier_qids(part_lines):
	sparql_query = """# Find items with 'has part' or 'calculated from' QIDs
	SELECT ?item ?itemLabel ?formula ?parts ?partsLabel WHERE {
		%s
		?item wdt:P2534 ?formula.
		SERVICE wikibase:label {
		bd:serviceParam wikibase:language "en" .
		}
	}""" % part_lines
	return sparql_query

# get sparql query for identifier symbols in mathml string
def get_sparql_string_identifier_symbols(contains_line):
	sparql_query = """#find items with defining formula containing identifier symbols
	SELECT ?item ?itemLabel ?formula WHERE {
	  ?item wdt:P2534 ?formula.
	  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
	  FILTER(%s)
	}""" % contains_line.strip(" && ")
	return sparql_query

# get sparql results for sparql query string
def get_sparql_results(sparql_query_string):
	sparql = SPARQLWrapper.SPARQLWrapper("https://query.wikidata.org/sparql")
	sparql.setQuery(sparql_query_string)
	try:
		# stream with the results in XML, see <http://www.w3.org/TR/rdf-sparql-XMLres/>
		sparql.setReturnFormat(SPARQLWrapper.JSON)
		result = sparql.query().convert()
	except:
		result = None
	return result

# get formulae using identifier names
def get_sparql_query_results_identifier_names(identifier_names_list):

	# get part query lines
	has_part_lines = ""
	calculated_from_lines = ""
	symbol_represents_lines = ""
	for identifier_name in identifier_names_list:
		identifier_qid = get_identifier_qid(identifier_name)
		if identifier_qid is not None:
			# 'has part' (P527) query lines
			has_part_lines += "\t?item wdt:P527 wd:" + identifier_qid + ".\n"
			# 'calculated from' (P4934) query lines
			calculated_from_lines += "\t?item wdt:P4934 wd:" + identifier_qid + ".\n"
			# symbol represents
			symbol_represents_lines += "\t?item p:P7235 / pq:P9758 wd:" + identifier_qid + ".\n"
	has_part_lines += "?item wdt:P527 ?parts.\n"
	calculated_from_lines += "?item wdt:P4934 ?parts.\n"
	symbol_represents_lines += "?item p:P7235 / pq:P9758 ?parts.\n"

	# get sparql queries
	sparql_query_has_part = get_sparql_string_identifier_qids(has_part_lines)
	sparql_query_calculated_from = get_sparql_string_identifier_qids(calculated_from_lines)
	sparql_query_symbol_represents = get_sparql_string_identifier_qids(symbol_represents_lines)

	return get_sparql_results(sparql_query_has_part),\
		   get_sparql_results(sparql_query_calculated_from),\
		   get_sparql_results(sparql_query_symbol_represents)

# get formulae using identifier qids
def get_sparql_query_results_identifier_qids(identifier_qid_list):

	# get part query lines
	has_part_lines = ""
	calculated_from_lines = ""
	for identifier_qid in identifier_qid_list:
		if identifier_qid is not None:
			# 'has part' (P527) query lines
			has_part_lines += "\t?item wdt:P527 wd:" + identifier_qid + ".\n"
			# 'calculated from' (P4934) query lines
			calculated_from_lines += "\t?item wdt:P4934 wd:" + identifier_qid + ".\n"
	has_part_lines += "?item wdt:P527 ?parts.\n"
	calculated_from_lines += "?item wdt:P4934 ?parts.\n"

	# get sparql queries
	sparql_query_has_part = get_sparql_string_identifier_qids(has_part_lines)
	sparql_query_calculated_from = get_sparql_string_identifier_qids(calculated_from_lines)

	return [get_sparql_results(sparql_query_has_part),\
		   get_sparql_results(sparql_query_calculated_from)]

# get formulae using identifier symbols
def get_sparql_query_results_identifier_symbols(identifier_symbols_list):

	# get contains query lines
	contains_line = ""
	calculated_from_lines = ""
	for identifier_symbol in identifier_symbols_list:
		contains_line += "CONTAINS(STR(?formula), '<mi>"\
						 + identifier_symbol + "</mi>') && "

	# get sparql queries
	sparql_query_symbols = get_sparql_string_identifier_symbols(contains_line)

	return get_sparql_results(sparql_query_symbols)

def search_formulae_by_identifiers_Wikidata(identifiers,result_limit):

	# Decide if identifier names or symbols
	symbols = True #all([len(i)==1 for i in identifiers])
	if symbols:
		sparql_results = get_sparql_query_results_identifier_symbols(
			identifier_symbols_list=identifiers)
	else:
		sparql_results = get_sparql_query_results_identifier_names(
			identifier_names_list=identifiers)

	query_results = []

	try:
		results = sparql_results['results']['bindings']
	except:
		results = []

	rank = 1
	for result in results:
		try:
			mathml = result['formula']['value']
			formula = (mathml.split('alttext="{'))[1].split('}">')[0]
			query_results.append((formula,rank))
		except:
			pass
		rank += 1

	return query_results[:result_limit]

def get_formula(sparql_results):
	first_hit = sparql_results['results']['bindings'][0]
	mathml = first_hit['formula']['value']
	formula = (mathml.split('alttext="{'))[1].split('}">')[0]
	return formula

def search_formulae_by_concept_name_Wikidata(name):

	sparql_query_string = """SELECT distinct ?item ?itemLabel ?itemDescription ?formula WHERE{  
		?item ?label "%s"@en.
		?item wdt:P2534 ?formula.
		?article schema:about ?item.
		?article schema:inLanguage "en".
		?article schema:isPartOf <https://en.wikipedia.org/>. 
		SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }    
		}""" % name

	try:
		sparql_results = get_sparql_results(sparql_query_string)
		formula = get_formula(sparql_results)
	except:
		formula = get_formula(sparql_results)

	return formula

def get_wikidata_formula_index():

	try:
		# load formula index
		with open('formula_index_wikidata.json','r',encoding='utf-8') as f:
			formula_index = json.load(f)

	except:

		formula_index = {}

		sparql_query_string = """# find all items with defining formula
		SELECT ?formula ?item ?itemLabel WHERE {
			?item wdt:P2534 ?formula.
			SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
			}"""

		sparql_results = get_sparql_results(sparql_query_string)['results']['bindings']

		nr_results = len(sparql_results)

		result_nr = 1
		nr_successful = 0
		for result in sparql_results:
			# get formula tex
			formula_string = result['formula']['value']
			try:
				formula_tex_string = re.search('%s(.*)%s' % ('alttext="', '">'),
											formula_string).group(1)
				print(formula_tex_string)
				# populate index with formula and qid
				formula_qid = result['item']['value'].split('/')[-1]
				formula_name = result['itemLabel']['value']
				formula_index[formula_tex_string] = {'name': formula_name, 'qid': formula_qid}
				nr_successful += 1
			except:
				print('failed')

			# display progress
			print('Processed: ' + str(result_nr / nr_results * 100) + '%')
			print('Successful: ' + str(nr_successful / result_nr * 100) + '%')

			result_nr += 1

		# save formula index
		with open('formula_index_wikidata.json', 'w', encoding='utf-8') as f:
			json.dump(formula_index,f)

	return formula_index

def search_formulae_by_fuzzystring_wikidata(formula_input_string, result_limit):

	formula_catalog = get_wikidata_formula_index()

	match_candidates = {}

	for formula_index_entry in formula_catalog.items():
		formula_index_string = formula_index_entry[0]
		fuzz_ratio = rapidfuzz.fuzz.ratio(formula_index_string, formula_input_string)
		match_candidates[formula_index_string] = fuzz_ratio

	match_candidates_sorted = sorted(match_candidates.items(), key=lambda kv: kv[1], reverse=True)
	query_results = match_candidates_sorted[:result_limit]

	return query_results

# EXECUTE

# formula = 'E = m c^2'
# identifiers = ['E','m','c']
#
# result_limit = 10
#
# query_results_wikidata_formula = search_formulae_by_fuzzystring(formula_input_string=formula,
# 																result_limit=result_limit)
# query_results_wikidata_identifiers = search_formulae_by_identifiers_Wikidata(identifiers)

#print('end')