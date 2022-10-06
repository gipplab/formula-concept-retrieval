import pandas as pd
import math

# Load result table

result_table = pd.read_csv('fcd_evaluation/FCR_search_rankings_10examples.csv')

def get_MRR_score(source):
	# from https://en.wikipedia.org/wiki/Mean_reciprocal_rank
	# \text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|}
	ranks = source['ranks']
	Q = len(ranks)
	MRR_score = 1/Q*[1/rank for rank in ranks]
	return MRR_score

print('end')