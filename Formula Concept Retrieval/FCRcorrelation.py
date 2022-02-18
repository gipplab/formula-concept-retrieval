from scipy.stats import pearsonr

formula_length = [1,2,3,4,5]
matches_without_mmltag = [2711,1826,1916,16,5]
matches_with_mmltag = [379,13,42,1,3]

corr1 = pearsonr(formula_length,matches_without_mmltag)
corr2 = pearsonr(formula_length,matches_with_mmltag)

print('end')