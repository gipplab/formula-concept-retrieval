import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# set paths
example_nr_prefix = ['three','ten'][1]
example_path = example_nr_prefix + " examples/"
full_path = "diff_eqns_examples/" + example_path

# load eval dict
eval_dict_path = full_path + 'results/' + 'eval_dict.json'
with open(eval_dict_path,'r') as f:
    eval_dict = json.load(f)

# iterate results for encodings and number classes
for encoding in eval_dict.items():
    encoding_string = encoding[0]
    encoding_results = encoding[1]
    result_json_lines = []
    for choices in encoding_results.items():
        n_classes = choices[0]
        choice_results = choices[1]
        # add meta
        choice_results['encoding'] = encoding_string
        choice_results['n_classes'] = n_classes
        #del choice_results['nr_choices']
        # normalize
        #choice_results['nr_choices'] /= 252
        # append
        #result_json_lines.append(choice_results)
        result_json_lines.append({'n_classes': choice_results['n_classes'],
                                  'n_choices': choice_results['nr_choices'],
                                'metric': 'accuracy',
                                'value':choice_results['mean_accuracy']})
        result_json_lines.append({'n_classes': choice_results['n_classes'],
                                  'n_choices': choice_results['nr_choices'],
                                'metric': 'purity',
                                'value':choice_results['mean_purity']})
        #print(choice_results)

    # plot
    plt.clf()
    result_table = pd.DataFrame(result_json_lines)
    # distribution
    #sns.barplot(data=result_table,x='n_classes',y='n_choices')
    #plt.title('Binomial distribution of formula class choices')
    #plt.text(x=5,y=125,s='total: 1275')
    # performance
    sns.barplot(data=result_table,x='n_classes',y='value',hue='metric')
    plt.title('Encoding: ' + encoding_string)
    plt.show()

print()