from bs4 import BeautifulSoup
import re

# define location of TeX string in MathML tag
# arXiv
start = 'alttext="'
end = '" display='
# mmlben
# start = 'alttext="'
# end1 = '" class='
# end2 = '" display='

def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i + 1)

def getFormulae(doc_str,mode):
    '''Returns TeX string, operators, numbers, and identifiers
    of all formulae contained within a given html document string.'''

    # retrieve all formulae
    if mode=="math":
        formulae = BeautifulSoup(doc_str, 'html.parser').find_all('math')
    if mode=="formula":
        formulae = BeautifulSoup(doc_str, 'html.parser').find_all('formula')

    # retrieve parts of each formula
    formula_catalog = {}
    for formula in formulae:

        # retrieve tex string
        s = str(formula)
        #s = str(formula).replace("\n","")
        try:
            formulaTeX= re.search('%s(.*)%s' % (start, end), s).group(1)
            # print(formulaTeX)
            # create new item for formula in catalog dict
            formula_catalog[formulaTeX] = {}
            formula_catalog[formulaTeX]['operators'] = set()
            formula_catalog[formulaTeX]['identifiers'] = set()
            # formula_catalog[formulaTeX]['numbers'] = []

            # extract operators/identifiers from formulae
            if mode == "math":
                prefix = '</m'
            if mode == "formula":
                prefix = '</m:m'

            formulaStr = str(formula)

            # retrieve operators (<mo>)
            for i in findall(prefix + 'o', formulaStr):
                try:
                    tmp = formulaStr[i - 5:i]
                    # cut off tag brackets
                    character = tmp.split('>')[1]
                    formula_catalog[formulaTeX]['operators'].add(character)
                except:
                    pass

            # retrieve operators (<mi>)
            for i in findall(prefix + 'i', formulaStr):
                try:
                    tmp = formulaStr[i - 5:i]
                    # cut off tag brackets
                    character = tmp.split('>')[1]
                    formula_catalog[formulaTeX]['identifiers'].add(character)
                except:
                    pass

            # retrieve numbers (<mn>)
            # for i in findall(prefix + 'n', formulaStr):
            #     try:
            #         tmp = formulaStr[i - 5:i]
            #         # cut off tag brackets
            #         character = tmp.split('>')[1]
            #         formula_catalog[formulaTeX]['numbers'].append(character)
            #     except:
            #         pass

        except:
            pass

    return formula_catalog