import gensim

def docs2vec(docData,docNames):

    # Create iterator for Doc2Vec vocabulary

    class LabeledLineSentence(object):

        def __init__(self, doc_list, labels_list):

            self.labels_list = labels_list
            self.doc_list = doc_list

        def __iter__(self):

            for idx, doc in enumerate(self.doc_list):
                  yield gensim.models.doc2vec.TaggedDocument(doc, [self.labels_list[idx]])

    iterator = LabeledLineSentence(docData, docNames)

    # Build Doc2Vec model

    model = gensim.models.Doc2Vec(vector_size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025, epochs=100)#
    model.build_vocab(iterator)

    # Train Doc2Vec model
    for epoch in range(10):#10
        print('iteration' + str(epoch+1))
        model.train(iterator, epochs=model.epochs, total_examples=model.corpus_count)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
        model.total_examples = model.corpus_count

    # Generate Doc2Vec vectors

    docVecs = []
    for Name in docNames:
        docVecs.append(model.docvecs[Name])

    return model,docVecs