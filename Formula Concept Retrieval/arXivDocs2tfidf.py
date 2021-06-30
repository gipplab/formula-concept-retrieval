from sklearn.feature_extraction.text import TfidfVectorizer

def docs2tfidf(docData):

    # Generate tf_idf vectors

    vectorizer = TfidfVectorizer()
    docVecs = vectorizer.fit_transform(docData)

    return docVecs