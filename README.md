fic-prototype
=============

Python implementation of BNS for Bi-Normal Separation, a high performance alternative to Tf-Idf scoring for text classification.

BnsTransformer and BnsVectorizer are structured to be used with Scikit-Learn.

Implemented to follow the Scikit-learn convention and usage pattern:

    from sklearn.datasets import load_files

    corpus = "/path/to/text/corpus" #note: positive and negative docs are in seperate folders
    label_names = ['positive', 'negative']
    
    textfiles = load_files(corpus, categories=label_names, load_content=False)
    data = [open(filename, 'r').read() for filename in textfiles.filenames]

    #Tokenize and get counts for all documents, becomes a numpy data structure
    #in a format suitable for scikit-learn
    countVec = CountVectorizer(stop_words="english", binary=True,
                               ngram_range=(1, 1), strip_accents='unicode')
    X = countVec.fit_transform(data)
    Y = textfiles.target

    #the counts are now used to compute the BNS score, we are ready to classify/cluster/whatever
    vocab = countVec.vocabulary_
    bns = BnsTransformer(y=Y, vocab=vocab)
    X_bns = bns.transform(X)

Reference: 
BNS Feature Scaling: An Improved Representation over TF·IDF for SVM Text Classification.  G. Forman.  CIKM'08.  HPL-2007-32R1.
link to paper: 
http://www.hpl.hp.com/techreports/2007/HPL-2007-32R1.html
