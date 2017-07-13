'''
Created on Feb 2, 2013

@author: MathieuXPS
'''

import scipy.sparse as sp
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from ficlearn.util.statslib import ltqnorm

class BnsTransformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a normalized BNS or tf–BNS representation

    Tf means term-frequency while BNS means Bi-Normal Seperation. This is a 
    term weighting scheme that has been proposed by Georges Forman of HPlabs
    for use in document classification with SVM in the paper: LINK.

    The goal of using BNS instead of Tf-idf is to scale down the impact of tokens 
    that occur very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus, while taking into account the relative frequency of the word with regards
    to its label.

    Parameters
    ----------
    y     : numpy array, required
        The y array for the training examples, used in the calculation of BNS
        
    vocab : map, required
        A mapping for the document vocabulary words and their indices in the 
        X matrix
        
    use_tf : boolean, optional
        Enable inverse-document-frequency reweighting.

    References
    ----------

    """

    def __init__(self, rate_range=(0.0005, 1 - 0.0005)):
        self.rate_range = rate_range
        self.bns_scores = []

    def fit(self, X, y, verbose=False):
        """Learn the bns vector (global term weights)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """

        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=True)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=True)
            
        # # THIS IS NOT GOOD, I SHOULD WORK WITH SPARSE MATRIX
        # X = np.matrix(X.todense(), dtype=np.float64)
        # # THIS IS NOT GOOD, I SHOULD WORK WITH SPARSE MATRIX

        classes = np.array(list(set(y)))
        self.bns_scores = np.zeros((len(classes),X.shape[1]))
        print("Fitting inputs")
        for index, target_class in enumerate(classes):
            print("%s / %s" % (index, len(classes)))
            class_mask = np.array(y == target_class,dtype=int)
            self.bns_scores[index, :] = self._generate_bns_score(X,class_mask)

        print("Fitting done")
        self.bns_scores = np.max(self.bns_scores, axis=0)

        return self

    def _generate_bns_score(self, X, class_mask, verbose=False):

        positive_doc = np.sum(class_mask)
        negative_doc = len(class_mask)-positive_doc

        bns_scores = np.ravel(np.zeros((1,X.shape[1])))

        for index, word in enumerate(X.T[:]):
            word_vector = np.ravel(word.toarray())
            bns_scores[index] = self._compute_partial_bns(word_vector, positive_doc, negative_doc, class_mask, verbose)


        return bns_scores

    def _compute_partial_bns(self, word_vector, pos, neg, class_mask, verbose=False):
        """ compute the BNS score of the word of the vocabulary at the index wordIndex """
        tp = np.sum(word_vector * class_mask)
        tn = np.sum(word_vector * np.abs(class_mask-1))

        tpr = self.bounded_value(float(tp) / pos, self.rate_range[0], self.rate_range[1])
        tnr = self.bounded_value(float(tn) / neg, self.rate_range[0], self.rate_range[1])

        bns_score = abs(ltqnorm(tpr) - ltqnorm(tnr))
        if verbose:
            print("tp={0} tn={1} tpr={2} tnr={3} bns_score={4}".format(tp, tn, tpr, tnr, bns_score))

        return bns_score

    def transform(self, X, copy=True):
        """Transform a count matrix to a bns or tf-bns representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)
        
        # # THIS IS NOT GOOD, I SHOULD WORK WITH SPARSE MATRIX
        # X = np.matrix(X.todense(), dtype=np.float64)
        # # THIS IS NOT GOOD, I SHOULD WORK WITH SPARSE MATRIX

        print("Transforming inputs")
        for it, index in enumerate(list(set(X.indices))):
            X.T[index] *= self.bns_scores[index]
            print("%s / %s" % (it, X.T[:].shape[0]))

        return sp.coo_matrix(X, dtype=np.float64)
        
    def is_word_feature(self, word, verbose=False):
        """ false if word is not alphanumeric (alphabet letters or numbers 0-9)        
            true otherwise
        """
        if not word.isalnum() and " " not in word:
            return False
        elif " " in word:
            parts = word.split(" ")
            first = parts[0]; second = parts[1]
            if not first.isalnum() or not second.isalnum():                
                return False
        return True
    
    def bounded_value(self, value, lower, upper):
        """ bound the value in the range [lower,upper] """
        
        if value < lower: value = lower
        elif value > upper: value = upper
        return value
