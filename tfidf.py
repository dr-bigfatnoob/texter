from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

__author__ = "bigfatnoob"

TOKEN_PATTERN = r"(?u)\b[a-zA-Z_]{3,100}\b"
STOP_WORDS = text.ENGLISH_STOP_WORDS


class StemmedCountVectorizer(CountVectorizer):
  def __init__(self, stemmer, **params):
    """
    Vectorizer capable of stemming.
    :param stemmer: Instance of stemmer. For eg. nltk.stem.porter.PorterStemmer()
    :param params: args of sklearn.feature_extraction.text.CountVectorizer
    """
    super(StemmedCountVectorizer, self).__init__(**params)
    self.stemmer = stemmer

  def build_analyzer(self):
    analyzer = super(StemmedCountVectorizer, self).build_analyzer()
    return lambda doc: (self.stemmer.stem(w) for w in analyzer(doc))


class TFIDF:
  def __init__(self, stemmer=None, stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN, smooth_idf=True, verbose=False):
    """
    TD
    :param stemmer: Instance of Stemmer
    :param stop_words: Set of stopwords
    :param token_pattern: Pattern of token used for extraction
    :param smooth_idf: If True, IDF is made smooth
    :param verbose: If True, logs are printed
    """
    if stemmer is not None:
      vectorizer = StemmedCountVectorizer(stemmer, stop_words=stop_words, token_pattern=token_pattern)
    else:
      vectorizer = CountVectorizer(stop_words=stop_words, token_pattern=token_pattern)
    self.smooth_idf = int(smooth_idf)
    self.analyzer = vectorizer.build_analyzer()
    self.idf = None
    self.vocab = None
    self.reverse_vocab = None
    self.scores = None
    self.verbose = verbose

  def _compute_weights(self, documents):
    """
    Compute IDF scores, Generate vocabulary and reverse vocabulary for corpus.
    :param documents: Documents used for training. Array of sentences
    :return: TF scores for each document
    """
    doc_tf = []
    term_doc_counts = {}
    self.log("Computing TF for each Doc ... ")
    vocab, reverse_vocab = {}, {}
    for i in xrange(len(documents)):
      if i > 0 and i % 100 == 0:
        self.log("Doc : %d" % i)
      doc = documents[i]
      tf = {}  # TF for each document
      for word in self.analyzer(doc):
        if word not in vocab:
          index = len(vocab)
          vocab[word] = index
          reverse_vocab[index] = word
        count = tf.get(word, 0)
        if count == 0:
          term_doc_counts[word] = term_doc_counts.get(word, 0) + 1
        tf[word] = count + 1
      den = sum([v ** 2 for v in tf.values()]) ** 0.5
      for word in tf.keys():  # Normalizing
        tf[word] /= den
      doc_tf.append(tf)
    del documents  # Freeing memory
    # IDF
    self.log("Computing IDF for each Word ... ")
    n_docs = len(doc_tf) + self.smooth_idf
    for word in term_doc_counts:
      term_doc_counts[word] = np.log(n_docs / (term_doc_counts[word] + self.smooth_idf)) + 1
    self.idf = term_doc_counts
    self.vocab = vocab
    self.reverse_vocab = reverse_vocab
    return doc_tf

  def fit_transform(self, documents):
    """
    :param documents: Documents to be transformed. Array of sentences.
    :return: Fit Documents and transform into normalized TFIDF scores
    """
    doc_tf = self._compute_weights(documents)
    tf_idf = {}
    transformed = np.zeros((len(documents), len(self.vocab)))
    for i in xrange(len(doc_tf)):
      if i > 0 and i % 100 == 0:
        self.log("TF Doc : %d" % i)
      tf = doc_tf[i]
      tot = 0
      for word in tf.keys():
        score = tf[word] * self.idf[word]
        transformed[i][self.vocab[word]] = score
        tot += score**2
        count = tf_idf.get(word, (0, 0))
        tf_idf[word] = (count[0] + score, count[1] + 1)  # Sum, count.
      if tot > 0:
        transformed[i] /= tot**0.5
    self.scores = tf_idf
    return transformed

  def get_feature_names(self):
    """
    :return: Dictionary where key is the feature_index and value is feature_name
    """
    return self.reverse_vocab

  def get_scores(self):
    """
    :return: Average TF * IDF for each Term in the corpus in descending order.
    """
    features = []
    for word in self.scores.keys():
      count = self.scores[word]
      features.append((word, count[0] / count[1]))
    return sorted(features, key=lambda x: x[1], reverse=True)

  def log(self, txt):
    """
    Print to console
    :param txt:
    """
    if self.verbose:
      print(txt)


def test():
  mydoclist = ['Julie loves me more than Linda loves me',
               'Jane likes me more than Julie loves me',
               'He likes basketball more than baseball']
  vectorizer = TFIDF(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN, verbose=True)
  print("### This vectorizer")
  print(vectorizer.fit_transform(mydoclist))
  print(vectorizer.get_feature_names())
  print(vectorizer.get_scores())
  print("\n### Scikit Learn's vectorizer")
  from sklearn.feature_extraction.text import TfidfVectorizer
  sk_vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN)
  print(sk_vectorizer.fit_transform(mydoclist).todense())
  print(sk_vectorizer.get_feature_names())


if __name__ == "__main__":
  test()
