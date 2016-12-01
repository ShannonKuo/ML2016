from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import manifold
from nltk.stem import RegexpStemmer
from nltk.tokenize import RegexpTokenizer
from nltk import stem
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
  
import sklearn 
import nltk
import logging
from optparse import OptionParser
import sys
import math
import matplotlib.pyplot as plt
from time import time

import numpy as np
import string


t0 = time()
true_k = 120
plot = 1

op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")
(opts, args) = op.parse_args()

opts.n_components = 20

file1 = open(sys.argv[1] + "title_StackOverflow.txt", "r")
dataset = []
cnt = 0
for line in file1.readlines():
   cnt += 1
   porter = stem.porter.PorterStemmer()
   lan = LancasterStemmer()
   lemma = WordNetLemmatizer()

   line = line.decode('utf-8')
   line = line.lower()
   tokenizer = RegexpTokenizer(r'\w+')
   line = tokenizer.tokenize(line)
   stop = set(stopwords.words('english')) 
   line = [i for i in line if i not in stop and len(i) >= 2]
   
   line = [porter.stem(j) for j in line ]
   line = [lan.stem(j) for j in line]
   line = [lemma.lemmatize(j) for j in line]
   newline = ""
   for i in range(len(line)):
       newline += line[i] + " "
   dataset.append(newline)
vectorizer = TfidfVectorizer(max_df=0.4, max_features=None,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(dataset)


if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()

ans = sklearn.metrics.pairwise.cosine_similarity(X)
"""
###############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=40,
                         init_size=1000, batch_size=1000, max_iter = 10000, max_no_improvement = 100, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
km.fit(X)

print (km.labels_)
if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()


if plot:
    #plt.figure(figsize = (10,10))
    #plt.scatter(X[:,0], X[:,1])
    #plt.show()
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    svd = TruncatedSVD(10)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()


    tsne = manifold.MDS(n_components=2, random_state=0)
    trans_data = tsne.fit_transform(X)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(2, 5, 10)
    plt.scatter(trans_data[:, 0], trans_data[:, 1], c=colors, cmap=plt.cm.rainbow)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    plt.show()# output prediction
"""
file2 = open(sys.argv[1] + "check_index.csv", "r")
file3 = open(sys.argv[2], "w")
file3.write("ID,Ans\n")
check_id = []
cnt = 0
for line in file2.readlines():
    if cnt > 0:
       line = line.split(",")
       line = line[1:]
       check_id.append(line)
    cnt += 1
for i in range(len(check_id)):
    if ans[int(check_id[i][0])][int(check_id[i][1])] > 0.85:
       file3.write(str(i) + ",1\n")
    else : file3.write(str(i) + ",0\n")
