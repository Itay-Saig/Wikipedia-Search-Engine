import gcsfs
import time
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from  updated_inverted_index import*

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    start_time = time.time()
    while(time.time() - start_time < 32): ##just in case there's a hard to handle query
        res = []
        query = request.args.get('query', '')
        # query = "political forms"
        if len(query) == 0:
            return jsonify(res)
        # BEGIN SOLUTION
        query_to_search = tokenize(query)
        if len(query_to_search) <= 3 :
        # or (l > 3 and query[-1] == "?"):  # The query contains 2 terms or fewer
            q_top_n_docs_dict = dict(search_title_scores_for_small_query(query_to_search))  # Dictionary of {doc_id: score}, sorted by score
            for doc_id in q_top_n_docs_dict.keys():
                q_top_n_docs_dict[doc_id] += (norm_pr.get(doc_id,0) * prw + norm_pv.get(doc_id,0) * pvw)
            res = [(doc_id, id_title[doc_id]) for doc_id in q_top_n_docs_dict.keys()][:5]
        else:  # The query contains at least 3 terms
            # body_res =   # List of (doc_id, score), sorted by score
            # body_res_sorted_doc = sorted(body_res, key=lambda x: x[0])  # List of (doc_id, score), sorted by doc_id
            q_top_n_docs_dict = dict(search_body_scores(query_to_search))
            title_res = search_title_scores(query_to_search)  # List of (doc_id, score), sorted by score
            for doc_id, score in title_res:
                q_top_n_docs_dict.get(doc_id, 0) + score * tw
            # anchor_res = search_anchor_scores(query_to_search)  # List of (doc_id, score), sorted by score
            # for doc_id, score in anchor_res:
            #     q_top_n_docs_dict.get(doc_id, 0) + score * aw
            for doc_id in q_top_n_docs_dict.keys():
                q_top_n_docs_dict[doc_id] += norm_pr.get(doc_id,0) * prw + norm_pv.get(doc_id,0) * pvw
            q_top_n_docs_list = sorted(q_top_n_docs_dict.items(), key=lambda x: x[1])[:5]  # List of all the (doc_id, score), sorted by score
            for doc_id, score in q_top_n_docs_list:
                res.append((doc_id, id_title[doc_id]))
        # END SOLUTION
        return jsonify(res)
    return []
@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    # query = "political forms"
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query_to_search = tokenize(query)
    query_dict = {}  # Dictionary of {term: tf-idf} for the unique terms in the query
    candidates_scores = {}  # Dictionary of {(doc, term): tf-idf} for the candidate documents
    doc_score = {}  # Dictionary of {doc: cosine score} for the unique terms in the query
    counter = Counter(query_to_search)  # Dict of tf for each term in the doc
    for term in set(query_to_search):
        if idx_body.df.get(term, -10) != -10:  # avoid terms that do not appear in the index  @TODO change the condition to getter on dicts
            query_dict[term] = (counter[term] / len(query_to_search)) * IDF[term]  # {term: tf-idf}
            term_pls = read_posting_list(idx_body, term)  # Posting list of the term
            for doc_id, tf in term_pls:
                candidates_scores[(doc_id, term)] = (tf / DL[doc_id]) * IDF[term]
                if doc_score.get(doc_id, -10) == -10:  # New doc is added
                    doc_score[doc_id] = 0

    for doc_id in doc_score.keys():
        score = 0
        for term in set(query_to_search):
            if candidates_scores.get((doc_id, term), -10) != -10:
                score += (candidates_scores[(doc_id, term)] * query_dict[term])
            doc_score[doc_id] = score  # Score is equal to doc * query

    cosine_dict = {doc_id: doc_score[doc_id] / (np.linalg.norm(list(query_dict.values())) * NORM[doc_id]) for doc_id, score in doc_score.items()}

    top_n_res = sorted([(doc_id, score) for doc_id, score in cosine_dict.items()], key=lambda x: x[1], reverse=True)[:60]
    res = [(doc_id, id_title[doc_id]) for doc_id, score in top_n_res]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    scores_doc = {}
    # query = "chemistry"
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tok_query = tokenize(query)

    for term in tok_query:  # Iterates over all the words in the query
        if term not in idx_title.df:
            continue
        doc_tf_list = read_posting_list(idx_title, term)  # List of tuples of template [(doc_id, tf), (doc_id, tf),...]
        for doc_tf in doc_tf_list:
            doc_id = doc_tf[0]
            scores_doc[doc_id] = scores_doc.get(doc_id, 0) + 1  # Gives score of 1 for each doc that contains word from the query
    doc_score_list = list(scores_doc.items())  # List of [(doc_id, score), (doc_id, score),...]
    doc_score_list.sort(key=lambda x: x[1])  # Sorted by score
    doc_score_list = map(lambda x: x[0], doc_score_list)  # List of all the relevant docs, sorted by relevance
    for doc_id in doc_score_list:
        res.append((doc_id, id_title[doc_id]))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with an anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    scores_doc = {}
    # query = "anarchism"
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tok_query = tokenize(query)
    for term in tok_query:  # Iterates over all the words in the query
        if term not in idx_anchor.df:
            continue
        doc_tf_list = read_posting_list(idx_anchor, term)  # List of tuples of template [(doc_id, tf), (doc_id, tf),...]
        for doc_tf in doc_tf_list:
            doc_id = doc_tf[0]
            scores_doc[doc_id] = scores_doc.get(doc_id, 0) + 1  # Gives score of 1 for each doc that contains word from the query
    doc_score_list = list(scores_doc.items())  # List of [(doc_id, score), (doc_id, score),...]
    doc_score_list.sort(key=lambda x: x[1])  # Sorted by score
    doc_score_list = map(lambda x: x[0], doc_score_list)  # List of all the relevant docs, sorted by relevance
    for doc_id in doc_score_list:
        res.append((doc_id, id_title[doc_id]))

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)

    for id in wiki_ids:
        res.append(page_rank.get(id,0))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    for id in wiki_ids:
        res.append(page_views.get(id,0))
    # END SOLUTION
    return jsonify(res)

def search_body_scores(query_to_search):
    ''' Returns up to a 100 search results for the query using TFIDF and COSINE
        SIMILARITY of the BODY of articles only.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, score).
    '''
    # BEGIN SOLUTION
    query_dict = {}  # Dictionary of {term: tf-idf} for the unique terms in the query
    candidates_scores = {}  # Dictionary of {(doc, term): tf-idf} for the candidate documents
    doc_score = {}  # Dictionary of {doc: cosine score} for the unique terms in the query
    counter = Counter(query_to_search)  # Dict of tf for each term in the doc
    for term in set(query_to_search):
        if idx_body.df.get(term, -10) != -10:  # avoid terms that do not appear in the index  @TODO change the condition
            query_dict[term] = (counter[term] / len(query_to_search)) * IDF[term]  # {term: tf-idf}
            term_pls = read_posting_list(idx_body, term)  # Posting list of the term
            for doc_id, tf in term_pls:
                candidates_scores[(doc_id, term)] = (tf / DL[doc_id]) * IDF[term]
                if doc_score.get(doc_id, -10) == -10:  # New doc is added
                    doc_score[doc_id] = 0

    for doc_id in doc_score.keys():
        score = 0
        for term in set(query_to_search):
            if candidates_scores.get((doc_id, term), -10) != -10:
                score += (candidates_scores[(doc_id, term)] * query_dict[term])
            doc_score[doc_id] = score  # Score is equal to doc * query

    cosine_dict = {doc_id: doc_score[doc_id] / (np.linalg.norm(list(query_dict.values())) * NORM[doc_id]) for doc_id, score in doc_score.items()}
    return sorted([(doc_id, score * bw) for doc_id, score in cosine_dict.items()], key=lambda x: x[1], reverse=True)[:60]
def search_title_scores(query_to_search):
    ''' Returns up to a 100 search results for the query using TFIDF and COSINE
        SIMILARITY of the TITLE of articles only.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, score).
    '''
    query_dict = {}  # Dictionary of {term: tf-idf} for the unique terms in the query
    candidates_scores = {}  # Dictionary of {(doc, term): tf-idf} for the candidate documents
    doc_score = {}  # Dictionary of {doc: cosine score} for the unique terms in the query
    counter = Counter(query_to_search)  # Dict of tf for each term in the doc
    for term in set(query_to_search):
        if idx_title.df.get(term, -10) != -10:  # avoid terms that do not appear in the index  @TODO change the condition
            query_dict[term] = (counter[term] / len(query_to_search)) * title_IDF[term]  # {term: tf-idf}
            term_pls = read_posting_list(idx_title, term)  # Posting list of the term
            for doc_id, tf in term_pls:
                candidates_scores[(doc_id, term)] = (tf / title_DL[doc_id]) * title_IDF[term]
                if doc_score.get(doc_id, -10) == -10:  # New doc is added
                    doc_score[doc_id] = 0

    for doc_id in doc_score.keys():
        score = 0
        for term in set(query_to_search):
            if candidates_scores.get((doc_id, term), -10) != -10:
                score += (candidates_scores[(doc_id, term)] * query_dict[term])
            doc_score[doc_id] = score  # Score is equal to doc * query

    cosine_dict = {doc_id: doc_score[doc_id] / (np.linalg.norm(list(query_dict.values())) * title_NORM[doc_id]) for doc_id, score in doc_score.items()}
    return sorted([(doc_id, score) for doc_id, score in cosine_dict.items()], key=lambda x: x[1], reverse=True)[:60]
def search_title_scores_for_small_query(query_to_search):
    ''' Returns up to a 100 search results for the query using TFIDF and COSINE
        SIMILARITY of the TITLE of articles only.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, score).
    '''
    query_dict = {}  # Dictionary of {term: tf-idf} for the unique terms in the query
    candidates_scores = {}  # Dictionary of {(doc, term): tf-idf} for the candidate documents
    doc_score = {}  # Dictionary of {doc: cosine score} for the unique terms in the query
    counter = Counter(query_to_search)  # Dict of tf for each term in the doc
    for term in set(query_to_search):
        if idx_title.df.get(term, -10) != -10:  # avoid terms that do not appear in the index  @TODO change the condition
            query_dict[term] = (counter[term] / len(query_to_search)) * title_IDF[term]  # {term: tf-idf}
            term_pls = read_posting_list(idx_title, term)  # Posting list of the term
            for doc_id, tf in term_pls:
                candidates_scores[(doc_id, term)] = (tf / title_DL[doc_id]) * title_IDF[term]
                if doc_score.get(doc_id, -10) == -10:  # New doc is added
                    doc_score[doc_id] = 0

    for doc_id in doc_score.keys():
        score = 0
        for term in set(query_to_search):
            if candidates_scores.get((doc_id, term), -10) != -10:
                score += (candidates_scores[(doc_id, term)] * query_dict[term])
            doc_score[doc_id] = score  # Score is equal to doc * query

    cosine_dict = {doc_id: doc_score[doc_id] / (np.linalg.norm(list(query_dict.values())) * title_NORM[doc_id]) for doc_id, score in doc_score.items()}
    return sorted([(doc_id, score * stw) for doc_id, score in cosine_dict.items()], key=lambda x: x[1], reverse=True)[:60]

def search_anchor_scores(query_to_search):
    ''' Returns up to a 100 search results for the query using TFIDF and COSINE
        SIMILARITY of the ANCHOR of articles only.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, score).
    '''
    query_dict = {}  # Dictionary of {term: tf-idf} for the unique terms in the query
    candidates_scores = {}  # Dictionary of {(doc, term): tf-idf} for the candidate documents
    doc_score = {}  # Dictionary of {doc: cosine score} for the unique terms in the query
    counter = Counter(query_to_search)  # Dict of tf for each term in the doc
    for term in set(query_to_search):
        if idx_anchor.df.get(term, -10) != -10:  # avoid terms that do not appear in the index  @TODO change the condition
            query_dict[term] = (counter[term] / len(query_to_search)) * IDF[term]  # {term: tf-idf}
            term_pls = read_posting_list(idx_anchor, term)  # Posting list of the term
            for doc_id, tf in term_pls:
                candidates_scores[(doc_id, term)] = (tf / DL[doc_id]) * IDF[term]
                if doc_score.get(doc_id, -10) == -10:  # New doc is added
                    doc_score[doc_id] = 0

    for doc_id in doc_score.keys():
        score = 0
        for term in set(query_to_search):
            if candidates_scores.get((doc_id, term), -10) != -10:
                score += (candidates_scores[(doc_id, term)] * query_dict[term])
            doc_score[doc_id] = score  # Score is equal to doc * query

    cosine_dict = {doc_id: doc_score[doc_id] / (np.linalg.norm(list(query_dict.values())) * NORM[doc_id]) for doc_id, score in doc_score.items()}
    return sorted([(doc_id, score) for doc_id, score in cosine_dict.items()], key=lambda x: x[1], reverse=True)[:60]


def load_global():
    global bw, tw, aw, prw, pvw, stw
    bw, tw, aw, prw, pvw, stw = 0.3, 0.5, 0.1, 0.1, 0.1, 0.8 #weights for ranks from indices
    global idx_body, idx_title, idx_anchor, DL, id_title, IDF, NORM, norm_pr, norm_pv, title_IDF, title_DL,title_NORM, page_rank ,page_views
    # # Open the file from the bucket
    with open('/mnt/disks/data/index.pkl', 'rb') as f:
        idx_body = pickle.load(f)
    # Open the file from the bucket
    with open('/mnt/disks/data/title_index.pkl', 'rb') as f:
        # Load the pickle file
        idx_title = pickle.load(f)

    with open('/mnt/disks/data/anchor_index.pkl', 'rb') as f:
        # Load the pickle file
        idx_anchor = pickle.load(f)

    with open('/mnt/disks/data/d2dl_dict.pkl', 'rb') as f:
        # Load the pickle file
        DL = pickle.load(f)

    with open('/mnt/disks/data/id2title_dict.pkl', 'rb') as f:
        # Load the pickle file
        id_title = pickle.load(f)

    with open('/mnt/disks/data/w2idf_dict.pkl', 'rb') as f:
        # Load the pickle file
        IDF = pickle.load(f)

    with open('/mnt/disks/data/norm2doc_dict.pkl', 'rb') as f:
        # Load the pickle file
        NORM = pickle.load(f)

    with open('/mnt/disks/data/page_views.pkl', 'rb') as f:
        # Load the pickle file
        page_views = pickle.load(f)

    with open('/mnt/disks/data/title_d2dl_dict.pkl', 'rb') as f:
        # Load the pickle file
        title_DL = pickle.load(f)

    with open('/mnt/disks/data/title_norm2doc_dict.pkl', 'rb') as f:
        # Load the pickle file
        title_NORM = pickle.load(f)
    with open('/mnt/disks/data/title_w2idf_dict.pkl', 'rb') as f:
        # Load the pickle file
        title_IDF = pickle.load(f)

    page_rank = pd.read_csv('/mnt/disks/data/pr.gz', compression='gzip', engine='python', index_col=0,header=None).squeeze('columns').to_dict()

    # Normalized page rank
    pr_values = list(page_rank.values())
    min_rank = min(pr_values)
    max_rank = max(pr_values)
    norm_pr_values = [(rank - min_rank) / (max_rank - min_rank) for rank in pr_values]
    norm_pr = dict(zip(page_rank.keys(), norm_pr_values))

    # Normalized page views
    pv_values = list(page_views.values())
    min_views = min(pv_values)
    max_views = max(pv_values)
    norm_pv_values = [(views - min_views) / (max_views - min_views) for views in pv_values]
    norm_pv = dict(zip(page_views.keys(), norm_pv_values))


if __name__ == '__main__':
    load_global()
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)


