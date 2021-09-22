import collections
from copy import copy
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as PS
import numpy as np
import os
import string
import json
from pprint import pprint


def my_index_multi(l, x):
    return [i for i, _x in enumerate(l) if _x == x]


# remove punctuations, and stemming
def preprocess(query):
    punctuations = list(string.punctuation)
    for punctuation in punctuations:
        query = query.replace(punctuation, "")

    ps = PS()
    stop_words = stopwords.words('english')
    query = [ps.stem(i) for i in word_tokenize(query.lower()) if i not in stop_words]

    return query


def preprocess_tf_table():
    with open('./paper.json', 'r') as f:
        papers = json.load(f)
    stop_words = stopwords.words('english') + list(string.punctuation)

    # create dictionary that contains df, unique_words for each id
    ps = PS()
    tf_table = {}
    for i, paper in enumerate(papers):
        # stop words and punctuation removal and stemming
        word_list = [ps.stem(i) for i in word_tokenize(paper["abstract"].lower()) if i not in stop_words]

        # calculate term frequency for each document
        tf_data = {}
        for word in word_list:
            tf_data[word] = my_index_multi(word_list, word)
        tf_table[i] = {}
        tf_table[i].update(tf=tf_data, unique_words=list(set(word_list)))

    with open('./tables/tf_table.json', 'w') as f:
        json.dump(tf_table, f, indent=4)


def preprocess_df_table():
    with open('./tables/tf_table.json', 'r') as f:
        tf_table = json.load(f)
    # collect unique_keywords from all the documents into a list
    unique_word_list = []
    for key in tf_table.keys():
        unique_word_list.extend(tf_table[key]["unique_words"])

    df_table = {}
    for unique_word in set(unique_word_list):
        df_table[unique_word] = {}

        # search all documents to see if it has the keyword
        for key in tf_table.keys():
            if unique_word in tf_table[key]["tf"]:
                tmp = {}
                tmp[key] = tf_table[key]["tf"][unique_word]
                df_table[unique_word].update(tmp)

    with open('./tables/df_table.json', 'w') as f:
        json.dump(df_table, f, indent=4)


def preprocess_original_tf_table():
    with open('./paper.json', 'r') as f:
        papers = json.load(f)
    punctuations = list(string.punctuation)

    # create dictionary that contains df, unique_words for each id
    ps = PS()
    tf_table = {}
    for i, paper in enumerate(papers):
        # stop words and punctuation removal and stemming
        word_list = [ps.stem(i) for i in word_tokenize(paper["abstract"].lower())]

        # calculate term frequency for each document
        tf_data = {}
        for word in word_list:
            tf_data[word] = my_index_multi(word_list, word)
        tf_table[i] = {}
        tf_table[i].update(tf=tf_data, unique_words=list(set(word_list)))

    with open('./tables/original_tf_table.json', 'w') as f:
        json.dump(tf_table, f, indent=4)


def preprocess_original_df_table():
    with open('./tables/original_tf_table.json', 'r') as f:
        tf_table = json.load(f)
    # collect unique_keywords from all the documents into a list
    unique_word_list = []
    for key in tf_table.keys():
        unique_word_list.extend(tf_table[key]["unique_words"])

    df_table = {}
    for unique_word in set(unique_word_list):
        df_table[unique_word] = {}

        # search all documents to see if it has the keyword
        for key in tf_table.keys():
            if unique_word in tf_table[key]["tf"]:
                tmp = {}
                tmp[key] = tf_table[key]["tf"][unique_word]
                df_table[unique_word].update(tmp)

    with open('./tables/original_df_table.json', 'w') as f:
        json.dump(df_table, f, indent=4)


def calc_scores(query, tf_table, df_table):
    # retrieve query-relevant documents
    relevant_doc_id_list = []
    for q in query:
        try:
            relevant_doc_id_list += list(df_table[q].keys())
        except KeyError:
            continue

    relevant_doc_id_list = [x for x in set(relevant_doc_id_list) if relevant_doc_id_list.count(x) == len(query)]

    # if document couldn't be found
    if len(relevant_doc_id_list) == 0:
        return False

    # create word-collection
    word_collection = []
    for relevant_doc_id in relevant_doc_id_list:
        word_collection.extend(tf_table[relevant_doc_id]["unique_words"])
    word_collection = list(set(word_collection))

    ###################################
    #   calc TF-IDF part              #
    ###################################
    word_fq_dict = {}
    for relevant_doc_id in relevant_doc_id_list:
        word_fq_dict[relevant_doc_id] = {}
        # count words
        for word in word_collection:
            if word in tf_table[relevant_doc_id]["tf"]:
                _tmp = {word: int(len(tf_table[relevant_doc_id]["tf"][word]))}
                word_fq_dict[relevant_doc_id].update(_tmp)
            else:
                _tmp = {word: 0}
                word_fq_dict[relevant_doc_id].update(_tmp)
    # convert fq to tf*idf score for all words
    tf_idf_score_dict = word_fq_dict.copy()
    for doc_id, word_fq in tf_idf_score_dict.items():
        sum_fq = sum(word_fq.values())
        # calc tf*idf
        for word, fq in word_fq.items():
            tf_idf = (fq / sum_fq) * (np.log2(len(tf_table.keys()) / len(relevant_doc_id_list)) + 1)
            _tmp = {word: tf_idf}
            tf_idf_score_dict[doc_id].update(_tmp)
    ###################################
    #   calc cosine similarity part   #
    ###################################
    # convert query into vector
    vectorized_query = dict.fromkeys(word_fq_dict[next(iter(word_fq_dict))], 0)
    query = collections.Counter(query)
    for k, v in query.items():
        vectorized_query[k] = v
    for word, fq in vectorized_query.items():
        sum_fq = sum(vectorized_query.values())
        tf_idf = (fq / sum_fq) * (np.log2(len(tf_table.keys()) / len(relevant_doc_id_list)) + 1)
        vectorized_query[word] = tf_idf

    # calc cosine similarity for each document on query
    similarity_scores = {}
    v = [v for v in vectorized_query.values()]
    for doc_id, tf_idf_score in tf_idf_score_dict.items():
        u = [u for u in tf_idf_score.values()]
        score = (np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v))))
        similarity_scores[doc_id] = {"similarity": score, "weights": tf_idf_score}

    return similarity_scores


def create_doc_vec(doc_id, tf_table, df_table):
    doc_vec = {}
    for word, fq in tf_table[doc_id]["tf"].items():
        doc_vec[word] = int(len(fq))

    sum_fq = sum(doc_vec.values())
    for word, fq in doc_vec.items():
        tf_idf = (fq / sum_fq) * (np.log2(len(tf_table.keys()) / len(df_table[word].keys()) + 1))
        doc_vec[word] = tf_idf

    return doc_vec


def retrieve_top_5_doc(similarity_scores, query, tf_table, df_table, original_tf_table, original_df_table):
    sorted_scores_list = sorted(similarity_scores.items(), key=lambda x: x[1]["similarity"], reverse=True)

    # to print results
    print("\n===================================================")
    print(f"Results on the query: {query}")
    print("===================================================\n")
    for doc in sorted_scores_list[:5]:
        # preprocessing
        doc_id = doc[0]
        similarity_score = doc[1]["similarity"]
        bow = original_tf_table[doc_id]["unique_words"]
        doc_vec = create_doc_vec(doc_id, tf_table, df_table)
        v = list(doc_vec.values())
        l2 = np.sqrt(np.dot(v, v))
        sorted_doc_vec = sorted(doc_vec.items(), key=lambda x: x[1], reverse=True)

        # print results
        print(f"Document ID: {doc_id}")
        print("Top 5 highest weighted words")
        for i, vec in enumerate(sorted_doc_vec[:5]):
            _tmp = ""
            for k, v in original_df_table[vec[0]].items():
                positions = ",".join(map(str, v))
                _tmp += f"D{k}: {positions} | "
            print(f"{i + 1}. {vec[0]} -> {_tmp}")
        print(f"Number of unique keywords: {len(bow)}")
        print(f"Magnitude of L2 norm: {l2}")
        print(f"Similarity score: {similarity_score}")
        print("--------------------------------------------")

    return sorted_scores_list


def search_api(query):
    if not os.path.exists("./tables"):
        os.mkdir("./tables")

    if not os.path.isfile("./tables/tf_table.json"):
        print("Creating term frequency table....")
        preprocess_tf_table()

    if not os.path.isfile("./tables/df_table.json"):
        print("Creating document frequency table....")
        preprocess_df_table()

    if not os.path.isfile("./tables/original_tf_table.json"):
        print("Creating ORIGINAL term frequency table....")
        preprocess_original_tf_table()

    if not os.path.isfile("./tables/original_df_table.json"):
        print("Creating ORIGINAL document frequency table....")
        preprocess_original_df_table()

    with open('./tables/tf_table.json', 'r') as f:
        tf_table = json.load(f)

    with open('./tables/df_table.json', 'r') as f:
        df_table = json.load(f)

    with open('./tables/original_tf_table.json', 'r') as f:
        original_tf_table = json.load(f)

    with open('./tables/original_df_table.json', 'r') as f:
        original_df_table = json.load(f)

    p_query = preprocess(query)

    print("\n*************************************************")
    print("Working on search_api...")
    print(f"original query: {query}")
    print("preprocessed query:", p_query)
    print("*************************************************\n")

    similarity_scores = calc_scores(p_query, tf_table, df_table)

    if not similarity_scores:
        return [{"title": "No Results", "authors": "", "abstract":"", "year": ""}]

    results = retrieve_top_5_doc(similarity_scores, query, tf_table, df_table, original_tf_table, original_df_table)

    with open('./paper.json', 'r') as f:
        papers = json.load(f)

    docs = []
    for result in results:
        doc_id = int(result[0])
        docs.append(papers[doc_id])

    return docs

if __name__ == "__main__":
    with open('./query.txt', 'r') as f:
        queries = f.readlines()

    for query in queries:
        search_api(query)

