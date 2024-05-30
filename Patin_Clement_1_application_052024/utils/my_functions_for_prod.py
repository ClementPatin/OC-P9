


def get_top_n(predictions, n=10):
    """
    Return the top-N recommendation for each user from a set of predictions.

    parameters :
    ------------
    predictions - list of Prediction objects : The list of predictions, as returned by the test method of an algorithm.
    n - int : The number of recommendation to output for each user. By default : 10

    returns :
    ---------
    top_n - dict : where keys are user (raw) ids and values are lists of tuples: [(raw item id, rating estimation), ...] of size n.
    """

    # import
    from collections import defaultdict
    
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n






def cf_top_5(user_id, data_cf, trained_model) :
    '''
    return the top_5 recommendations for a given user and a given model

    parameters :
    ------------
    user_id - int
    data_cf - DataFrame : clicks data as DataFrame, with "user_id", "article_id" and "rating" columns, for Surprise compatibility
    trained_model string : Surprise model, already trained

    return :
    --------
    top_5 - list : 5 top articles

    '''
    # import 
    import pandas as pd
    import numpy as np
    from surprise import Reader, Dataset
    import gc
    from joblib import load
    from surprise import dump as surprise_dump

    # filter on user_id
    X_filtered = data_cf.loc[data_cf["user_id"] == user_id]
    # articles already read
    read_articles = X_filtered["article_id"].unique()
    # all articles
    all_articles = data_cf["article_id"].unique()
    # get articles_ids NOT already read/clicked by this user
    new_articles_id = np.setdiff1d(all_articles, read_articles, assume_unique=True)

    # create an dataframe for Surprise with this user_id, theses article_ids
    anti_X = pd.DataFrame(columns=data_cf.columns)

    anti_X["article_id"] = new_articles_id
    anti_X["user_id"] = user_id
    anti_X["rating"] = 0
    # display(anti_X)

    # create a test surprise dataset
    # create reader with rating scale from 0 to 1
    reader = Reader(rating_scale=(0,1))
    # create Dataset
    anti_X_surprise = Dataset.load_from_df(anti_X, reader=reader)
    # create test set
    anti_X_test = anti_X_surprise.construct_testset(raw_testset=anti_X_surprise.raw_ratings)

    # predict
    preds_anti = trained_model.test(anti_X_test)
    # display(preds_anti)

    # top5
    top_5 = get_top_n(predictions=preds_anti, n=5)[user_id]
    top_5 = [pair[0] for pair in top_5]

    del X_filtered, read_articles, all_articles, new_articles_id, anti_X, anti_X_surprise, anti_X_test, preds_anti
    gc.collect()

    return top_5






def get_user_top2_categories_and_list_of_ids(data_cb, user_id) :
    '''
    for a given user and clicks data (with only "user_id", "click_timestamp", "article_id", "category_id" columns), get a his top 2 categories and related read articles.
    the top2 is based on how recently the articles of each category have been read, and the number of occurences of this category

    parameters :
    ------------
    data_cb - dataframe : clicks data (with only "user_id", "click_timestamp", "article_id", "category_id" columns)
    user_id - int

    return :
    --------
    user_dict - dict : {category_id : list of article_ids}, with 2 keys


    '''

    # imports
    import pandas as pd
    import gc

    # filter on user_id
    data_user = data_cb.loc[data_cb["user_id"]==user_id].copy()

    ## create a rating based on recency and category popularity
    # create a normalized rating based on recency
    click_ts_user = data_user["click_timestamp"]
    data_user["timestamp_rating"] = (click_ts_user-click_ts_user.min())/(click_ts_user.max()-click_ts_user.min())

    # create a rating based on the number of occurences of the "category_id"
    cat_id_user_count = data_user["category_id"].astype(int).value_counts()
    data_user["cat_rating"] = data_user["category_id"].astype(int).apply(lambda x : cat_id_user_count.loc[x]/cat_id_user_count.sum())

    # unify scoring (geo mean)
    data_user["rating"] = data_user[["timestamp_rating", "cat_rating"]].prod(axis=1)**(1/2)

    ## group by "category_id", 
    # keeping the most recent "rating" (data_cb is sorted by "click_timestamp", so we take the first)
    # keeping the list of all "article_id"s
    gb = data_user.groupby("category_id", observed=True).agg({"rating" : ["first"], "article_id" : [lambda s : list(s.values)]})
    # rename columns
    gb.columns = ["rating", "list_article_ids"]
    # sort by "rating"
    gb = gb.sort_values(by="rating", ascending=False)
    # keep the top 2 "category_id"s
    gb = gb.iloc[:2]
    # put "list_article_ids" in a dict, with the 2 "category_id"s as keys
    user_dict = gb["list_article_ids"].to_dict()

    del data_cb, data_user, click_ts_user, cat_id_user_count, gb
    gc.collect()

    return user_dict






def get_cat_embeddings(cat_ids_list, emb, meta) :

    '''
    from a given list of "category_id"s and articles embeddings and "articles_metadata.csv" files, create a dictionnary with "category_id"s as keys and filtered embeddings as values

    parameters :
    ------------
    cat_ids_list - list of int
    emb - array - articles embeddings
    meta - dataframe : articles metadata (with "article" and "category_id" columns)

    return :
    --------
    dict_of_embs - dict of dataframes : {cat_id : filtered embeddings on this cat_id articles}
    '''

    # imports 
    import pandas as pd

    # read emb_path
    emb_df = pd.DataFrame(emb)

    # create a dict to store filtered embeddings
    dict_of_embs = {}

    # iterate on cat_ids
    for cat_id in cat_ids_list :
        # filter on cat_id
        articles_filtered = meta.loc[meta["category_id"]==cat_id, "article_id"].values
        # filter embeddings and put in dict 
        dict_of_embs[cat_id] = emb_df.loc[articles_filtered]

    return dict_of_embs






def get_5_content_base_reco_from_dicts(user_dict, emb_dict) : 

    '''
    given a user dictionnary (with some "category_id"s as keys and their list of "article_id"s) and an embeddings dictionnary (with the same keys and their filtered articles embeddings on each of these categories),
    produce content base recommendations (3 for the first "category_id", 2 for the second)
    
    '''

    # import
    from sklearn.metrics.pairwise import cosine_similarity
    import gc
    import pandas as pd

    # initiate a list of recommendations
    rec = []

    # iterate on category_id
    for i, (cat_id, list_of_ids) in enumerate(user_dict.items()) :
        # filter embeddings on the list_of_ids (the ones read by the user)
        with_ids_emb = emb_dict[cat_id].loc[list_of_ids]
        # synthesize embeddings, using the mean
        # we use ".transpose" method to keep the embedding features as columns (for cosine_similarity compatibility)
        with_ids_emb = with_ids_emb.mean(axis=0).to_frame().transpose()

        # get the other articles_ids
        rest_of_ids = list(set(emb_dict[cat_id].index) - set(list_of_ids))
        # filter embeddings on these other ids
        without_ids_emb = emb_dict[cat_id].loc[rest_of_ids]

        # compute cosine similarities
        cos_sim = cosine_similarity(without_ids_emb, with_ids_emb)

        # put results in a dataframe, with "article_id" as index 
        cos_sim = pd.DataFrame(cos_sim, index=rest_of_ids)
        # sort by the value of cosine similarities
        cos_sim = cos_sim.sort_values(by=0, ascending=False)

        # extract recommended article_ids
        # for first cat_id, recommend 3 articles. For second one, recommend 2
        n = 3-i
        # handle the case with only one category
        if len(user_dict) == 1 :
            n = 5
        rec = rec + cos_sim.index.to_list()[:n]

        del with_ids_emb, rest_of_ids, without_ids_emb, cos_sim
        gc.collect()

    return rec







def cb_top_5 (user_id, data_cb, emb, meta) :
    '''
    given a user_id and the paths to data files :
        - clicks/meta data (with only "user_id", "click_timestamp", "article_id", "category_id" columns)
        - embeddings of articles
        - articles metadata
    return a list of 5 articles recommendations

    parameters :
    ------------
    user_id - int
    data_cb - dataframe : clicks data (with only "user_id", "click_timestamp", "article_id", "category_id" columns)
    emb - array - articles embeddings
    meta - dataframe : articles metadata (with "article" and "category_id" columns)

    return :
    --------
    rec - list of int : 5 "article_id"s srecommendations 

    '''

    # use the custom function to get the user's top 2 categories and related read articles
    user_dict = get_user_top2_categories_and_list_of_ids(data_cb=data_cb, user_id=user_id)
    # use custom function to get the embeddings of all articles belonging to each category 
    emb_dict = get_cat_embeddings(cat_ids_list=list(user_dict.keys()), emb=emb, meta=meta)
    # use custmo function to get the recommendations from user_dict and emb_dict
    rec = get_5_content_base_reco_from_dicts(user_dict=user_dict, emb_dict=emb_dict)

    return rec





def cf_and_cb_mix_top_5(
    user_id, 
    n_cf,
    data, 
    trained_model, 
    emb,
    meta
    ) :
    '''
    5 recommendations mixing collaborative filtering and content base

    parameters :
    ------------
    user_id - int
    n_cf - int in [0, 5] : number of articles recommended by collaborative filtering
    data - dataframe : clicks data, with "user_id", "article_id" and "rating", "click_timestamp", "category_id" columns
    trained_model : Surprise model, already trained
    emb - array - articles embeddings
    meta - dataframe : articles metadata (with "article" and "category_id" columns)

    '''
    # get top 5 for collaborative filtering and content base
    if n_cf != 0 :
        cf_5 = cf_top_5(user_id=user_id, data_cf=data[["user_id", "article_id", "rating"]], trained_model=trained_model)
    else :
        cf_5 = []
    
    if n_cf != 5 :
        cb_5 = cb_top_5(user_id=user_id, data_cb=data[["user_id", "click_timestamp", "article_id", "category_id"]], emb=emb, meta=meta)
    else :
        cb_5 = []

    # handle possible duplicates
    cb_5 = [article_id for article_id in cb_5 if article_id not in cf_5]

    # mix them
    mix_top_5 = cf_5[:n_cf] + cb_5[:(5-n_cf)]

    return mix_top_5

