# My Content
# BUILD RECOMMENDER MODEL

# (the methodology is explained and illustrated in "1_eda_and_model_selection.ipynb")

print("Build hybrid recommendation model")

# 0 - Preparation

## 0.1 - import custom functions and packages

import myFunctions as mf
from joblib import load, dump
import numpy as np
import pandas as pd
import datetime
import os
import gc
from surprise import Reader, Dataset, SVDpp

## 0.2 - prepare paths

# parent path
parent_path = os.path.abspath("..")
# for "clicks" directory
clicks_folder_path = os.path.join(parent_path, "news-portal-user-interactions-by-globocom/clicks")
# for "articles_metadata"
meta_path = os.path.join(parent_path, "news-portal-user-interactions-by-globocom/articles_metadata.csv")
# for "articles_embeddings"
emb_path = os.path.join(parent_path, "news-portal-user-interactions-by-globocom/articles_embeddings.pickle")


# 1 - Collaborative Filtering

print("*"*30)
print("1 - Collaborative Filtering :")

## 1.1 - select 100000 users

# already selected in "1_eda_and_model_selection.ipynb"
kept_user_ids = load("mySaves/dev_files/kept_user_ids.joblib")

## 1.2 - create a Surprise compatible dataset

# use custom func to load "clicks" data and create a DataFrame compatible with Surprise (with columns "user_id", "article_id" and "rating")
if "X_cf_light.joblib" not in os.listdir("mySaves/dev_files") :
    X_cf_light = mf.read_clicks_and_prepare_rating(
        clicks_folder_path=clicks_folder_path,  
        ref_date_for_seniority=datetime.datetime(year=2017, month=10, day=18),
        kept_user_ids = kept_user_ids,
        min_seniority=0,
        max_seniority=None,
        minimum_count=10,
        min_count_for_best=1000
        )
    # save
    dump(X_cf_light, "mySaves/dev_files/X_cf_light.joblib")
else :
    # load
    X_cf_light = load("mySaves/dev_files/X_cf_light.joblib")

# create a Surprise Dataset
# create reader with rating scale from 0 to 1
reader = Reader(rating_scale=(0,1))
# create Dataset
X_cf_light_surprise = Dataset.load_from_df(X_cf_light, reader=reader)

## 1.3 - create and train a Surprise model

# train SVDpp on X_cf_light_surprise
if "collab_model_light.joblib" not in os.listdir("mySaves/prod_files") :
    # get best parameters from our work on "1_eda_and_model_selection.ipynb"
    cf_results = load("mySaves/cf_results/cf_results.joblib")
    best_parameters = cf_results.loc[cf_results["model_name"]=="SVDpp_opti", "parameters"].values[0]
    # initiate model
    light_model = SVDpp(**best_parameters)
    # use custom function "fit_and_save"
    mf.fit_and_save(
        X_surprise=X_cf_light_surprise, 
        model_surprise=light_model,
        path="mySaves/prod_files/collab_model_light.joblib")
    print("Collaborative Filtering model created and trained.")
else :
    # load existing trained model
    light_model = load("mySaves/prod_files/collab_model_light.joblib")
    print("Collaborative Filtering model loaded.")

## 1.4 - test on some users

print("Test on 5 users : ")
for i in range(5) :
    user_id = X_cf_light["user_id"].sample(1, random_state=i).values[0]
    print(
        "user #",
        user_id,
        " : ",
        mf.cf_top_5(
            user_id=user_id, 
            data_cf=X_cf_light, 
            trained_model=light_model
            )
        )
    


# 2 - Content Base

print("*"*30)
print("Content Base : ")

## 2.1 - Create a dataframe to load, prepare and merge "clicks" and "articles_metadata" CSVs

if "data_cb_light.joblib" not in os.listdir("mySaves/dev_files") :
    # necessary columns
    clicks_usecols = ['user_id', 'click_article_id', 'click_timestamp']
    meta_usecols = ['article_id', 'category_id']
    # unnecessary columns (only used for preparation (sorting) and merging)
    drop_after_prep_cols = ["click_article_id"]
    # use custom function
    data_cb_light = mf.read_clicks_and_meta(
        clicks_folder_path=clicks_folder_path,
        meta_path=meta_path,
        kept_user_ids=kept_user_ids,
        clicks_usecols=clicks_usecols,
        meta_usecols=meta_usecols,
        drop_after_prep_cols=drop_after_prep_cols
        )
    # save
    dump(data_cb_light, "mySaves/dev_files/data_cb_light.joblib")
else :
    data_cb_light = load("mySaves/dev_files/data_cb_light.joblib")


## 2.2 - load and reduce "articles_embeddings"

# perform PCA on articles embeddings
if "emb_reduced.joblib" not in os.listdir("mySaves/prod_files") :
    # use custom function "embedding_reduction"
    emb_reduced = mf.embedding_reduction(
        emb_path=emb_path, 
        n_components=0.95, 
        articles_ids = None
        )
    # save
    dump(emb_reduced, "mySaves/prod_files/emb_reduced.joblib")
else :
    # load saved embeddings reduction
    emb_reduced = load("mySaves/prod_files/emb_reduced.joblib")


## 2.3 - read "articles metadata"

# use custom function "read_meta"
if "meta_cb.joblib" not in os.listdir("mySaves/prod_files") :
    meta_cb = mf.read_meta(meta_path, usecols = ["article_id", "category_id"]).astype(int)
    # save
    dump(meta_cb, "mySaves/prod_files/meta_cb.joblib")
else :
    meta_cb = load("mySaves/prod_files/meta_cb.joblib")


## 2.4 - test on some users

# use custom function "cb_top_5" on 5 users
print("Test on 5 users : ")
for i in range(5) :
    user_id = X_cf_light["user_id"].sample(1, random_state=i).values[0]
    print(
        "user #",
        user_id,
        " : ",
        mf.cb_top_5(
            user_id=user_id, 
            data_cb=data_cb_light, 
            emb=emb_reduced,
            meta=meta_cb
            )
        )
    

# 3 - Hybrid Model (for production)

print("*"*30)
print("Hybrid Model : ")

## 3.1 - merge "cf" and "cb" data
if "data_light.joblib" not in os.listdir("mySaves/prod_files") :
    # filter "data_cb_light" on articles contained in "X_cf_light"
    mask = data_cb_light["article_id"].isin(X_cf_light["article_id"].unique())
    data_light = data_cb_light.loc[mask]
    # merge
    # (If "on" is None and not merging on indexes then this defaults to the intersection of the columns in both DataFrames.)
    data_light = data_light.merge(X_cf_light, how="inner", on=None)
    # save
    dump(data_light, "mySaves/prod_files/data_light.joblib")

else :
    data_light = load("mySaves/prod_files/data_light.joblib")

## 3.2 - test on some users

# use custom function "cf_and_cb_mix_top_5" on 5 users
# number of recommendations from collaborative filtering
n_cf = 3
print("Test on 5 users (3 recs from CF, 2 recs from CB) : ")
for i in range(5) :
    user_id = X_cf_light["user_id"].sample(1, random_state=i).values[0]
    print(
        "user #",
        user_id,
        " : ",
        mf.cf_and_cb_mix_top_5(
            user_id=user_id, 
            n_cf=n_cf,
            data=data_light, 
            trained_model=light_model,
            emb=emb_reduced,
            meta=meta_cb
        )
    )