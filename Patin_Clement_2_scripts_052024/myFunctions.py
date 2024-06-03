def myDateDistri(s, nBars=100, ax=None):
    """
    plot empirical distribution of a datetime column

    parameters :
    ------------
    s - Series : datetime dtype
    n_bars - int : maximum number of bars desired. By default 15
    ax - axes : By default None
    """
    # imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # put s in a dataframe
    df = s.copy().to_frame()

    # extract the date and set a datetimeIndex with it, creating a timeSeries
    df["date"] = pd.to_datetime(s.dt.date)
    df.set_index("date", inplace=True)

    # resample the timeSeries

    # investigate on the date period to choose frequency of the resample, and xtick labels
    period = (s.max() - s.min()).ceil("d").days
    if period > 365:
        # adapt period
        periodMonths = period / 365 * 12
        # choose frequence
        digitFreq = int(np.ceil(periodMonths / nBars))
        freq = str(digitFreq) + "M"
        # set xticklabels date_format
        date_format = "%b-%Y"

    else:
        digitFreq = int(np.ceil(period / nBars))
        # choose frequence
        freq = str(digitFreq) + "d"
        # set xticklabels date_format
        date_format = "%d-%b"

    # resample
    df = df.resample(freq).count()

    # plot
    sns.barplot(x=df.index, y=df[s.name], ax=ax)

    # adjust x tick labels
    labels = [e.strftime(date_format) for e in df.index]
    if not ax:
        ax = plt.gca()
    ax.set_xticklabels(labels, rotation=45, ha="right")











def myDescribe(dataframe):
    """displays a Pandas .describe with options : quantitaves columns, qualitatives columns, all columns.
    If a dict is given as an input : {"df1Name" : df1, "df2Name" : df2, etc.}, one can choose the dataframe

    parameters :
    ------------
    dataframe : Pandas dataframe or a Dict

    """

    import ipywidgets as widgets  # import library
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display

    # def main function
    def myDescribeForOnlyOneDf(df):
        # if df is a dictionnary key, we get its value
        if type(df) == str:
            df = dataframe[df]

        # widget with .describe() display options
        widDescribe = widgets.RadioButtons(
            options=["quantitative", "qualitative", "all"],
            value="all",
            description="Which features :",
            disabled=False,
            style={"description_width": "initial"},
        )

        # widget to select column
        widColList = widgets.Dropdown(
            options={"all": None} | {col: col for col in list(df.columns)},
            value=None,
            description="Which column :",
            disabled=False,
            style={"description_width": "initial"},
        )

        def handle_widDescribe_change(
            change,
        ):  # linking col dropdwn with the type of describe display option
            if change.new == "qualitative":
                widColList.options = {"all": None} | {
                    col: col
                    for col in list(df.select_dtypes(["O", "category"]).columns)
                }
            if change.new == "quantitative":
                widColList.options = {"all": None} | {
                    col: col
                    for col in list(
                        df.select_dtypes(
                            [
                                "float64",
                                "float32",
                                "float16",
                                "int64",
                                "int32",
                                "int16",
                                "int8",
                                "uint8",
                            ]
                        ).columns
                    )
                }
            if change.new == "all":
                widColList.options = {"all": None} | {
                    col: col for col in list(df.columns)
                }

        widDescribe.observe(handle_widDescribe_change, "value")

        # sub function used in final output
        def describeFunct(df, whichTypes, columnName=None):
            if whichTypes == "qualitative":
                include = ["O", "category"]
                exclude = [
                    "float64",
                    "float32",
                    "float16",
                    "int64",
                    "int32",
                    "int16",
                    "int8",
                    "uint8",
                ]
            elif whichTypes == "quantitative":
                include = None
                exclude = None
            elif whichTypes == "all":
                include = "all"
                exclude = None
            if columnName:
                df = df[[columnName]]
            describeTable = df.describe(include=include, exclude=exclude)
            # add dtypes
            describeTable.loc["dtype"] = describeTable.apply(
                lambda s: df[s.name].dtype
            ).values.tolist()
            describeTable.loc["%NaN"] = describeTable.apply(
                lambda s: (round(df[s.name].isna().mean() * 100, 1)).astype(str) + "%"
            ).values.tolist()
            describeTable = pd.concat(
                [describeTable.iloc[-1:], describeTable.iloc[:-1]]
            )

            # decide which kind of display

            # for columns other than "O", we can plot distribution next to .describe() table
            if columnName and df[columnName].dtype.kind not in "O":
                # create fig and 2 axes, one for the table, one for the plot
                fig, (ax1, ax2) = plt.subplots(
                    1, 2, width_ratios=[1, 4], figsize=(14, 4)
                )
                # set lines colors, "grey" every other line
                colors = [
                    "#F5F5F5" if i % 2 == 1 else "w" for i in range(len(describeTable))
                ]
                # plot table
                ax1.table(
                    cellText=describeTable.values,
                    rowLabels=describeTable.index,
                    bbox=[0, 0, 1, 1],
                    colLabels=describeTable.columns,
                    cellColours=[[color] for color in colors],
                    rowColours=colors,
                )
                ax1.axis(False)
                # plot a box plot if column is numerical and not datetime
                if df[columnName].dtype.kind not in "mM":
                    sns.boxplot(data=df, x=columnName, ax=ax2)
                # if datatime, use myDateDistri function
                else:
                    myDateDistri(s=df[columnName], ax=ax2)
                plt.show()

            else:
                display(describeTable)

        # output
        out = widgets.interactive_output(
            describeFunct,
            {
                "df": widgets.fixed(df),
                "whichTypes": widDescribe,
                "columnName": widColList,
            },
        )
        display(widgets.HBox([widDescribe, widColList]), out)

    # if input is a dataframe, use above function
    if type(dataframe) != dict:
        myDescribeForOnlyOneDf(dataframe)

    # if input is a dict, add a widget to select a dataframe
    else:
        widDfList = widgets.Dropdown(
            options=list(dataframe.keys()),
            value=list(dataframe.keys())[0],
            description="Which dataframe :",
            disabled=False,
            style={"description_width": "initial"},
        )

        out = widgets.interactive_output(myDescribeForOnlyOneDf, {"df": widDfList})
        display(widDfList, out)





def read_all_clicks(clicks_folder_path, usecols = None, kept_user_ids = None) : 
    '''
    load all "clisks_hour_xxx" .csv and put them in a unique dataframe

    parameters :
    ------------
    clicks_folder_path - string : path of the folder containing all .csv
    usecols - list of strings or None : list of column names to use. By default : None (read all columns)
    kept_user_ids - list_like : for sampling, "user_id"s we want to keep. By default : None (keep all of them)

    return :
    --------
    clicks - dataframe : containing all concatenated "clisks_hour_xxx"s
    '''

    # imports
    import pandas as pd
    import os

    # create files paths
    files_paths = [os.path.join(clicks_folder_path, file_name) for file_name in os.listdir(clicks_folder_path)]

    # read all files and concatenate them
    clicks = pd.concat(
        [
            pd.read_csv(file_path, usecols=usecols)
            for file_path in files_paths
        ],
        ignore_index=True
    )

    # handle sampling
    if kept_user_ids is not None :
        clicks = clicks.loc[clicks["user_id"].isin(kept_user_ids)]

    # handle dtypes
    for col in clicks.columns :
        if ("timestamp" in col) or ("start" in col) :
            clicks[col] = pd.to_datetime(clicks[col], unit = "ms")
        elif ("size" in col) :
            clicks[col] = clicks[col].astype("uint8")
        else :
            clicks[col] = pd.Categorical(clicks[col])

    # for some features, replace the codes for their respective categories
    # "click_environment"
    if "click_environment" in clicks.columns :
        clicks["click_environment"] = clicks["click_environment"].map(
            {1 : "Facebook Instant Article", 2 : "Mobile App", 3 : "AMP (Accelerated Modile Pages)", 4 : "Web"}
        )
    # "click_deviceGroup"
    if "click_deviceGroup" in clicks.columns :
        clicks["click_deviceGroup"] = clicks["click_deviceGroup"].map(
            {1 : "Tablet", 2 : "TV", 3 : "empty", 4 : "Mobile", 5 : "Desktop"}
        )
    
    # sort by "click_timestamp"
    if "click_timestamp" in clicks.columns :
        clicks = clicks.sort_values(by = "click_timestamp", ascending=False)

    return clicks





def featureDistrib(df,featureName,dictPalette=None,ax=None) :
    '''
    A function to draw the empirical distribution of a given feature
    
    parameters
    ----------
    df : dataframe
    
    optionnal
    ---------
    dictPalette : dictionnary, with features names for keys and colors for values. By default None
    ax = axe position, if used within a matplotlib subplots
    

    '''
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # common parameters
    myColor=dictPalette[featureName] if dictPalette else None # set the color of graph with dictPalette
    myStat="density" # graph will display percentages
    
    # for numerical features
    if (df[featureName].dtype.kind in "biufc") and df[featureName].nunique()!=2 : 
        # draw
        sns.histplot(data=df,
                     x=featureName,
                     color=myColor,
                     kde=True, # show density estimate
                     ax=ax,
                     stat=myStat,
                    )
    # for other categorical features
    else : 
        # sort feature categories by number of appearence
        myOrder=df[featureName].value_counts().index.tolist() 
        # create a categorical Series with that order
        myOrderedCatSeries=pd.Categorical(df[featureName],myOrder) 
        # draw
        sns.histplot(x=myOrderedCatSeries,
                     color=myColor,
                     ax=ax,
                     stat=myStat

                   )
    # set title if ax=None
    if not ax :
        plt.title(featureName+" - Empirical distribution")
        plt.show()






def unique_values(series) :
    '''
    extract unique values of a Series as a list
    '''
    # import
    import pandas as pd

    # use ".unique()" method and put the result in a list 
    return list(series.unique())







def n_duplicates(series) :
    '''
    returns the number of duplicated values in a Series
    '''
    # import 
    import pandas as pd
    
    # use ".duplicated" method and sum the result
    return series.duplicated().sum()




def read_meta(meta_path, usecols = None) : 
    '''
    load and prepare "articles_metadata.csv"

    parameters :
    ------------
    meta_path - string : path of "articles_metadata.csv"
    usecols - list of strings or None : list of column names to use. By default : None (read all columns)

    return :
    --------
    meta - dataframe 
    '''

    # imports
    import pandas as pd
    import os

    # read
    meta = pd.read_csv(meta_path, usecols=usecols)

    # handle dtypes
    for col in meta.columns :
        if col == "created_at_ts" :
            meta[col] = pd.to_datetime(meta[col], unit = "ms")
        elif col == "words_count" :
            meta[col] = meta[col].astype("uint16")
        else :
            meta[col] = pd.Categorical(meta[col])

   
    # sort by "created_at_ts"
    if "created_at_ts" in meta.columns :
        meta = meta.sort_values(by = "created_at_ts", ascending=False)

    return meta




def read_clicks_and_meta(clicks_folder_path, meta_path, kept_user_ids = None, clicks_usecols = None, meta_usecols = None, drop_after_prep_cols = None) :
    '''
    load all "clisks_hour_xxx.csv"s and the "articles_metadata.csv", prepare them, and merge them

    parameters :
    ------------
    clicks_folder_path - string : path of the folder containing all .csv
    meta_path - string : path of "articles_metadata.csv"
    kept_user_ids - list_like : for sampling, "user_id"s we want to keep. By default : None (keep all of them)
    clicks_usecols - list of strings or None : list of column names to use in "clicks". By default : None (read all columns)
    meta_usecols - list of strings or None : list of column names to use in "articles_metadata". By default : None (read all columns)
    drop_after_prep_cols - list of strings or None : list of column names to drop after preparation and merging. By default : None (keep all columns)

    return :
    --------
    data - dataframe
    '''

    # import
    import pandas as pd

    # read clicks with custom function
    clicks = read_all_clicks(clicks_folder_path=clicks_folder_path, usecols = clicks_usecols, kept_user_ids=kept_user_ids)
    # drop duplicates (articles present several times for the same user)
    clicks.drop_duplicates(subset=["user_id", "click_article_id"], inplace=True, keep="last")

    # read meta with custom function
    meta = read_meta(meta_path=meta_path, usecols = meta_usecols)

    # merge
    data = pd.merge(
        left=clicks,
        right=meta,
        left_on="click_article_id",
        right_on="article_id",
        how="left"
    )

    # remove of unnecessary cols
    if drop_after_prep_cols : 
        data.drop(labels=drop_after_prep_cols, inplace=True, axis=1)

    return data






def my_seniority_rating(ts_series, ref_date, min_seniority=None, max_seniority=None) :
    '''
    for collaborative filtering, create a "rating" feature based on the seniority of the interactions

    parameters :
    ------------
    ts_series - Series : "click_timestamp" feature, from "clicks" CSVs
    ref_date - datatime : reference date, reference date, from which seniority is measured
    min_seniority - int : minimum seniority of each click. By default : None (duration between dates is note lower-bounded)
    max_seniority - int : maximum seniority of each click. By default : None (duration between dates is note upper-bounded)

    return :
    --------
    seniority_rating - Series

    '''
    # imports
    import pandas as pd

    # compute difference between date, the clicks seniorities, in days
    seniority_rating = (ref_date - ts_series).dt.days
    # clip
    seniority_rating = seniority_rating.clip(lower=min_seniority, upper=max_seniority)
    # 1 - normalise
    # rating in [0, 1]
    # 0 for oldest interactions, 1 for most recent ones
    seniority_rating = 1 - (seniority_rating - seniority_rating.min()) / (seniority_rating.max() - seniority_rating.min())

    return seniority_rating





def my_popularity_rating(article_series, min_count_for_best = None) :
    '''
    for collaborative filtering, create a "rating" feature based on the normalized number of occurrences of the article

    parameters :
    ------------
    article_series - Series : "click_article_id" feature, from "clicks" CSVs
    min_count_for_best - int : minimum number of occurrences of an article to obtain maximum popularity. By default : None (max popularity for max count)

    return :
    --------
    popularity_rating - Series

    '''
    # imports
    import pandas as pd

    # "article_id" value_count
    article_counter = article_series.astype(int).value_counts()
    # normalise
    # first, limit the number of occurrences to min_count_for_best
    article_counter.clip(0, min_count_for_best)
    article_counter_norm = (article_counter - article_counter.min())/(article_counter.max()-article_counter.min())
    # create rating
    popularity_rating = article_series.astype(int).apply(lambda x : article_counter_norm.loc[x])

    return popularity_rating







def my_interaction_rating(user_series, article_series) :
    '''
    for collaborative filtering, create a "rating" feature based on the inverse of the number of articles read by the user

    parameters :
    ------------
    user_series - Series : "user_id" feature, from "clicks" CSVs
    article_series - Series : "click_article_id" feature, from "clicks" CSVs

    return :
    --------
    interaction_rating - Series

    '''
    # imports
    import pandas as pd

    # contat
    df = pd.concat(objs=[user_series, article_series], axis=1)
    df.columns = ["user_id", "article_id"]

    # get the number of articles for each user
    aggDict = {"article_id" : ["nunique"]}
    n_articles_by_uid = df.groupby(by=["user_id"], observed=True).agg(func=aggDict)

    # create rating : 1/n_read_articles 
    interaction_rating = df["user_id"].astype(int).apply(lambda x : 1/n_articles_by_uid.loc[x].values[0])

    return interaction_rating






def read_clicks_and_prepare_rating(clicks_folder_path, ref_date_for_seniority, kept_user_ids=None, min_seniority=None, max_seniority=None, minimum_count=None, min_count_for_best=None) :
    '''
    for collaborative filtering, 
        - load all "clisks_hour_xxx" .csv and put them in a unique dataframe
        - create a "rating" feature based on the seniority of the interactions and the inverse of the number of articles read by the user

    parameters :
    ------------
    clicks_folder_path - string : path of the folder containing all .csv
    ref_date - datatime : reference date, reference date, from which seniority is measured
    kept_user_ids - list_like : for sampling, "user_id"s we want to keep. By default : None (keep all of them)
    min_seniority - int : minimum seniority of each click. By default : None (duration between dates is note lower-bounded)
    max_seniority - int : maximum seniority of each click. By default : None (duration between dates is note upper-bounded)
    minimum_count - int : minimum number of times an article must have been consulted. By default : None (we keep all observations)
    min_count_for_best - int : minimum number of occurrences of an article to obtain maximum popularity. By default : None (max popularity for max count)

    return :
    --------
    data - dataframe : containing "user_id", "article_id", "rating", for surprise compatibility
    '''

    # imports
    import pandas as pd

    # use custom function to load "clicks"
    data = read_all_clicks(
        clicks_folder_path=clicks_folder_path, 
        usecols=["user_id", "click_article_id", "click_timestamp"],
        kept_user_ids=kept_user_ids
        )
    
    # drop duplicates (articles present several times for the same user)
    # we keep "last", the oldest (data is sorted by "click_timestamp")
    data.drop_duplicates(subset=["user_id", "click_article_id"], inplace=True, keep="last")

    # delete articles that have been read less than "minimum_count"
    if minimum_count :
        article_counter = data["click_article_id"].value_counts()
        article_counter_filtered = article_counter.loc[article_counter >= minimum_count]
        mask = data["click_article_id"].isin(article_counter_filtered.index)
        data = data.loc[mask]


    # use custom function to create seniority_rating
    data["seniority_rating"] = my_seniority_rating(
        ts_series=data["click_timestamp"],
        ref_date=ref_date_for_seniority,
        min_seniority=min_seniority,
        max_seniority=max_seniority
    )

    # use custom function to create my_popularity rating
    data["popularity_rating"] = my_popularity_rating(
        article_series=data["click_article_id"],
        min_count_for_best=min_count_for_best
    )

    # use custom function to create my_interaction_rating
    data["interaction_rating"] = my_interaction_rating(
        user_series=data["user_id"],
        article_series=data["click_article_id"]
    )

    # create final rating, the geometric mean
    data["rating"] = (data[["seniority_rating", "popularity_rating", "interaction_rating"]].prod(axis=1))**(1/3)

    # drop useless columns
    data = data[["user_id", "click_article_id", "rating"]]

    # rename "click_article_id"
    data.rename(columns={"click_article_id" : "article_id"}, inplace=True)

    # reset index
    data.reset_index(drop=True, inplace=True)

    return data






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




def fit_and_save(X_surprise, model_surprise, path) :
    '''
    train and save a surprise model

    parameters :
    ------------
    X_surprise - surprise dataset
    model_surprise - surprise algorithm instance
    path - string
    '''
    # imports
    import os
    from joblib import dump
    # build trainset
    X = X_surprise.build_full_trainset()
    # fit
    model_surprise.fit(X)
    # dump model
    dump(model_surprise, path)



    


def embedding_reduction(emb_path, n_components, articles_ids = None) :
    '''
    load articles embeddings, performs preprocessing and PCA

    parameters :
    ------------
    emb_path - string
    n_components - int or float : number of dimensions of PCA, or ratio of explained variance (float in [0, 1])
    articles_ids - listlike : ids of articles one wants to keep. By default : None (keep all rows/articles)
    '''
    # imports
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    # load
    emb = pd.read_pickle(emb_path)

    # select wanted articles
    if articles_ids is not None :
        emb = emb[articles_ids]

    # preprocessing
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)
    # perform PCA
    pca = PCA(n_components=n_components)
    principal_comps = pca.fit_transform(emb_scaled)

    return principal_comps





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

