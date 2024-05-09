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





def read_all_clicks(clicks_folder_path, usecols = None) : 
    '''
    load all "clisks_hour_xxx" .csv and put them in a unique dataframe

    parameters :
    ------------
    clicks_folder_path - string : path of the folder containing all .csv
    usecols - list of strings or None : list of column names to use. By default : None (read all columns)

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




def read_clicks_and_meta(clicks_folder_path, meta_path, clicks_usecols = None, meta_usecols = None, drop_after_prep_cols = None) :
    '''
    load all "clisks_hour_xxx.csv"s and the "articles_metadata.csv", prepare them, and merge them

    parameters :
    ------------
    clicks_folder_path - string : path of the folder containing all .csv
    meta_path - string : path of "articles_metadata.csv"
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
    clicks = read_all_clicks(clicks_folder_path=clicks_folder_path, usecols = clicks_usecols)

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







def my_interaction_rating(user_series, article_series) :
    '''
    for collaborative filtering, create a "rating" feature based on the inverse of the number of articles read by the user

    parameters :
    ------------
    ts_series - Series : "click_timestamp" feature, from "clicks" CSVs
    ref_date - datatime : reference date, reference date, from which seniority is measured
    min_seniority - int : minimum seniority of each click. By default : None (duration between dates is note lower-bounded)
    max_seniority - int : maximum seniority of each click. By default : None (duration between dates is note upper-bounded)

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
    interaction_rating = df["user_id"].apply(lambda x : 1/n_articles_by_uid.loc[x].values[0])

    return interaction_rating




def read_clicks_and_prepare_rating(clicks_folder_path, ref_date_for_seniority, min_seniority=None, max_seniority=None) :
    '''
    for collaborative filtering, 
        - load all "clisks_hour_xxx" .csv and put them in a unique dataframe
        - create a "rating" feature based on the seniority of the interactions and the inverse of the number of articles read by the user

    parameters :
    ------------
    clicks_folder_path - string : path of the folder containing all .csv
    ref_date - datatime : reference date, reference date, from which seniority is measured
    min_seniority - int : minimum seniority of each click. By default : None (duration between dates is note lower-bounded)
    max_seniority - int : maximum seniority of each click. By default : None (duration between dates is note upper-bounded)

    return :
    --------
    data - dataframe : containing "user_id", "article_id", "rating", for surprise compatibility
    '''

    # imports
    import pandas as pd

    # use custom function to load "clicks"
    data = read_all_clicks(
        clicks_folder_path=clicks_folder_path, 
        usecols=["user_id", "click_article_id", "click_timestamp"]
        )

    # use custom function to create seniority_rating
    data["seniority_rating"] = my_seniority_rating(
        ts_series=data["click_timestamp"],
        ref_date=ref_date_for_seniority,
        min_seniority=min_seniority,
        max_seniority=max_seniority
    )

    # use custom function to create my_interaction_rating
    data["interaction_rating"] = my_interaction_rating(
        user_series=data["user_id"],
        article_series=data["click_article_id"]
    )

    # drop duplicates (articles present several times for the same user)
    # we keep "last", the oldest (data is sorted by "click_timestamp")
    data.drop_duplicates(subset=["user_id", "click_article_id"], inplace=True, keep="last")

    # create final rating, the mean
    data["rating"] = data[["seniority_rating", "interaction_rating"]].mean(axis=1)

    # drop useless columns
    data = data[["user_id", "click_article_id", "rating"]]

    # rename "click_article_id"
    data.rename(columns={"click_article_id" : "article_id"}, inplace=True)

    # reset index
    data.reset_index(drop=True, inplace=True)

    return data