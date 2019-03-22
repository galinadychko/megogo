from sklearn import preprocessing

import pandas as pd
import numpy as np
import os

from multiprocessing import Pool
from functools import partial

def read_data(path_to_data):
    """
    Read lines from data file, check if first line is a header.
    :param path_to_data: string; path to file with data
    :return: pandas data frame; format: "reviewerID" "asin" "overall"
    """
    with open(path_to_data) as f:
        line = [elem[:-1].split(",") for elem in f.readlines()]
    f.close()
    line_all = [elem[2] for elem in line]
    final_lines = [elem[:3] for elem in line]
    try:
        np.array(line_all).astype(np.float)
    except ValueError:
        print("Caution! Item id is not a float. Empty result will be returned.")
        return pd.DataFrame(columns=["revieverID", "asin", "overall"])
    return pd.DataFrame(final_lines, columns=["reviewerID", "asin", "overall"])

def read_data_with_time(path_to_data):
    """
    Read lines from data file, check if first line is a header.
    :param path_to_data: string; path to file with data
    :return: pandas data frame; format: "reviewerID" "asin" "overall"
    """
    with open(path_to_data) as f:
        line = [elem[:-1].split(",") for elem in f.readlines()]
    f.close()
    line_all = [elem[2] for elem in line]
    final_lines = [elem[:4] for elem in line]
    try:
        np.array(line_all).astype(np.float)
    except ValueError:
        print("Caution! Item id is not a float. Empty result will be returned.")
        return pd.DataFrame(columns=["revieverID", "asin", "overall", "time"])
    return pd.DataFrame(final_lines, columns=["reviewerID", "asin", "overall", "time"])


def encode_train(train_file, var_name):
    """
    Encode column values of train file to new integer labels.
    :param train_file: pandas data frame; train data frame
    :param var_name: string; name of column to encode
    :return: pandas data frame; with changed appropriate column
             LabelEncoder(); fitted by train appropriate column
    """
    if train_file.empty:
        return train_file
    train_ = train_file.copy()
    le_reviewer = preprocessing.LabelEncoder()
    le_reviewer.fit(sorted(train_file[var_name].unique()))
    train_.loc[:, var_name] = le_reviewer.transform(train_file[var_name])
    return train_, le_reviewer


def decode_train(train_df, var_name, preproc_train):
    """
    Decode column values, which was encoded by encode_train function.
    :param train_df: pandas data frame; train data frame
    :param var_name: string; name of column to decode
    :param preproc_train: LabelEncoder; fitted appropriate train column
    :return: pandas data frame; with appropriate decoded column
    """
    if train_df.empty:
        return train_df
    train_ = train_df.copy()
    train_.loc[:, var_name] = preproc_train.inverse_transform(train_df[var_name])
    return train_


def encode_test(test_file, train_file, var_name, le_reviewer):
    """
    Encode test column according to the appropriate encoded train column,
    if there is a new value, the custom dictionary encoding is used.
    :param test_file: pandas data frame; test or validation data set
    :param train_file:pandas data frame; train data set
    :param var_name: string; variable name for encoding
    :param le_reviewer:  LabelEncoder; label encoder fitted by the appropriate train column
    :return: pandas data frame; data set with encoded appropriate column;
             dictionary; key and values custom encoding.
    """
    if test_file.empty or train_file.empty:
        return test_file
    test_ = test_file.copy()
    val_diff = list(set(test_file[var_name].unique()) - set(train_file[var_name].unique()))
    test_[var_name] = test_file[var_name].astype("str")
    test_.index = test_[var_name]
    n = len(train_file[var_name].unique())
    if val_diff != []:
        new_var_id = list(range(n, n + len(val_diff)))
        dict_new_unique = {}
        for i, var_id in zip(val_diff, new_var_id):
            dict_new_unique[str(var_id)] = str(i)
            test_.loc[str(i), "new_class"] = var_id
        custom_coding, auto_coding = test_[test_["new_class"].notnull()], test_[test_["new_class"].isnull()]
        auto_coding["new_class"] = le_reviewer.transform(auto_coding[var_name])
        final_ = pd.concat([custom_coding, auto_coding])
        final_[var_name] = final_.new_class.astype("int")
        del final_["new_class"]
        final_.reset_index(drop=True)
        final_.index = range(0, final_.shape[0])
        return final_, dict_new_unique
    else:
        auto_coding = test_
        auto_coding["new_class"] = le_reviewer.transform(auto_coding[var_name])
        final_ = auto_coding
        final_[var_name] = final_.new_class.astype("int")
        del final_["new_class"]
        final_.reset_index(drop=True)
        final_.index = range(0, final_.shape[0])
        return final_, {}


def decode_test(test_file, var_name, dict_new_unique, le_reviewer):
    """
    Decode test file which was encoded by encode_test.
    :param test_file: pandas data frame; test set
    :param var_name: string; variable name to decode
    :param dict_new_unique: dictionary; dictionary with custom encoding
    :param le_reviewer: LabelEncoder; label encoder fitted by the appropriate train column
    :return: pandas data frame; test set with decoded column appropriately
    """
    if test_file.empty:
        return test_file
    new_test_ = test_file.copy()
    new_test_[var_name] = new_test_[var_name].astype("int")
    new_test_.index = new_test_[var_name]
    try:
        new_test_custom = new_test_.loc[
                          set(map(int, new_test_.index)).intersection(set(map(int, list(dict_new_unique.keys())))), :]
    except KeyError:
        new_test_custom = pd.DataFrame(columns=list(new_test_.columns))
    try:
        new_test_auto = new_test_.loc[
                        list(set(map(int, new_test_.index)) - set(map(int, list(dict_new_unique.keys())))), :]
    except KeyError:
        new_test_auto = pd.DataFrame(columns=list(new_test_.columns))
    if not new_test_auto.empty:
        new_test_auto.loc[:, var_name] = le_reviewer.inverse_transform(list(map(int, list(new_test_auto[var_name]))))
    if not new_test_custom.empty:
        for i, row in new_test_custom.iterrows():
            new_test_custom.loc[i, var_name] = dict_new_unique[str(int(row[var_name]))]
    decoded_test = pd.concat([new_test_custom, new_test_auto], ignore_index=True)
    return decoded_test

def decode_test_in_parallel(df, var_name, dict_new_unique, le_reviewer, max_threads):
    df__ = df.copy()
    df__.index = df__.reviewerID
    list_range_across = np.array_split(df__.reviewerID.unique(), max_threads)
    all_parts = []
    results = pd.DataFrame(columns=["user", "items", "ratings"])
    for user_ids in list_range_across:
        all_parts.append(df__.loc[user_ids, :])
    with Pool(max_threads) as p:
        prod_x = partial(decode_test, 
                         var_name=var_name, 
                         dict_new_unique=dict_new_unique, 
                         le_reviewer=le_reviewer)
        results = results.append(p.map(prod_x, all_parts))
    print("DECODE in parallel was performed successfully")
    return results


def save_data(cache_directory, file_name, data):
    """
    Save pandas data frame to the appropriate place.
    :param cache_directory: string; place to store to
    :param file_name: string; file name with the appropriate format
    :param data: data frame; to save to
    :return: None
    """
    os.system("if [ ! -d " + cache_directory + " ]; then mkdir -p " + cache_directory + "; fi")
    data.to_csv(cache_directory + "/" + file_name, sep="\t", header=False, index=False)


def prepare_train_file(cache_folder, path_to_train, train_name="train"):
    """
    Make some preparation for train set: read data appropriately, encode it and save into the file.
    :param cache_folder: string: place to store
    :param path_to_train: string; place where train file is located
    :return: pandas data frame; train set with initial data
            pandas data frame; train set with encoded reviewerID and itemID
            LabelEncoder; encoder for reviewerID
            LabelEncoder; encoder for itemID

    """
    train = read_data(path_to_train)
    new_train, le_reviewerID = encode_train(train, "reviewerID")
    new_train, le_asin = encode_train(new_train, "asin")
    save_data(cache_folder + "/data", train_name, new_train)
    print("The data was successfully prepared for training")
    return train, new_train, le_reviewerID, le_asin

def prepare_train_file_with_time(cache_folder, path_to_train, train_name="train"):
    """
    Make some preparation for train set: read data appropriately, encode it and save into the file.
    :param cache_folder: string: place to store
    :param path_to_train: string; place where train file is located
    :return: pandas data frame; train set with initial data
            pandas data frame; train set with encoded reviewerID and itemID
            LabelEncoder; encoder for reviewerID
            LabelEncoder; encoder for itemID

    """
    train = read_data_with_time(path_to_train)
    new_train, le_reviewerID = encode_train(train, "reviewerID")
    new_train, le_asin = encode_train(new_train, "asin")
    save_data(cache_folder + "/data", train_name, new_train)
    print("The data was successfully prepared for training")
    return train, new_train, le_reviewerID, le_asin


def item_dstatistics(train_df):
    """
    Count mean and std for each items from train data
    :param train_df: pandas data frame; train data set
    :return: dictionary; with descriptive statistics in the format: {itemID1: [mean1, std1], ..., itemIDK: [meanK, stdK]}
    """
    tpgrp = train_df.groupby("asin")
    tpgrpdict = dict(list(tpgrp))
    d = {}
    for k, v in zip(list(tpgrpdict.keys()), list(tpgrpdict.values())):
        d[k] = [v.overall.astype("float").values.mean(), v.overall.astype("float").values.std()]
    return d


def sort_approriate(data, dict_of_items):
    """
    Sort items by rating.
    If rating the same - by item average.
    If average is the same - by std.
    If std is the same - by item_index.
    If item is new for train file - ordinary sorting.
    :param data: dictionary {"user": , "items": np.array([...]), "ratings": np.array([...])}
    :param dict_of_items: dictionary {item1: [item1_mean, item1_std], item2: [item2_mean, item1_std], ...}
    :return: dictionary {"user": , "items": np.array([...]), "ratings": np.array([...])}
    """
    if str(type(data)) != "<class 'dict'>" or str(type(dict_of_items)) != "<class 'dict'>":
        raise Exception("Caution! Data and dict_of_items must be dictionaries!")
    df_char = pd.DataFrame.from_dict(dict_of_items).transpose()
    user = pd.DataFrame.from_dict(data)
    df_char.columns = ["average", "std"]
    df_char["items"] = df_char.index
    #try:
        #df_char["items"], user["items"] = df_char.index.astype("int"), user["items"].astype(int)
    #except:
        #raise Exception(
            #"Caution! Type conversion error!\nItems in data or in dict_of_items cannot be conversioned to int!")
    all_ = pd.merge(user, df_char, on=["items"])
    if all_.empty or all_.shape[0] < len(data["items"]):
        print("Caution! Function returns ordinary sorting!")
        i = data["ratings"].argsort()[::-1]
        d = dict(user=data["user"], items=data["items"][i], ratings=data["ratings"][i])
    else:
        all_.sort_values(by=["ratings", "average", "std"], ascending=[False, False, True], inplace=True)
        d = dict(user=all_["user"].unique()[0], items=all_["items"].values, ratings=all_["ratings"].values)
    return d


def generate_user_file(test_df, cache_folder, folder_name, train_df):
    """
    Create one file for each user from test_df with items, which were not in train_df.
    :param cache_folder: string; path for storing
    :param folder_name: string; solder to store
    :param test_df: pandas data frame; test set, for instance
    :param train_df: pandas data frame; train set, for instance
    :return: None
    """
    os.system("if [ ! -d " + cache_folder + folder_name + " ]; then mkdir -p " + cache_folder + folder_name + "; fi")
    unique_user_test = test_df.reviewerID.unique()
    unique_it_train = train_df.asin.unique()
    for i in range(0, len(unique_user_test)):
        s = set(unique_it_train) - set(train_df[train_df.reviewerID == unique_user_test[i]].asin.values)
        d = pd.DataFrame.from_dict(dict(reviewerID=[unique_user_test[i]] * len(s), asin=list(s), overall=0))
        name = cache_folder + folder_name + "/" + str(unique_user_test[i])
        d.to_csv(name, index=False, header=False, sep="\t", columns=["reviewerID", "asin", "overall"])
    print("The data for prediction was successfully generated")


def generate_user_file_in_parallel(max_threads, cache_folder, folder_name, test_df, train_df):
    """
    Split test_data into max_threads parts and generate files for users in parallel
    :param max_threads: int: number of threads to use
    :param cache_folder: string: place to store path
    :param folder_name: string: name of folder to store generated files
    :param test_df: pandas df: data frame from which users should be taken
    :param train_df: pandas df: data frame for deletion already reviewed items
    :return: None
    """
    test_df_ = test_df.copy()
    test_df_.index = test_df_.reviewerID
    list_range_across = np.array_split(test_df_.reviewerID.unique(), max_threads)
    all_parts = []
    for user_ids in list_range_across:
        all_parts.append(test_df_.loc[user_ids, :])
    with Pool(max_threads) as p:
        prod_x = partial(generate_user_file,
                         cache_folder=cache_folder,
                         folder_name=folder_name,
                         train_df=train_df)
        p.map(prod_x, all_parts)
    print("Files were generated successfully")

def generate_file_list_name(list_name):
    """
    Make a string of files` names, separated by \t
    :param list_name: list; list of names;
    :return: string; all elements of list written in a raw
    """
    res=""
    for elem in list_name:
        res+=str(elem)+" "
    return res


# Mykola Trohymovych:
# Function for transformation dataset in form of list of {"user" : , "items": np.array([...]), "ratings": np.array([...])}
# It transforms dataset to friendly, appropriate for Recommender.py form
# Input: test - pandas dataframe,
#        user_var_name - name of column of users,
#        item_var_name - name of column of items,
#        rating_var_name - name of column of ratings.
# Output: list of dictionaries in form of {"user" : , "items": np.array([...]), "ratings": np.array([...])}
def transform(test, user_var_name, item_var_name, rating_var_name):
    users_in_dataset = test[user_var_name].unique()
    testset = []
    for user_id in users_in_dataset:
        d = {}
        d['user'] = user_id
        d['items'] = []
        d['ratings'] = []
        inf = test[test[user_var_name] == user_id]
        items = np.array(inf[item_var_name].values)
        ratings = np.array(inf[rating_var_name].values)
        d['items'] = np.array(items)
        d['ratings'] = np.array(ratings)
        testset.append(d)
    return testset
