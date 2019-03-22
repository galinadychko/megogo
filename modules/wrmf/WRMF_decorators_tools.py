import pandas as pd
import numpy as np
from multiprocessing import Pool


def parse_prediction_unit(line):
    data = pd.DataFrame(columns=["user", "items", "preferences"])
    for elem in line:

        user_info = {}
        items = []
        preferences = []
        
        rest = elem[1][1:len(elem[1]) - 1]

        if rest != "":
            for el in rest.split(","):
                sp = el.split(":")
                items.append(int(sp[0]))
                preferences.append(float(sp[1]))
        else:
            pass
        user_info["items"] = np.array(items)
        user_info["preferences"] = np.array(preferences)
        user_info["user"] = np.array([elem[0]] * len(items))
        data = data.append(pd.DataFrame(user_info), sort=False)
    return data


def parse_prediction_in_parallel(prediction_path, save_path, max_threads):
    with open(prediction_path) as f:
        line = [elem[:-1].split("\t") for elem in f.readlines()]
    f.close()
    line_parts_list = np.array_split(np.array(line), max_threads)
    results = pd.DataFrame(columns=["user", "items", "preferences"])
    with Pool(max_threads) as p:
        res = p.map(parse_prediction_unit, line_parts_list)
        results = results.append(res, sort=False)
    results.to_csv(save_path, header=False, index=False)


def parse_prediction_no_parallel(prediction_path, save_path):
    with open(prediction_path) as f:
        line = [elem[:-1].split("\t") for elem in f.readlines()]
    f.close()

    data = []
    for elem in line:

        user_info = {}
        items = []
        preferences = []

        user_info["user"] = elem[0]
        rest = elem[1][1:len(elem[1]) - 1]

        if rest != "":
            for el in rest.split(","):
                sp = el.split(":")
                items.append(int(sp[0]))
                preferences.append(float(sp[1]))
        else:
            pass

        user_info["items"] = np.array(items)
        user_info["preferences"] = np.array(preferences)
        data.append(user_info)
    df = pd.concat(pd.DataFrame(elem) for elem in data)
    df.to_csv(save_path, header=False, index=False)


def prepare_file(file_path, save_path):
    with open(file_path) as f:
        line = [elem[:-1].split(",") for elem in f.readlines()]
    f.close()
    line_all = [elem[2] for elem in line]
    final_lines = [elem[:3] for elem in line]
    try:
        np.array(line_all).astype(np.float)
    except ValueError:
        print("Caution! Item id is not a float. Empty result will be returned.")
        return pd.DataFrame(columns=["revieverID", "asin", "overall"])
    (pd.DataFrame(final_lines,
                  columns=["reviewerID", "asin", "overall"])).to_csv(save_path,
                                                                     sep="\t",
                                                                     header=False, index=False)