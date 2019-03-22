import os
import numpy as np


def recommender_parameters(params_dict):
	options = ""
	for elem in params_dict:
		options += elem + "=" + str(params_dict[elem]) + " "
	if options.strip() != "":
		options = " --recommender-options=" + "\'" + options + "\'"
	return options


def predict_file(path_to_mymedialite, train_path, test_path, users_path, prediction_path,
				 recommender_name, k, random_seed, model_path, params):
	os.system("cd " + path_to_mymedialite + " &&" +
		" ./bin/item_recommendation" +
        " --file-format=DEFAULT" +
        " --training-file=" + train_path +
        " --test-file=" + test_path +
        " --test-users=" + users_path +
        " --recommender=" + recommender_name + " --random-seed=" + str(random_seed) +
        " --predict-items-number=" + str(k) +
        " --prediction-file=" + prediction_path +
        " --load-model=" + model_path + 
        params)


def parse_array(u_info):
    l = np.array([elem.split(" ") for elem in list(u_info)])
    d = {}
    d["internal_id"] = l[0, 0]
    for i in range(len(l)):
    	d[i] = l[i, 2]
    return d


def parse_feature(feat_array, nelem):
	splited_u = np.array_split(feat_array, nelem)
	dict_u = np.apply_along_axis(parse_array, 1, np.array(splited_u))
	return dict_u


def parse_model(path, num_user, num_item, num_feat):
    with open(path) as f:
        line = [elem[:-1] for elem in f.readlines()]
    f.close()
    i = [line.index(elem) for elem in line if elem == (str(num_item) + " " + str(num_feat))][0]
    user_feat = np.array(line[3:(3+num_user*num_feat)])
    item_feat = np.array(line[(i + 1):(i+1 + num_item * num_feat)])
    del line
    dict_u = parse_feature(user_feat, num_user)
    dict_i = parse_feature(item_feat, num_item)
    

    return dict_u, dict_i