from modules.BMF_tools import *
from modules.MML_istalation import *

import pandas as pd
import numpy as np
import os

from itertools import chain
import multiprocessing
from multiprocessing import Pool
from functools import partial
import pickle

import warnings
warnings.filterwarnings("ignore")

max_threads = multiprocessing.cpu_count()-1
#max_threads=1

class BMF:
    def __init__(self, path_to_train="/home/g_dychko/downloads/MovieLens/train_real_entrance.csv",
                 mymedialite_folder="/home/g_dychko/libraries/MyMediaLite",
                 cache_folder="/home/g_dychko/cache"):
        self.path_to_train = path_to_train
        self.mymedialite_folder = mymedialite_folder
        self.cache_folder = cache_folder
        self.rec_options_names = ["num_factors", "bias_reg", "reg_u_i", "frequency_regularization",
                                  "learn_rate", "bias_learn_rate", "num_iter", "bold_driver", "loss",
                                  "naive_parallelization"]

    def clean_cache(self):
        """
        Check if the cache directory exists and delete if enything is already stored there.
        If there is no such direcrtoy it will be created.
        :return: None
        """
        os.system("if [ -d " + self.cache_folder + " ]; then rm -rf " + self.cache_folder + " && mkdir -p " + self.cache_folder + "; fi")
        os.system("if [ ! -d "+self.cache_folder+" ]; then mkdir -p " + self.cache_folder+"; fi")

    def clean_before_rec_with_cache(self):
        """
        Check if the cache directory with prediction files and data for new(according to train set) users exists,
        delete if enything is already stored there.
        :return: None
        """
        os.system("if [ -d " + self.cache_folder+"/prediction/prediction_for_new"+" ]; then rm -rf "
                  + self.cache_folder+"/prediction/prediction_for_new"+" && mkdir -p "
                  + self.cache_folder+"/prediction/prediction_for_new"+"; fi")
        os.system("if [ -d " + self.cache_folder+"/data/user_not_in_cache"+" ]; then rm -rf "
                  + self.cache_folder+"/data/user_not_in_cache"+" && mkdir -p "
                  + self.cache_folder+"/data/user_not_in_cache"+"; fi")
        os.system("if [ ! -d " + self.cache_folder+"/prediction/prediction_for_new"+" ]; then mkdir -p "
                  + self.cache_folder+"/prediction/prediction_for_new"+"; fi")
        os.system("if [ ! -d " + self.cache_folder+"/data/user_not_in_cache"+" ]; then mkdir -p "
                  + self.cache_folder+"/data/user_not_in_cache"+"; fi")



    def clean_before_rec(self):
        """
        Check if the cache directory with data and prediction files exists,
        delete if enything is already stored there.
        :return: None
        """
        os.system("if [ -d " + self.cache_folder+"/prediction/prediction_for_new"+" ]; then rm -rf "
                  + self.cache_folder+"/prediction/prediction_for_new"+" && mkdir -p "
                  + self.cache_folder+"/prediction/prediction_for_new"+"; fi")
        os.system("if [ ! -d " + self.cache_folder+"/data/unique_user_data"+" ]; then mkdir -p "
                  + self.cache_folder+"/data/unique_user_data"+"; fi")



    def checking_parameters(self):
        """
        Checking if the model parameters for training is given,
        try  to convert into integer number of factors and niterations.
        Make a string in appropriate format for MyMediaLite model.
        Set default number of threads (max - 1), the same parameter for user and item regularisation.
        :return: string; with the given parameters in the format: "num_factors=10 bias_reg=0.01 ... naive_parallelization=False"
        """
        try:
            self.max_threads=max_threads
        except:
            self.max_threads = 1
        rec_options=""
        #rec_options = "max_threads="+str(self.max_threads)+" "
        for opt_names in self.rec_options_names:
            try:
                if self.rec_options_var[opt_names] is not None:
                    if opt_names == "reg_u_i":
                        rec_options += "reg_u={} reg_i={}".format(str(self.rec_options_var[opt_names]),
                                                                 str(self.rec_options_var[opt_names]))+" "
                    elif opt_names in ["num_factors", "num_iter"]:
                        rec_options += opt_names + "=" + str(int(self.rec_options_var[opt_names])) + " "
                    else:
                        rec_options += opt_names + "=" + str(self.rec_options_var[opt_names]) + " "
                else:
                    rec_options += ""
            except:
                rec_options += ""
        return rec_options

    def train_model(self, model_name, train_name):
        """
        Run the train and the save commands for MyMediaLite model with a terminal help,
        using given (or default) parameters,
        use train file (which was save to the "/data" cache folder after preprocessing).
        Create folder "/model" if it does not exist yet and save there the model.
        :param model_name: string; the name for saving model
        :param train_name: string; the name of train dataset
        :return: None
        """
        os.system("if [ ! -d " + self.cache_folder + "/model" + " ]; then mkdir -p " + self.cache_folder + "/model" + "; fi")
        self.rec_options = self.checking_parameters()
        if self.rec_options != "":
            os.system(self.mymedialite_folder + "/bin/rating_prediction --training-file="
                      + self.cache_folder + "/data/"+train_name+
                      " --recommender=BiasedMatrixFactorization" +
                      " --recommender-options=" + "\'{}\'".format(self.rec_options) +
                      " --random-seed="
                      + str(self.random_seed) +
                      " --save-model=" + self.cache_folder + "/model/" + model_name)
        else:
            os.system(self.mymedialite_folder + "/bin/rating_prediction --training-file="
                      + self.cache_folder + "/data/"+train_name+
                      " --recommender=BiasedMatrixFactorization" +
                      " --random-seed="
                      + str(self.random_seed) +
                      " --save-model=" + self.cache_folder + "/model/" + model_name)

    def train(self,
              parameters=dict(random_seed=42, num_factors=10, bias_reg=0.01, reg_u_i=0.015,
                              frequency_regularization=False, learn_rate=0.01, bias_learn_rate=1,
                              num_iter=30, bold_driver=False, loss="RMSE",
                              naive_parallelization=False),
              model_name="BiasedMatrixFactorisation",
              train_name="train"):
        """
        Final method for model trainig with the input (or default) parameters.
        Convert train file to the appropriate format for MyMediaLite,
        save as a train file in "/data" folder,
        train and save the trained model in "/model" folder.
        If random seed cannot be converted to the int type, random seed is set to 42.
        :param parameters: dictionary:
                            :param random_seed: int or float; random seed
                            :param num_factors: int; number of the latent users and item features
                            :param bias_reg: float; regularisation parameter for the general bias
                            :param reg_u_i: float; regularisation parameter for the user and item biases
                            :param frequency_regularization: boolean; include or not the frequency factor to the model
                            :param learn_rate: float; learning rate
                            :param bias_learn_rate: int; learn rate factor for the bias terms
                            :param num_iter: int; number of iteration for optimisation problem
                            :param bold_driver: boolean; indicate if use bold driver heuristics for learning rate adaption
                            :param loss: string; loss for the last iteration, used by bold driver heuristics
                            :param max_threads: int; number of threads to use
                            :param naive_parallelization: boolean; the exact sequence of updates depends on the
                                                          thread scheduling
        :param: model_name: string; the model name, which will be used
        :param: train_name: string; the train dataset`s name
        :return: None
        """
        self.rec_options_var = parameters
        try:
            self.random_seed = int(parameters["random_seed"])
        except:
            self.random_seed = 42
        self.train, self.new_train, self.le_reviewerID, self.le_asin = prepare_train_file(self.cache_folder,
                                                                                          self.path_to_train,
                                                                                          train_name)
        self.train_model(model_name, train_name)
        print("The model was successfully trained")

    def predict_from_path_to_path(self, list_dir_data, data_path, path_to_store):
        """
        Take files from data_path and store predictions to the files in the path_to_store.
        :param list_dir_data: list; with files names.
        :param data_path: string; path to data (where files from list_dir_data locate)
        :param path_to_store: string; folder to which store
        :return: None
        """
        for u in list_dir_data:
            self.file_prediction(data_path, u, path_to_store, u)

    def train_with_cache(self,
                         parameters=dict(random_seed=42, num_factors=10, bias_reg=0.01, reg_u_i=0.015,
                                         frequency_regularization=False, learn_rate=0.01, bias_learn_rate=1,
                                         num_iter=30, bold_driver=False, loss="RMSE",
                                         naive_parallelization=False)):
        """
        Train model as "train" method,
        create and save data sets in format: one file corresponds one user,
        each user file contains items id of not raited(in "train" file) products,
        save the predictions as separated files for each user (from the train file)
        with the predicted items ratings in "/prediction_cache".
        :param parameters: the same as in train function
        :return: None
        """
        os.system("if [ ! -d " + self.cache_folder + "/data/train_cache" + " ]; then mkdir -p " +
                  self.cache_folder + "/data/train_cache" + "; fi")
        os.system("if [ ! -d " + self.cache_folder + "/prediction" + " ]; then mkdir -p " +
                  self.cache_folder + "/prediction" + "; fi")
        os.system("if [ ! -d " + self.cache_folder + "/prediction/prediction_cache" + " ]; then mkdir -p " +
                  self.cache_folder + "/prediction/prediction_cache" + "; fi")

        self.rec_options_var = parameters
        try:
            self.random_seed = int(parameters["random_seed"])
        except:
            self.random_seed = 42

        self.train(parameters)
        generate_user_file_in_parallel(self.max_threads, self.cache_folder, "/data/train_cache", self.new_train,
                                       self.new_train)
        self.predict_from_path_to_path(os.listdir(self.cache_folder + "/data/train_cache"),
                                       "/data/train_cache", "/prediction/prediction_cache")
        print("The model was successfully trained with cache")

    def save_trained_bmf(self):
        """
        Save trained example of the previously trained class object with all parameters
        in created "/pickle_trained" folder.
        :return: None
        """
        os.system("if [ ! -d " + self.cache_folder + "/pickle_trained" + " ]; then mkdir -p " + self.cache_folder + "/pickle_trained" + "; fi")
        with open(self.cache_folder + "/pickle_trained/" + "BMF" + '.pickle', 'wb') as f:
            pickle.dump(self, f)

    def load_trained_bmf(self):
        """
        Load the previously saved trained class object
        from the default "/pickle_trained" folder.
        :return: None
        """
        with open(self.cache_folder + "/pickle_trained/" + 'BMF.pickle', 'rb') as f:
            data_new = pickle.load(f)
        return data_new

    def file_prediction(self,
                        test_folder, test_name,
                        save_folder, save_name,
                        model_name="BiasedMatrixFactorisation", train_name="train", max_threads=1):
        """
        The terminal command for making prediction with MyMediaLite model for a test file
        and save it in a file.
        Method uses initial train file, which was initialised with class object.
        :param test_folder: string; folder name, where test file is located
        :param test_name: string; test file name
        :param save_folder: string; folder name, where prediction will be saved
        :param save_name: string; prediction file name
        :param model_name: string; the name of model
        :param train_name: string; the train dataset`s name
        :return: None
        """
        os.system("if [ ! -d " + self.cache_folder + save_folder + " ]; then mkdir -p " + self.cache_folder + save_folder + "; fi")
        if self.rec_options != "":
            os.system(self.mymedialite_folder + "/bin/rating_prediction --training-file="
                      + self.cache_folder + "/data/"+train_name+
                      " --recommender=BiasedMatrixFactorization" +
                      " --test-file="
                      + self.cache_folder + test_folder + "/" + test_name +
                      " --recommender-options=" + "\'{}\'".format(self.rec_options+"max_threads="+str(max_threads)) +
                      " --random-seed="
                      + str(self.random_seed) +
                      " --prediction-file="
                      + self.cache_folder + save_folder + "/" + save_name +
                      " --load-model="
                      + self.cache_folder + "/model/"+model_name)
        else:
            os.system(self.mymedialite_folder + "/bin/rating_prediction --training-file="
                      + self.cache_folder + "/data/"+train_name+
                      " --recommender=BiasedMatrixFactorization" +
                      " --test-file="
                      + self.cache_folder + test_folder + "/" + test_name +
                      " --random-seed="
                      + str(self.random_seed) +
                      " --prediction-file="
                      + self.cache_folder + save_folder + "/" + save_name +
                      " --load-model="
                      + self.cache_folder + "/model/"+model_name)

    def rank_for_one_user(self, data, dict_of_items):
        """
        Get from the all items top-k the most relevant recommendation for specific user
        with special sorting for the same integer rating predictions.
        :param data: dictionary {"user": , "items": np.array([...]), "ratings": np.array([...])}
        :param dict_of_items: dictionary {item1: [item1_mean, item1_std], item2: [item2_mean, item1_std], ...}
        :return: {"user" : , "items": np.array([...]), "ratings": np.array([...])}
        """
        i = (-data["ratings"]).argsort()
        if len(data["items"]) != len(data["ratings"]):
            print("Items and Ratings have different lengths!")
            raise
        else:
            ratings_ = data["ratings"][i]
            if len([(i, list(chain.from_iterable(np.argwhere(i == ratings_))))
                    for ii in np.unique(ratings_) if list(ratings_).count(ii) > 1]) > 0:
                sorted_result = sort_approriate(data, dict_of_items)
                return dict(user=sorted_result["user"], items=sorted_result["items"],
                            ratings=sorted_result["ratings"])
            else:
                return dict(user=data["user"], items=data["items"][i],
                            ratings=data["ratings"][i])

    def rank_file(self, df, d):
        """
        Rank all items for each user from the given data frame.
        :param df: pandas data frame; data set for ranking users items
        :param d: dictionary; with mean and std statistics of items from train set.
        :return: pandas data frame; with the appropriate sorted recommendations
        """
        test_for_rank = transform(df, "reviewerID", "asin", "overall")
        final_df = pd.DataFrame()
        for recommendation in test_for_rank:
            final_df = pd.concat([final_df, pd.DataFrame(self.rank_for_one_user(recommendation, d))], ignore_index=True)
        print("rank_file ENDED")
        return final_df

    def rank_file_in_paralllel(self, df, d):
        """
        Split data frame for rating into parts(number of parts=max threads numbers).
        :param df:pandas data frame: data set for ranking users items
        :param d: dictionary; with mean and std statistics of items from train set.
        :return: pandas data frame; with the appropriate sorted recommendations
        """
        df__ = df.copy()
        df__.index = df__.reviewerID
        list_range_across = np.array_split(df__.reviewerID.unique(), self.max_threads)
        all_parts = []
        results = pd.DataFrame(columns=["user", "items", "ratings"])
        for user_ids in list_range_across:
            all_parts.append(df__.loc[user_ids, :])
        with Pool(self.max_threads) as p:
            prod_x = partial(self.rank_file, d=d)
            results = results.append(p.map(prod_x, all_parts))
        print("Rank in parallel was performed successfully")
        return results

    def rank(self, path_to_test, path_with_name_to_store,
             model_name="BiasedMatrixFactorisation",
             train_name="train", test_name="test", rank_name="rank"):
        """
        Final method to rank in parallel items from test file.
        Count mean and std statistics of items from train set,
        save in the appropriate format, make prediction to the modified test file,
        read it, decode and rank.
        Read data from path for test file, encode reviewers and items,
        :param path_to_test: string; full path with file name to make prediction
        :param path_with_name_to_store; full path with name to store
        :param model_name: string; the model`s name
        :param test_name: string; the name of test data set
        :param rank_name: string; the name of ranked file
        :return: None
        """
        d_ = item_dstatistics(self.train)
        validation = read_data(path_to_test)
        new_test, dict_asin = encode_test(validation, self.train, "asin", self.le_asin)
        self.new_test, dict_new_reviewerID = encode_test(new_test, self.train, "reviewerID", self.le_reviewerID)
        save_data(self.cache_folder + "/data", test_name, self.new_test)
        self.file_prediction("/data", test_name, "/ranking", rank_name,
                             model_name, train_name, self.max_threads)
        pred = pd.read_csv(self.cache_folder + "/ranking/" + rank_name,
                           names=["reviewerID", "asin", "overall"], sep="\t")
        df_ = decode_test_in_parallel(pred, "asin", dict_asin, self.le_asin, self.max_threads)
        df_ = decode_test_in_parallel(df_, "reviewerID", dict_new_reviewerID, self.le_reviewerID, self.max_threads)
        (self.rank_file_in_paralllel(df_, d_)).to_csv(path_with_name_to_store,
                   columns=["user", "items", "ratings"],
                   index=False, header=False)
        print("Ranking was successfully performed")
    
    def action_preparation_before_predict(self, data, separate_mml_folder="/parallel"):
        folders=["/data", "/prediction", "/prediction/prediction_no_cache"]
        
        for folder_name in folders:
            os.system("if [ ! -d "
                      +self.cache_folder+separate_mml_folder+"/"+data[1]+folder_name
                      +" ]; then mkdir -p "
                      +self.cache_folder+separate_mml_folder+"/"+data[1]+folder_name
                      +"; fi")
        
        mml = MML(self.cache_folder+separate_mml_folder)
        mml.make_install("/"+data[1])
        data[0].to_csv(self.cache_folder+separate_mml_folder+"/"+data[1]+"/data/test",
                      header=False,index=False)
#         self.new_train.to_csv(self.cache_folder+separate_mml_folder+"/"+data[1]+"/data/train",
#                               header=False,index=False)
        
    def parallel_preparaton(self, list_data, separate_mml_folder="/parallel"):
        """
        Only for the ranking task.
        Make preparation for each grid search folder in parallel.
        :param separate_mml_folder: string; subfolder for the additional grid search counting
        :return: None
        """
        with Pool(3) as p:
            prod_x = partial(self.action_preparation_before_predict,
                             separate_mml_folder=separate_mml_folder)
            p.map(prod_x, list_data)
    
    def action_for_pedict(self, list_of_files, path_to_list_of_files, 
                          separate_mml_folder, path_to_store):
        self.mymedialite_folder = self.cache_folder+separate_mml_folder+"/"+list_of_files[1]+"/MyMediaLite"
        self.predict_from_path_to_path(os.listdir(self.cache_folder+separate_mml_folder+"/"+list_of_files[1]+"/data"),
                                       separate_mml_folder+"/"+list_of_files[1]+"/data",
                                       separate_mml_folder+"/"+list_of_files[1]+path_to_store)
        df = pd.read_csv(self.cache_folder+separate_mml_folder+"/"+list_of_files[1]+path_to_store+"/test", sep="\t",
                        names=["reviewerID", "asin", "overall"])
        print("Action for predict in parallel was performed successfully")
        return df
        
    def predict_parallel(self, list_data, separate_mml_folder, list_dir_data, data_path, path_to_store):
        results = pd.DataFrame(columns=["reviewerID", "asin", "overall"])
        os.system("if [ ! -d "
                  +self.cache_folder+separate_mml_folder
                  +" ]; then mkdir -p "
                  +self.cache_folder+separate_mml_folder
                  +"; fi")

        with Pool(self.max_threads) as p:
            prod_x = partial(self.action_for_pedict,
                             path_to_list_of_files=data_path,
                             separate_mml_folder=separate_mml_folder,
                             path_to_store=path_to_store)
            results = results.append(p.map(prod_x, list_data))
            p.close()
            p.join()
        print("The results are ready")
        return results
        
    def predict(self, path_to_test, path_with_name_to_store, 
             model_name="BiasedMatrixFactorisation",
             train_name="train", test_name="test", rank_name="rank"):
        """
        """
        validation = read_data(path_to_test)
        
        new_test, dict_asin = encode_test(validation, self.train, "asin", self.le_asin)
        self.new_test, dict_new_reviewerID = encode_test(new_test, self.train, "reviewerID", self.le_reviewerID)
        
        df__ = self.new_test.copy()
        df__.index = df__.reviewerID
        list_range_across = np.array_split(df__.reviewerID.unique(), self.max_threads)
        all_parts = []

        for i, user_ids in enumerate(list_range_across):
            all_parts.append((df__.loc[user_ids, :], str(i)))
        
        self.parallel_preparaton(list_data=all_parts, separate_mml_folder="/parallel")
        
        pred = self.predict_parallel(list_data=all_parts, 
                                     separate_mml_folder="/parallel",
                                     list_dir_data=os.listdir(self.cache_folder+"/data"),
                                     data_path="/data",
                                     path_to_store="/prediction")
        df_ = decode_test_in_parallel(pred, "asin", dict_asin, self.le_asin, self.max_threads)
        df_ = decode_test_in_parallel(df_, "reviewerID", dict_new_reviewerID, self.le_reviewerID, self.max_threads)
        df_.to_csv(path_with_name_to_store,
                   columns=["reviewerID", "asin", "overall"],
                   index=False, header=False)
        print("Prediction was successfully done")

    def rank_no_parallel(self, path_to_test, path_with_name_to_store,
                         model_name="BiasedMatrixFactorisation",
                         train_name="train", test_name="test", rank_name="rank"):
        """
        Final method to rank items from test file without using multiprocessing.
        Count mean and std statistics of items from train set,
        save in the appropriate format, make prediction to the modified test file,
        read it, decode and rank.
        Read data from path for test file, encode reviewers and items,
        :param path_to_test: string; full path with file name to make prediction
        :param path_with_name_to_store; full path with name to store
        :param model_name: string; the model`s name
        :param test_name: string; the name of test data set
        :param rank_name: string; the name of ranked file
        :return: None
        """
        d_ = item_dstatistics(self.train)
        validation = read_data(path_to_test)
        new_test, dict_asin = encode_test(validation, self.train, "asin", self.le_asin)
        self.new_test, dict_new_reviewerID = encode_test(new_test, self.train, "reviewerID", self.le_reviewerID)
        save_data(self.cache_folder + "/data", test_name, self.new_test)
        self.file_prediction("/data", test_name, "/ranking", rank_name,
                             model_name, train_name)
        pred = pd.read_csv(self.cache_folder + "/ranking/" + rank_name, names=["reviewerID", "asin", "overall"], sep="\t")
        df_ = decode_test(pred, "asin", dict_asin, self.le_asin)
        df_ = decode_test(df_, "reviewerID", dict_new_reviewerID, self.le_reviewerID)
        (self.rank_file(df_, d_)).to_csv(path_with_name_to_store,
                                         columns=["user", "items", "ratings"],
                                         index=False, header=False)
        print("Ranking was successfully performed")



    def predict_k_for_user(self, data, dict_of_items, k_recommendations=5):
        """
        Get from the all items top-k the most relevant recommendation for the given user
        with the special sort for the same rating predictions.
        :param data: dictionary {"user": , "items": np.array([...]), "ratings": np.array([...])}
        :param dict_of_items: dictionary {item1: [item1_mean, item1_std], item2: [item2_mean, item1_std], ...}
        :param k_recommendations: integer
        :return: {"user" : , "items": np.array([...]), "ratings": np.array([...])}
        """
        if str(type(k_recommendations)) != "<class 'int'>":
            print("Caution! k_recommendations was coerced to int")
            k_recommendations = int(k_recommendations)
        i = (-data["ratings"]).argsort()
        if len(data["items"]) != len(data["ratings"]):
            print("Items and Ratings have different lengths!")
            raise
        else:
            ratings_ = data["ratings"][i][:k_recommendations]
            if (len(data["items"])) < k_recommendations:
                print("Caution! Not enough elements. Only firsts are used!")
            if len([(i, list(chain.from_iterable(np.argwhere(i == ratings_))))
                    for ii in np.unique(ratings_) if list(ratings_).count(ii) > 1]) > 0:
                sorted_result = sort_approriate(data, dict_of_items)
                return dict(user=sorted_result["user"], items=sorted_result["items"][:k_recommendations],
                            ratings=sorted_result["ratings"][:k_recommendations])
            else:
                return dict(user=data["user"], items=data["items"][i][:k_recommendations],
                            ratings=data["ratings"][i][:k_recommendations])

    def recommend_from_path(self, list_dir_tostore, path_to_store, dict_of_items, k_recommendations=5):
        """
        Sort and choose top k recommendation for each file from path_to_store directory.
        :param list_dir_tostore: list; full paths to files;
        :param path_to_store: string; folder name, where prediction were stored;
        :param dict_of_items: dictionary; mean and std statistics for each item from train set
        :param k_recommendations: int; number of items to chose;
        :return: pandas data frame; with k recommendation for each user
        """
        all_recommendation = pd.DataFrame(columns=["user", "items", "ratings"])
        for u in list_dir_tostore:
            k_ = k_recommendations
            p = pd.read_csv(self.cache_folder + path_to_store + "/" + u, sep="\t",
                            names=["reviewerID", "asin", "overall"])
            if (k_ == None or k_ > len(p.asin.unique())):
                print("Caution! k is not correct. All possible predictions is returned.")
                k_ = len(p.asin.unique())
            decodded_p = decode_test(p, "reviewerID", self.dict_new_reviewerID, self.le_reviewerID)
            decodded_p = decode_test(decodded_p, "asin", self.dict_asin, self.le_asin)

            k_pred = self.predict_k_for_user(dict(user=decodded_p.reviewerID.unique()[0],
                                                  ratings=decodded_p.overall.values,
                                                  items=decodded_p.asin.values), dict_of_items, k_)
            all_recommendation = pd.concat([all_recommendation, pd.DataFrame(k_pred)], ignore_index=True)
        return all_recommendation

    def recommend_from_path_in_parallel(self, list_dir_tostore, path_to_store, dict_of_items, k_recommendations=5):
        """
        Make recommendation for all files from the chosen directory in parallel.
        Split list of all files into max_threads number of files folds
        and perfom recommendation using recommend_from_path
        :param list_dir_tostore: list; full paths to files;
        :param path_to_store: string; folder name, where prediction were stored;
        :param dict_of_items: dictionary; mean and std statistics for each item from train set
        :param k_recommendations: int; number of items to chose;
        :return: pandas data frame; with k recommendation for each user
        """
        list_range_across = np.array_split(list_dir_tostore, self.max_threads)
        results = pd.DataFrame(columns=["user", "items", "ratings"])
        with Pool(self.max_threads) as p:
            prod_x = partial(self.recommend_from_path,
                             path_to_store=path_to_store,
                             dict_of_items=dict_of_items,
                             k_recommendations=k_recommendations)
            results = results.append(p.map(prod_x, list_range_across))
        print("Recommendation in parallel was performed successfully")
        return results

    def recommend(self, path_to_test, path_with_name_to_store, k_recommendations=5):
        """
        Final method to create recommendation.
        Count mean and std statistics of items from train set,
        read test data, encode reviewers and items,
        generate required files for prediction,
        recommend k items for each user (in parallel for each folder of users).
        :param path_to_test: string;
        :param path_with_name_to_store: string; path which included file name for saving result
        :param k_recommendations: int; umber of recommendations to make to
        :return: None
        """
        d = item_dstatistics(self.train)
        validation = read_data(path_to_test)

        new_test, self.dict_asin = encode_test(validation, self.train, "asin", self.le_asin)
        self.new_test, self.dict_new_reviewerID = encode_test(new_test, self.train, "reviewerID", self.le_reviewerID)

        generate_user_file_in_parallel(self.max_threads, self.cache_folder, "/data/unique_user_data", self.new_test,
                                       self.new_train)
        recommendation = self.predict_from_path_to_path_parallel(
                                                separate_mml_folder="/parallel",
                                                list_dir_data=os.listdir(self.cache_folder+"/data/unique_user_data"),
                                                data_path="/data/unique_user_data",
                                                path_to_store="/prediction/prediction_no_cache",
                                                d=d,
                                                k_recommendations=k_recommendations)
        print("Recommmend: Predict from path to path in parallel was done!")
        #self.predict_from_path_to_path(os.listdir(self.cache_folder + "/data/unique_user_data"),
                                       #"/data/unique_user_data", "/prediction/prediction_no_cache")
        recommendation.to_csv(path_with_name_to_store,
                              columns=["user", "items", "ratings"],
                              header=False, index=False)
        print("Recommendaion was successfully written to: %s" %path_with_name_to_store)

    def action_for_prfrom_path_to_path_parallel(self,
                                                list_of_files, path_to_list_of_files,
                                                separate_mml_folder, path_to_store, d, k_recommendations=5):
        """
        Function for the parallel execution of prediction from path to path.
        Clean appropriate folders, install MyMediaLite (max_threads-1 times),
        make prediction for group, make recommendation
        :param list_of_files: list; with files` names
        :param path_to_list_of_files: string; path to all files from list_of_files
        :param separate_mml_folder: string; name of folder to make installation MML into
        :param path_to_store: string: path to store predicted recommendations
        :param d: dictionary: with items characteristics (rating mean, std)
        :param k_recommendations: integer; number of recommendations to make to
        :return: pandas data frame: with appropriate recommendations.
        """
        folders=["", "/data", "/prediction", "/prediction/prediction_no_cache"]
        for folder in folders:
            os.system("if [ -d "
                      +self.cache_folder+separate_mml_folder+"/"+list_of_files[0]+folder
                      +" ]; then rm -rf "
                      +self.cache_folder+separate_mml_folder+"/"+list_of_files[0]+folder
                      +" && mkdir -p "
                      + self.cache_folder+separate_mml_folder+"/"+list_of_files[0]+folder
                      +"; fi")
            os.system("if [ ! -d "
                      +self.cache_folder+separate_mml_folder+"/"+list_of_files[0]+folder
                      +" ]; then mkdir -p "
                      +self.cache_folder+separate_mml_folder+"/"+list_of_files[0]+folder
                      +"; fi")
        mml = MML(self.cache_folder+separate_mml_folder)
        mml.make_install("/"+list_of_files[0])
        self.mymedialite_folder = self.cache_folder+separate_mml_folder+"/"+list_of_files[0]+"/MyMediaLite"
        string_of_list = generate_file_list_name(list_of_files[1])
        os.system("cd "+self.cache_folder+path_to_list_of_files
                  +" && cp "+ string_of_list
                  +self.cache_folder+separate_mml_folder+"/"+list_of_files[0]+"/data")
        self.predict_from_path_to_path(os.listdir(self.cache_folder+separate_mml_folder+"/"+list_of_files[0]+"/data"),
                                       separate_mml_folder+"/"+list_of_files[0]+"/data",
                                       separate_mml_folder+"/"+list_of_files[0]+path_to_store)
        batch_recommendation = self.recommend_from_path(os.listdir(self.cache_folder
                                                          +separate_mml_folder+"/"+list_of_files[0]+
                                                          path_to_store),
                                 separate_mml_folder+"/"+list_of_files[0]+path_to_store,
                                 d, k_recommendations)
        print("Action for recommendation in parallel was performed successfully")
        return batch_recommendation


    def predict_from_path_to_path_parallel(self, separate_mml_folder,
                                           list_dir_data, data_path, path_to_store, d, k_recommendations=5):
        """
        Execute action_for_prfrom_path_to_path_parallel inparallel
        :param separate_mml_folder: string; name of folder to make installation MML into
        :param list_dir_data: listdir: directory with data, which would be devided into max_threads parts
        :param data_path: string: the path to data
        :param path_to_store: string: path to store predicted recommendations
        :param d: dictionary: with items characteristics (rating mean, std)
        :param k_recommendations: integer; number of recommendations to make to
        :return: data frame: the all neccessary predictions.
        """
        results = pd.DataFrame(columns=["user", "items", "ratings"])
        os.system("if [ -d "
                  +self.cache_folder+separate_mml_folder
                  +" ]; then rm -rf "
                  +self.cache_folder+separate_mml_folder
                  +" && mkdir -p "
                  + self.cache_folder+separate_mml_folder
                  +"; fi")
        os.system("if [ ! -d "
                  +self.cache_folder+separate_mml_folder
                  +" ]; then mkdir -p "
                  +self.cache_folder+separate_mml_folder
                  +"; fi")

        list_files_batches = np.array_split(list_dir_data, self.max_threads)
        list_files_batches = [[str(i), elem] for i, elem in zip(range(0, self.max_threads), list_files_batches)]
        with Pool(self.max_threads) as p:
            prod_x = partial(self.action_for_prfrom_path_to_path_parallel,
                             path_to_list_of_files=data_path,
                             separate_mml_folder=separate_mml_folder,
                             path_to_store=path_to_store,
                             d=d,
                             k_recommendations=k_recommendations)
            results = results.append(p.map(prod_x, list_files_batches))
            p.close()
            p.join()
        print("The results are ready")
        return results



    def recommend_from_cache(self, path_to_test, path_with_name_to_store, k_recommendations):
        """
        Count mean and std for each item from train set,
        read data from test, encode it,
        split test data to new users(which were not in train set) and
        old, which are in train set,
        generate users files, make prediction and recommend for new according to train set users,
        and just choose top-k for old users (in parallel for each folder of users).
        :param path_to_test: string; path to test set
        :param path_with_name_to_store: string; path which included file name for saving result
        :param k_recommendations: int; number of recommendations to make to
        :return: None
        """
        d = item_dstatistics(self.train)
        validation = read_data(path_to_test)

        new_test, self.dict_asin = encode_test(validation, self.train, "asin", self.le_asin)
        self.new_test, self.dict_new_reviewerID = encode_test(new_test, self.train, "reviewerID", self.le_reviewerID)
        (self.new_test).index = self.new_test["reviewerID"]
        for_making_prediction = (self.new_test).loc[
                                set(self.new_test["reviewerID"]) - set(self.new_train["reviewerID"]), :]
        generate_user_file_in_parallel(self.max_threads, self.cache_folder, "/data/user_not_in_cache",
                                       for_making_prediction, self.new_train)

        self.predict_from_path_to_path(os.listdir(self.cache_folder + "/data/user_not_in_cache"),
                                       "/data/user_not_in_cache", "/prediction/prediction_for_new")
        new_recommendation = self.recommend_from_path_in_parallel(
            os.listdir(self.cache_folder + "/prediction/prediction_for_new"), "/prediction/prediction_for_new", d,
            k_recommendations)
        print("New recommendation was successfully done")

        from_cache = (self.new_test).loc[
                     set(self.new_test["reviewerID"]).intersection(set(self.new_train["reviewerID"])), :]
        old_users = list(set(os.listdir(self.cache_folder + "/prediction/prediction_cache")).intersection(
            set(from_cache.reviewerID.astype("str"))))
        cache_recommendation = self.recommend_from_path_in_parallel(old_users, "/prediction/prediction_cache", d,
                                                                    k_recommendations)
        print("Cached recommendation was successfully done")
        recommendation = pd.concat([new_recommendation, cache_recommendation], ignore_index=True)
        recommendation.to_csv(path_with_name_to_store,
                              columns=["user", "items", "ratings"],
                              header=False, index=False)
        print("Recommendation from cache was successfully done")
