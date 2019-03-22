from modules.wrmf.WRMF_decorators import *
from modules.wrmf.WRMF_tools import *
from multiprocessing import cpu_count
import subprocess


max_threads = cpu_count()-1


class ImplicitRecommendation:

    @create_folders
    def __init__(self, path_to_mymedialite,
                 train_path, cache_folder, random_seed):
        self.path_to_mymedialite = path_to_mymedialite
        self.train_path = train_path
        self.cache_folder = cache_folder
        self.random_seed = random_seed
        self.model_path = cache_folder + "/models"
        self.data_path = cache_folder + "/data"
        self.prediction_path = cache_folder + "/prediction"
        self.prediction_file = None
        self.save_results = None
        self.parameters = None
        self.recommender = None


    @prepare_train_file
    def train(self, **kwargs):
        self.recommender = kwargs["recommender"]
        k=kwargs["k"]
        del kwargs["recommender"], kwargs["k"]

        self.parameters = recommender_parameters(kwargs)
        os.system("cd " + self.path_to_mymedialite + " &&" +
                  " ./bin/item_recommendation" +
                  " --file-format=DEFAULT" +
                  " --training-file=" + self.data_path + "/train_spec.txt" +
                  " --recommender=" + self.recommender + 
                  self.parameters + 
                  " --random-seed=" + str(self.random_seed) +
                  " --predict-items-number=" + str(k) +
                  " --save-model=" + self.model_path + "/" + self.recommender)

    
    @parse_prediction(max_threads=max_threads)
    @prepare_test_file
    @make_user_file
    def predict_no_parallel(self, test_path, save_results, k):
        self.prediction_file = self.prediction_path + "/prediction.txt"
        self.save_results = save_results
        predict_file(path_to_mymedialite=self.path_to_mymedialite, 
                     train_path=self.data_path + "/train_spec.txt", 
                     test_path=self.data_path + "/test_spec.txt", 
                     users_path=self.data_path + "/test_user.txt", 
                     prediction_path=self.prediction_file, 
                     recommender_name=self.recommender, k=k, random_seed=self.random_seed, 
                     model_path=self.model_path + "/" + self.recommender,
                     params=self.parameters)

    @parse_prediction(max_threads=1)
    def predict_unit(self, test_batch, k):
        test_df, save_name = test_batch
        test_path = self.data_path + "/" + save_name
        users_path = self.data_path + "/user_" + save_name
        self.prediction_file = self.prediction_path + "/p_" + save_name
        self.save_results = self.prediction_path + "/" + save_name

        test_df.to_csv(test_path, sep="\t", header=False, index=False)
        pd.DataFrame(test_df["user"].unique()).to_csv(users_path, header=False, index=False)

        del test_df, test_batch

        predict_file(path_to_mymedialite=self.path_to_mymedialite, 
                     train_path=self.data_path + "/train_spec.txt", 
                     test_path=test_path, 
                     users_path=users_path, 
                     prediction_path=self.prediction_file, 
                     recommender_name=self.recommender, k=k, random_seed=self.random_seed, 
                     model_path=self.model_path + "/" + self.recommender,
                     params=self.parameters)

    def predict_in_parallel(self, test_path, save_results, k, max_threads=max_threads):
        save_path=self.data_path + "/test_spec.txt"
        prepare_file(file_path=test_path, save_path=save_path)
        test = pd.read_csv(save_path, names=["user", "item", "rating"], sep="\t")
        splited_test_users = np.array_split(np.array(test["user"].unique()), max_threads)
        test_parts = [test[test.user.isin(u)] for u in splited_test_users]
        names_parts = [str(i+1) for i  in range(max_threads)]
        test_batches = zip(test_parts, names_parts)
        with Pool(max_threads) as p:
            part = functools.partial(self.predict_unit, k=k)
            res = p.map(part, test_batches)
        result = pd.concat(res)
        result.to_csv(save_results, index=False)
        return result

    #@clean_cache
    def predict(self, test_path, save_results, k, parallel=True, max_threads=max_threads):
        if parallel:
            p = self.predict_in_parallel(test_path=test_path, save_results=save_results, 
                                         k=k, max_threads=max_threads)
        else:
            p = self.predict_no_parallel(test_path=test_path, save_results=save_results, k=k)
        return p

    def get_latent_features(self, num_user, num_item, num_feat):
        user_features, item_features = parse_model(path=self.model_path + "/" + self.recommender, 
                                                   num_user=num_user, num_item=num_item, 
                                                   num_feat=num_feat)
        return pd.DataFrame(list(user_features)), pd.DataFrame(list(item_features))
