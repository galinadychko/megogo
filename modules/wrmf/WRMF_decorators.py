from modules.wrmf.WRMF_decorators_tools import *
import os
import re
import functools


def prepare_train_file(func):
    @functools.wraps(func)
    def wrapper(self, **kwargs):
        prepare_file(file_path = self.train_path, 
                     save_path = self.data_path + "/train_spec.txt")
        func_output = func(self, **kwargs)
        return func_output
    return wrapper


def prepare_test_file(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        test_path=kwargs["test_path"]
        save_path=self.data_path + "/test_spec.txt"
        prepare_file(file_path = test_path, save_path = save_path)
        
        func(self, *args, **kwargs)
    return wrapper



def make_user_file(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        test_df = pd.read_csv(self.data_path + "/test_spec.txt", sep="\t", names=["user", "item", "rating"])
        pd.DataFrame(test_df.user.unique(), columns=["user"]).to_csv(self.data_path + "/test_user.txt",
                                                                     sep="\t", header=False, index=False)
        func(self, *args, **kwargs)
    return wrapper


def create_folders(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self,  *args, **kwargs)
        list_of_pathes = [self.cache_folder, self.model_path, self.data_path, self.prediction_path]
        for folder in list_of_pathes:
            os.system("if [ ! -d " + folder + " ]; then mkdir -p " + folder + "; fi")
    return wrapper


def parse_prediction(max_threads=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            if max_threads > 1:
                p = parse_prediction_in_parallel(prediction_path=self.prediction_file, 
                                                 save_path=self.save_results,
                                                 max_threads=max_threads)
            else:
                p = parse_prediction_no_parallel(prediction_path=self.prediction_file, 
                                                 save_path=self.save_results)
            return pd.read_csv(self.save_results, names=["user", "item", "score"])
        return wrapper
    return decorator

def clean_cache(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        f = func(self, *args, **kwargs)
        os.system("rm -rf " + self.cache_folder)
        return f
    return wrapper