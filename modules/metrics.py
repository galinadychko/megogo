import pandas as pd
import numpy as np
import scipy.stats as stat
import math
from multiprocessing import Pool
import multiprocessing


class metrics:

    def __init__(self, train, test, predict, train_columns=('user', 'item', 'rating'),
                       test_columns=('user', 'item', 'rating'), predict_columns=('user', 'item', 'rating')):

        self.train = train[list(train_columns)].copy()
        self.train.rename(columns={train_columns[0]: 'user', train_columns[1]: 'item', train_columns[2]: 'rating'}, inplace=True)
        self.train[['user', 'item']] = self.train[['user', 'item']].astype(str)
        self.train.rating = self.train.rating.astype(float)

        self.test = test[list(test_columns)].copy()
        self.test.rename(columns={test_columns[0]: 'user', test_columns[1]: 'item', test_columns[2]: 'rating'}, inplace=True)
        self.test[['user', 'item']] = self.test[['user', 'item']].astype(str)
        self.test.rating = self.test.rating.astype(float)

        self.predict = predict[list(predict_columns)].copy()
        self.predict.rename(columns={predict_columns[0]: 'user', predict_columns[1]: 'item', predict_columns[2]: 'rating'}, inplace=True)
        self.predict[['user', 'item']] = self.predict[['user', 'item']].astype(str)
        self.predict.rating = self.predict.rating.astype(float)


    def drop_bad_ratings(self):

        train = self.train.groupby('user').median().reset_index()
        try:
            train = train.drop('item', axis=1)
        except:
            pass

        _test = pd.merge(self.test, train, on='user')
        _test = _test[_test.rating_x >= _test.rating_y]
        _test = _test.drop(['rating_y'], axis=1)
        _test.columns = ['user', 'item', 'rating']

        return _test


    def precision_at_k(self, k=5):

        _test = self.drop_bad_ratings()
        _predict = self.predict[self.predict.user.isin(set(_test.user))].copy()

        for _user in set(_predict.user):
            _predict = _predict.drop(_predict[_predict.user == _user].iloc[k:].index, axis=0)

        merge = pd.merge(_predict, _test, on=['user', 'item'], how='left')
        precision = len(merge.dropna()) / (len(set(_predict.user)) * k)
        # print('precision@{} = {}'.format(k, precision))

        return precision

    def recall(self):

        _test = self.drop_bad_ratings()
        _predict = self.predict[self.predict.user.isin(set(_test.user))].copy()
        merge = pd.merge(_test, _predict, on=['user', 'item'], how='left')
        merge.rating_y = [0 if np.isnan(elem) else 1 for elem in merge.rating_y]
        merge = merge.groupby('user')['rating_y'].agg(['sum', 'count']).reset_index()
        merge['recall'] = merge['sum'] / merge['count']
        recall = np.sum(merge.recall) / len(set(_test.user))
        #print('recall =', recall)

        return recall

    def dcg_batch(self, user_list):
        local_res = []
        for user in user_list:
            sorted_rating = sorted(self.test[self.test.user == user].rating, reverse=True)
            dcg_p = sum([rating / np.log2(2 + i) for i, rating in enumerate(sorted_rating)])
            local_res.append((user, dcg_p))
        return local_res


    def ndcg(self):

        k = len(self.predict[self.predict.user == list(set(self.predict.user))[0]])
        merge = pd.merge(self.predict, self.test, on=['user', 'item'], how='left')
        merge.rating_x = list(range(1, k+1)) * len(set(self.predict.user))
        merge.columns = ['user', 'item', 'rank', 'true_rating']
        merge = merge.fillna(0)
        merge['dcg'] = merge.true_rating / np.log2(1 + merge['rank'])
        merge = merge.groupby('user').sum().reset_index()[['user', 'dcg']]

        # one_chunk = int(len(set(self.test.user))/(multiprocessing.cpu_count() - 1))
        #
        # chunks = [list(self.test.user)[i*one_chunk:(i+1)*one_chunk] for i in range((multiprocessing.cpu_count() - 1))]
        # chunks[-1] = chunks[-1].extend(list(self.test.user)[((multiprocessing.cpu_count() - 1)+1)*one_chunk:])
        #
        # all_dcg_p = []
        # p = Pool()
        # all_dcg_pp = p.map(self.dcg_batch, chunks)
        #
        all_dcg_p = []
        # for i in range(len(all_dcg_pp)):
        #     all_dcg_p = all_dcg_p.extend(all_dcg_pp[i])
        for user in set(self.test.user):
            sorted_rating = sorted(self.test[self.test.user == user].rating, reverse=True)
            dcg_p = sum([rating / np.log2(2 + i) for i, rating in enumerate(sorted_rating)])
            all_dcg_p.append((user, dcg_p))

        all_dcg_p = pd.DataFrame.from_records(all_dcg_p, columns=['user', 'dcg_p'])
        ndcg_df = pd.merge(merge, all_dcg_p, on='user')
        ndcg_df['ndcg'] = ndcg_df.dcg / ndcg_df.dcg_p
        ndcg = sum(ndcg_df.ndcg) / len(set(self.predict.user))
        # print('nDCG =', ndcg)

        return ndcg


    def map_helper(self, user, merge):
        relevant = list(merge[merge.user == user].rating_y)
        avg_prec = [sum(relevant[:i + 1]) / (i + 1) for i, elem in enumerate(relevant) if elem != 0]

        if avg_prec != []:
            avg_prec = sum(avg_prec) / len(avg_prec)
            local_map += avg_prec
        return local_map


    def mean_avg_precision(self):

        _test = self.drop_bad_ratings()
        _predict = self.predict[self.predict.user.isin(set(_test.user))].copy()
        merge = pd.merge(_predict, _test, on=['user', 'item'], how='left')
        merge = merge.fillna(0)
        merge.rating_y = [1 if elem != 0 else elem for elem in merge.rating_y]
        mean_avg_precision = 0

        for user in set(merge.user):
            relevant = list(merge[merge.user == user].rating_y)
            avg_prec = [sum(relevant[:i + 1]) / (i + 1) for i, elem in enumerate(relevant) if elem != 0]

            if avg_prec != []:
                avg_prec = sum(avg_prec) / len(avg_prec)
                mean_avg_precision += avg_prec

        mean_avg_precision /= len(set(_predict.user))

        return mean_avg_precision


    def rec_sys_report(self, metrics_list, k=5, verbose=False):

        report_dict = {}
        error = ''

        for rec_metric in metrics_list:
            try:
                if rec_metric == 'precision_at_k':
                    metric_value = getattr(self, rec_metric)(k)
                    rec_metric = 'precision@' + str(k)
                elif rec_metric == 'recall':
                    metric_value = getattr(self, rec_metric)()
                    rec_metric = 'recall@' + str(k)
                else: metric_value = getattr(self, rec_metric)()

            except Exception as e:
                error = 'Failed in {} with exception: {}'.format(rec_metric, e)
                if verbose:
                    print(error)
                continue

            report_dict[rec_metric] = metric_value

        return (report_dict, error)


    #### Mean Reciprocal Rank (MRR) ########################################
    #### for recommendation task only ######################################

    def mrr(self):
        """
        mrr = Mean Reciprocal Rank
        """
        _test = self.drop_bad_ratings()
        merged = pd.merge(left=_test, right=self.predict, on=['user', 'item'], how='right')[
            ['user', 'item', 'rating_x', 'rating_y']]
        nott = np.vectorize(lambda x: not x)
        mrrs = []
        for user in merged.user.unique():
            frame = merged[merged.user == user].sort_values(by='rating_y', ascending=False)
            true_ratings = frame.rating_x.values
            positions = np.where(nott(np.isnan(true_ratings)))[0]
            if len(positions) > 0:
                mrrs.append(1 / (positions[0] + 1))
            else:
                mrrs.append(0)

        return sum(mrrs) / len(mrrs)





    def num_of_ordered_positive(self, frame, col1, col2):
        res = 0
        if frame.shape[0] > 1:
            true = frame[col1].values
            predicted = frame[col2].values
            for i in range(true.shape[0]):
                for j in range(i, true.shape[0]):
                    if ((true[i] > true[j]) and (predicted[i] > predicted[j])) or \
                            ((true[i] < true[j]) and (predicted[i] < predicted[j])):
                        res += 1
        return res

    def num_of_ordered_negative(self, frame, col1, col2):
        res = 0
        if frame.shape[0] > 1:
            true = frame[col1].values
            predicted = frame[col2].values
            for i in range(true.shape[0]):
                for j in range(i, true.shape[0]):
                    if ((true[i] > true[j]) and (predicted[i] < predicted[j])) or \
                            ((true[i] < true[j]) and (predicted[i] > predicted[j])):
                        res += 1
        return res

    def num_of_ordered(self, frame, col):
        res = 0
        if frame.shape[0] > 1:
            true = frame[col].values
            for i in range(true.shape[0]):
                for j in range(i, true.shape[0]):
                    if true[i] > true[j] or true[i] < true[j]:
                        res += 1
        return res


    #### nDPM ###########################################################
    #### for ranking task only ##########################################

    def ndpm(self):
        """
        ndpm = Normalized Distance-Based Performance Measure

        nDPM measure gives a perfect score of 0 to systems that correctly predicts every preference relation
        asserted by the reference. The worst score is 1.
        """

        merged = pd.merge(left=self.test, right=self.predict, on=['user', 'item'], how='inner')[
            ['user', 'rating_x', 'rating_y']]
        ndpms = []
        for user in merged.user.unique():
            frame = merged[merged.user == user]
            if frame.shape[0] <= 1:
                continue
            C_plus = self.num_of_ordered_positive(frame, 'rating_x', 'rating_y')
            C_minus = self.num_of_ordered_negative(frame, 'rating_x', 'rating_y')
            C_u = self.num_of_ordered(frame, 'rating_x')
            if C_u == 0:
                continue
            C_s = self.num_of_ordered(frame, 'rating_y')
            C_u0 = C_u - (C_plus + C_minus)
            ndpms.append(1 - (C_minus + 0.5 * C_u0) / C_u)

        return sum(ndpms) / len(ndpms)




    #### Spearman correlation (deprecated) ############################################
    #### for ranking task only ###########################################

    def spearman_corr(self):

        merged = pd.merge(left=self.test, right=self.predict, on=['user', 'item'], how='inner')[
            ['user', 'item', 'rating_x', 'rating_y']]
        corrs = []
        for user in merged.user.unique():
            frame = merged[merged.user == user]
            corr = stat.spearmanr(frame.rating_x.values, frame.rating_y.values)[0]
            corrs.append(corr if not math.isnan(corr) else 1)

        return sum(corrs) / len(corrs)

    #### COVERAGE METRICS ###############################################
    #### for recommendation task only ###################################
    #####################################################################

    #### Percentage of the recommended items in the set of all items ####
    def percentage(self):

        return self.predict.item.unique().shape[0] / min(self.train.item.unique().shape[0],
                                                         self.predict.item.values.shape[0])

    #### Gini Index #####################################################
    def gini_index(self):
        '''
        The index is 0 when all items are chosen equally often, and 1 when a single item is always chosen.
        '''
        proportions = (self.predict.item.value_counts() / self.predict.shape[0]).sort_values().values
        n = min(self.train.item.unique().shape[0], self.predict.item.values.shape[0])
        proportions = np.concatenate([np.zeros(n - proportions.shape[0]), proportions])
        gini = 0
        for j in range(n):
            gini += (2 * (j + 1) - n - 1) * proportions[j]
        gini /= n - 1

        return gini

    #### Shannon Entropy ################################################
    def normed_entropy(self):
        '''
        The Normed Entropy is 0 when a single item is always chosen or recommended,
        and 1 when n items are chosen or recommended equally often.
        '''
        proportions = (self.predict.item.value_counts() / self.predict.shape[0]).sort_values().values
        x_logx = np.vectorize(lambda x: x * math.log2(x))
        n = min(self.train.item.unique().shape[0], self.predict.item.values.shape[0])

        return - x_logx(proportions).sum() / math.log2(n)

class metrics_for_each:

    def __init__(self, train, test, predict, k, train_columns=('user', 'item', 'rating'),
                       test_columns=('user', 'item', 'rating'), predict_columns=('user', 'item', 'rating'), metrics_list=['ndcg']):

        self.train = train[list(train_columns)].copy()
        self.train.rename(columns={train_columns[0]: 'user', train_columns[1]: 'item', train_columns[2]: 'rating'}, inplace=True)
        self.train[['user', 'item']] = self.train[['user', 'item']].astype(str)
        self.train.rating = self.train.rating.astype(float)

        self.test = test[list(test_columns)].copy()
        self.test.rename(columns={test_columns[0]: 'user', test_columns[1]: 'item', test_columns[2]: 'rating'}, inplace=True)
        self.test[['user', 'item']] = self.test[['user', 'item']].astype(str)
        self.test.rating = self.test.rating.astype(float)

        self.predict = predict[list(predict_columns)].copy()
        self.predict.rename(columns={predict_columns[0]: 'user', predict_columns[1]: 'item', predict_columns[2]: 'rating'}, inplace=True)
        self.predict[['user', 'item']] = self.predict[['user', 'item']].astype(str)
        self.predict.rating = self.predict.rating.astype(float)


        self.users = list(self.predict['user'].unique())
        self._metric_per_user = {}
        self.metrics_list = metrics_list

        try:
            self.k = int(k)
        except:
            self.k=5


    def one_user_helper(self, users):
        local_metric_per_user = {}
        for user in users:
            one_predict = self.predict[self.predict.user == user]
            one_test = self.test[self.test.user == user]
            one_evaluator = metrics(self.train, one_test, one_predict)
            one_res, error = one_evaluator.rec_sys_report(self.metrics_list, k=self.k)
            if not error :
                local_metric_per_user[user] = one_res
        return local_metric_per_user

    def metric_per_user(self,):

        cpus_to_use = (multiprocessing.cpu_count() - 1)

        one_chunk = int(len(self.users)/cpus_to_use)

        chunks = [self.users[i*one_chunk:(i+1)*one_chunk] for i in range(cpus_to_use)]

        for u in self.users[(cpus_to_use+1)*one_chunk:]:
            chunks[-1].append(u)

        # print(chunks[-1])
        # print(cpus_to_use)
        # print(len(chunks))

        all_met = []
        p = Pool()
        all_met = p.map(self.one_user_helper, chunks)
        _metric_per_user ={}

        for el in all_met:
            _metric_per_user.update(el)

        # for user in users:
        #     one_predict = predict[predict.user == user]
        #     one_test = test[test.user == user]
        #     one_evaluator = metrics(train, one_test, one_predict)
        #     one_res, error = one_evaluator.rec_sys_report(metrics_list, k=k)
        #     if not error :
        #         _metric_per_user[user] = one_res

        return _metric_per_user



if __name__ == '__main__':
    train = pd.read_csv('data/u1_trans.base', sep='\t', header=0, names=['user', 'item', 'rating', 'time'])[['user', 'item', 'rating']]
    # test = pd.read_csv('data/u1_trans.test', sep='\t', header=0, names=['user', 'item', 'rating', 'time'])[['user', 'item', 'rating']]
    test = pd.read_csv('data/u1_dataset_for_ranking_5.csv', sep=',', header=None, names=['user', 'item', 'rating'])

    # predict = pd.read_csv('data/recommendations_k=5_result_model_mf_simple.csv')
    predict = pd.read_csv('data/recommendations_k=5_result_model_mf_simple.csv')
    predict.columns = ['user', 'item', 'rating']
    predict['user'] = predict.user.apply(int)
    evaluator = metrics(train, test, predict)
    print(evaluator.rec_sys_report(['ndcg', 'precision_at_k', 'mean_avg_precision', 'mrr', 'ndpm'], k=5))
