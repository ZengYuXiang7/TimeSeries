# coding : utf-8
# Author : yuxiang Zeng
import collections
import time
import pickle
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import torch

from utils.exp_config import get_config
from utils.exp_logger import Logger
from exp.exp_metrics import ErrorMetrics
from utils.utils import set_settings, set_seed

global log
torch.set_default_dtype(torch.double)

class Model(torch.torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.log = args.log
    def forward(self, adjacency, features):
        pass

    def set_runid(self, runid):
        self.runid = runid
        self.log.only_print('-' * 80)
        self.log.only_print(f'Runid : {self.runid + 1}')

    def machine_learning_model_train_evaluation(self, train_x, train_y, valid_x, valid_y, test_x, test_y, max_value):
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import ParameterGrid
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.svm import SVR
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        # print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape, test_x.shape, test_y.shape)
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            # 'KNeighborsRegressor': KNeighborsRegressor(),
            'SVR': SVR(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
        }
        param_grids = {
            'Ridge': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
            'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
            # 'KNeighborsRegressor': {'n_neighbors': [3, 5, 7, 9, 11, 15]},
            'SVR': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear']},
            'DecisionTreeRegressor': {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10]
            },
            'RandomForestRegressor': {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoostingRegressor': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 1],
                'max_depth': [3, 5, 7]
            }
        }
        results_dict = {}
        for name, model in models.items():
            self.log.only_print(f"模型: {name}")
            if name in param_grids:
                best_score = float('inf')
                best_params = None
                for params in ParameterGrid(param_grids[name]):
                    model.set_params(**params)
                    model.fit(train_x, train_y)
                    predictions = model.predict(valid_x)
                    score = mean_squared_error(valid_y, predictions)
                    if score < best_score:
                        best_score = score
                        best_params = params
                # print(f"{name} 最佳参数: {best_params}")
                model.set_params(**best_params)
                model.fit(train_x, train_y)
            else:
                model.fit(train_x, train_y)
            predict_test_y = model.predict(test_x)
            results_test = ErrorMetrics(predict_test_y * max_value, test_y * max_value, self.args)
            self.log.only_print(f"测试集上的表现 - MAE={results_test['MAE']:.4f}, RMSE={results_test['RMSE']:.4f}, NMAE={results_test['NMAE']:.4f}, NRMSE={results_test['NRMSE']:.4f}")
            self.log.only_print(f"Acc = [1%={results_test['Acc_1']:.4f}, 5%={results_test['Acc_5']:.4f}, 10%={results_test['Acc_10']:.4f}]  ")
            results_dict[name] = results_test
        return results_dict


    def machine_learning_model_train_evaluation_classification(self, train_x, train_y, valid_x, valid_y, test_x, test_y):
        models = {
            # 'KNeighborsClassifier': KNeighborsClassifier(),
            # 'SVC': SVC(),
            # 'DecisionTreeClassifier': DecisionTreeClassifier(),
            # 'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
        }

        param_grids = {
            # 'KNeighborsClassifier': {'n_neighbors': [3, 5, 7, 9, 11, 15]},
            # 'SVC': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear']},
            # 'DecisionTreeClassifier': {
            #     'max_depth': [None, 5, 10, 15, 20],
            #     'min_samples_split': [2, 5, 10]
            # },
            # 'RandomForestClassifier': {
            #     'n_estimators': [10, 50, 100],
            #     'max_depth': [None, 5, 10, 15, 20],
            #     'min_samples_split': [2, 5, 10]
            # },
            'GradientBoostingClassifier': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 1],
                'max_depth': [3, 5, 7]
            }
        }

        results_dict = {}
        for name, model in models.items():
            self.log.only_print(f"模型: {name}")
            if name in param_grids:
                best_score = float('-inf')
                best_params = None
                for params in ParameterGrid(param_grids[name]):
                    model.set_params(**params)
                    model.fit(train_x, train_y)
                    predictions = model.predict(valid_x)
                    score = accuracy_score(valid_y, predictions)
                    if score > best_score:
                        best_score = score
                        best_params = params
                # print(f"{name} 最佳参数: {best_params}")
                model.set_params(**best_params)
                model.fit(train_x, train_y)
            else:
                model.fit(train_x, train_y)

            predict_test_y = model.predict(test_x)
            results_test = ErrorMetrics(predict_test_y * 1, test_y * 1, self.args)
            results_dict[name] = results_test

        return results_dict


def RunOnce(args, runId, Runtime, log):
    # Set seed
    set_seed(args.seed + runId)
    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(args)
    model.set_runid(runId)
    dataset_info = pickle.load(open(f'./datasets/flow/{args.dataset}_info_{args.flow_length_limit}.pickle', 'rb'))
    max_packet_length = dataset_info['max_packet_length']
    train_x, train_y = np.array(datamodule.train_set.x)[:,args.flow_length_limit:], np.array(datamodule.train_set.y)
    valid_x, valid_y = np.array(datamodule.valid_set.x)[:,args.flow_length_limit:], np.array(datamodule.valid_set.y)
    test_x, test_y = np.array(datamodule.test_set.x)[:,args.flow_length_limit:], np.array(datamodule.test_set.y)
    train_x /= max_packet_length
    valid_x /= max_packet_length
    test_x /= max_packet_length
    results = model.machine_learning_model_train_evaluation_classification(train_x, train_y, valid_x, valid_y, test_x, test_y)
    return results


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(args.rounds):
        runHash = int(time.time())
        results = RunOnce(args, runId, runHash, log)
        for model_name, model_results in results.items():
            for metric_name, metric_value in model_results.items():
                metrics[f"{model_name}_{metric_name}"].append(metric_value)
    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')
    if args.record:
        log.save_result(metrics)
    log('*' * 20 + 'Experiment Success' + '*' * 20 + '\n')

    return metrics



if __name__ == '__main__':
    args = get_config()
    set_settings(args)
    args.model = 'ML'
    args.dimension = None
    args.rounds = 5
    exper_detail = f"Dataset : {args.dataset.upper()}, Model : {args.model}, Train_size : {args.train_size}, Bs : {args.bs}, Rank : {args.rank}, "
    log_filename = f'Model_{args.model}_{args.dataset}_S{args.train_size}_R{args.rank}'
    print(log_filename)
    log = Logger(log_filename, exper_detail, args)
    args.log = log
    log(str(args))
    RunExperiments(log, args)



