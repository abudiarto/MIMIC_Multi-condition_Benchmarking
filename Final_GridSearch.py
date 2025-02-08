import pandas as pd
import numpy as np
import sklearn

#statistics
from scipy.stats import chi2_contingency, ttest_ind

# import cudf #gpu-powered DataFrame (Pandas alternative)

#imbalance handling
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, RepeatedEditedNearestNeighbours
from imblearn.pipeline import Pipeline

#preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, MinMaxScaler

#internal validation
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold, cross_val_score, GridSearchCV, PredefinedSplit, train_test_split

#performance metrices
from sklearn.metrics import make_scorer, confusion_matrix, classification_report, f1_score, balanced_accuracy_score, matthews_corrcoef, auc, average_precision_score, roc_auc_score, balanced_accuracy_score, roc_curve, accuracy_score


#hyperparameter search
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

#Models selection
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier



#save and load trained model
import pickle

#visualisation
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter


random_state = 42


target_diseases = ['overall', 'neoplasm', 'mental', 'circulatory', 'respiratory'] #all target diseases

for target_disease in target_diseases:

    if target_disease == 'overall':
        outcomes_vars = ['outcome_hospitalization', 'outcome_icu_transfer_12h']
    else:
        outcomes_vars = ['outcome_icu_transfer_12h']
        
    model_names = ['LR', 'DT', 'RF', 'XGB', 'MLP']
    output = []

    for outcome in outcomes_vars:
        #GRID SEARCH
        print(target_disease, outcome)
        # Data loader
        X_train, y_train, X_val, y_val, X_grid, y_grid = pickle.load(open('../mimic4diseases/'+target_disease+'/'+outcome+'_dataset.pkl', 'rb'))
        
        print(X_grid.shape)
        print(y_grid.shape)
        scale_pos_ratio = y_grid.value_counts()[0]/y_grid.value_counts()[1]

        #MODELS
        lr_model = LogisticRegression(class_weight='balanced', 
                                      random_state=random_state)
        dt_model = DecisionTreeClassifier(class_weight='balanced', 
                                          random_state=random_state)
        rf_model = RandomForestClassifier(class_weight='balanced', 
                                          random_state=random_state)
        xgb_model = xgb.XGBClassifier(objective ='binary:logistic', 
                                      tree_method='gpu_hist', 
                                      gpu_id=0,  verbosity = 0,
                                      scale_pos_weight = scale_pos_ratio, 
                                      random_state=random_state)
        mlp_model = MLPClassifier(hidden_layer_sizes=(32,32,),
                                  activation='relu',                                  
                                  learning_rate='adaptive',
                                  early_stopping=True,
                                  validation_split=.2,
                                  random_state=random_state)

        #PARAMS
        lr_params = {'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
                     'C': [0.1, 1.0, 10.0],
                     'max_iter': [80.0, 100.0, 120.0]}
        dt_params = {'max_depth': [3,5,7,9]}
        rf_params = {'n_estimators': [100, 200, 300],
                    'max_depth': [3,5,7,9]}
        xgb_params = {'n_estimators': [100, 200, 300],
                    'max_depth': [3,5,7,9],
                     'learning_rate': [1e-1, 1e-2, 1e-3],
                     'reg_alpha': [0.3, 0.5, 0.7],
                     'reg_lambda': [0.3, 0.5, 0.7],}
        mlp_params = {'learning_rate_init': [1e-1, 1e-2, 1e-3],
                      'solver': ['sgd', 'adam']
                     }


        #Models and params in DICT
        models_to_be_trained = [
            {'model_name': 'LR', 'model': lr_model, 'params': lr_params},
            {'model_name': 'DT', 'model': dt_model, 'params': dt_params},
            {'model_name': 'RF', 'model': rf_model, 'params': rf_params},
            {'model_name': 'XGB', 'model': xgb_model, 'params': xgb_params},
            {'model_name': 'MLP', 'model': mlp_model, 'params': mlp_params},
        ]

        scoring = {
            'auc': make_scorer(roc_auc_score)
            }

        for item in models_to_be_trained:
            print(item['model_name'])
            gs = BayesSearchCV(item['model'],
                              search_spaces=item['params'],
                              scoring=make_scorer(roc_auc_score),
                               n_iter = 50,
                              cv=5,
                              verbose=3, 
                               n_jobs=5,
                               n_points=10,
                                random_state = 1234)
            gs.fit(X_grid, y_grid)
            output.append([outcome, item['model_name'], gs.best_params_, gs.best_score_])
            
        output = pd.DataFrame(output, columns=['outcome', 'model', 'best_params', 'best_score'])
        pickle.dump(output, open('../mimic4diseases/gs/gs_result_'+target_disease+'_'+outcome+'.pkl', 'wb'))