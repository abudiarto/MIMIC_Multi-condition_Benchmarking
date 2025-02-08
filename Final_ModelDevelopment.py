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
from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score, matthews_corrcoef, auc, average_precision_score, roc_auc_score, balanced_accuracy_score, roc_curve, accuracy_score

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


#Model evaluation function

def summariseResult (testX, testY, model):
    preds = model.predict_proba(testX)
    preds = [x[1] for x in preds]
    aucscore = roc_auc_score(testY, preds)
    auprc = average_precision_score(testY, preds)
    # plot_confusion_matrix(model, testX, testY, cmap='viridis')  
    return np.round(aucscore,4), np.round(auprc,4)

#Fix model name for visualisation

def modelNameFixer(x):
    if 'LogisticRegression' in x:
        return 'LR'
    elif 'DecisionTreeClassifier' in x:
        return 'DT'
    elif 'RandomForestClassifier' in x:
        return 'RF'
    elif 'MLPClassifier' in x:
        return 'MLP'
    elif 'XGB' in x:
        return 'XGB'

# instantiate the model (using the default parameters)
def build_models (outcome, X_train, y_train, split_counter, params_dict):
    models = [] #list to store all the models
    print("Building models . . . .")
      
    #LR
    params = params_dict[params_dict.model=='LR'].best_params.values[0]
    lr_model = LogisticRegression(class_weight='balanced',
                                  solver=params['solver'],
                                  max_iter = params['max_iter'],
                                  C = params['C'],
                                  random_state=random_state)
    lr_model.fit(X_train,y_train)
    modelname =str(split_counter) + 'LRModel_'
    models.append([modelname, y_train.value_counts()[1]/y_train.value_counts()[0], split_counter])
    pickle.dump(lr_model, open('../mimic4diseases/models/'+ target_disease + '_'+ outcome + '_' + modelname + '.sav', 'wb')) 
    print("LR done")
    
#     #SVM
#     svm_model = SVC(C = params['C'], 
#                     kernel = 'rbf',
#                     class_weight='balanced', 
#                     probability=True,
#                     random_state=random_state)
#     svm_model.fit(X_train,y_train)
#     modelname =str(split_counter) + 'SVMModel_'
#     models.append([modelname, y_train.value_counts()[1]/y_train.value_counts()[0], split_counter])
#     pickle.dump(svm_model, open('../mimic4diseases/models/'+ target_disease + '_'+ modelname + '.sav', 'wb')) 
#     print("SVM done")
    
    
    #DT
    params = params_dict[params_dict.model=='DT'].best_params.values[0]
    dt_model = DecisionTreeClassifier(max_depth = params['max_depth'], 
                                      class_weight='balanced', 
                                      random_state=random_state)
    dt_model.fit(X_train,y_train)
    modelname =str(split_counter) + 'DTModel_'
    models.append([modelname, y_train.value_counts()[1]/y_train.value_counts()[0], split_counter])
    pickle.dump(dt_model, open('../mimic4diseases/models/'+ target_disease + '_'+ outcome + '_' + modelname + '.sav', 'wb')) 
    print("DT done")
    
    
    #RF
    params = params_dict[params_dict.model=='RF'].best_params.values[0]
    rf_model = RandomForestClassifier(n_estimators = params['n_estimators'], 
                                      max_depth = params['max_depth'], 
                                      class_weight='balanced', 
                                      random_state=random_state)
    rf_model.fit(X_train,y_train)
    modelname =str(split_counter) + 'RFModel_'
    models.append([modelname, y_train.value_counts()[1]/y_train.value_counts()[0], split_counter])
    pickle.dump(rf_model, open('../mimic4diseases/models/'+ target_disease + '_'+ outcome + '_' + modelname + '.sav', 'wb')) 
    print("RF done")



    #XGB
    params = params_dict[params_dict.model=='XGB'].best_params.values[0]
    scale_pos_ratio = y_train.value_counts()[0]/y_train.value_counts()[1]
    xgb_model = xgb.XGBClassifier(objective ='binary:logistic', 
                                  max_depth = params['max_depth'], 
                                  n_estimators = params['n_estimators'],  
                                  tree_method='gpu_hist', 
                                  learning_rate = params['learning_rate'], 
                                  reg_alpha = params['reg_alpha'],
                                  reg_lambda = params['reg_lambda'],
                                  scale_pos_weight = scale_pos_ratio, 
                                  gpu_id=0, verbosity = 0, random_state = random_state)

    xgb_model.fit(X_train,y_train)
    #save model
    modelname = str(split_counter) + 'XGBoostModel'
    models.append([modelname, y_train.value_counts()[1]/y_train.value_counts()[0], split_counter])
    pickle.dump(xgb_model, open('../mimic4diseases/models/'+ target_disease + '_'+ outcome + '_' + modelname + '.sav', 'wb')) 
    print("XGB done")
    
    

    #MLP
    params = params_dict[params_dict.model=='MLP'].best_params.values[0]
    mlp_model = MLPClassifier(hidden_layer_sizes=(32,32,),
                              solver='sgd',
                              activation='relu',
                              learning_rate_init=params['learning_rate_init'],
                              learning_rate='adaptive',
                              early_stopping=True,
                              random_state=random_state)
    mlp_model.fit(X_train,y_train)
    modelname =str(split_counter) + 'MLPModel_'
    models.append([modelname, y_train.value_counts()[1]/y_train.value_counts()[0], split_counter])
    pickle.dump(mlp_model, open('../mimic4diseases/models/'+ target_disease + '_' + outcome + '_' + modelname + '.sav', 'wb')) 
    print("MLP done")
    

    
    return models
    # return [xgb_model]    

target_diseases = ['overall', 'neoplasm', 'mental', 'circulatory', 'respiratory'] #all target diseases

for target_disease in target_diseases:
    if target_disease == 'overall':
        outcomes_vars = ['outcome_hospitalization', 'outcome_icu_transfer_12h']
    else:
        outcomes_vars = ['outcome_icu_transfer_12h']

    
    summary_result = []
    cols = ['model_name', 'class_ratio', 'outcome', 'auc', 'auprc']

    for outcome in outcomes_vars:
        print(target_disease)
        print(outcome)
        # Data loader
        X_train, y_train, X_val, y_val, X_grid, y_grid = pickle.load(open('../mimic4diseases/'+target_disease+'/'+outcome+'_dataset.pkl', 'rb'))
        params = pickle.load(open('../mimic4diseases/gs/gs_result_'+target_disease+'_'+outcome+'.pkl', 'rb'))

        print('X_train shape: ', X_train.shape)
        print('y_train shape: ', y_train.shape)
        models = pd.DataFrame(columns=['modelname', 'class_ratio', 'imbalance_method', 'split_counter'])

        split_counter = 99 #to flag this as test

        #Build models -> it can be commented if the models have been trained
        models_temp = pd.DataFrame(build_models(outcome, X_train, y_train, split_counter, params), columns=['modelname', 'class_ratio', 'split_counter'])
        models = pd.concat([models,models_temp]).reset_index(drop=True)


        #evaluate model
        for modelname, classratio, split_counter in models_temp.values:
            # print('======================================================================')
            print(modelname)
            model = pickle.load(open('../mimic4diseases/models/'+ target_disease + '_'+ outcome + '_' + modelname + '.sav', 'rb'))
            summary_result.append((str(model), classratio, outcome) + summariseResult (X_val, y_val, model) )     

        split_counter+=1


    summary_result = pd.DataFrame(summary_result, columns=cols)
    summary_result['model_num'] = summary_result.index
    
    summary_result['model_name'] = summary_result.model_name.apply(lambda x: modelNameFixer(x))
    summary_result.to_csv('../mimic4diseases/models/summaryResult_'+target_disease+'.csv', index_label=False, index=False)