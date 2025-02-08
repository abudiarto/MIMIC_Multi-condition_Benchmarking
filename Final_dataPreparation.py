import pandas as pd
import pyreadr
import numpy as np
import pickle
random_state = 42
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


target_diseases = ['overall', 'neoplasm', 'mental', 'circulatory', 'respiratory'] #all target diseases

#define relevant variables
identifier = ['subject_id', 'intime']
cont_vars = ['age', 
             'n_hosp_365d', 'n_icu_365d', 'n_ed_365d', 
             'triage_temperature','triage_heartrate', 'triage_resprate', 'triage_o2sat',
             'triage_sbp', 'triage_dbp', 'triage_pain', 'triage_acuity', 'n_med','n_medrecon']

mulcat_vars = ['ethnicity', 
               'arrival_transport',  
               # 'insurance', #highly correlated to the outcomes (especially hospitalisation)
              ]

cat_vars = ['gender', 'chiefcom_chest_pain', 'chiefcom_abdominal_pain',
       'chiefcom_headache', 'chiefcom_shortness_of_breath',
       'chiefcom_back_pain', 'chiefcom_cough', 'chiefcom_nausea_vomiting',
       'chiefcom_fever_chills', 'chiefcom_syncope', 'chiefcom_dizziness',
       'cci_MI', 'cci_CHF', 'cci_PVD', 'cci_Stroke', 'cci_Dementia',
       'cci_Pulmonary', 'cci_Rheumatic', 'cci_PUD', 'cci_Liver1',
       'cci_DM1', 'cci_DM2', 'cci_Paralysis', 'cci_Renal', 'cci_Cancer1',
       'cci_Liver2', 'cci_Cancer2', 'cci_HIV', 'eci_CHF',
       'eci_Arrhythmia', 'eci_Valvular', 'eci_PHTN', 'eci_PVD',
       'eci_HTN1', 'eci_HTN2', 'eci_Paralysis', 'eci_NeuroOther',
       'eci_Pulmonary', 'eci_DM1', 'eci_DM2', 'eci_Hypothyroid',
       'eci_Renal', 'eci_Liver', 'eci_PUD', 'eci_HIV', 'eci_Lymphoma',
       'eci_Tumor2', 'eci_Tumor1', 'eci_Rheumatic', 'eci_Coagulopathy',
       'eci_Obesity', 'eci_WeightLoss', 'eci_FluidsLytes',
       'eci_BloodLoss', 'eci_Anemia', 'eci_Alcohol', 'eci_Drugs',
       'eci_Psychoses', 'eci_Depression',]

outcomes_vars = ['outcome_hospitalization', 'outcome_icu_transfer_12h', 'outcome_ed_revisit_30d']
print(len(identifier), len(cont_vars), len(cat_vars), len(mulcat_vars), len(outcomes_vars))
print(f'Total variables: {len(cont_vars)+len(cat_vars)+len(mulcat_vars)+len(outcomes_vars)+len(identifier)}')


#function to simplify ethnicity variable
asian = ['ASIAN - SOUTH EAST ASIAN','ASIAN - CHINESE','ASIAN - ASIAN INDIAN', 'ASIAN','ASIAN - KOREAN',]
hispanic = ['HISPANIC/LATINO - PUERTO RICAN', 'HISPANIC/LATINO - CUBAN','HISPANIC OR LATINO', 'HISPANIC/LATINO - DOMINICAN','HISPANIC/LATINO - CENTRAL AMERICAN','HISPANIC/LATINO - GUATEMALAN',
           'SOUTH AMERICAN',  'HISPANIC/LATINO - MEXICAN', 'HISPANIC/LATINO - SALVADORAN','HISPANIC/LATINO - HONDURAN', 'HISPANIC/LATINO - COLUMBIAN',]
white = ['WHITE', 'WHITE - BRAZILIAN','WHITE - RUSSIAN', 'AMERICAN INDIAN/ALASKA NATIVE', 'WHITE - OTHER EUROPEAN','PORTUGUESE','WHITE - EASTERN EUROPEAN',]
black = ['BLACK/AFRICAN AMERICAN','BLACK/AFRICAN','BLACK/CARIBBEAN ISLAND','BLACK/CAPE VERDEAN', ]
other = ['OTHER','NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER','MULTIPLE RACE/ETHNICITY']
unknown = ['UNKNOWN', 'PATIENT DECLINED TO ANSWER','UNABLE TO OBTAIN',]

def ethnicity_simplification(x):
    if x in asian:
        return 'Asian'
    elif x in hispanic:
        return 'Hispanic/Latino'
    elif x in white:
        return 'White'
    elif x in black:
        return 'Black/African American'
    elif x in other:
        return 'Other'
    elif x in unknown:
        return 'Unknown'
    


for target_disease in target_diseases: #loop over target diseases
    # Data loader
    print(target_disease)
    
    masterData = pd.read_csv("../mimic4diseases/"+target_disease+"/master_dataset.csv") #load master data for each target disease
    
    masterData[['intime']] = masterData[['intime']].apply(pd.to_datetime) #transform intime var
    
    masterData['ethnicity'] = masterData.apply(lambda x: ethnicity_simplification(x.ethnicity), axis=1) #transform ethnicity variable
    
    masterData = masterData[identifier + cont_vars+cat_vars+mulcat_vars+outcomes_vars] #select only relevant variables

    masterData['gender'] = masterData.gender.apply(lambda x: 1 if x=='F' else 0) #transform gender variable from F/M to 1/0

    for outcome in outcomes_vars:
        masterData[outcome] = masterData[outcome].replace({True: 1, False: 0}) #transform True/False to 0/1

    chiefcom_vars = [x for x in cat_vars if x.startswith('chiefcom')]
    for var in chiefcom_vars:
        masterData[var] = masterData[var].replace({True: 1, False: 0}) #transform True/False to 0/1
        
    # one hot encoding for categorical variables
    masterData = pd.get_dummies(masterData, columns=mulcat_vars)
    print('Data shape after one-hot encoding: ', masterData.shape)
    
    #check missing values
    print(f'Missing values: {(masterData.shape[0] - masterData.dropna().shape[0]) / masterData.shape[0] * 100}%')
    
    #remove missing values
    masterData = masterData.dropna()
    
    #separate predictors and outcome
    masterData = masterData.groupby('subject_id').last() #select only last record from each patient
    outcomes = masterData[outcomes_vars]
    features = masterData[masterData.columns[~masterData.columns.str.contains('|'.join(['intime']+outcomes_vars))]]

    #print shape for features and outcomes
    print(f'outcomes shape: {outcomes.shape}')
    print(f'features shape: {features.shape}')

    #save preprocessed features and outcomes into pickle file
    dataset = [features, outcomes]
    pickle.dump(dataset, open('../mimic4diseases/'+target_disease+'/featuresAndOutcomes.pkl', 'wb')) 
    print('dataset is successfully stored')


    #data split based on outcome in target disease
    if target_disease == 'overall':
        outcomes_vars = ['outcome_hospitalization', 'outcome_icu_transfer_12h']
    else:
        outcomes_vars = ['outcome_icu_transfer_12h']

    for outcome in outcomes_vars:
        print(target_disease)
        print(outcome)
        print(outcomes[outcome].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
    
        #Split data into grid (10%), train(70%), val(20%)
        X_train, X_val, y_train, y_val = train_test_split(features, outcomes[[outcome]], test_size=0.2, stratify=outcomes[[outcome]], random_state=random_state)
        X_train, X_grid, y_train, y_grid = train_test_split(X_train, y_train[[outcome]], test_size=0.125, stratify=y_train[[outcome]], random_state=random_state)

        #Scaling continous variable into 0-1 range
        #define variables to be scalled
        cont_vars = ['age', 
                     'n_hosp_365d', 'n_icu_365d', 'n_ed_365d', 
                     'triage_temperature','triage_heartrate', 'triage_resprate', 'triage_o2sat',
                     'triage_sbp', 'triage_dbp', 'triage_pain', 'triage_acuity', 'n_med','n_medrecon']

        # define scaler
        scaler = MinMaxScaler()

        # transform data
        result = scaler.fit_transform(X_train[cont_vars])
        result = pd.DataFrame(result, columns=scaler.get_feature_names_out())
        result.index = X_train.index

        #save scaler
        pickle.dump(scaler, open('../mimic4diseases/models/scaler_'+target_disease+'_'+outcome, 'wb'))

        X_train = pd.concat([X_train.loc[:,~X_train.columns.isin(cont_vars)],result], axis=1)

        print('Data shape after scaling: ', X_train.shape)

        #scalling for other sets
        result = scaler.fit_transform(X_val[cont_vars])
        result = pd.DataFrame(result, columns=scaler.get_feature_names_out())
        result.index = X_val.index
        X_val = pd.concat([X_val.loc[:,~X_val.columns.isin(cont_vars)],result], axis=1)

        result = scaler.fit_transform(X_grid[cont_vars])
        result = pd.DataFrame(result, columns=scaler.get_feature_names_out())
        result.index = X_grid.index
        X_grid = pd.concat([X_grid.loc[:,~X_grid.columns.isin(cont_vars)],result], axis=1)

        print('Data shape after scaling: ', X_val.shape)
        print('Data shape after scaling: ', X_grid.shape)

        dataset = [X_train, y_train, X_val, y_val, X_grid, y_grid]
        pickle.dump(dataset,open('../mimic4diseases/'+target_disease+'/'+outcome+'_dataset.pkl', 'wb'))