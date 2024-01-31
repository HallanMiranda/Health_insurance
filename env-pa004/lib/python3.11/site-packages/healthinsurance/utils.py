# General Propose
import numpy as np
import pandas as pd

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency

# Hipo Test
from scipy import stats
from scipy.stats import f_oneway
from scipy.stats import ttest_ind

# ML
import xgboost as xgb
import lightgbm as lgb
from skopt import forest_minimize
from healthinsurance import HealthInsurance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit as sss
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, f1_score, recall_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class utils:

    def __init__(self):
        pass

    # Function to select the top null variables
    def top_nulls(self, df, rate):
        null_vars = df.isna().mean().sort_values(ascending=False)
        return null_vars.loc[null_vars>rate].index.to_list()

    # Function to convert rs to brl
    def convert_rs_to_brl(self, x, rate):
        return x*rate

    # Function to categorize the variable age
    def cat_age(self, x):
        
        if x > 20 and x <= 25 :
            return 'Faixa_20_25'
        elif x > 25 and x <= 30:
            return 'Faixa_25_30'
        else:
            return 'Faixa_maior_30'
        
    def Myheat_map(dataset, variaveis):

        df_corr = dataset[variaveis].corr()

        fig, ax = plt.subplots(figsize=(16, 10))
        # mask
        mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
        # adjust mask and df
        mask = mask[1:, :-1]
        corr = df_corr.iloc[1:,:-1].copy()
        # color map
        cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

        # plot heatmap
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    linewidths=5, cmap=cmap, vmin=-1, vmax=1, 
                    cbar_kws={"shrink": .8}, square=True)
        yticks = [i.upper() for i in corr.index]
        xticks = [i.upper() for i in corr.columns]
        plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
        plt.xticks(plt.xticks()[0], labels=xticks, rotation=20)

        # title
        title = 'CORRELATION MATRIX\n'
        plt.title(title, loc='left', fontsize=18)
        plt.show()
        
    def cramer_v(self, var_x, var_y):

        # builds contigency matrix (or confusion matrix)
        confusion_matrix_v = pd.crosstab(var_x, var_y).values

        # gets the sum of all values in the matrix
        n = confusion_matrix_v.sum()

        # gets the rows, cols
        r, k = confusion_matrix_v.shape

        # gets the chi-squared
        chi2 = chi2_contingency(confusion_matrix_v)[0]

        # makes the bias correction
        chi2corr = max(0, chi2 - (k-1) * (r-1) / (n-1))
        kcorr = k - (k-1) ** 2 / (n-1)
        rcorr = r - (r-1) ** 2 / (n-1)

        # returns cramér V
        return np.sqrt((chi2corr/n) / min(kcorr-1, rcorr-1))

    def hipo_test(self, *samples):

        samples = samples

        try:
            if len(samples) == 2:
                stat, p = ttest_ind(*samples)
            elif len(samples) > 2:
                stat, p = f_oneway(*samples)
        except:
            raise Exception("Deve ser fornecido pelo menos duas samples!!!")

        if p < 0.05:
            print(f'O valor de p é: {p}')
            print('Provável haver diferença')
        else:
            print(f'O valor de p é: {p}')
            print('Provável que não haja diferença')

        return stat, p

    # Evaluate model by precision_at_k metric
    def precision_at_k(self, data, k= 20000):
        
        # Reseting index
        data = data.reset_index(self, drop=True)
        
        # Creating ranking order
        data['ranking'] = data.index + 1
        data['precision_at_k'] = data['response'].cumsum()/data['ranking']
        
        return data.loc[k, 'precision_at_k']

    # Evaluate model by recall_at_k metric
    def recall_at_k(self, data, k=20000):
        
        # Reseting index
        data = data.reset_index(drop=True)
        data['recall_at_k'] = data['response'].cumsum() / data['response'].sum()
        
        return data.loc[k, 'recall_at_k']


    # Function to tune the parameters
    def tunnig_gridsearch(self, Xtrain, model, param_grid, cv, scoring, refit):

        """
        Função para tunnig de parâmetros utilizando o RandomizedSearchCV"""
        
        X_= Xtrain.drop('target', axis=1)
        y_= Xtrain['target']
        

        search = RandomizedSearchCV(estimator=model,
                                param_distributions=param_grid,
                                scoring=scoring,
                                refit=refit,
                                    
                                cv=cv,
                                verbose=1,
                                n_jobs=-1,
                                return_train_score=True)
        search.fit(X_, y_)

        return search.best_params_, search.cv_results_


    # Function to Model Selection
    def model_selection(self, data_train, data_test, model, model_name, k=20000):
        
        # Split Data
        Xtrain = data_train.drop(['response'], axis=1)
        prec_k_pred = np.round(np.mean(prec_klist), 4).astype(str) + '+/-' + np.round(np.std(prec_k_list), 4).astype(str) 
        rec_k_pred =  np.round(np.mean(rec_k_list), 4).astype(str) + '+/-' + np.round(np.std(prec_k_list), 4).astype(str)
        
        return pd.DataFrame({'Model name': model_name,
                            'precision_at_k': prec_k_pred,
                            'recall_at_k': rec_k_pred}, index=[0])

        return search.best_params_, search.cv_results_


    # Function to Model Selection
    def model_selection(self, data_train, data_test, model, model_name, k=20000):
        
        # Split Data
        Xtrain = data_train.drop(['response'], axis=1)
        ytrain = data_train.response
        
        Xtest = data_test.drop(['response'], axis=1)
        ytest = data_test.response
        
        # fit and predict_proba
        model.fit(Xtrain, ytrain)
        pred = model.predict(Xtest)
        proba = model.predict_proba(Xtest)
        
        # 1d array
        proba_1d = proba[:,1].tolist()
        
        # Include in dataframe
        test_data = data_test.copy()
        test_data['score'] = proba_1d
        
        # Sort
        test_data = test_data.sort_values('score', ascending=False)

        # Painel
        painel_df = pd.DataFrame({'Model name': model_name,
                                'precision_at_k': precision_at_k(test_data, k),
                                'recall_at_k': recall_at_k(test_data, k)}, index=[0])
        
        return painel_df, model


    # Training the model with the tunned parameters
    def train_evaluate_model(self, data_train, data_test, model, model_name, k=20000):
        
        # Split Data
        Xtrain = data_train.drop(['response'], axis=1)
        ytrain = data_train.response
        
        Xtest = data_test.drop(['response'], axis=1)
        ytest = data_test.response
        
        # fit and predict_proba
        model.fit(Xtrain, ytrain)
        pred = model.predict(Xtest)
        proba = model.predict_proba(Xtest)
        
        # 1d array
        proba_1d = proba[:,1].tolist()
        
        # Include in dataframe
        test_data = data_test.copy()
        test_data['score'] = proba_1d
        
        # Sort
        test_data = test_data.sort_values('score', ascending=False)


        # Salvando o modelo em pickle
        with open(f'{model_name}', 'wb') as f:
            pickle.dump(model, f)

        # Painel
        painel_df = pd.DataFrame({'Model name': model_name,
                                'precision_at_k': precision_at_k(test_data, k),
                                'recall_at_k': recall_at_k(test_data, k)}, index=[0])
        
        
        _, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
        plot_cumulative_gain(ytest, proba, ax=ax1)
        plot_lift_curve(ytest, proba, ax=ax2)
        return painel_df, model

    # CrossValidation
    def cross_validation(self, model, model_name, training_Data, k_top, kfolds, verbose=False):
        
        xtrain = training_data.drop(['target'], axis=1)
        ytrain = training_data.target
        
        # cross-validation
        cv = sss(n_splits=kfolds)
        prec_k_list = []
        rec_k_list = []
        for train_index, val_index in cv,split(xtrain, ytrain):
            X_train, Xval = xtrain.iloc[train_index], xtrain.iloc[val_index]
            y_train, yval = ytrain.iloc[train_index], ytrain.iloc[val_index]
            
            # fit and predict_proba
            model.fit(X_train, y_train)
            yhat_proba = model.predict_proba(Xval)
            
            #transform yhat_proba to 1D-array
            yhat_proba_1d = yhat_proba[:, 1].tolist()
            
            #reconstruct dataframe
            full_val = pd.concat([Xval, yval], axis=1)
            full_val['score'] = yhat_proba_1d
            full_val = full_val.sort_values('score', ascending=False)
            
            #evaluate metrics and store in list
            prec_k_list.append(precision_at_k(full_val, k_top))
            rec_k_list.append(recall_at_k(full_val, k_top))
            
        # evaluate mean an std
        prec_k_pred = np.round(np.mean(prec_klist), 4).astype(str) + '+/-' + np.round(np.std(prec_k_list), 4).astype(str) 
        rec_k_pred =  np.round(np.mean(rec_k_list), 4).astype(str) + '+/-' + np.round(np.std(prec_k_list), 4).astype(str)
        
        return pd.DataFrame({'Model name': model_name,
                            'precision_at_k': prec_k_pred,
                            'recall_at_k': rec_k_pred}, index=[0])
