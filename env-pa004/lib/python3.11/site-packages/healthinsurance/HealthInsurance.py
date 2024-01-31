import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin


class HealthInsurance(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        Xtemp = X.copy()
        
        # FEATTURING ENGINNERING
        # Soma de previously_insured by gender
        pi_by_gender = pd.DataFrame(Xtemp.groupby(by='gender').sum()['previously_insured'])
        pi_by_gender.reset_index(inplace=True)

        Xtemp = Xtemp.merge(pi_by_gender, left_on='gender', right_on='gender', how='left')
        Xtemp.rename(columns={'previously_insured_y':'previously_insured_bygender','previously_insured_x':'previously_insured'}, inplace=True)
        
        # Annual_premium_brl - Exchange Rate - BRL X R = 0.07490
        Xtemp['annual_premium_brl'] = Xtemp['annual_premium'].apply(lambda x: x*0.07490)
        
        # Annual_premium_brl_ squared
        Xtemp['annual_premium_brl_squared'] = Xtemp['annual_premium_brl'].apply(lambda x: x**2)
        
        # Annual_premium_brl by region_code
        annual_premium_byregioncode = pd.DataFrame(Xtemp.groupby(by='region_code').sum()['annual_premium_brl'])
        annual_premium_byregioncode.reset_index(inplace=True)

        Xtemp = Xtemp.merge(annual_premium_byregioncode, left_on='region_code', right_on='region_code', how='left')
        Xtemp.rename(columns={'annual_premium_brl_y':'annual_premium_brl_byregionCode','annual_premium_brl_x':'annual_premium_brl'}, inplace=True)

        # Annual_premium_brl by policy_sales_channel
        annual_premium_bypolicySalesChannel = pd.DataFrame(Xtemp.groupby(by='policy_sales_channel').sum()['annual_premium_brl'])
        annual_premium_bypolicySalesChannel.reset_index(inplace=True)

        Xtemp = Xtemp.merge(annual_premium_bypolicySalesChannel, left_on='policy_sales_channel', right_on='policy_sales_channel', how='left')
        Xtemp.rename(columns={'annual_premium_brl_y':'annual_premium_bypolicySalesChannel','annual_premium_brl_x':'annual_premium_brl'}, inplace=True)

        # Annual_premium_brl divided by vintage
        Xtemp['annual_premium/vintage'] = Xtemp['annual_premium_brl']/Xtemp['vintage']
        
        # Age Categorical
        Xtemp['cat_age'] = Xtemp['age'].apply(lambda x: 'Faixa_20_25' if x > 20 and x <= 25 else 'Faixa_25_30' if  x > 25 and x <= 30 else 'Faixa_maior_30')
        
        # Median by policy_sales_channel
        dados_temp = Xtemp.groupby('policy_sales_channel')['annual_premium_brl'].median()
        dados_temp = pd.DataFrame(dados_temp)
        
        Xtemp = Xtemp.merge(dados_temp, on='policy_sales_channel')
        Xtemp.rename(columns={'annual_premium_brl_y': 'median_by_policy',
                             'annual_premium_brl_x':'annual_premium_brl'}, inplace=True)
        
        # Frequency Encoding region_cod
        df_temp = pd.DataFrame(Xtemp.groupby('region_code')['id'].count())
        df_temp.reset_index(inplace=True)
        
        Xtemp = Xtemp.merge(df_temp, on='region_code')
        Xtemp.rename(columns={'id_y': 'Freq_region_code',
                             'id_x':'id'}, inplace=True)
        
        # FILTERING CATEGORICALS
        categoricals = Xtemp.select_dtypes(include = ['object']).columns.to_list()
        
        # Encoder
        enc = OrdinalEncoder()
        for cat in categoricals:
            Xtemp[cat] = enc.fit_transform(np.array(Xtemp[cat]).reshape(-1,1))

        # FILTERING NUMERICALS
        numericals = Xtemp.select_dtypes(include = ['int64', 'float64']).columns.to_list()
        numericals.remove('annual_premium_brl')
        
        # Scaler
        scaler = StandardScaler()

        # Encoding
        for num in numericals:
            Xtemp[num] = scaler.fit_transform(np.array(Xtemp[num]).reshape(-1,1))
            
        #RobustScaler
        var = 'annual_premium_brl'
        rscaler = RobustScaler()
        
        Xtemp[var] = scaler.fit_transform(np.array(Xtemp[var]).reshape(-1,1))
        
        # FEATURE SELECTION
        variaveis = ['annual_premium_bypolicySalesChannel', 'vehicle_damage', 'median_by_policy', 'previously_insured', 'annual_premium_brl_byregionCode', 'annual_premium_brl', 'policy_sales_channel', 'Freq_region_code',
                     'region_code', 'age']
        
        Xtemp = Xtemp[variaveis]
        
        return Xtemp