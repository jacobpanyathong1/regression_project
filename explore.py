import os
import pandas as pd
import explore as e

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
import warnings
from urllib.request import urlopen
import plotly.express as px
import env 
# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures



def sql_query_zillow():
    '''Acquire function to pull sql query for zillow
    '''
    sql_connection = env.get_connection('zillow')
    filename = "zillow_2017.csv"       #iris CSV
    if os.path.isfile(filename):
        return pd.read_csv(filename)        # filename returned in system files
    else:
        df = pd.read_sql(
            '''
            SELECT *
            FROM properties_2017 as a 
            JOIN propertylandusetype as b 
            USING (propertylandusetypeid) 
            WHERE propertylandusedesc = 'Single Family Residential';''', sql_connection
            )      # Read the SQL query into a dataframe
        df.to_csv(filename)        # Write that dataframe to save csv file. //"caching" the data for later.
        return df
def roun_fun():
    r2 = lambda x: round(x,2)
    return r2
def clean_zillow(df):
    ''' Clean zillow data function
    '''
    drop_col = ['Unnamed: 0', 
     'id', 
     'airconditioningtypeid', 
     'architecturalstyletypeid', 
     'buildingclasstypeid', 
     'finishedsquarefeet13',
     'finishedsquarefeet15',
     'decktypeid', 
     'finishedsquarefeet6',
     'hashottuborspa',
     'heatingorsystemtypeid', 
     'pooltypeid10', 
     'pooltypeid2',
     'pooltypeid7',
     'poolsizesum',
    'storytypeid',
    'typeconstructiontypeid',
    'yardbuildingsqft17',
    'yardbuildingsqft26',
    'fireplaceflag',
    'propertycountylandusecode',
    'propertylandusedesc',
    'landtaxvaluedollarcnt',
    'structuretaxvaluedollarcnt',
    'taxamount',
    'taxdelinquencyyear']

    df.drop(columns=df[drop_col], inplace=True)
    df.basementsqft.fillna(value=647.2, inplace=True)
    df.bathroomcnt.fillna(value=2.00, inplace=True)
    df.bedroomcnt.fillna(value=3.0, inplace=True)
    df.buildingqualitytypeid.fillna(value=6.3, inplace=True)
    df.calculatedbathnbr.fillna(value=2.3, inplace=True)
    df.finishedfloor1squarefeet.fillna(value=1379.8, inplace =True)
    df.calculatedfinishedsquarefeet.fillna(value=1831.5, inplace =True)
    df.finishedsquarefeet12.fillna(value=1831.5, inplace =True)
    #df.finishedsquarefeet13.fillna(value=1178.9, inplace =True)
    #df.finishedsquarefeet15.fillna(value=2754.9, inplace =True)
    df.finishedsquarefeet50.fillna(value=1392.0, inplace =True)
    #df.finishedsquarefeet6.fillna(value=2427.6, inplace =True)
    df.fireplacecnt.fillna(value=1.0, inplace =True)
    df.fullbathcnt.fillna(value=2.0, inplace =True)
    df.garagecarcnt.fillna(value=2.0, inplace =True)
    df.garagetotalsqft.fillna(value=383.2, inplace =True)
    #df.hashottuborspa.fillna(value=0, inplace =True)
    df.lotsizesquarefeet.fillna(value=22603.8, inplace =True)
    df.poolcnt.fillna(value=0, inplace =True)
    #df.poolsizesum.fillna(value=519.7, inplace =True)
    #df.pooltypeid7.fillna(value=0.0, inplace =True)
    df.regionidcity.fillna(value=12447.0, inplace =True)
    df.regionidneighborhood.fillna(value=118208.0, inplace =True)
    df.regionidzip.fillna(value=96987.0, inplace =True)
    df.roomcnt.fillna(value=0.0, inplace =True)
    df.threequarterbathnbr.fillna(value=1.0, inplace =True)
    df.unitcnt.fillna(value=1.0, inplace =True)
    #df.yardbuildingsqft17.fillna(value=321.5, inplace =True)
    #df.yardbuildingsqft26.fillna(value=278.4, inplace =True)
    df.yearbuilt.fillna(value=1955.0, inplace =True)
    df.numberofstories.fillna(value=1.0, inplace =True)
    #df.fireplaceflag.fillna(value=0, inplace =True)
    #df.structuretaxvaluedollarcnt.fillna(value=178142.9, inplace =True)
    df.taxvaluedollarcnt.fillna(value=450000.0, inplace =True)
    #df.landtaxvaluedollarcnt.fillna(value=268455.8, inplace =True)
    df.assessmentyear.fillna(value=2016.0, inplace =True)
    #df.taxamount.fillna(value=5408.95, inplace =True)
    #df.taxdelinquencyyear.fillna(value=15.0, inplace =True)
    df.censustractandblock.fillna(value=60486644377348.1, inplace =True)
    df.drop(columns=['propertyzoningdesc'], inplace=True)
    df.taxdelinquencyflag.replace({'Y' : 1}, inplace=True)
    df.taxdelinquencyflag.fillna(value=0, inplace=True)

    df.rename(columns={'propertylandusetypeid':'proplandusetypeid',
    'parcelid' : 'parid', 
    'basementsqft' : 'basesqft', 
    'bathroomcnt':'bathrooms', 
    'bedroomcnt' : 'bedrooms', 
    'buildingqualitytypeid': 'buildingqualityid',
    'calculatedbathnbr' : 'calcbathnbed',
    'finishedfloor1squarefeet' : 'finfloorsqft1',
    'calculatedfinishedsquarefeet' : 'calcfinsqft',
    'finishedsquarefeet12' : 'finfloorsqft12',
    'finishedsquarefeet13' : 'finfloorsqft13',
    'finishedsquarefeet15' : 'finfloorsqft15',
    'finishedsquarefeet50' : 'finfloorsqft50',
    'finishedsquarefeet6' : 'finfloorsqft6',
    'fips' : 'fed_info_proc_std',
    'fireplacecnt' : 'fireplaces',
    'fullbathcnt' : 'fullbaths',
    'garagecarcnt' : 'garagecars',
    'garagetotalsqft' : 'garagetotsqft',
    'hashottuborspa' : 'hottuborspa',
    'latitude' : 'lat',
    'longitude' : 'long',
    'lotsizesquarefeet' : 'lotsqft',
    'poolcnt' : 'pools',
    'poolsizesum' : 'poolsizetot',
    'rawcensustractandblock' : 'census_tract_block',
    'regionidcity' : 'regcityid',
    'regionidcounty' : 'regcontyid',
    'regionidneighborhood' : 'regidnbrhood',
    'taxvaluedollarcnt' : 'homevalue'}, inplace=True)
    return df 

def rts_formu(): # Clean return df.
    df = sql_query_zillow()
    df = clean_zillow(df)
    return df 

def train_validate_test(df):
    train_validate, test = train_test_split(df, train_size =.2, random_state = 100)
    train, validate = train_test_split(train_validate, train_size=.4, random_state = 100)
    return test, validate, train

def pull_tvt_formu():
    df = rts_formu()
    test, validate, train = train_validate_test(df)
    return test, validate, train


def xy_formula(train, validate, test):
    x_train = train.drop(columns=['homevalue'])
    y_train = train['homevalue']

    x_validate = validate.drop(columns=['homevalue'])
    y_validate = validate['homevalue']

    x_test = test.drop(columns=['homevalue'])
    y_test = test['homevalue']

    return x_train, y_train, x_validate, y_validate, x_test, y_test

def plot_variable_pairs():
    train, validate, test = pull_tvt_formu()
    x_train, y_train, x_validate, y_validate, x_test, y_test = xy_formula(train, validate, test)
    cont_var = ['bedrooms', 'bathrooms', 'sqft','homevalue', 'year', 'taxes', 'fips']

    for col in cont_var:
        plt.hist(train[col], bins =25)
        plt.title(f'{col} distribution')
        plt.show()
        sns.relplot(x='homevalue', y='taxes', data=train)
        plt.show()
        sns.lmplot(x='taxdollars', y='taxes', data=train)
        plt.show()
def rs_formula(train):
    rs_scaler = RobustScaler()
    rs_scaler.fit(train[['sqft', 'taxdollars', 'taxes', ]])
    more_trouble = rs_scaler.transform((train[['sqft', 'taxdollars', 'taxes', ]]))
    plt.subplot(121)
    plt.hist(train[['sqft', 'taxdollars', 'taxes']], bins=8)
    plt.title('Original data')

    plt.subplot(122)
    plt.hist(more_trouble, bins=8)
    plt.title('Transformed data')
    plt.show()


def alpha():
    alpha = 0.05
    return alpha

def null_hyp_variables(variable_1, variable_2):
    null_hypothesis = print(f'there is no significant relationship between {variable_1} and {variable_2} are independent on each other')
    return null_hypothesis

def alt_hyp_variables(var1, var2):
    alt_hypothesis = print(f'there is a significant relationship between {var1} and {var2} are dependent upon each other')
    return alt_hypothesis


def pearson_findings(p, alpha, varable):
   
    if p < alpha:
        print(f'We reject the null hypothesis that,', null_hypothesis)
    
    else:
        print(f'We fail to reject the null hypothesis that, {null_hyp}')
        print(null_hyp)


def corr_ranks(df):
    pc = .70
    min_null = pc * df.shape[0] + 1
    df.dropna(axis=1 ,thresh=min_null ,inplace=True)
    df.corr()[(df.corr() >= 0.50) & (df.corr() < 1)]['taxvaluedollarcnt'].dropna()

    
    sns.set_theme(style='ticks')
    f, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Correlation of weighted ranks for each feature")
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(data=df.corr(), mask=mask, center = 0, cmap=cmap, vmax=.3, cbar_kws={"shrink": .5})

    plt.show()

def remove_out_lot(df):
    lot = df.lotsqft.values.copy()

    q1 = np.percentile(lot,25,interpolation="midpoint")
    q3 = np.percentile(lot,75,interpolation="midpoint")

    iqr = q3 - q1

    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    upper_index = np.where(df.lotsqft >= upper)[0]
    lower_index = np.where(df.lotsqft <= lower)[0]

    df.drop(upper_index,inplace=True)
    df.drop(lower_index,inplace=True)

    df.reset_index(inplace=True)
    return df

def remove_out_bth(df):
    bthrms= df.bathrooms.values.copy()	
    q1 = np.percentile(bthrms,25,interpolation="midpoint")
    q3 = np.percentile(bthrms,75,interpolation="midpoint")

    iqr = q3 - q1

    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    upper_index = np.where(df.bathrooms >= upper)[0]
    lower_index = np.where(df.bathrooms <= lower)[0]

    df.drop(upper_index,inplace=True)
    df.drop(lower_index,inplace=True)

    df.reset_index(inplace=True)
    return df

def remove_out_bed(df):
    bdrms = df.bedrooms.values.copy()
    q1 = np.percentile(bdrms,25,interpolation="midpoint")
    q3 = np.percentile(bdrms,75,interpolation="midpoint")

    iqr = q3 - q1

    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    upper_index = np.where(df.bedrooms >= upper)[0]
    lower_index = np.where(df.bedrooms <= lower)[0]

    df.drop(upper_index,inplace=True)
    df.drop(lower_index,inplace=True)

    
    return df

def remove_out_tot(df):
    totarea = df.finfloorsqft15.values.copy()
    q1 = np.percentile(totarea,25,interpolation="midpoint")
    q3 = np.percentile(totarea,75,interpolation="midpoint")

    iqr = q3 - q1

    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    upper_index = np.where(df.finfloorsqft15 >= upper)[0]
    lower_index = np.where(df.finfloorsqft15 <= lower)[0]

    df.drop(upper_index,inplace=True)
    df.drop(lower_index,inplace=True)
    
    return df

def remove_out_rmcnt(df):
    rmcnt = df.roomcnt.values.copy()
        
    q1 = np.percentile(rmcnt,25,interpolation="midpoint")
    q3 = np.percentile(rmcnt,75,interpolation="midpoint")

    iqr = q3 - q1

    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    upper_index = np.where(df.roomcnt >= upper)[0]
    lower_index = np.where(df.roomcnt <= lower)[0]

    df.drop(upper_index,inplace=True)
    df.drop(lower_index,inplace=True)
    return df

def remove_outliers():
    df = sql_query_zillow()
    df = clean_zillow(df)
    df = remove_out_lot(df)
    df = remove_out_bth(df)
    df = remove_out_bed(df)
    df = remove_out_rmcnt(df)
    return df 



def slr(y_train, y_validate):
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # 1. Predict G3_pred_mean
    baseline = y_train['homevalue'].mean()
    y_train['homevalue_pred_mean'] = baseline
    y_validate['homevalue_pred_mean'] = baseline

    # 2. compute G3_pred_median
    y_pred_median = y_train['homevalue'].median()
    y_train['homevalue_pred_median'] = y_pred_median
    y_validate['homevalue_pred_median'] = y_pred_median

    # 3. RMSE of G3_pred_mean
    rmse_train = mean_squared_error(y_train.homevalue, y_train.homevalue_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.homevalue, y_validate.homevalue_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
        "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    # 4. RMSE of G3_pred_median
    rmse_train = mean_squared_error(y_train.homevalue, y_train.homevalue_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.homevalue, y_validate.homevalue_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
        "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

def ll(x_train_scaled, x_validate_scaled, y_train, y_validate):
    # create the model object
    lars = LassoLars(alpha=1.0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(x_train_scaled, y_train)

    # predict train
    y_train['homevalue_pred_lars'] = lars.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train, y_train.homevalue_pred_lars)**(1/2)

    # predict validate
    y_validate['homevalue_pred_lars'] = lars.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.homevalue, y_validate.homevalue_pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
        "\nValidation/Out-of-Sample: ", rmse_validate)

def tr(x_train_scaled, x_validate_scaled, y_train, y_validate):
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0.95)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(x_train_scaled, y_train)

    # predict train
    y_train['homevalue_pred_glm'] = glm.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train, y_train.homevalue_pred_glm)**(1/2)

    # predict validate
    y_validate['homevalue_pred_glm'] = glm.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate, y_validate.homevalue_pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=1 & alpha=0.95\nTraining/In-Sample: ", rmse_train, 
        "\nValidation/Out-of-Sample: ", rmse_validate)


def lm2_form(x_train_scaled, x_validate_scaled, x_test_scaled, y_train, y_validate):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    x_train_degree2 = pf.fit_transform(x_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    x_validate_degree2 = pf.transform(x_validate_scaled)
    x_test_degree2 = pf.transform(x_test_scaled)

    # create the model object
    lm2 = LinearRegression()

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(x_train_degree2, y_train)

    # predict train
    y_train['homevalue_pred_lm2'] = lm2.predict(x_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train, y_train.homevalue_pred_lm2)**(1/2)

    # predict validate
    y_validate['homevalue_pred_lm2'] = lm2.predict(x_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate, y_validate.homevalue_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
        "\nValidation/Out-of-Sample: ", rmse_validate)

def model_viz(y_validate):
    # y_validate.head()
    plt.figure(figsize=(8,6))
    plt.plot(y_validate, y_validate.homevalue_pred_mean, alpha=.5, color="gray", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Mean", (16, 9.5))
    plt.plot(y_validate, y_validate, alpha=.5, color="blue", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5), rotation=15.5)

    plt.scatter(y_validate, y_validate.homevalue_pred_lm, 
                alpha=.5, color="red", s=100, label="Model: LinearRegression")
    plt.scatter(y_validate, y_validate.homevalue_pred_glm, 
                alpha=.5, color="yellow", s=100, label="Model: TweedieRegressor")
    plt.scatter(y_validate, y_validate.homevalue_pred_lm2, 
                alpha=.5, color="green", s=100, label="Model 2nd degree Polynomial")
    plt.legend()
    plt.xlabel("Actual Home Value")
    plt.ylabel("Predicted Home Value")
    plt.title("Where are predictions more extreme? More modest?")
    # plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
    # plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
    plt.show()

def total_model_viz(y_validate):
    # plot to visualize actual vs predicted. 
    plt.figure(figsize=(8,6))
    plt.hist(y_validate, color='#0072BD', alpha=.5, label="Actual Home values", histtype='step')
    plt.hist(y_validate.homevalue_pred_lm, color="#A2142F", alpha=.5, label="Model: LinearRegression", histtype='step')
    plt.hist(y_validate.homevalue_pred_glm, color='#EDB120', alpha=.5, label="Model: TweedieRegressor")
    plt.hist(y_validate.homevalue_pred_lm2, color='#77AC30', alpha=.5, label="Model 2nd degree Polynomial", histtype='step')
    plt.hist(y_validate.homevalue_pred_lars, color='seagreen', alpha=.5, label="Model: LassoLars")
    plt.xlabel("Final Prediction")
    plt.ylabel("Number of Houses")
    plt.title("Comparing the Distribution of Actual Home Values to Distributions of Predicted Home Values for the Top Models")
    plt.legend()
    plt.show()

def best_test(x_test_scaled):
    y_test = pd.DataFrame(y_test)
    lm2 = LinearRegression()
    lm2.fit(x_test_scaled, y_test)
    # predict on test
    y_test['homevalue_pred_lm'] = lm2.predict(x_test_scaled)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test, y_test.homevalue_pred_lm)**(1/2)

    print("RMSE for OLS Model using LinearRegression\nOut-of-Sample Performance: ", rmse_test)