import os
import os
import pandas as pd
import acquire as a
import numpy as np
import explore as e

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



def remove_out_lot(df):
    lot = df.lotsizesqft.values.copy()

    q1 = np.percentile(lot,25,interpolation="midpoint")
    q3 = np.percentile(lot,75,interpolation="midpoint")

    iqr = q3 - q1

    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    upper_index = np.where(df.lotsizesqft >= upper)[0]
    lower_index = np.where(df.lotsizesqft <= lower)[0]

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

    df.reset_index(inplace=True)
    return df

def remove_out_tot(df):
    totarea = df.totalarea.values.copy()
    q1 = np.percentile(totarea,25,interpolation="midpoint")
    q3 = np.percentile(totarea,75,interpolation="midpoint")

    iqr = q3 - q1

    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    upper_index = np.where(df.totalarea >= upper)[0]
    lower_index = np.where(df.totalarea <= lower)[0]

    df.drop(upper_index,inplace=True)
    df.drop(lower_index,inplace=True)
    df.reset_index(inplace=True)
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


def remove_outliers():
    df = e.sql_query_zillow()
    df = clean_zillow(df)
    df = remove_out_lot(df)
    df = remove_out_bth(df)
    df = remove_out_bed(df)
    df = remove_out_tot(df)
    df = remove_out_rmcnt(df)
    return df 
