# Acquire File 
import os       # import os filepaths
import env      #importing get_connection function
import pandas as pd     #import Pandas library as pd

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