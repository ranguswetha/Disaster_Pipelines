import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_CSVdata(messages_filepath, categories_filepath):
    """
    Load and merge datasets messages and categories 
    
    Inputs:
        Path to the CSV file containing messages
        Path to the CSV file containing categories
    Output:
        dataframe with merged data containing messages and categories
    """
    #reading messages and categories
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merge datasets
    df = pd.merge(messages,categories,on='id')
    return df 


def clean_dataframe(df):
    """
    Clean dataframe by removing duplicates and converting categories to binary values
    
    Inputs:
        dataframe with Combined data containing messages and categories
    Outputs:
        cleaned version of input dataframe 
    """
    
    #create df for each individual category
    categories = df['categories'].str.split(pat=';',expand=True)
    
    #access 1st row to get column names
    row = categories.iloc[[1]]
    
    # separate - from column value
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    
    # drop duplicates and concatenate category column to dataframe
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    
    return df


def save_datatoDB(df, database_filename):
    """
    Save transformed Data to SQLite Database 
    
    Inputs:
        marged data containing messages and categories with categories cleaned up
        Path to SQLite database
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_CSVdata(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_dataframe(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_datatoDB(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()