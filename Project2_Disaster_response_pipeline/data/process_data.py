import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Purpose: Load data from 2 filepaths:
    Input: 
    * messages_filepath: disaster_messages.csv filepath
    * categories_filepath: disaster_categories.csv filepath

    Output:
    * df: dataframe containing data for training model
    """
    # Load df_message from messages_filepath
    df_messages = pd.read_csv(messages_filepath)
    # Load df_categories from categories_filepath
    df_categories = pd.read_csv(categories_filepath)
    # merge 2 datasets
    df = df_messages.merge(df_categories, on=['id'])

    return df    


def clean_data(df):
    """
    Purpose: Clean data (split, Clear NA value, ...)

    Input: 
    * df: dataframe want to clean

    Output:
    * df: clean dataframe
    """
    # Process in categories columns
    # Create categories_df from categories columns and split by ;
    categories_df = df['categories'].str.split(";", expand=True)
    # Create new column names for categories_df from first row of the categories_df
    category_colnames = categories_df.iloc[0].apply(lambda x: x[:-2]).tolist()
    # Use category_colnames for rename the columns of categories_df
    categories_df.columns = category_colnames

    # Take the last character of each value in each column and convert all of them to integer type
    categories_df = categories_df.apply(lambda x: x.astype(str).str[-1]).astype(int)

    # Concatenate the original df with the new categories_df
    df = pd.concat([df, categories_df], axis=1)

    # Drop the original categories column because it is no longer needed
    df.drop('categories', axis=1, inplace=True)

    # Delete rows in df when related column has value 2
    df = df[df['related'] != 2]

    # Remove rows where the 'message' column has a NaN value
    df = df.dropna(subset=['message'])

    return df


def save_data(df, database_filename):
    """
    Purpose: Stores df in a SQLite database.

    Input:
    * df: dataframe want to save
    * database_filename: name of sqlite database

    Output:
    (Database was saved)
    """
    engine = create_engine('sqlite:///'+ str(database_filename))
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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