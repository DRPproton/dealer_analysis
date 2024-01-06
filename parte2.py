"""
Create Database
"""

import sqlite3
import pandas as pd


def clean_num_values(df, columns):
    """_summary_

    Args:
        df DataFrame: DataFrame with '$ and ',' in a number column
        columns List: List of columns needs to be alter

    Returns:
        DataFrame: Remove '$ and ',' from a number column
    """
    df_temp = df.copy()
    for col in columns:
        df_temp[col] = df_temp[col].str.replace('$', '').str.replace(',', '').astype(float)
        
    return df_temp


def convert_data(df):
    """_summary_

    Args:
        df DataFrame: DataFrame needs to be altered

    Returns:
        DataFrame: Clean DataFrame
    """
    columns_to_clean = [col for col in df.columns if '$' in col]  # Getting columns that needs to be clean
    
    return clean_num_values(df, columns_to_clean)


def create_db():
    # Define the name of the SQLite database file (change it to your desired name)
    db_filename = 'database.db'
    # Establish a connection to the SQLite database (this will create the database if it doesn't exist)
    conn = sqlite3.connect(db_filename)

    # Read the CSV files into a DataFrames
    df_directorio = pd.read_csv('Directorio.csv',  encoding = 'ISO-8859-1')
    df_directorio.drop(columns='Unnamed: 2', inplace=True) # Eliminar columna repetida
    
    df_Q1_2021 = pd.read_csv('Q1_2021 (1).csv')
    df_Q1_2021_clean = convert_data(df_Q1_2021)
    
    df_clean = pd.read_csv('clean_data.csv')
    
    # Write the DataFrames to a tables in the database
    df_directorio.to_sql('Directorio', conn, if_exists='replace', index=False)
    df_Q1_2021_clean.to_sql('Q1_2021', conn, if_exists='replace', index=False)
    df_clean.to_sql('Q1_y_Q2_2021', conn, if_exists='replace', index=False)


    # Commit changes to the database and close the connection
    conn.commit()
    conn.close()


# Uncomment to make the DB for the first time. If the DB exist comment the line
# create_db()

