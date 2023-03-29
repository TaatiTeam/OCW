import numpy as np
import pandas as pd
import string
import json
import warnings

warnings.filterwarnings("ignore")
# each episode has 2 walls and each wall has 16 items
ITEM_NUM = 2*16

class DataFrameUtils:
    def __init__(self, df_text='only_connect_dataset_new_july_2022.xlsx', series_dict_path='series_dict.json'):
        self.df_text = df_text
        self.series_dict_path = series_dict_path

    def load_df(self):
        return pd.read_excel(self.df_text)

    def load_series_dict(self):
        with open(self.series_dict_path, 'r') as f:
            json_dict = json.load(f)
            # convert dict keys, values to int
            json_dict = {int(k): int(v) for k, v in json_dict.items()}
            return json_dict

    def accumulated_series_dict(self):
        temp_dict = self.load_series_dict().copy()
        for key in temp_dict:
            if key == 0:
                continue
            else:
                temp_dict[key] += temp_dict[key - 1]
        return temp_dict

    # function to preprocess text dataframe
    def preprocess_text(self):
        # make copy of dataframe
        df = self.load_df().copy()
        df = df[['Names', 'Answer']]
        df['Series'] = 0
        series_dict = self.accumulated_series_dict()
        # set series flag
        for key in series_dict:
            df['Series'][series_dict[key] * ITEM_NUM:] = key+1

        # remove punctuation
        df['Names'] = df['Names'].str.replace(' ', '-')
        df['Names'] = df['Names'].str.replace('=', '-')
        df['Names'] = df['Names'].str.replace('&', 'and')
        df['Names'] = df['Names'].str.replace('!', '')
        df['Names'] = df['Names'].str.replace(',', '')
        df['Names'] = df['Names'].str.replace('.', '')
        df['Names'] = df['Names'].str.replace('(', '')
        df['Names'] = df['Names'].str.replace(')', '')
        df['Names'] = df['Names'].str.replace('1/3', '33')
        df['Names'] = df['Names'].str.replace('2/3', '66')
        df['Answer'] = df['Answer'].str.replace('____ ', '')
        df['Answer'] = df['Answer'].str.replace(' ____', '')
        df['Answer'] = df['Answer'].str.replace('____', '')
        df['Answer'] = df['Answer'].str.replace(' ___', '')
        df['Answer'] = df['Answer'].str.replace('___ ', '')
        df['Answer'] = df['Answer'].str.replace('-', ' ')
        df['Answer'] = df['Answer'].str.replace('___', '')
        #edge cases
        df['Names'][265] = "".join([char for char in df['Names'][265] if char not in string.punctuation]).replace(' ', '-')
        df['Names'][578] = "".join([char for char in df['Names'][578] if char not in string.punctuation]).replace(' ', '-')
        # df['Names'][4813] = 'Theleme'
        # df['Names'][4584] = 'Jamon'
        df['Names'][4640] = df['Names'][4640].replace(' ', '-')
        #lowercase
        df['Names'] = df['Names'].str.lower()
        df['Answer'] = df['Answer'].str.lower()
        #convert to string
        df['Names'] = df['Names'].astype(str)


        return df
