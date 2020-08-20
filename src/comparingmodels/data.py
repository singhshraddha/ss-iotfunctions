import pandas as pd
import numpy as np
from . import utils
from sklearn.preprocessing import StandardScaler


class DataProcessor(object):

    def __init__(self, filepath, timestampcolumn):
        '''
        Data Processor will read in raw data, and
        make it compatible to the Monitor pipeline
        '''
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath, index_col=False, parse_dates=[timestampcolumn])
        self.ss = StandardScaler()

    def get_data(self):
        return self.df

    def add_entity(self, entityname):
        self.df['entity'] = entityname

    def add_derived_column(self, columnname, fromcolumn, addvalue):
        self.df[columnname] = self.df[fromcolumn] + addvalue

    def change_column_name(self, columnname, fromcolumn):
        self.df[columnname] = self.df[fromcolumn]

    def drop_columns(self, columns):
        self.df = self.df.drop(columns=columns)

    def process_data(self):
        # and sort it by timestamp
        self.df = self.df.sort_values(by='timestamp')
        self.df = self.df.set_index(['entity', 'timestamp']).dropna()

    def filter_data(self, filterlist):
        self.df = self.df.filter(filterlist, axis=1)
        return self.df

    def standard_scaler(self, df, columnname):
        self.df = df
        self.df[columnname] = self.ss.fit_transform(np.array(self.df[columnname]).reshape(-1, 1))
        return self.df

    def scaler_transform(self, df, columnname):
        df[columnname] = self.ss.transform(np.array(df[columnname]).reshape(-1, 1))
        return df
