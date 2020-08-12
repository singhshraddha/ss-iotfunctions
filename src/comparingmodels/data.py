import pandas as pd
import numpy as np
from . import utils


class DataProcessor(object):

    def __init__(self, filepath, timestampcolumn):
        '''
        Data Processor will read in raw data, and
        make it compatible to the Monitor pipeline
        '''
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath, index_col=False, parse_dates=[timestampcolumn])

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
