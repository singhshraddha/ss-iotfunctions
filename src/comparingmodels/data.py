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


class MonitorData(object):

    def __init__(self, data_sel=None):
        """
        Make functions to get processed data - ready for anomaly detection
        """
        self.data_sel = data_sel

    def get_data(self, datapath):
        if self.data_sel.lower() == 'temperature':
            return self.get_temperature_data(datapath)
        if self.data_sel.lower() == 'vibrations':
            return self.get_amrstark_data(datapath)
        if self.data_sel.lower() == 'floatvalue':
            return self.get_cakebreak_data(datapath)
        if self.data_sel.lower() == 'pressure':
            return self.get_anomaly_sample_data(datapath)

    def get_temperature_data(self, datapath, timestampcolumn='timestamp'):
        dp = DataProcessor(filepath=datapath, timestampcolumn=timestampcolumn)
        dp.add_entity('MyRoom')
        dp.add_derived_column('Temperature', 'value', 20)
        dp.process_data()
        df_i = dp.get_data()
        df_i = df_i.drop(columns=['value'])
        return df_i

    def get_amrstark_data(self, datapath, timestampcolumn='RCV_TIMESTAMP_UTC'):
        # customer data
        dp = DataProcessor(filepath=datapath, timestampcolumn=timestampcolumn)
        dp.change_column_name('entity', 'DEVICE_ID')
        dp.change_column_name('timestamp', timestampcolumn)
        dp.process_data()
        df_i = dp.get_data()
        df_i.describe()

        #filter data to specific needs
        listAttr = ['timestamp', 'entity', 'vibrations', 'rms', 'accel_speed', 'accel_power_0', 'accel_power_1',
                    'accel_power_2', 'accel_power_3', 'accel_power_4']
        utils.l2norm(df_i, 'vibrations', 'VIBRATIONS_XAXIS', 'VIBRATIONS_YAXIS', 'VIBRATIONS_ZAXIS')
        utils.l2norm(df_i, 'rms', 'RMS_X', 'RMS_Y', 'RMS_Z')
        utils.l2norm(df_i, 'accel_speed', 'ACCEL_SPEED')
        utils.unrollAccel(df_i)

        df_i = df_i.filter(listAttr, axis=1)
        return df_i

    def get_cakebreak_data(self, datapath, timestampcolumn='timestamp'):
        dp = DataProcessor(filepath=datapath, timestampcolumn=timestampcolumn)
        dp.change_column_name('entity', 'deviceid')
        dp.process_data()
        df_i = dp.get_data()
        return df_i

    def get_anomaly_sample_data(self, datapath, timestampcolumn='EVT_TIMESTAMP'):
        dp = DataProcessor(filepath=datapath, timestampcolumn=timestampcolumn)
        dp.change_column_name('entity', 'DEVICEID')
        dp.change_column_name('timestamp', 'EVT_TIMESTAMP')
        dp.drop_columns(columns=['LOGICALINTERFACE_ID', 'EVENTTYPE', 'FORMAT', 'TURBINE_ID'])
        dp.process_data()
        df = dp.get_data()

        return df
