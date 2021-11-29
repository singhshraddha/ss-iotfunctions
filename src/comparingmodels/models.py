from iotfunctions.anomaly import (SaliencybasedGeneralizedAnomalyScore,
                                  SpectralAnomalyScore,
                                  FFTbasedGeneralizedAnomalyScore,
                                  SaliencybasedGeneralizedAnomalyScoreV2,
                                  FFTbasedGeneralizedAnomalyScoreV2,
                                  KMeansAnomalyScoreV2)

from ..anomalymodels.anomaly import KMeansAnomalyScore

from iotfunctions.db import Database
from iotfunctions.dbtables import FileModelStore

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func

from . import utils

import numpy as np
import pandas as pd
import seaborn as sns
import json
import time
from scipy import stats

sns.set()
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import stumpy

##SET GLOBALS
# setting to make life easier
kmeans = 'KmeansAnomalyScore'
fft = 'FFTAnomalyScore'
spectral = 'SpectralAnomalyScore'
sal = 'SaliencyAnomalyScore'
gen = 'GeneralizedAnomalyScore'
kmeansA = 'kmeansAnomaly'
kmeansB = 'kmeansAnomalyB'
spectralA = 'spectralAnomaly'
fftA = 'fftAnomaly'
salA = 'salAnomaly'
genA = 'genAnomaly'

path_to_credentials = '../dev_resources/credentials_as_dev.json'

def get_job_settings():
    # set up a db object with a FileModelStore to support scaling
    with open(path_to_credentials, encoding='utf-8') as F:
        credentials = json.loads(F.read())
    db_schema = None
    fm = FileModelStore()
    db = Database(credentials=credentials, model_store=fm)
    jobsettings = dict(db=db, _db_schema='public', save_trace_to_file=True)

    return jobsettings


class ExistingModels(object):
    kmeans_break = 50000
    spectral_break = 100
    fft_break = 1000
    sal_break = 1000
    gen_break = 300

    def __init__(self, columnname):
        '''
        run existing anomaly
        '''
        self.columnname = columnname

    def run_saliency(self, df, normalize=False, windowsize=12):
        if normalize:
            sali = SaliencybasedGeneralizedAnomalyScoreV2(self.columnname, 12, normalize, sal)
        else:
            sali = SaliencybasedGeneralizedAnomalyScore(self.columnname, windowsize, sal)
        df_ret = self.execute(sali, df)
        return df_ret

    def run_kmeans(self, df, nclusters=40, contamination=0.1, normalize=False, windowsize=12):
        if normalize:
            kmi = KMeansAnomalyScoreV2(self.columnname, windowsize, normalize, kmeans)
        else:
            kmi = KMeansAnomalyScore(self.columnname, windowsize, kmeans, nclusters=nclusters, contamination=contamination)
        df_ret = self.execute(kmi, df)
        return df_ret

    def run_fft(self, df, normalize=False, windowsize=12):
        if normalize:
            ffti = FFTbasedGeneralizedAnomalyScoreV2(self.columnname, windowsize, normalize, fft)
        else:
            ffti = FFTbasedGeneralizedAnomalyScore(self.columnname, windowsize, fft)
        df_ret = self.execute(ffti, df)
        return df_ret

    def run_spectral(self, df, windowsize=12):
        spsi = SpectralAnomalyScore(self.columnname, windowsize, spectral)
        df_ret = self.execute(spsi, df)
        return df_ret

    def run_all(self, df, normalize=False, windowsize=12):
        df_ret = self.run_spectral(df, windowsize=windowsize)
        df_ret = self.run_saliency(df_ret, normalize=normalize, windowsize=windowsize)
        df_ret = self.run_kmeans(df_ret, normalize=normalize, windowsize=windowsize)
        df_ret = self.run_fft(df_ret, normalize=normalize, windowsize=windowsize)

        return df_ret

    def execute(self, model, df):
        jobsettings = get_job_settings()
        et = model._build_entity_type(columns=[Column(self.columnname, Float())], **jobsettings)
        model._entity_type = et
        return model.execute(df=df)

    def visualize_kmeans(self, df, threshold=kmeans_break):
        plots = 2
        fig, ax = plt.subplots(plots, 1, figsize=(25, 15))

        self.kmeans_break = threshold
        print(f'kmeans anomaly threshold: {self.kmeans_break}')

        df[kmeans] = df[kmeans].astype(float)

        self.plot_anomaly_column(df, ax, 0)
        self.plot_kmeans(df, ax, 1)

    def plot_all(self, df, scaling_factor=100):
        plots = 5
        fig, ax = plt.subplots(plots, 1, figsize=(16, 24))

        self.plot_anomaly_column(df, ax, 0)
        self.plot_kmeans(df, ax, 1, scaling_factor=scaling_factor)
        self.plot_fft(df, ax, 2, scaling_factor=scaling_factor)
        self.plot_spectral(df, ax, 3, scaling_factor=scaling_factor)
        self.plot_sal(df, ax, 4, scaling_factor=scaling_factor)

    def plot_anomaly_column(self, df, ax, cnt):
        ax[cnt].plot(df.index, df[self.columnname], linewidth=1, color='black',
                     label=self.columnname)
        ax[cnt].legend(bbox_to_anchor=(1.1, 1.05))
        ax[cnt].set_ylabel('Input', fontsize=14, weight="bold")

    def plot_spectral(self, df, ax, cnt, scaling_factor=100):
        df[spectralA] = df[spectral]
        df[spectralA].values[df[spectralA] < self.spectral_break] = np.nan
        df[spectralA].values[df[spectralA] > self.spectral_break] = self.spectral_break

        ax[cnt].plot(df.index, df[self.columnname], linewidth=1, color='black', label=self.columnname)
        ax[cnt].plot(df.index, df[spectral] * scaling_factor, linewidth=2, color='dodgerblue', label=spectral)
        ax[cnt].plot(df.index, df[spectralA] / self.spectral_break, linewidth=10, color='red')
        ax[cnt].legend(bbox_to_anchor=(1.1, 1.05))
        ax[cnt].set_ylabel('Spectral \n like FFT for less "CPU"\n less sensitive', fontsize=13)

    def plot_sal(self, df, ax, cnt, scaling_factor=100):
        df[salA] = df[sal]
        df[salA].values[df[salA] < self.sal_break] = np.nan
        df[salA].values[df[salA] > self.sal_break] = self.sal_break

        ax[cnt].plot(df.index, df[self.columnname], linewidth=1, color='black', label=self.columnname)
        ax[cnt].plot(df.index, df[sal] * scaling_factor, linewidth=2, color='chartreuse', label=sal)
        ax[cnt].plot(df.index, df[salA] / self.sal_break, linewidth=10, color='red')
        ax[cnt].legend(bbox_to_anchor=(1.1, 1.05))
        ax[cnt].set_ylabel('Saliency \n like FFT, part of Azure\'s approach', fontsize=13)

    def plot_fft(self, df, ax, cnt, scaling_factor=100):
        df[fftA] = df[fft]
        df[fftA].values[df[fftA] < self.fft_break] = np.nan
        df[fftA].values[df[fftA] > self.fft_break] = self.fft_break

        ax[cnt].plot(df.index, df[self.columnname], linewidth=1, color='black', label=self.columnname)
        ax[cnt].plot(df.index, df[fft] * scaling_factor, linewidth=2, color='darkgreen', label=fft)
        ax[cnt].plot(df.index, df[fftA] / self.fft_break, linewidth=10, color='red')
        ax[cnt].legend(bbox_to_anchor=(1.1, 1.05))
        ax[cnt].set_ylabel('FFT \n detects frequency changes', fontsize=13)

    def plot_kmeans(self, df, ax, cnt, scaling_factor=100):
        df[kmeansA] = df[kmeans]
        df[kmeansA].values[df[kmeansA] > self.kmeans_break] = self.kmeans_break
        #df[kmeansA].values[df[kmeansA] < self.kmeans_break] = np.nan

        #ax[cnt].plot(df.index, df[self.columnname], linewidth=1, color='black', label=self.columnname)
        ax[cnt].plot(df.index, df[kmeans] * scaling_factor, linewidth=2, color='magenta', label=kmeans)
        #ax[cnt].plot(df.index, df[kmeansA] / self.kmeans_break, linewidth=10, color='red')
        ax[cnt].legend(bbox_to_anchor=(1.1, 1.05))
        ax[cnt].set_ylabel('KMeans \n detects changes in "steepness"', fontsize=13)


class MatrixProfile(object):
    def __init__(self, df, columnname, timestamp='timestamp', windowsize=12, z_normalized=True):
        #description
        self.originaldf = df
        self.windowsize = windowsize
        self.columnname = columnname
        self.timestamp = timestamp
        self.mp = None
        self.dfmp = None
        self.z_normalized = z_normalized

    def top_k_discords_idx(self, df, k=1, exclusion_zone=12):
        """
        :param df: dataframe with values of mp
        :param k: num of discords
        :param exclusion_zone: area to exclude around discord when finding next discord
        :return: the top k discords with exclusion zone enabled. Only the largest discord
        in a ~2*exclusion zone window is considered
        """
        tic = time.time()
        exclusion_zone = exclusion_zone

        index = np.argsort(df['mp'].to_list())[::-1]
        index_discords = []
        for idx in index:
            if not any(item in list(range(idx - exclusion_zone, idx + exclusion_zone)) for item in index_discords):
                index_discords.append(idx)

            if len(index_discords) >= k:
                break
        toc = time.time()
        print(f'Execution time (top_k_discords_idx): {toc - tic}')
        return index_discords

    def zscore_threshold_idx(self, df, threshold=3, exclusion_zone=12):
        """
        :param df pd.DataFarame data
        :param threshold number of standard deviation above mean to retain
        :param exclusion_zone number {area +/-} to exclude around discord when finding next discord
        :return: all discord value above threshold
        """
        tic = time.time()
        df["mp_zscore"] = stats.zscore(df["mp"].values)
        #threshold = df["mp"].max() - threshold * df["mp"].std()
        # index for all anomalies
        # anomalies are defined as mp with zscore above threshold
        # above_threshold_index = df.index[df["mp_zscore"] > threshold].tolist()
        above_threshold_index = df.index[df["mp"] > threshold].tolist()
        # idx for anomalies with exclusion zone
        zscore_index = []
        for idx in above_threshold_index:
            if not utils.is_in_exclusion_zone(idx, zscore_index, exclusion_zone):
                zscore_index.append(idx)
        toc = time.time()
        print(f'Selected threshold (zscore_threshold_idx) {threshold}')
        print(f'Execution time (zscore_threshold_idx): {toc - tic}')

        return zscore_index

    def plot_mp(self, topk=1, threshold=3, scaling_factor=1, printdata=False, motif=False, usethreshold=True):
        """

        :param usethreshold: bool when true use zscore threshold, when false use top-k discords
        :param topk: nsmallest values to label as discords
        :param threshold: discords (z-scored mp) above threshold are considered anomalies
        :param shiftoriginal: shift the univariate signal on the plot
        :param printdata: print a few discords
        :param motif: plot pattern
        :return: data plotted as dataframe
        """
        #appends mp to original df
        df_mp_anomaly = self.append_to_original()

        # mark the X highest values as discord, where X corresponds to user input for threshold
        if usethreshold:
            anomaly_index = self.zscore_threshold_idx(df_mp_anomaly, threshold=threshold,exclusion_zone=self.windowsize)
        else:
            anomaly_index = self.top_k_discords_idx(df_mp_anomaly, k=topk, exclusion_zone=self.windowsize)

        df_mp_anomaly['anomaly'] = np.nan
        df_mp_anomaly.loc[anomaly_index, 'anomaly'] = df_mp_anomaly['mp'].max() + 1 + 1
        xindex_anomaly = anomaly_index
        xpositions_anomaly = df_mp_anomaly.loc[anomaly_index]['timestamp'].to_list()

        # mark the 2 lowest values as subsequence
        if motif:
            df_mp_anomaly['pattern'] = np.nan
            df_mp_anomaly.loc[df_mp_anomaly.mp == df_mp_anomaly.mp.min(), 'pattern'] = df_mp_anomaly['mp'].min() - 1
            df_pattern = df_mp_anomaly[df_mp_anomaly.mp == df_mp_anomaly.mp.min()]
            xpositions_patterns = df_mp_anomaly[df_mp_anomaly.mp == df_mp_anomaly.mp.min()]['timestamp'].to_list()

        if printdata:
            print(f'Data for Unique discords\n '
                  f'{df_mp_anomaly.loc[anomaly_index][[self.timestamp, self.columnname,"mp"]]}')
            print(f'Data for patterns\n {df_pattern[[self.timestamp, self.columnname, "mp"]]}')

        fig, ax = plt.subplots(figsize=(20, 5))

        ax.plot(df_mp_anomaly['timestamp'], df_mp_anomaly[self.columnname], linewidth=1, color='black',
                label=self.columnname)
        #plot matrix profile
        ax.plot(df_mp_anomaly['timestamp'], df_mp_anomaly['mp'] * scaling_factor, linewidth=2, color='magenta', 
                label='mp')
        #plotting discords
        if anomaly_index:
            ax.scatter(df_mp_anomaly['timestamp'], df_mp_anomaly['anomaly'], linewidth=5, color='red')
        for idx, xc in enumerate(xpositions_anomaly):
            plt.axvline(x=xc, color='r', linestyle='--', linewidth=1)
            loc = min(df_mp_anomaly.shape[0] - 1, xindex_anomaly[idx] + self.windowsize)
            start = mdates.date2num(xc)
            end = mdates.date2num(df_mp_anomaly.iloc[loc]['timestamp'])
            rect = Rectangle((start, 0), end-start, df_mp_anomaly['mp'].mean(), facecolor='lightgrey')
            ax.add_patch(rect)
        #plot motif/pattern
        if motif:
            ax.scatter(df_mp_anomaly['timestamp'], df_mp_anomaly['pattern'], linewidth=5, color='blue')
            for xc in xpositions_patterns:
                plt.axvline(x=xc, color='k', linestyle='--', linewidth=1)

        ax.legend(bbox_to_anchor=(1.1, 1.05))

        if usethreshold:
            ax.set_ylabel(f'Matrix Profile \n uses threshold={threshold} to detect anomalies with input z-normalization'
                          f'={self.z_normalized}',fontsize=13)
        else:
            ax.set_ylabel(
                f'Matrix Profile \n detects k={threshold} discords with input z-normalization={self.z_normalized}',
                fontsize=13)

        return df_mp_anomaly

    def run_model(self):
        #during data processing it is sorted by time
        self.originaldf[self.columnname] = self.originaldf[self.columnname].astype(float)
        if self.z_normalized:
            self.mp = stumpy.stump(self.originaldf[self.columnname], m=self.windowsize)
        else:
            self.mp = stumpy.aamp(self.originaldf[self.columnname], m=self.windowsize)
        self.dfmp = pd.DataFrame(self.mp)
        return self.mp

    def append_to_original(self):
        df = self.originaldf.reset_index()
        mp_inc = list(self.dfmp[0])
        #mp_inc.extend(([self.dfmp[0].mean()] * (self.windowsize - 1)))
        df = df.iloc[:-(self.windowsize-1)]
        df['mp'] = mp_inc
        return df