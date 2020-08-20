from iotfunctions.anomaly import (SaliencybasedGeneralizedAnomalyScore, SpectralAnomalyScore,
                                  FFTbasedGeneralizedAnomalyScore)

from ..anomalymodels.anomaly import KMeansAnomalyScore

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func

import numpy as np
import pandas as pd
import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt
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


# kmeans_break = 1
# spectral_break = 100
# fft_break = 100
# sal_break = 2.8
# gen_break = 300


class ExistingModels(object):
    kmeans_break = 1
    spectral_break = 100
    fft_break = 100
    sal_break = 2.8
    gen_break = 300

    def __init__(self, columnname):
        '''
        run existing anomaly
        '''
        self.columnname = columnname

    def run_saliency(self, df):
        sali = SaliencybasedGeneralizedAnomalyScore(self.columnname, 12, sal)
        df_ret = self.execute(sali, df)
        return df_ret

    def run_kmeans(self, df, nclusters=40, contamination=0.1):
        kmi = KMeansAnomalyScore(self.columnname, 12, kmeans, nclusters=nclusters, contamination=contamination)
        df_ret = self.execute(kmi, df)
        return df_ret

    def run_fft(self, df):
        ffti = FFTbasedGeneralizedAnomalyScore(self.columnname, 12, fft)
        df_ret = self.execute(ffti, df)
        return df_ret

    def run_spectral(self, df):
        spsi = SpectralAnomalyScore(self.columnname, 12, spectral)
        df_ret = self.execute(spsi, df)
        return df_ret

    def run_all(self, df):
        df_ret = self.run_spectral(df)
        df_ret = self.run_saliency(df_ret)
        df_ret = self.run_kmeans(df_ret)
        df_ret = self.run_fft(df_ret)

        return df_ret

    def execute(self, model, df):
        et = model._build_entity_type(columns=[Column(self.columnname, Float())])
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

    def plot_all(self, df):
        plots = 5
        fig, ax = plt.subplots(plots, 1, figsize=(16, 24))

        self.plot_anomaly_column(df, ax, 0)
        self.plot_kmeans(df, ax, 1)
        self.plot_fft(df, ax, 2)
        self.plot_spectral(df, ax, 3)
        self.plot_sal(df, ax, 4)

    def plot_anomaly_column(self, df, ax, cnt):
        ax[cnt].plot(df.index, df[self.columnname], linewidth=1, color='black',
                     label=self.columnname)
        ax[cnt].legend(bbox_to_anchor=(1.1, 1.05))
        ax[cnt].set_ylabel('Input', fontsize=14, weight="bold")

    def plot_spectral(self, df, ax, cnt):
        df[spectralA] = df[spectral]
        df[spectralA].values[df[spectralA] < self.spectral_break] = np.nan
        df[spectralA].values[df[spectralA] > self.spectral_break] = self.spectral_break

        ax[cnt].plot(df.index, df[self.columnname], linewidth=1, color='black', label=self.columnname)
        ax[cnt].plot(df.index, df[spectral] / self.spectral_break, linewidth=2, color='dodgerblue', label=spectral)
        ax[cnt].plot(df.index, df[spectralA] / self.spectral_break, linewidth=10, color='red')
        ax[cnt].legend(bbox_to_anchor=(1.1, 1.05))
        ax[cnt].set_ylabel('Spectral \n like FFT for less "CPU"\n less sensitive', fontsize=13)

    def plot_sal(self, df, ax, cnt):
        df[salA] = df[sal]
        df[salA].values[df[salA] < self.sal_break] = np.nan
        df[salA].values[df[salA] > self.sal_break] = self.sal_break

        ax[cnt].plot(df.index, df[self.columnname], linewidth=1, color='black', label=self.columnname)
        ax[cnt].plot(df.index, df[sal] / self.sal_break, linewidth=2, color='chartreuse', label=sal)
        ax[cnt].plot(df.index, df[salA] / self.sal_break, linewidth=10, color='red')
        ax[cnt].legend(bbox_to_anchor=(1.1, 1.05))
        ax[cnt].set_ylabel('Saliency \n like FFT, part of Azure\'s approach', fontsize=13)

    def plot_fft(self, df, ax, cnt):
        df[fftA] = df[fft]
        df[fftA].values[df[fftA] < self.fft_break] = np.nan
        df[fftA].values[df[fftA] > self.fft_break] = self.fft_break

        ax[cnt].plot(df.index, df[self.columnname], linewidth=1, color='black', label=self.columnname)
        ax[cnt].plot(df.index, df[fft] / self.fft_break, linewidth=2, color='darkgreen', label=fft)
        ax[cnt].plot(df.index, df[fftA] / self.fft_break, linewidth=10, color='red')
        ax[cnt].legend(bbox_to_anchor=(1.1, 1.05))
        ax[cnt].set_ylabel('FFT \n detects frequency changes', fontsize=13)

    def plot_kmeans(self, df, ax, cnt):
        df[kmeansA] = df[kmeans]
        df[kmeansA].values[df[kmeansA] > self.kmeans_break] = self.kmeans_break
        df[kmeansA].values[df[kmeansA] < self.kmeans_break] = np.nan

        #ax[cnt].plot(df.index, df[self.columnname], linewidth=1, color='black', label=self.columnname)
        ax[cnt].plot(df.index, df[kmeans] / self.kmeans_break, linewidth=2, color='magenta', label=kmeans)
        ax[cnt].plot(df.index, df[kmeansA] / self.kmeans_break, linewidth=10, color='red')
        ax[cnt].legend(bbox_to_anchor=(1.1, 1.05))
        ax[cnt].set_ylabel('KMeans \n detects changes in "steepness"', fontsize=13)


class MatrixProfile(object):
    def __init__(self, df, columnname, windowsize=12):
        #description
        self.originaldf = df
        self.windowsize = windowsize
        self.columnname = columnname
        self.mp = None
        self.dfmp = None

    def plot_mp(self, threshold=0, columnsub=0):
        #description
        df_mp_anomaly = self.append_to_original()
        threshold = df_mp_anomaly['mp'].max() - threshold
        df_mp_anomaly['anomaly'] = df_mp_anomaly['mp']
        df_mp_anomaly['anomaly'].values[df_mp_anomaly['mp'] < threshold] = np.nan
        df_mp_anomaly['anomaly'].values[df_mp_anomaly['mp'] >= threshold] = 1

        fig, ax = plt.subplots(figsize=(25, 10))

        ax.plot(df_mp_anomaly['timestamp'], df_mp_anomaly[self.columnname] - columnsub, linewidth=1, color='black',
                label=self.columnname)
        ax.plot(df_mp_anomaly['timestamp'], df_mp_anomaly['mp'], linewidth=2, color='magenta', label='mp')
        ax.scatter(df_mp_anomaly['timestamp'], df_mp_anomaly['anomaly'], linewidth=10, color='red')
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_ylabel('Matrix Profile \n detects true anomalies', fontsize=13)

        return 1

    def run_model(self):
        self.mp = stumpy.stump(self.originaldf[self.columnname], m=self.windowsize)
        self.dfmp = pd.DataFrame(self.mp)
        return self.mp

    def append_to_original(self):
        df = self.originaldf.reset_index()
        mp_inc = list(self.dfmp[0])
        mp_inc.extend(([self.dfmp[0].min()] * (self.windowsize - 1)))
        df['mp'] = mp_inc
        return df
