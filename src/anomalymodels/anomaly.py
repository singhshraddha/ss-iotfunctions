# *****************************************************************************
# © Copyright IBM Corp. 2018-2020.  All Rights Reserved.
#
# This program and the accompanying materials
# are made available under the terms of the Apache V2.0
# which accompanies this distribution, and is available at
# http://www.apache.org/licenses/LICENSE-2.0
#
# *****************************************************************************

'''
The Built In Functions module contains preinstalled functions
'''

import datetime as dt
import numpy as np
import scipy as sp

#  for Spectral Analysis
from scipy import signal, fftpack
# from scipy.stats import energy_distance
from sklearn.utils import check_array
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.covariance import MinCovDet
from sklearn import ensemble
from sklearn import linear_model

#   for KMeans
#  import skimage as ski
from skimage import util as skiutil  # for nifty windowing
from pyod.models.cblof import CBLOF

# for gradient boosting
import lightgbm

import pandas as pd
import logging
# import warnings
# import json
# from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
from iotfunctions.base import (BaseTransformer, BaseRegressor, BaseEstimatorFunction, BaseSimpleAggregator)
from iotfunctions.bif import (AlertHighValue)
from iotfunctions.ui import (UISingle, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti)

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/ibm-watson-iot/functions.git@'
_IS_PREINSTALLED = True

Error_SmallWindowsize = 0.0001
Error_Generic = 0.0002

FrequencySplit = 0.3
DefaultWindowSize = 12
SmallEnergy = 1e-20

# KMeans_normalizer = 100 / 1.3
KMeans_normalizer = 1
Spectral_normalizer = 100 / 2.8
FFT_normalizer = 1
Saliency_normalizer = 1
Generalized_normalizer = 1 / 300



def min_delta(df):
    # minimal time delta for merging

    if len(df.index.names) > 1:
        df2 = df.copy()
        df2.index = df2.index.droplevel(list(range(1, df.index.nlevels)))
    else:
        df2 = df

    try:
        mindelta = df2.index.to_series().diff().min()
    except Exception as e:
        logger.debug('Min Delta error: ' + str(e))
        mindelta = pd.Timedelta('5 seconds')

    if mindelta == dt.timedelta(seconds=0) or pd.isnull(mindelta):
        mindelta = pd.Timedelta('5 seconds')

    return mindelta, df2


def set_window_size_and_overlap(windowsize, trim_value=2 * DefaultWindowSize):
    # make sure it is positive and not too large
    trimmed_ws = np.minimum(np.maximum(windowsize, 1), trim_value)

    # overlap
    if trimmed_ws == 1:
        ws_overlap = 0
    else:
        # ws_overlap = trimmed_ws - np.maximum(trimmed_ws // DefaultWindowSize, 1)
        # larger overlap - half the window
        ws_overlap = trimmed_ws // 2

    return trimmed_ws, ws_overlap


def merge_score(dfEntity, dfEntityOrig, column_name, score, mindelta):
    '''
    Fit interpolated score to original entity slice of the full dataframe
    '''

    # equip score with time values, make sure it's positive
    score[score < 0] = 0
    dfEntity[column_name] = score

    # merge
    dfEntityOrig = pd.merge_asof(dfEntityOrig, dfEntity[column_name],
                                 left_index=True, right_index=True, direction='nearest', tolerance=mindelta)

    if column_name + '_y' in dfEntityOrig:
        merged_score = dfEntityOrig[column_name + '_y'].to_numpy()
    else:
        merged_score = dfEntityOrig[column_name].to_numpy()

    return merged_score


class KMeansAnomalyScore(BaseTransformer):
    '''
    An unsupervised anomaly detection function.
     Applies a k-means analysis clustering technique to time series data.
     Moves a sliding window across the data signal and applies the anomaly model to each window.
     The window size is typically set to 12 data points.
     Try several anomaly models on your data and use the one that fits your databest.
    '''

    def __init__(self, input_item, windowsize, output_item, nclusters=40, contamination=0.1):
        super().__init__()
        logger.debug(input_item)
        self.input_item = input_item

        # use 12 by default
        self.windowsize, windowoverlap = set_window_size_and_overlap(windowsize)

        self.nclusters = nclusters
        self.contamination = contamination

        # step
        self.step = self.windowsize - windowoverlap

        # assume 1 per sec for now
        self.frame_rate = 1

        self.output_item = output_item

        self.whoami = 'KMeans'

    def prepare_data(self, dfEntity):

        logger.debug(self.whoami + ': prepare Data')

        # operate on simple timestamp index
        if len(dfEntity.index.names) > 1:
            index_names = dfEntity.index.names
            dfe = dfEntity.reset_index().set_index(index_names[0])
        else:
            index_names = None
            dfe = dfEntity

        # interpolate gaps - data imputation
        try:
            dfe = dfe.interpolate(method="time")
        except Exception as e:
            logger.error('Prepare data error: ' + str(e))

        # one dimensional time series - named temperature for catchyness
        temperature = dfe[[self.input_item]].fillna(0).to_numpy().reshape(-1, )

        return dfe, temperature

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        df_copy[self.output_item] = 0

        # check data type
        if df_copy[self.input_item].dtype != np.float64:
            return (df_copy)

        for entity in entities:
            # per entity - copy for later inplace operations
            dfe = df_copy.loc[[entity]].dropna(how='all')
            dfe_orig = df_copy.loc[[entity]].copy()

            # get rid of entityid part of the index
            # do it inplace as we copied the data before
            dfe.reset_index(level=[0], inplace=True)
            dfe.sort_index(inplace=True)
            dfe_orig.reset_index(level=[0], inplace=True)
            dfe_orig.sort_index(inplace=True)

            # minimal time delta for merging
            mindelta, dfe_orig = min_delta(dfe_orig)

            logger.debug('Timedelta:' + str(mindelta))

            # interpolate gaps - data imputation by default
            #   for missing data detection we look at the timestamp gradient instead
            dfe, temperature = self.prepare_data(dfe)

            logger.debug('Module KMeans, Entity: ' + str(entity) + ', Input: ' + str(self.input_item) +
                         ', Windowsize: ' + str(self.windowsize) + ', Output: ' + str(self.output_item) +
                         ', Overlap: ' + str(self.step) + ', Inputsize: ' + str(temperature.size))

            if temperature.size > self.windowsize:
                logger.debug(str(temperature.size) + ',' + str(self.windowsize))

                # Chop into overlapping windows
                slices = skiutil.view_as_windows(temperature, window_shape=(self.windowsize,), step=self.step)

                if self.windowsize > 1:
                    n_cluster = self.nclusters
                else:
                    n_cluster = 20

                n_cluster = np.minimum(n_cluster, slices.shape[0] // 2)

                logger.debug('KMeans params, Clusters: ' + str(n_cluster) + ', Slices: ' + str(slices.shape))

                cblofwin = CBLOF(n_clusters=n_cluster,
                                 n_jobs=1,
                                 contamination=self.contamination)
                try:
                    cblofwin.fit(slices)
                except Exception as e:
                    logger.info('KMeans failed with ' + str(e))
                    self.trace_append('KMeans failed with' + str(e))
                    continue

                pred_score = cblofwin.decision_scores_.copy() * KMeans_normalizer
                threshold = cblofwin.threshold_
                logger.info(f'For entity {entity} CBLOF threshold: {threshold}')
                # np.savetxt('kmeans.csv', pred_score)

                # length of time_series_temperature, signal_energy and ets_zscore is smaller than half the original
                #   extend it to cover the full original length
                diff = temperature.size - pred_score.size

                time_series_temperature = np.linspace(
                    self.windowsize // 2, temperature.size - self.windowsize // 2 + 1,
                    temperature.size - diff)
                #     temperature.size - self.windowsize + 1)

                #time_series_temperature = np.linspace(diff // 2 + diff % 2, temperature.size - diff//2,
                #                                      temperature.size - diff)

                linear_interpolateK = sp.interpolate.interp1d(
                    time_series_temperature, pred_score, kind='linear', fill_value='extrapolate')

                zScoreII = merge_score(dfe, dfe_orig, self.output_item,
                                       linear_interpolateK(np.arange(0, temperature.size, 1)), mindelta)

                #zScoreII = np.where(zScoreII > threshold, 1, 0)

                # np.savetxt('kmeans2.csv', zScoreII)

                idx = pd.IndexSlice
                df_copy.loc[idx[entity, :], self.output_item] = zScoreII

        msg = 'KMeansAnomalyScore'
        self.trace_append(msg)
        return (df_copy)

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(
            name='input_item',
            datatype=float,
            description='Data item to analyze'
        ))

        inputs.append(UISingle(
            name='windowsize',
            datatype=int,
            description='Size of each sliding window in data points. Typically set to 12.'
        ))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(
            name='output_item',
            datatype=float,
            description='Anomaly score (kmeans)'
        ))
        return (inputs, outputs)
