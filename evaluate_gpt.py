import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
from df_utils import DataFrameUtils
from utils import *
import warnings

class EvaluateGPT:
    def __init__(self, model_name='gpt-4', series_dict_path='series_dict.json', results_path='./results'):
        self.model_name = model_name
        self.NUM_SEASONS = 5
        self.series_dict_path = series_dict_path
        self.results_path = results_path

    def evaluation(self):
        warnings.filterwarnings("ignore")
        oc_results_1 = pd.read_excel('only_connect_results_hmean_updated.xlsx', index_col=None)
        df_results = pd.DataFrame(columns=['season', 'wall', 'ars', 'rs', 'amis', 'nmis', 'correct_groups', 'full_wall'])
        ars_total = []
        rs_total = []
        amis_total = []
        nmis_total = []
        cnt_df = -1
        total_wall, total_correct_groups_wall = 0, 0
        series_dict = DataFrameUtils().load_series_dict()
        series_dict_accumulated = DataFrameUtils().accumulated_series_dict()
        set_seed(seed=42)
        for season in range(1, self.NUM_SEASONS + 1):
            full_wall, correct_groups_wall = 0, 0

            ars = []
            rs = []
            amis = []
            nmis = []
            lst_nan = ['nan']*16
            for idx in range(1, series_dict[season] * 2 + 1):
                print('Season: {}, Wall: {}'.format(season, idx))
                cnt_df += 1
                try:
                    dict_wall = wall_evaluator_gpt(num_default=series_dict_accumulated[season-1],
                                               wall_skip=idx)
                    if dict_wall['scores']['ars'] == 1.0:
                        full_wall += 1

                    correct_groups_wall += dict_wall['correct_groups']

                    ars.append(dict_wall['scores']['ars'])
                    rs.append(dict_wall['scores']['rs'])
                    amis.append(dict_wall['scores']['amis'])
                    nmis.append(dict_wall['scores']['nmis'])
                    df_results.loc[cnt_df] = [season, idx, dict_wall['scores']['ars'], dict_wall['scores']['rs'],
                                              dict_wall['scores']['amis'], dict_wall['scores']['nmis'],
                                              dict_wall['correct_groups'], full_wall]
                except:
                    print(self.model_name + " generation formating failed for wall {}".format(idx))
                    df_results.loc[cnt_df] = [season, idx, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                    # ars.append(lst_nan)
                    # rs.append(lst_nan)
                    # amis.append(lst_nan)
                    # nmis.append(lst_nan)
            oc_results_1['Full_wall'][season - 1] = int(full_wall)
            oc_results_1['Almost_full_wall'][season - 1] = int(correct_groups_wall)
            oc_results_1['ARS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(ars)), 3)
            oc_results_1['ARS_hmean'][season - 1] = np.round(hmean(np.abs(ars)), 3)
            oc_results_1['ARS'][season - 1] = np.round(np.mean(ars), 3)
            # oc_results_1['ARS_lst'][season - 1] = ars
            oc_results_1['RS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(rs)), 3)
            oc_results_1['RS_hmean'][season - 1] = np.round(hmean(np.abs(rs)), 3)
            oc_results_1['RS'][season - 1] = np.round(np.mean(rs), 3)
            # oc_results_1['RS_lst'][season - 1] = rs
            oc_results_1['AMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(amis)), 3)
            oc_results_1['AMIS_hmean'][season - 1] = np.round(hmean(np.abs(amis)), 3)
            oc_results_1['AMIS'][season - 1] = np.round(np.mean(amis), 3)
            # oc_results_1['AMIS_lst'][season - 1] = amis
            oc_results_1['NMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(nmis)), 3)
            oc_results_1['NMIS_hmean'][season - 1] = np.round(hmean(np.abs(nmis)), 3)
            oc_results_1['NMIS'][season - 1] = np.round(np.mean(nmis), 3)
            # oc_results_1['NMIS_lst'][season - 1] = nmis
            # oc_results_1['Triplet_accuracy'][season-1] = triplet_accuracy

            total_wall += full_wall
            total_correct_groups_wall += correct_groups_wall

            for i in range(len(ars) - 1):
                ars_total += [abs(j) for j in ars]
                rs_total += [abs(j) for j in rs]
                amis_total += [abs(j) for j in amis]
                nmis_total += [abs(j) for j in nmis]

                # excluded_walls += int(oc_results_1['Excluded_walls'][i])
            print('----------------end of season {}----------------'.format(season))
        oc_results_1['ARS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(ars_total)), 3)
        oc_results_1['ARS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(ars_total)), 3)
        oc_results_1['ARS'][self.NUM_SEASONS] = np.round(np.mean(ars_total), 3)
        oc_results_1['RS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(rs_total)), 3)
        oc_results_1['RS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(rs_total)), 3)
        oc_results_1['RS'][self.NUM_SEASONS] = np.round(np.mean(rs_total), 3)
        oc_results_1['AMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(amis_total)), 3)
        oc_results_1['AMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(amis_total)), 3)
        oc_results_1['AMIS'][self.NUM_SEASONS] = np.round(np.mean(amis_total), 3)
        oc_results_1['NMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(nmis_total)), 3)
        oc_results_1['NMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(nmis_total)), 3)
        oc_results_1['NMIS'][self.NUM_SEASONS] = np.round(np.mean(nmis_total), 3)
        oc_results_1['Full_wall'][self.NUM_SEASONS] = int(total_wall)
        oc_results_1['Almost_full_wall'][self.NUM_SEASONS] = int(total_correct_groups_wall)
        # oc_results_1['Excluded_walls'][self.NUM_SEASONS] = excluded_walls

        with pd.ExcelWriter(self.results_path + '/only_connect_results_'
                            + self.model_name + '.xlsx', engine='xlsxwriter') as writer:
            oc_results_1.to_excel(writer, 'scores', index=False)
            df_results.to_excel(writer, 'wall_results', index=False)
        print('results saved to: ', self.results_path)


if __name__ == '__main__':
    EvaluateGPT().evaluation()