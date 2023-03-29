import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
from sentence_transformers import SentenceTransformer
# custom library import
from df_utils import DataFrameUtils
from utils import *
import warnings

class Evaluate:
    def __init__(self, model_name='elmo', series_dict_path='series_dict.json', results_path='./results'):
        self.model_name = model_name
        self.NUM_SEASONS = 15
        self.series_dict_path = series_dict_path
        self.results_path = results_path

    def evaluation(self):
        warnings.filterwarnings("ignore")
        oc_results_1 = pd.read_excel('only_connect_results_hmean_updated.xlsx', index_col=None)
        oc_results_2 = oc_results_1.copy()
        oc_results_3 = oc_results_1.copy()
        oc_results_4 = oc_results_1.copy()
        oc_results_5 = oc_results_1.copy()
        oc_results_6 = oc_results_1.copy()
        ars_6_total = []
        rs_6_total = []
        amis_6_total = []
        nmis_6_total = []
        ars_5_total = []
        rs_5_total = []
        amis_5_total = []
        nmis_5_total = []
        ars_4_total = []
        rs_4_total = []
        amis_4_total = []
        nmis_4_total = []
        ars_3_total = []
        rs_3_total = []
        amis_3_total = []
        nmis_3_total = []
        ars_2_total = []
        rs_2_total = []
        amis_2_total = []
        nmis_2_total = []
        ars_1_total = []
        rs_1_total = []
        amis_1_total = []
        nmis_1_total = []
        total_wall_m1, total_almost_wall_m1 = 0, 0
        total_wall_m2, total_almost_wall_m2 = 0, 0
        total_wall_m3, total_almost_wall_m3 = 0, 0
        total_wall_m4, total_almost_wall_m4 = 0, 0
        total_wall_m5, total_almost_wall_m5 = 0, 0
        total_wall_m6, total_almost_wall_m6 = 0, 0
        series_dict = DataFrameUtils().load_series_dict()
        series_dict_accumulated = DataFrameUtils().accumulated_series_dict()
        set_seed(seed=42)
        if self.model_name == 'elmo':
            spare_model = ELMoEmbeddings(embedding_mode='top')
        for season in range(1, self.NUM_SEASONS + 1):
            # set_seed(seed=42)
            # if self.model_name == 'elmo':
            #     spare_model = ELMoEmbeddings(embedding_mode='top')
            if self.model_name != 'elmo':
                model_save_path = './model/' + self.model_name + '/' + str(season)
                if not os.path.exists(model_save_path):
                    raise Exception('Model path does not exist')
                spare_model = TransformerWordEmbeddings(model_save_path, layers='-1')

            full_wall_m1, almost_full_wall_m1 = 0, 0
            full_wall_m2, almost_full_wall_m2 = 0, 0
            full_wall_m3, almost_full_wall_m3 = 0, 0
            full_wall_m4, almost_full_wall_m4 = 0, 0
            full_wall_m5, almost_full_wall_m5 = 0, 0
            full_wall_m6, almost_full_wall_m6 = 0, 0
            # excluded_walls = 0

            ars_1 = []
            rs_1 = []
            amis_1 = []
            nmis_1 = []

            ars_2 = []
            rs_2 = []
            amis_2 = []
            nmis_2 = []

            ars_3 = []
            rs_3 = []
            amis_3 = []
            nmis_3 = []

            ars_4 = []
            rs_4 = []
            amis_4 = []
            nmis_4 = []

            ars_5 = []
            rs_5 = []
            amis_5 = []
            nmis_5 = []

            ars_6 = []
            rs_6 = []
            amis_6 = []
            nmis_6 = []
            for idx in range(1, series_dict[season] * 2 + 1):
                print('Season: {}, Wall: {}'.format(season, idx))
                # try:
                dict_wall = wall_evaluator(num_default=series_dict_accumulated[season-1],
                                           wall_skip=idx, embeddings=spare_model)
                if dict_wall['clf_embeds'][1]['ars'] == 1.0:
                    full_wall_m1 += 1
                elif dict_wall['clf_embeds'][1]['ars'] == 0.6875:
                    almost_full_wall_m1 += 1

                if dict_wall['clf_embeds_ssm'][1]['ars'] == 1.0:
                    full_wall_m2 += 1
                elif dict_wall['clf_embeds_ssm'][1]['ars'] == 0.6875:
                    almost_full_wall_m2 += 1

                if dict_wall['clf_pca_embeds'][1]['ars'] == 1.0:
                    full_wall_m3 += 1
                elif dict_wall['clf_pca_embeds'][1]['ars'] == 0.6875:
                    almost_full_wall_m3 += 1

                if dict_wall['clf_pca_embeds_ssm'][1]['ars'] == 1.0:
                    full_wall_m4 += 1
                elif dict_wall['clf_pca_embeds_ssm'][1]['ars'] == 0.6875:
                    almost_full_wall_m4 += 1

                if dict_wall['clf_tsne_embeds'][1]['ars'] == 1.0:
                    full_wall_m5 += 1
                elif dict_wall['clf_tsne_embeds'][1]['ars'] == 0.6875:
                    almost_full_wall_m5 += 1

                if dict_wall['clf_tsne_embeds_ssm'][1]['ars'] == 1.0:
                    full_wall_m6 += 1
                elif dict_wall['clf_tsne_embeds_ssm'][1]['ars'] == 0.6875:
                    almost_full_wall_m6 += 1

                ars_1.append(dict_wall['clf_embeds'][1]['ars'])
                rs_1.append(dict_wall['clf_embeds'][1]['rs'])
                amis_1.append(dict_wall['clf_embeds'][1]['amis'])
                nmis_1.append(dict_wall['clf_embeds'][1]['nmis'])

                ars_2.append(dict_wall['clf_embeds_ssm'][1]['ars'])
                rs_2.append(dict_wall['clf_embeds_ssm'][1]['rs'])
                amis_2.append(dict_wall['clf_embeds_ssm'][1]['amis'])
                nmis_2.append(dict_wall['clf_embeds_ssm'][1]['nmis'])

                ars_3.append(dict_wall['clf_pca_embeds'][1]['ars'])
                rs_3.append(dict_wall['clf_pca_embeds'][1]['rs'])
                amis_3.append(dict_wall['clf_pca_embeds'][1]['amis'])
                nmis_3.append(dict_wall['clf_pca_embeds'][1]['nmis'])

                ars_4.append(dict_wall['clf_pca_embeds_ssm'][1]['ars'])
                rs_4.append(dict_wall['clf_pca_embeds_ssm'][1]['rs'])
                amis_4.append(dict_wall['clf_pca_embeds_ssm'][1]['amis'])
                nmis_4.append(dict_wall['clf_pca_embeds_ssm'][1]['nmis'])

                ars_5.append(dict_wall['clf_tsne_embeds'][1]['ars'])
                rs_5.append(dict_wall['clf_tsne_embeds'][1]['rs'])
                amis_5.append(dict_wall['clf_tsne_embeds'][1]['amis'])
                nmis_5.append(dict_wall['clf_tsne_embeds'][1]['nmis'])

                ars_6.append(dict_wall['clf_tsne_embeds_ssm'][1]['ars'])
                rs_6.append(dict_wall['clf_tsne_embeds_ssm'][1]['rs'])
                amis_6.append(dict_wall['clf_tsne_embeds_ssm'][1]['amis'])
                nmis_6.append(dict_wall['clf_tsne_embeds_ssm'][1]['nmis'])
                # except:
                #     print("the following wall has unsupported characters: ", idx)

            oc_results_6['Full_wall'][season - 1] = int(full_wall_m6)
            oc_results_6['Almost_full_wall'][season - 1] = int(almost_full_wall_m6)
            oc_results_6['ARS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(ars_6)), 3)
            oc_results_6['ARS_hmean'][season - 1] = np.round(hmean(np.abs(ars_6)), 3)
            oc_results_6['ARS'][season - 1] = np.round(np.mean(ars_6), 3)
            oc_results_6['ARS_lst'][season - 1] = ars_6
            oc_results_6['RS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(rs_6)), 3)
            oc_results_6['RS_hmean'][season - 1] = np.round(hmean(np.abs(rs_6)), 3)
            oc_results_6['RS'][season - 1] = np.round(np.mean(rs_6), 3)
            oc_results_6['RS_lst'][season - 1] = rs_6
            oc_results_6['AMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(amis_6)), 3)
            oc_results_6['AMIS_hmean'][season - 1] = np.round(hmean(np.abs(amis_6)), 3)
            oc_results_6['AMIS'][season - 1] = np.round(np.mean(amis_6), 3)
            oc_results_6['AMIS_lst'][season - 1] = amis_6
            oc_results_6['NMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(nmis_6)), 3)
            oc_results_6['NMIS_hmean'][season - 1] = np.round(hmean(np.abs(nmis_6)), 3)
            oc_results_6['NMIS'][season - 1] = np.round(np.mean(nmis_6), 3)
            oc_results_6['NMIS_lst'][season - 1] = nmis_6
            # oc_results_6['Triplet_accuracy'][season-1] = triplet_accuracy

            oc_results_5['Full_wall'][season - 1] = int(full_wall_m5)
            oc_results_5['Almost_full_wall'][season - 1] = int(almost_full_wall_m5)
            oc_results_5['ARS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(ars_5)), 3)
            oc_results_5['ARS_hmean'][season - 1] = np.round(hmean(np.abs(ars_5)), 3)
            oc_results_5['ARS'][season - 1] = np.round(np.mean(ars_5), 3)
            oc_results_5['ARS_lst'][season - 1] = ars_5
            oc_results_5['RS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(rs_5)), 3)
            oc_results_5['RS_hmean'][season - 1] = np.round(hmean(np.abs(rs_5)), 3)
            oc_results_5['RS'][season - 1] = np.round(np.mean(rs_5), 3)
            oc_results_5['RS_lst'][season - 1] = rs_5
            oc_results_5['AMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(amis_5)), 3)
            oc_results_5['AMIS_hmean'][season - 1] = np.round(hmean(np.abs(amis_5)), 3)
            oc_results_5['AMIS'][season - 1] = np.round(np.mean(amis_5), 3)
            oc_results_5['AMIS_lst'][season - 1] = amis_5
            oc_results_5['NMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(nmis_5)), 3)
            oc_results_5['NMIS_hmean'][season - 1] = np.round(hmean(np.abs(nmis_5)), 3)
            oc_results_5['NMIS'][season - 1] = np.round(np.mean(nmis_5), 3)
            oc_results_5['NMIS_lst'][season - 1] = nmis_5
            # oc_results_5['Triplet_accuracy'][season-1] = triplet_accuracy

            oc_results_4['Full_wall'][season - 1] = int(full_wall_m4)
            oc_results_4['Almost_full_wall'][season - 1] = int(almost_full_wall_m4)
            oc_results_4['ARS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(ars_4)), 3)
            oc_results_4['ARS_hmean'][season - 1] = np.round(hmean(np.abs(ars_4)), 3)
            oc_results_4['ARS'][season - 1] = np.round(np.mean(ars_4), 3)
            oc_results_4['ARS_lst'][season - 1] = ars_4
            oc_results_4['RS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(rs_4)), 3)
            oc_results_4['RS_hmean'][season - 1] = np.round(hmean(np.abs(rs_4)), 3)
            oc_results_4['RS'][season - 1] = np.round(np.mean(rs_4), 3)
            oc_results_4['RS_lst'][season - 1] = rs_4
            oc_results_4['AMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(amis_4)), 3)
            oc_results_4['AMIS_hmean'][season - 1] = np.round(hmean(np.abs(amis_4)), 3)
            oc_results_4['AMIS'][season - 1] = np.round(np.mean(amis_4), 3)
            oc_results_4['AMIS_lst'][season - 1] = amis_4
            oc_results_4['NMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(nmis_4)), 3)
            oc_results_4['NMIS_hmean'][season - 1] = np.round(hmean(np.abs(nmis_4)), 3)
            oc_results_4['NMIS'][season - 1] = np.round(np.mean(nmis_4), 3)
            oc_results_4['NMIS_lst'][season - 1] = nmis_4
            # oc_results_4['Triplet_accuracy'][season-1] = triplet_accuracy

            oc_results_3['Full_wall'][season - 1] = int(full_wall_m3)
            oc_results_3['Almost_full_wall'][season - 1] = int(almost_full_wall_m3)
            oc_results_3['ARS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(ars_3)), 3)
            oc_results_3['ARS_hmean'][season - 1] = np.round(hmean(np.abs(ars_3)), 3)
            oc_results_3['ARS'][season - 1] = np.round(np.mean(ars_3), 3)
            oc_results_3['ARS_lst'][season - 1] = ars_3
            oc_results_3['RS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(rs_3)), 3)
            oc_results_3['RS_hmean'][season - 1] = np.round(hmean(np.abs(rs_3)), 3)
            oc_results_3['RS'][season - 1] = np.round(np.mean(rs_3), 3)
            oc_results_3['RS_lst'][season - 1] = rs_3
            oc_results_3['AMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(amis_3)), 3)
            oc_results_3['AMIS_hmean'][season - 1] = np.round(hmean(np.abs(amis_3)), 3)
            oc_results_3['AMIS'][season - 1] = np.round(np.mean(amis_3), 3)
            oc_results_3['AMIS_lst'][season - 1] = amis_3
            oc_results_3['NMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(nmis_3)), 3)
            oc_results_3['NMIS_hmean'][season - 1] = np.round(hmean(np.abs(nmis_3)), 3)
            oc_results_3['NMIS'][season - 1] = np.round(np.mean(nmis_3), 3)
            oc_results_3['NMIS_lst'][season - 1] = nmis_3
            # oc_results_3['Triplet_accuracy'][season-1] = triplet_accuracy

            oc_results_2['Full_wall'][season - 1] = int(full_wall_m2)
            oc_results_2['Almost_full_wall'][season - 1] = int(almost_full_wall_m2)
            oc_results_2['ARS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(ars_2)), 3)
            oc_results_2['ARS_hmean'][season - 1] = np.round(hmean(np.abs(ars_2)), 3)
            oc_results_2['ARS'][season - 1] = np.round(np.mean(ars_2), 3)
            oc_results_2['ARS_lst'][season - 1] = ars_2
            oc_results_2['RS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(rs_2)), 3)
            oc_results_2['RS_hmean'][season - 1] = np.round(hmean(np.abs(rs_2)), 3)
            oc_results_2['RS'][season - 1] = np.round(np.mean(rs_2), 3)
            oc_results_2['RS_lst'][season - 1] = rs_2
            oc_results_2['AMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(amis_2)), 3)
            oc_results_2['AMIS_hmean'][season - 1] = np.round(hmean(np.abs(amis_2)), 3)
            oc_results_2['AMIS'][season - 1] = np.round(np.mean(amis_2), 3)
            oc_results_2['AMIS_lst'][season - 1] = amis_2
            oc_results_2['NMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(nmis_2)), 3)
            oc_results_2['NMIS_hmean'][season - 1] = np.round(hmean(np.abs(nmis_2)), 3)
            oc_results_2['NMIS'][season - 1] = np.round(np.mean(nmis_2), 3)
            oc_results_2['NMIS_lst'][season - 1] = nmis_2
            # oc_results_2['Triplet_accuracy'][season-1] = triplet_accuracy

            oc_results_1['Full_wall'][season - 1] = int(full_wall_m1)
            oc_results_1['Almost_full_wall'][season - 1] = int(almost_full_wall_m1)
            oc_results_1['ARS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(ars_1)), 3)
            oc_results_1['ARS_hmean'][season - 1] = np.round(hmean(np.abs(ars_1)), 3)
            oc_results_1['ARS'][season - 1] = np.round(np.mean(ars_1), 3)
            oc_results_1['ARS_lst'][season - 1] = ars_1
            oc_results_1['RS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(rs_1)), 3)
            oc_results_1['RS_hmean'][season - 1] = np.round(hmean(np.abs(rs_1)), 3)
            oc_results_1['RS'][season - 1] = np.round(np.mean(rs_1), 3)
            oc_results_1['RS_lst'][season - 1] = rs_1
            oc_results_1['AMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(amis_1)), 3)
            oc_results_1['AMIS_hmean'][season - 1] = np.round(hmean(np.abs(amis_1)), 3)
            oc_results_1['AMIS'][season - 1] = np.round(np.mean(amis_1), 3)
            oc_results_1['AMIS_lst'][season - 1] = amis_1
            oc_results_1['NMIS_gmean'][season - 1] = np.round(geo_mean_overflow(np.abs(nmis_1)), 3)
            oc_results_1['NMIS_hmean'][season - 1] = np.round(hmean(np.abs(nmis_1)), 3)
            oc_results_1['NMIS'][season - 1] = np.round(np.mean(nmis_1), 3)
            oc_results_1['NMIS_lst'][season - 1] = nmis_1
            # oc_results_1['Triplet_accuracy'][season-1] = triplet_accuracy

            total_wall_m1 += full_wall_m1
            total_almost_wall_m1 += almost_full_wall_m1

            total_wall_m2 += full_wall_m2
            total_almost_wall_m2 += almost_full_wall_m2

            total_wall_m3 += full_wall_m3
            total_almost_wall_m3 += almost_full_wall_m3

            total_wall_m4 += full_wall_m4
            total_almost_wall_m4 += almost_full_wall_m4

            total_wall_m5 += full_wall_m5
            total_almost_wall_m5 += almost_full_wall_m5

            total_wall_m6 += full_wall_m6
            total_almost_wall_m6 += almost_full_wall_m6
            for i in range(len(ars_1) - 1):
                ars_1_total += [abs(j) for j in ars_1]
                rs_1_total += [abs(j) for j in rs_1]
                amis_1_total += [abs(j) for j in amis_1]
                nmis_1_total += [abs(j) for j in nmis_1]

                ars_2_total += [abs(j) for j in ars_2]
                rs_2_total += [abs(j) for j in rs_2]
                amis_2_total += [abs(j) for j in amis_2]
                nmis_2_total += [abs(j) for j in nmis_2]

                ars_3_total += [abs(j) for j in ars_3]
                rs_3_total += [abs(j) for j in rs_3]
                amis_3_total += [abs(j) for j in amis_3]
                nmis_3_total += [abs(j) for j in nmis_3]

                ars_4_total += [abs(j) for j in ars_4]
                rs_4_total += [abs(j) for j in rs_4]
                amis_4_total += [abs(j) for j in amis_4]
                nmis_4_total += [abs(j) for j in nmis_4]

                ars_5_total += [abs(j) for j in ars_5]
                rs_5_total += [abs(j) for j in rs_5]
                amis_5_total += [abs(j) for j in amis_5]
                nmis_5_total += [abs(j) for j in nmis_5]

                ars_6_total += [abs(j) for j in ars_6]
                rs_6_total += [abs(j) for j in rs_6]
                amis_6_total += [abs(j) for j in amis_6]
                nmis_6_total += [abs(j) for j in nmis_6]

                # excluded_walls += int(oc_results_1['Excluded_walls'][i])
            print('----------------end of season {}----------------'.format(season))
        oc_results_1['ARS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(ars_1_total)), 3)
        oc_results_1['ARS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(ars_1_total)), 3)
        oc_results_1['ARS'][self.NUM_SEASONS] = np.round(np.mean(ars_1_total), 3)
        oc_results_1['RS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(rs_1_total)), 3)
        oc_results_1['RS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(rs_1_total)), 3)
        oc_results_1['RS'][self.NUM_SEASONS] = np.round(np.mean(rs_1_total), 3)
        oc_results_1['AMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(amis_1_total)), 3)
        oc_results_1['AMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(amis_1_total)), 3)
        oc_results_1['AMIS'][self.NUM_SEASONS] = np.round(np.mean(amis_1_total), 3)
        oc_results_1['NMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(nmis_1_total)), 3)
        oc_results_1['NMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(nmis_1_total)), 3)
        oc_results_1['NMIS'][self.NUM_SEASONS] = np.round(np.mean(nmis_1_total), 3)
        oc_results_1['Full_wall'][self.NUM_SEASONS] = int(total_wall_m1)
        oc_results_1['Almost_full_wall'][self.NUM_SEASONS] = int(total_almost_wall_m1)
        # oc_results_1['Excluded_walls'][self.NUM_SEASONS] = excluded_walls

        oc_results_2['ARS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(ars_2_total)), 3)
        oc_results_2['ARS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(ars_2_total)), 3)
        oc_results_2['ARS'][self.NUM_SEASONS] = np.round(np.mean(ars_2_total), 3)
        oc_results_2['RS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(rs_2_total)), 3)
        oc_results_2['RS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(rs_2_total)), 3)
        oc_results_2['RS'][self.NUM_SEASONS] = np.round(np.mean(rs_2_total), 3)
        oc_results_2['AMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(amis_2_total)), 3)
        oc_results_2['AMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(amis_2_total)), 3)
        oc_results_2['AMIS'][self.NUM_SEASONS] = np.round(np.mean(amis_2_total), 3)
        oc_results_2['NMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(nmis_2_total)), 3)
        oc_results_2['NMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(nmis_2_total)), 3)
        oc_results_2['NMIS'][self.NUM_SEASONS] = np.round(np.mean(nmis_2_total), 3)
        oc_results_2['Full_wall'][self.NUM_SEASONS] = int(total_wall_m2)
        oc_results_2['Almost_full_wall'][self.NUM_SEASONS] = int(total_almost_wall_m2)
        # oc_results_2['Excluded_walls'][self.NUM_SEASONS] = excluded_walls

        oc_results_3['ARS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(ars_3_total)), 3)
        oc_results_3['ARS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(ars_3_total)), 3)
        oc_results_3['ARS'][self.NUM_SEASONS] = np.round(np.mean(ars_3_total), 3)
        oc_results_3['RS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(rs_3_total)), 3)
        oc_results_3['RS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(rs_3_total)), 3)
        oc_results_3['RS'][self.NUM_SEASONS] = np.round(np.mean(rs_3_total), 3)
        oc_results_3['AMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(amis_3_total)), 3)
        oc_results_3['AMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(amis_3_total)), 3)
        oc_results_3['AMIS'][self.NUM_SEASONS] = np.round(np.mean(amis_3_total), 3)
        oc_results_3['NMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(nmis_3_total)), 3)
        oc_results_3['NMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(nmis_3_total)), 3)
        oc_results_3['NMIS'][self.NUM_SEASONS] = np.round(np.mean(nmis_3_total), 3)
        oc_results_3['Full_wall'][self.NUM_SEASONS] = int(total_wall_m3)
        oc_results_3['Almost_full_wall'][self.NUM_SEASONS] = int(total_almost_wall_m3)
        # oc_results_3['Excluded_walls'][self.NUM_SEASONS] = excluded_walls

        oc_results_4['ARS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(ars_4_total)), 3)
        oc_results_4['ARS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(ars_4_total)), 3)
        oc_results_4['ARS'][self.NUM_SEASONS] = np.round(np.mean(ars_4_total), 3)
        oc_results_4['RS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(rs_4_total)), 3)
        oc_results_4['RS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(rs_4_total)), 3)
        oc_results_4['RS'][self.NUM_SEASONS] = np.round(np.mean(rs_4_total), 3)
        oc_results_4['AMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(amis_4_total)), 3)
        oc_results_4['AMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(amis_4_total)), 3)
        oc_results_4['AMIS'][self.NUM_SEASONS] = np.round(np.mean(amis_4_total), 3)
        oc_results_4['NMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(nmis_4_total)), 3)
        oc_results_4['NMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(nmis_4_total)), 3)
        oc_results_4['NMIS'][self.NUM_SEASONS] = np.round(np.mean(nmis_4_total), 3)
        oc_results_4['Full_wall'][self.NUM_SEASONS] = int(total_wall_m4)
        oc_results_4['Almost_full_wall'][self.NUM_SEASONS] = int(total_almost_wall_m4)
        # oc_results_4['Excluded_walls'][self.NUM_SEASONS] = excluded_walls

        oc_results_5['ARS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(ars_5_total)), 3)
        oc_results_5['ARS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(ars_5_total)), 3)
        oc_results_5['ARS'][self.NUM_SEASONS] = np.round(np.mean(ars_5_total), 3)
        oc_results_5['RS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(rs_5_total)), 3)
        oc_results_5['RS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(rs_5_total)), 3)
        oc_results_5['RS'][self.NUM_SEASONS] = np.round(np.mean(rs_5_total), 3)
        oc_results_5['AMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(amis_5_total)), 3)
        oc_results_5['AMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(amis_5_total)), 3)
        oc_results_5['AMIS'][self.NUM_SEASONS] = np.round(np.mean(amis_5_total), 3)
        oc_results_5['NMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(nmis_5_total)), 3)
        oc_results_5['NMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(nmis_5_total)), 3)
        oc_results_5['NMIS'][self.NUM_SEASONS] = np.round(np.mean(nmis_5_total), 3)
        oc_results_5['Full_wall'][self.NUM_SEASONS] = int(total_wall_m5)
        oc_results_5['Almost_full_wall'][self.NUM_SEASONS] = int(total_almost_wall_m5)
        # oc_results_5['Excluded_walls'][self.NUM_SEASONS] = excluded_walls

        oc_results_6['ARS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(ars_6_total)), 3)
        oc_results_6['ARS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(ars_6_total)), 3)
        oc_results_6['ARS'][self.NUM_SEASONS] = np.round(np.mean(ars_6_total), 3)
        oc_results_6['RS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(rs_6_total)), 3)
        oc_results_6['RS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(rs_6_total)), 3)
        oc_results_6['RS'][self.NUM_SEASONS] = np.round(np.mean(rs_6_total), 3)
        oc_results_6['AMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(amis_6_total)), 3)
        oc_results_6['AMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(amis_6_total)), 3)
        oc_results_6['AMIS'][self.NUM_SEASONS] = np.round(np.mean(amis_6_total), 3)
        oc_results_6['NMIS_gmean'][self.NUM_SEASONS] = np.round(geo_mean_overflow(np.abs(nmis_6_total)), 3)
        oc_results_6['NMIS_hmean'][self.NUM_SEASONS] = np.round(hmean(np.abs(nmis_6_total)), 3)
        oc_results_6['NMIS'][self.NUM_SEASONS] = np.round(np.mean(nmis_6_total), 3)
        oc_results_6['Full_wall'][self.NUM_SEASONS] = int(total_wall_m6)
        oc_results_6['Almost_full_wall'][self.NUM_SEASONS] = int(total_almost_wall_m6)
        # oc_results_6['Excluded_walls'][self.NUM_SEASONS] = excluded_walls

        with pd.ExcelWriter(self.results_path + '/only_connect_results_'
                            + self.model_name + '.xlsx', engine='xlsxwriter') as writer:
            oc_results_6.to_excel(writer, 'ssm_tsne', index=False)
            oc_results_5.to_excel(writer, 'tsne', index=False)
            oc_results_4.to_excel(writer, 'ssm_pca', index=False)
            oc_results_3.to_excel(writer, 'pca', index=False)
            oc_results_2.to_excel(writer, 'ssm', index=False)
            oc_results_1.to_excel(writer, 'default', index=False)

        print('results saved to: ', self.results_path)


if __name__ == '__main__':
    Evaluate(model_name='elmo').evaluation()