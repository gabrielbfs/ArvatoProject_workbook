# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 20:21:07 2020

@author: gabriel bustamante
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

    
    
def get_attributes_list(information_level, attributes_description):
    
    attributes_list = list(attributes_description[attributes_description['information'] == information_level]['attribute'].unique())
    
    if information_level == '125m x 125m Grid':
        attributes_list += ['D19_BUCH_CD', 'D19_LETZTER_KAUF_BRANCHE', 'D19_LOTTO', 'D19_SOZIALES']
    if information_level == 'Building':
        attributes_list += ['FIRMENDICHTE', 'HH_DELTA_FLAG', 'KOMBIALTER']
    if information_level == 'Household':
        attributes_list += ['ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3', 'ALTER_KIND4', 
                            'ANZ_KINDER', 'ANZ_STATISTISCHE_HAUSHALTE', 
                            'D19_BANKEN_ANZ_12', 'D19_BANKEN_ANZ_24', 'D19_KONSUMTYP_MAX', 
                            'D19_TELKO_ANZ_12', 'D19_TELKO_ANZ_24', 'D19_TELKO_ONLINE_QUOTE_12', 'D19_VERSAND_ANZ_12',
                            'D19_VERSAND_ANZ_24', 'D19_VERSI_ANZ_12', 'D19_VERSI_ANZ_24', 'D19_VERSI_ONLINE_QUOTE_12', 
                            'KK_KUNDENTYP', 'KONSUMZELLE', 'UMFELD_ALT', 'STRUKTURTYP', 
                            'UMFELD_JUNG', 'UNGLEICHENN_FLAG', 'VERDICHTUNGSRAUM', 
                            'VHA', 'VHN', 'VK_DHT4A', 'VK_DISTANZ', 'VK_ZG11']
    if information_level == 'PLZ8':
        attributes_list += ['KBA13_ANTG1', 'KBA13_ANTG2', 'KBA13_ANTG3', 'KBA13_ANTG4', 'KBA13_BAUMAX', 'KBA13_CCM_1401_2500', 
                            'KBA13_CCM_3000', 'KBA13_CCM_3001', 'KBA13_GBZ', 'KBA13_HHZ', 'KBA13_KMH_210']
    if information_level == 'Person':
        attributes_list += ['ALTERSKATEGORIE_FEIN', 'AKT_DAT_KL', 
                            'CJT_KATALOGNUTZER', 'CJT_TYP_1', 'CJT_TYP_2', 'CJT_TYP_3', 'CJT_TYP_4', 'CJT_TYP_5', 'CJT_TYP_6', 
                            'DSL_FLAG', 'EINGEFUEGT_AM', 'EINGEZOGENAM_HH_JAHR', 
                            'RT_KEIN_ANREIZ', 'RT_SCHNAEPPCHEN', 'RT_UEBERGROESSE', 'SOHO_KZ']
    if information_level == 'Microcell (RR4_ID)':
        attributes_list += ['CAMEO_INTL_2015']
    if information_level == 'Postcode':
        attributes_list += ['GEMEINDETYP', 'EXTSEL992']
    if information_level == 'RR1_ID':
        attributes_list += ['MOBI_RASTER']
    
    return attributes_list



def plot_heatmap_isna(information_columns, title, data):
    plt.figure(figsize=(10,5))
    sns.heatmap(data[information_columns].isna(), cbar=False)
    plt.title(title)
    
    plt.show()

    
    