# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 21:48:29 2020

@author: gabrielbustamante
"""

import pandas as pd
import numpy as np
import gc


def preprocess_grid_125(df, info_list):
    gc.collect()
    data = df.copy()
    attributes_list = info_list.copy()
    
    data[['D19_LOTTO', 'D19_SOZIALES']] = data[['D19_LOTTO', 'D19_SOZIALES']].fillna(0.)
    
    dummy_D19_LETZTER_KAUF_BRANCHE = pd.get_dummies(data[['D19_LETZTER_KAUF_BRANCHE']], dummy_na=True)
    data = data.join(dummy_D19_LETZTER_KAUF_BRANCHE)
    data = data.drop('D19_LETZTER_KAUF_BRANCHE', axis='columns')
    
    attributes_list += list(dummy_D19_LETZTER_KAUF_BRANCHE.columns)
    attributes_list.remove('D19_LETZTER_KAUF_BRANCHE')
    
    data[attributes_list] = data[attributes_list].astype(float).fillna(0.)    
    return data, attributes_list



def preprocess_building(df, info_list):
    gc.collect()
    data = df.copy()
    attributes_list = info_list.copy()
    
    data[['ANZ_HAUSHALTE_AKTIV', 'ANZ_HH_TITEL']] = data[['ANZ_HAUSHALTE_AKTIV', 'ANZ_HH_TITEL']].fillna(-1.)
    data[['KBA05_HERSTTEMP', 'KBA05_MODTEMP']] = data[['KBA05_HERSTTEMP', 'KBA05_MODTEMP']].replace({-1.:0., 9.:0.}).fillna(0.)
    data[['KONSUMNAEHE', 'WOHNLAGE', 'HH_DELTA_FLAG', 'FIRMENDICHTE', 'KOMBIALTER']] = data[['KONSUMNAEHE', 'WOHNLAGE', 'HH_DELTA_FLAG', 'FIRMENDICHTE', 'KOMBIALTER']].fillna(-1.)
    data[['MIN_GEBAEUDEJAHR']] = data[['MIN_GEBAEUDEJAHR']].fillna(2017.)
    
    data[['GEBAEUDETYP']] = data[['GEBAEUDETYP']].replace({-1.:0.}).fillna(0.).astype('category')
    dummy_GEBAEUDETYP = pd.get_dummies(data[['GEBAEUDETYP']], dummy_na=True)
    data[['OST_WEST_KZ']] = data[['OST_WEST_KZ']].fillna(-1)
    dummy_OST_WEST_KZ = pd.get_dummies(data[['OST_WEST_KZ']], dummy_na=True)

    data = data.join(dummy_GEBAEUDETYP).join(dummy_OST_WEST_KZ)
    data = data.drop(['GEBAEUDETYP', 'OST_WEST_KZ'], axis='columns')
    
    attributes_list += list(dummy_GEBAEUDETYP.columns)
    attributes_list += list(dummy_OST_WEST_KZ.columns)
    attributes_list.remove('GEBAEUDETYP')
    attributes_list.remove('OST_WEST_KZ')
    
    data[attributes_list] = data[attributes_list].astype(float).fillna(0.)    
    return data, attributes_list



def preprocess_community(df, info_list):
    gc.collect()
    data = df.copy()
    attributes_list = info_list.copy()
    
    data[['ARBEIT']] = data[['ARBEIT']].fillna(-1.)
    data[['ORTSGR_KLS9']] = data[['ORTSGR_KLS9']].fillna(-1.)
    data[['RELAT_AB']] = data[['RELAT_AB']].replace({9.:-1.}).fillna(-1.)
    
    data[attributes_list] = data[attributes_list].astype(float).fillna(0.)    
    return data, attributes_list



def preprocess_household(df, info_list):
    gc.collect()
    data = df.copy()
    attributes_list = info_list.copy()
    
    data[['ALTER_HH', 'ANZ_PERSONEN', 'ANZ_TITEL']] = data[['ALTER_HH', 'ANZ_PERSONEN', 'ANZ_TITEL']].fillna(0.)
    data[['HH_EINKOMMEN_SCORE', 'W_KEIT_KIND_HH', 'WOHNDAUER_2008']] = data[['HH_EINKOMMEN_SCORE', 'W_KEIT_KIND_HH', 'WOHNDAUER_2008']].replace({-1:0}).fillna(0.)
    data[['ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3', 'ALTER_KIND4', 'ANZ_KINDER', 'ANZ_STATISTISCHE_HAUSHALTE']] = data[['ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3', 'ALTER_KIND4', 'ANZ_KINDER', 'ANZ_STATISTISCHE_HAUSHALTE']].fillna(-1.)
    data[['KONSUMZELLE', 'UMFELD_ALT', 'STRUKTURTYP']] = data[['KONSUMZELLE', 'UMFELD_ALT', 'STRUKTURTYP']].fillna(-1.)
    data[['VHA', 'VHN', 'VK_DHT4A', 'VK_DISTANZ', 'VK_ZG11']] = data[['VHA', 'VHN', 'VK_DHT4A', 'VK_DISTANZ', 'VK_ZG11']].fillna(-1.)
    data[['UMFELD_JUNG', 'UNGLEICHENN_FLAG', 'VERDICHTUNGSRAUM']] = data[['UMFELD_JUNG', 'UNGLEICHENN_FLAG', 'VERDICHTUNGSRAUM']].fillna(-1.)
    
    cols_household1 = ['D19_GESAMT_ANZ_12', 'D19_GESAMT_ANZ_24', 'D19_BANKEN_ANZ_12', 'D19_BANKEN_ANZ_24', 
                        'D19_TELKO_ANZ_12', 'D19_TELKO_ANZ_24', 'D19_VERSI_ANZ_12', 'D19_VERSI_ANZ_24', 
                        'D19_VERSAND_ANZ_12', 'D19_VERSAND_ANZ_24', 'D19_BANKEN_ANZ_12', 'D19_BANKEN_ANZ_24', 
                        'D19_TELKO_ANZ_12', 'D19_TELKO_ANZ_24', 'D19_VERSAND_ANZ_12', 'D19_VERSAND_ANZ_24', 
                        'D19_VERSI_ANZ_12', 'D19_VERSI_ANZ_24']
    data[cols_household1] = data[cols_household1].fillna(0.)
    
    cols_household2 = ['D19_GESAMT_OFFLINE_DATUM', 'D19_GESAMT_ONLINE_DATUM', 'D19_GESAMT_DATUM', 
                  'D19_BANKEN_OFFLINE_DATUM', 'D19_BANKEN_ONLINE_DATUM', 'D19_BANKEN_DATUM', 
                  'D19_TELKO_OFFLINE_DATUM', 'D19_TELKO_ONLINE_DATUM', 'D19_TELKO_DATUM', 
                  'D19_VERSAND_OFFLINE_DATUM', 'D19_VERSAND_ONLINE_DATUM', 'D19_VERSAND_DATUM', 
                  'D19_VERSI_OFFLINE_DATUM', 'D19_VERSI_ONLINE_DATUM', 'D19_VERSI_DATUM']
    data[cols_household2] = data[cols_household2].fillna(10.)
    
    cols_household3 = ['D19_GESAMT_ONLINE_QUOTE_12', 'D19_BANKEN_ONLINE_QUOTE_12', 
                  'D19_VERSAND_ONLINE_QUOTE_12', 'D19_TELKO_ONLINE_QUOTE_12', 
                  'D19_VERSI_ONLINE_QUOTE_12']
    data[cols_household3] = data[cols_household3].fillna(-1.)
    
    data[['D19_KONSUMTYP']] = data[['D19_KONSUMTYP']].fillna(0.).astype('category')
    dummy_D19_KONSUMTYP = pd.get_dummies(data[['D19_KONSUMTYP']], dummy_na=True)
    data[['D19_KONSUMTYP_MAX']] = data[['D19_KONSUMTYP_MAX']].fillna(0.).astype('category')
    dummy_D19_KONSUMTYP_MAX = pd.get_dummies(data[['D19_KONSUMTYP_MAX']], dummy_na=True)
    data[['KK_KUNDENTYP']] = data[['KK_KUNDENTYP']].fillna(0.).astype('category')
    dummy_KK_KUNDENTYP = pd.get_dummies(data[['KK_KUNDENTYP']], dummy_na=True)
    
    data = data.join(dummy_D19_KONSUMTYP).join(dummy_D19_KONSUMTYP_MAX).join(dummy_KK_KUNDENTYP)
    data = data.drop(['D19_KONSUMTYP', 'D19_KONSUMTYP_MAX', 'KK_KUNDENTYP'], axis='columns')
    
    attributes_list += list(dummy_D19_KONSUMTYP.columns)
    attributes_list += list(dummy_D19_KONSUMTYP_MAX.columns)
    attributes_list += list(dummy_KK_KUNDENTYP.columns)
    attributes_list.remove('D19_KONSUMTYP')
    attributes_list.remove('D19_KONSUMTYP_MAX')
    attributes_list.remove('KK_KUNDENTYP')
    
    data[attributes_list] = data[attributes_list].astype(float).fillna(0.)    
    return data, attributes_list



def preprocess_microcell_rr3(df, info_list):
    gc.collect()
    data = df.copy()
    attributes_list = info_list.copy()
    
    cols_microcell1 = ['KBA05_AUTOQUOT', 
                 'KBA05_CCM1', 'KBA05_CCM2', 'KBA05_CCM3', 'KBA05_CCM4', 
                 'KBA05_DIESEL', 'KBA05_FRAU', 
                 'KBA05_HERST1', 'KBA05_HERST2', 'KBA05_HERST2', 'KBA05_HERST3', 'KBA05_HERST4', 'KBA05_HERST5', 
                 'KBA05_KRSAQUOT', 'KBA05_KRSHERST1', 'KBA05_KRSHERST2', 'KBA05_KRSHERST3', 
                 'KBA05_KRSKLEIN', 'KBA05_KRSOBER', 'KBA05_KRSVAN', 'KBA05_KRSZUL', 
                 'KBA05_KW1', 'KBA05_KW2', 'KBA05_KW3', 
                 'KBA05_MAXAH', 'KBA05_MAXBJ', 'KBA05_MAXHERST', 'KBA05_MAXSEG', 'KBA05_MAXVORB', 
                 'KBA05_MOD1', 'KBA05_MOD2', 'KBA05_MOD3', 'KBA05_MOD4', 'KBA05_MOD8', 
                 'KBA05_MOTOR', 'KBA05_MOTRAD', 
                 'KBA05_SEG1', 'KBA05_SEG2', 'KBA05_SEG3', 'KBA05_SEG4', 'KBA05_SEG5', 
                 'KBA05_SEG6', 'KBA05_SEG7', 'KBA05_SEG8', 'KBA05_SEG9', 'KBA05_SEG10', 
                 'KBA05_VORB0', 'KBA05_VORB1', 'KBA05_VORB2', 'KBA05_ZUL1', 'KBA05_ZUL2', 'KBA05_ZUL3', 'KBA05_ZUL4']
    data[cols_microcell1] = data[cols_microcell1].replace({9.:-1.}).fillna(-1.)
    
    cols_microcell2 = ['KBA05_BAUMAX', 'KBA05_GBZ']
    data[cols_microcell2] = data[cols_microcell2].replace({0.:-1.}).fillna(-1.)
    
    data[attributes_list] = data[attributes_list].astype(float).fillna(0.)    
    return data, attributes_list



def preprocess_microcell_rr4(df, info_list):
    gc.collect()
    data = df.copy()
    attributes_list = info_list.copy()
    
    data['CAMEO_DEUG_2015'] = data['CAMEO_DEUG_2015'].str.replace(r"[a-zA-z]+", '-1.').fillna(-1.).astype(float)
    
    cols_microcell1 = ['KBA05_ALTER1', 'KBA05_ALTER2', 'KBA05_ALTER3', 'KBA05_ALTER4', 
                 'KBA05_ANHANG']
    data[cols_microcell1] = data[cols_microcell1].replace({9.:-1.}).fillna(-1.)
    
    cols_microcell2 = ['KBA05_ANTG1', 'KBA05_ANTG2', 'KBA05_ANTG3', 'KBA05_ANTG4']
    data[cols_microcell2] = data[cols_microcell2].fillna(-1.)
    
    data[['CAMEO_DEU_2015']] = data[['CAMEO_DEU_2015']].fillna('NA').astype('category')
    dummy_CAMEO_DEU_2015 = pd.get_dummies(data[['CAMEO_DEU_2015']], dummy_na=True)
    
    data = data.join(dummy_CAMEO_DEU_2015)
    data = data.drop(['CAMEO_DEU_2015'], axis='columns')
    
    attributes_list += list(dummy_CAMEO_DEU_2015.columns)
    attributes_list.remove('CAMEO_DEU_2015')
    
    data = data.drop(['CAMEO_INTL_2015'], axis='columns')
    attributes_list.remove('CAMEO_INTL_2015')
    
    data[attributes_list] = data[attributes_list].astype(float).fillna(0.)    
    return data, attributes_list



def preprocess_plz8(df, info_list):
    gc.collect()
    data = df.copy()
    attributes_list = info_list.copy()
    
    cols_plz81 = ['KBA13_ALTERHALTER_30', 'KBA13_ALTERHALTER_45', 'KBA13_ALTERHALTER_60', 'KBA13_ALTERHALTER_61', 
                'KBA13_AUDI', 'KBA13_AUTOQUOTE', 'KBA13_BMW', 'KBA13_FAB_ASIEN', 'KBA13_FAB_SONSTIGE', 'KBA13_FIAT', 
                'KBA13_FORD', 'KBA13_HERST_ASIEN', 'KBA13_HERST_AUDI_VW', 'KBA13_HERST_BMW_BENZ', 'KBA13_HERST_EUROPA', 
                'KBA13_HERST_FORD_OPEL', 'KBA13_HERST_SONST', 
                'KBA13_BJ_1999', 'KBA13_BJ_2000', 'KBA13_BJ_2004', 'KBA13_BJ_2006', 'KBA13_BJ_2008', 'KBA13_BJ_2009', 
                'KBA13_CCM_1000', 'KBA13_CCM_1200', 'KBA13_CCM_1400', 'KBA13_CCM_0_1400', 'KBA13_CCM_1500', 
                'KBA13_CCM_1600', 'KBA13_CCM_1800', 'KBA13_CCM_2000', 'KBA13_CCM_2500', 'KBA13_CCM_2501', 
                'KBA13_HALTER_20', 'KBA13_HALTER_25', 'KBA13_HALTER_30', 'KBA13_HALTER_35', 'KBA13_HALTER_40', 
                'KBA13_HALTER_45', 'KBA13_HALTER_50', 'KBA13_HALTER_55', 'KBA13_HALTER_60', 'KBA13_HALTER_65', 
                'KBA13_HALTER_66', 
                'KBA13_KMH_110', 'KBA13_KMH_140', 'KBA13_KMH_180', 'KBA13_KMH_0_140', 'KBA13_KMH_140_210', 
                'KBA13_KMH_211', 'KBA13_KMH_250', 'KBA13_KMH_251', 
                'KBA13_KRSAQUOT', 'KBA13_KRSHERST_AUDI_VW', 'KBA13_KRSHERST_BMW_BENZ', 'KBA13_KRSHERST_FORD_OPEL', 
                'KBA13_KRSSEG_KLEIN', 'KBA13_KRSSEG_OBER', 'KBA13_KRSSEG_VAN', 'KBA13_KRSZUL_NEU', 
                'KBA13_KW_30', 'KBA13_KW_40', 'KBA13_KW_50', 'KBA13_KW_60', 'KBA13_KW_0_60', 'KBA13_KW_70', 
                'KBA13_KW_61_120', 'KBA13_KW_80', 'KBA13_KW_90', 'KBA13_KW_110', 'KBA13_KW_120', 'KBA13_KW_121', 
                'KBA13_MAZDA', 'KBA13_MERCEDES', 'KBA13_MOTOR', 'KBA13_NISSAN', 'KBA13_OPEL', 'KBA13_PEUGEOT', 
                'KBA13_RENAULT', 'KBA13_TOYOTA', 'KBA13_VW', 
                'KBA13_SEG_GELAENDEWAGEN', 'KBA13_SEG_GROSSRAUMVANS', 'KBA13_SEG_KLEINST', 'KBA13_SEG_KLEINWAGEN', 
                'KBA13_SEG_KOMPAKTKLASSE', 'KBA13_SEG_MINIVANS', 'KBA13_SEG_MINIWAGEN', 'KBA13_SEG_MITTELKLASSE', 
                'KBA13_SEG_OBEREMITTELKLASSE', 'KBA13_SEG_OBERKLASSE', 'KBA13_SEG_SONSTIGE', 'KBA13_SEG_SPORTWAGEN', 
                'KBA13_SEG_UTILITIES', 'KBA13_SEG_VAN', 'KBA13_SEG_WOHNMOBILE', 
                'KBA13_SITZE_4', 'KBA13_SITZE_5', 'KBA13_SITZE_6', 
                'KBA13_VORB_0', 'KBA13_VORB_1', 'KBA13_VORB_1_2', 'KBA13_VORB_2', 'KBA13_VORB_3', 
                'PLZ8_ANTG1', 'PLZ8_ANTG2', 'PLZ8_ANTG3', 'PLZ8_ANTG4', 
                'PLZ8_BAUMAX', 'PLZ8_HHZ', 'PLZ8_GBZ', 
                'KBA13_ANTG1', 'KBA13_ANTG2', 'KBA13_ANTG3', 'KBA13_ANTG4', 
                'KBA13_BAUMAX', 'KBA13_CCM_1401_2500', 'KBA13_CCM_3000', 'KBA13_CCM_3001', 
                'KBA13_GBZ', 'KBA13_HHZ', 'KBA13_KMH_210']
    data[cols_plz81] = data[cols_plz81].fillna(-1.)
    
    data['KBA13_ANZAHL_PKW'] = data['KBA13_ANZAHL_PKW'].fillna(-1.)
    
    data[attributes_list] = data[attributes_list].astype(float).fillna(0.)    
    return data, attributes_list



def preprocess_person(df, info_list):
    gc.collect()
    data = df.copy()
    attributes_list = info_list.copy()
    
    cols_person1 = ['AGER_TYP', 'AKT_DAT_KL', 'DSL_FLAG', 'GFK_URLAUBERTYP', 'GREEN_AVANTGARDE', 
                    'HEALTH_TYP', 'LP_FAMILIE_FEIN', 'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 
                    'LP_STATUS_FEIN', 'LP_STATUS_GROB', 'RT_KEIN_ANREIZ', 'RT_SCHNAEPPCHEN', 'RT_UEBERGROESSE', 
                    'SHOPPER_TYP', 'SOHO_KZ', 'VERS_TYP']
    data[cols_person1] = data[cols_person1].fillna(-1.)
    
    cols_person2 = ['ALTERSKATEGORIE_GROB', 'ALTERSKATEGORIE_FEIN', 'ANREDE_KZ', 'CJT_GESAMTTYP', 'CJT_KATALOGNUTZER', 
                    'CJT_TYP_1', 'CJT_TYP_2', 'CJT_TYP_3', 'CJT_TYP_4', 'CJT_TYP_5', 'CJT_TYP_6', 
                    'FINANZTYP', 'NATIONALITAET_KZ', 'PRAEGENDE_JUGENDJAHRE', 'RETOURTYP_BK_S', 'TITEL_KZ']
    data[cols_person2] = data[cols_person2].replace({0.:-1}).fillna(-1.)
    
    cols_person3 = ['FINANZ_ANLEGER', 'FINANZ_HAUSBAUER', 'FINANZ_MINIMALIST', 
                    'FINANZ_SPARER', 'FINANZ_UNAUFFAELLIGER', 'FINANZ_VORSORGER']
    data[cols_person3] = data[cols_person3].replace({-1.:10.}).fillna(10.)
    
    cols_person4 = ['SEMIO_DOM', 'SEMIO_ERL', 'SEMIO_FAM', 'SEMIO_KAEM', 'SEMIO_KRIT', 'SEMIO_KULT', 
                    'SEMIO_LUST', 'SEMIO_MAT', 'SEMIO_PFLICHT', 'SEMIO_RAT', 'SEMIO_REL', 'SEMIO_SOZ', 
                    'SEMIO_TRADV', 'SEMIO_VERT', 'ZABEOTYP']
    data[cols_person4] = data[cols_person4].replace({-1.:9.}).fillna(9.)
    
    data[['EINGEFUEGT_AM']] = (pd.to_datetime('2017-12-31') - pd.to_datetime(data['EINGEFUEGT_AM'])).dt.days
    data[['EINGEFUEGT_AM']] = data[['EINGEFUEGT_AM']].fillna(10000)
    data[['EINGEZOGENAM_HH_JAHR']] = 2018. - data[['EINGEZOGENAM_HH_JAHR']].fillna(1980.)
    data[['GEBURTSJAHR']] = 2018. - data[['GEBURTSJAHR']].fillna(2018.)

    data[['LP_FAMILIE_GROB']] = data[['LP_FAMILIE_GROB']].replace({3.:4., 5.:4., 6.:7., 8.:7., 9.:10, 11.:10.}).fillna(-1.)

    cols_person5 = ['ALTERSKATEGORIE_GROB', 'ALTERSKATEGORIE_FEIN', 'ANREDE_KZ', 'CJT_GESAMTTYP', 'CJT_KATALOGNUTZER', 
            'CJT_TYP_1', 'CJT_TYP_2', 'CJT_TYP_3', 'CJT_TYP_4', 'CJT_TYP_5', 'CJT_TYP_6', 'DSL_FLAG', 
            'FINANZTYP', 'GFK_URLAUBERTYP', 'GREEN_AVANTGARDE', 'HEALTH_TYP', 'LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 
            'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 'LP_STATUS_FEIN', 'LP_STATUS_GROB', 'NATIONALITAET_KZ', 
            'PRAEGENDE_JUGENDJAHRE', 'RETOURTYP_BK_S', 'RT_KEIN_ANREIZ', 'RT_SCHNAEPPCHEN', 'RT_UEBERGROESSE', 
            'SHOPPER_TYP', 'SOHO_KZ', 'TITEL_KZ', 'VERS_TYP', 'ZABEOTYP']
    data[cols_person5] = data[cols_person5].astype('category')
    dummy_cols_person5 = pd.get_dummies(data[cols_person5], dummy_na=True)
    
    data = data.join(dummy_cols_person5)
    data = data.drop(cols_person5, axis='columns')
    
    attributes_list += list(dummy_cols_person5.columns)
    for col in cols_person5:
        attributes_list.remove(col)

    data[attributes_list] = data[attributes_list].astype(float).fillna(0.)    
    return data, attributes_list



def preprocess_postcode(df, info_list):
    gc.collect()
    data = df.copy()
    attributes_list = info_list.copy()
    
    cols_postcode1 = ['BALLRAUM', 'INNENSTADT']
    data[cols_postcode1] = data[cols_postcode1].replace({-1.:9.}).fillna(9.)
    
    cols_postcode2 = ['EWDICHTE']
    data[cols_postcode2] = data[cols_postcode2].fillna(-1.)
    
    cols_postcode3 = ['EWDICHTE']
    data[cols_postcode3] = data[cols_postcode3].astype('category')
    dummy_cols_postcode3 = pd.get_dummies(data[cols_postcode3], dummy_na=True)
    
    data = data.join(dummy_cols_postcode3)
    data = data.drop(cols_postcode3, axis='columns')
    
    attributes_list += list(dummy_cols_postcode3.columns)
    for col in cols_postcode3:
        attributes_list.remove(col)
    
    data = data.drop(['EXTSEL992', 'GEMEINDETYP'], axis='columns')
    attributes_list.remove('EXTSEL992')
    attributes_list.remove('GEMEINDETYP')
    
    data[attributes_list] = data[attributes_list].astype(float).fillna(0.)    
    return data, attributes_list



def preprocess_rr1(df, info_list):
    gc.collect()
    data = df.copy()
    attributes_list = info_list.copy()
    
    cols_rr1_1 = ['GEBAEUDETYP_RASTER', 'ONLINE_AFFINITAET']
    data[cols_rr1_1] = data[cols_rr1_1].fillna(-1.)

    cols_rr1_2 = ['KKK']
    data[cols_rr1_2] = data[cols_rr1_2].replace({-1.:5., 0.:5.}).fillna(5.)
    
    cols_rr1_3 = ['MOBI_RASTER', 'MOBI_REGIO']
    data[cols_rr1_3] = data[cols_rr1_3].fillna(6.)
    
    cols_rr1_4 = ['REGIOTYP']
    data[cols_rr1_4] = data[cols_rr1_4].replace({-1.:0.}).fillna(0.)
    
    cols_rr1_5 = ['GEBAEUDETYP_RASTER', 'REGIOTYP']
    data[cols_rr1_5] = data[cols_rr1_5].astype('category')
    dummy_cols_rr1_5 = pd.get_dummies(data[cols_rr1_5], dummy_na=True)
    
    data = data.join(dummy_cols_rr1_5)
    data = data.drop(cols_rr1_5, axis='columns')
    
    attributes_list += list(dummy_cols_rr1_5.columns)
    for col in cols_rr1_5:
        attributes_list.remove(col)
    
    data[attributes_list] = data[attributes_list].astype(float).fillna(0.)    
    return data, attributes_list



def preprocess_all(data, steps):
    gc.collect()
    M = data.copy()
    
    M, steps['125m x 125m Grid']['df_columns_preproc'] = preprocess_grid_125(M, steps['125m x 125m Grid']['df_columns'])
    M, steps['Building']['df_columns_preproc'] = preprocess_building(M, steps['Building']['df_columns'])
    M, steps['Community']['df_columns_preproc'] = preprocess_community(M, steps['Community']['df_columns'])
    M, steps['Household']['df_columns_preproc'] = preprocess_household(M, steps['Household']['df_columns'])
    M, steps['Microcell (RR3_ID)']['df_columns_preproc'] = preprocess_microcell_rr3(M, steps['Microcell (RR3_ID)']['df_columns'])
    M, steps['Microcell (RR4_ID)']['df_columns_preproc'] = preprocess_microcell_rr4(M, steps['Microcell (RR4_ID)']['df_columns'])
    M, steps['PLZ8']['df_columns_preproc'] = preprocess_plz8(M, steps['PLZ8']['df_columns'])
    M, steps['Person']['df_columns_preproc'] = preprocess_person(M, steps['Person']['df_columns'])
    M, steps['Postcode']['df_columns_preproc'] = preprocess_postcode(M, steps['Postcode']['df_columns'])
    M, steps['RR1_ID']['df_columns_preproc'] = preprocess_rr1(M, steps['RR1_ID']['df_columns'])
    
    return M, steps