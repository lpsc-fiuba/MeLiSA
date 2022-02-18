import pandas as pd
import numpy as np
import fasttext

to_be_removed_esp = [
     483197,483223,483442,483512,484318,484498,484586,484729,485267,485810,
     485903,486030,486154,486295,486525,486749,486773,486786,486791,486811,486814,
     486827,486842,486848,486855,486854,486869,486908,486917,486918,486922,486925,
     486931,486934,486935,486940,486941,486948,486956,486958,486959,486961,486962,
     486966,486968,486973,486974,486978,486981,486983,486987,486992,486999,487001,
     487003,487009,487019,487018,487021,487022,487026,487028,487029,487033,487037,
     487041,487044,487052,487060,487062,487065,487072,487074,487076,487081,487084,
     487085,487088,487092,487094,487095,487099,487103,487105,487106,487109,487110,
     487111,487112,487115,487122,487126,487133,487135,487138,487142,487143,487146,
     487154,487155,487156,487162,487164,487173,487179,487185,487189,487197,487198,
     487199,487204,487207,487210,487211,487216,487219,487225,487229,487233,487234,
     487235,487243,487244,487245,487251,487252,487253,487254,487255,487256,487258,
     487264,487273,487276,487282,487290,487292,487294,487298,487304,487303,487308,
     487318,487321,487323,487326,487327,487329,487330,487331,487334,487337,487346,
     487343,487347,487349,487350,487360,487361,487365,487366,487375,487379,487380,
     487386,487389,487391,487393,487396,487397,487399,487400,487401,487402,487411,
     487412,487414,487416,487417,487419,487421,487424,487425,487427,487433,487435,
     487437,487443,487449,487452,487453,487455,487456,487459,487463,487465,487466,
     487468,487471,487479,487481,487483,487485,487486,487487,487488,487489,487490,
     487493,487494,487496,487498,487500,487501,487502,487505,487506,487512,487517,
     487519,487528,487525,487529,487530,487537,487538,487541,487542,487545,487554,
     487555,487556,487558,487567,487569,487573,487574,487578,487582,487586,487587,
     487592,487596,487602,487603,487604,487607,487608,487609,487612,487613,487616,
     487617,487618,487621,487623,487624                
]

to_be_removed_por = [
    274310,274300,274299,274294,274287,274281,274265,274259,274256,274255,274232,274225,274226,274219,
    274213,274206,274200,274199,274194,274172,274171,274170,274167,274166,274165,274163,274153,274146,
    274143,274142,274136,274134,274130,274125,274123,274122,274109,274108,274079,274075,274073,274071,
    274068,274057,274054,274044,274043,274042,274030,274029,274019,274018,274017,274015,274014,274011,
    273998,273975,273969,273967,273951,273934,273924,273922,273914,273910,273909,273901,273899,273895,
    273889,273881,273876,273871,273875,273869,273820,273812,273799,273791,273786,273783,273781,273780,
    273779,273772,273768,273754,273750,273741,273739,273736,273732,273731,273727,273715,273703,273674,
    273596,273595,
]


# countries = ['MLB','MLA','MLM','MLU','MCO','MLC','MLV','MPE']
# esp_countries = ['MLA','MLM','MLU','MCO','MLC','MLV','MPE']
# rates = [1, 2, 3, 4, 5]

abbreviations = {
    'Hogar / Casa': 'HOGAR',
    'Tecnología y electrónica / Tecnologia e electronica': 'TEC',
    'Arte y entretenimiento / Arte e Entretenimiento': 'ARTE',
    'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal': 'SALUD',
    'Alimentos y Bebidas / Alimentos e Bebidas': 'ALIMENTOS'
}
inv_abbreviations = {v:k for k,v in abbreviations.items()}


def detect_lang_fasttext(df_es,df_pt):
    ds_es = (df_es['review_content'] + ' ' + df_es['review_title']).astype(str)
    ds_pt = (df_pt['review_content'] + ' ' + df_pt['review_title']).astype(str)
    
    model_predict = fasttext.load_model('../datav2/lid.176.bin').predict

    def apply_lang_detect(text):
        return dict(zip(*[('lang','prob'),next(zip(*model_predict(text, k=1)))]))

    lang_score_es = pd.DataFrame(ds_es.apply(apply_lang_detect).tolist())
    lang_score_pt = pd.DataFrame(ds_pt.apply(apply_lang_detect).tolist())
    
    lang_score_es.loc[lang_score_es['lang'] != '__label__es', 'prob'] = 0.
    df_es['lang_prob'] = lang_score_es['prob']
    df_es = df_es.sort_values(by=['lang_prob'],ascending=False).reset_index(drop=True)
    
    lang_score_pt.loc[lang_score_pt['lang'] != '__label__pt', 'prob'] = 0.
    df_pt['lang_prob'] = lang_score_pt['prob']
    df_pt = df_pt.sort_values(by=['lang_prob'],ascending=False).reset_index(drop=True)
    
    return df_es, df_pt


def train_test_split(
    df,
    samples,
    random_seed
):
    rs = np.random.RandomState(random_seed)
    test_indices = []
    for country in samples.keys():
        for cat, n in samples[country].items():
            if n == 0:
                continue
#             print(country, cat, n)
#             print(df.loc[
#                 (df['country'] == country) & (df['category'] == inv_abbreviations[cat]), "review_rate"
#             ])
            idx = df[
                (df['country'] == country) & (df['category'] == inv_abbreviations[cat])
            ].groupby('review_rate').sample(n=n,random_state=rs).index.tolist()
            test_indices.extend(idx)

    df_test = df.loc[
        test_indices, ['country','category','review_content','review_title','review_rate']
    ].reset_index(drop=True)
    
    train_indices = sorted(list(set(range(len(df))) - set(test_indices)))
    df_train = df.loc[
        train_indices, ['country','category','review_content','review_title','review_rate']
    ].reset_index(drop=True)
    
    return df_train, df_test

def main():
    # Se leen todos los comentarios descargados
    df_es = pd.read_csv('./reviews_es_full.csv')
    df_pt = pd.read_csv('./reviews_pt_full.csv')
    
    # Se ordenan por relevancia según idioma
    df_es, df_pt = detect_lang_fasttext(df_es,df_pt)
    
    ## ESPAÑOL
    # Se eliminan los que están en la lista to_be_removed_esp
    df_es = df_es.drop(set(to_be_removed_esp)).reset_index(drop=True)
    
    # Se extrae el conjunto de test
    es_country_samples = {
        'MLA':{'ALIMENTOS': 3,'ARTE':30,'HOGAR': 156,'SALUD':210,'TEC':315},
        'MLM':{'ALIMENTOS': 4,'ARTE':30,'HOGAR': 156,'SALUD':210,'TEC':315},
        'MLU':{'ALIMENTOS': 4,'ARTE':30,'HOGAR': 156,'SALUD':210,'TEC':315},
        'MCO':{'ALIMENTOS': 4,'ARTE':30,'HOGAR': 156,'SALUD':210,'TEC':315},
        'MLC':{'ALIMENTOS': 4,'ARTE':30,'HOGAR': 156,'SALUD':210,'TEC':315},
        'MLV':{'ALIMENTOS': 2,'ARTE':30,'HOGAR': 156,'SALUD':172,'TEC':353},
        'MPE':{'ALIMENTOS': 2,'ARTE':30,'HOGAR': 156,'SALUD':210,'TEC':315}
    }
    df_es_train, df_es_test = train_test_split(df_es,es_country_samples,random_seed=776436538)
    
    # Se extrae el conjunto de dev
    es_country_samples = {
        'MLA':{'ALIMENTOS': 10,'ARTE':30,'HOGAR': 200,'SALUD':200,'TEC':300},
        'MLM':{'ALIMENTOS': 10,'ARTE':30,'HOGAR': 200,'SALUD':200,'TEC':300},
        'MLU':{'ALIMENTOS': 10,'ARTE':30,'HOGAR': 200,'SALUD':200,'TEC':300},
        'MCO':{'ALIMENTOS': 10,'ARTE':40,'HOGAR': 200,'SALUD':200,'TEC':300},
        'MLC':{'ALIMENTOS': 20,'ARTE':60,'HOGAR': 200,'SALUD':200,'TEC':300},
        'MLV':{'ALIMENTOS': 0,'ARTE':30,'HOGAR': 20,'SALUD':0,'TEC':250},
        'MPE':{'ALIMENTOS': 0,'ARTE':0,'HOGAR': 1,'SALUD':0,'TEC':1}
    }
    df_es_train, df_es_dev = train_test_split(df_es_train,es_country_samples,random_seed=776436538)
    
    df_es_train.to_csv('./es/train.csv',index=False)
    df_es_dev.to_csv('./es/validation.csv',index=False)
    df_es_test.to_csv('./es/test.csv',index=False)
    
    ## PORTUGUÉS
    # Se eliminan los que están en la lista to_be_removed_por
    df_pt = df_pt.drop(set(to_be_removed_por)).reset_index(drop=True)
    
    # Se extrae el conjunto de test
    pt_country_samples = {'MLB':{'ALIMENTOS': 23,'ARTE':210,'HOGAR': 1092,'SALUD':1432,'TEC':2243}}
    df_pt_train, df_pt_test = train_test_split(df_pt,pt_country_samples,random_seed=776436538)
    
    # Se extrae el conjunto de dev
    pt_country_samples = {'MLB':{'ALIMENTOS': 20,'ARTE':200,'HOGAR': 1032,'SALUD':1400,'TEC':1400}}
    df_pt_train, df_pt_dev = train_test_split(df_pt,pt_country_samples,random_seed=776436538)
    
    df_pt_train.to_csv('./pt/train.csv',index=False)
    df_pt_dev.to_csv('./pt/validation.csv',index=False)
    df_pt_test.to_csv('./pt/test.csv',index=False)
    
    
if __name__ == "__main__":
    main()