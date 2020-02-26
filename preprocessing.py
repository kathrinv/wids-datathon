import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


def import_df(path: str = ''):
    """
    """
    df = pd.read_csv(path + 'training_v2.csv')
    
    return df

# composite function to clean dataset
def impute_missing_values(df):
    """
    """
    
    df_new = df.copy(deep=True)
    
    #find mean height and weight by gender
    height_weight_by_gender = df_new.groupby('gender').mean()[['height', 'weight']]
    avg_height = height_weight_by_gender['height'].mean()
    avg_weight = height_weight_by_gender['weight'].mean()
    
    #impute missing gender based on average height and weight
    #where F are < average and M are > average
    df_new.loc[(df_new['height']> avg_height) & (df_new['gender'].isna()), 'gender'] = 'M'
    df_new.loc[(df_new['height']< avg_height) & (df_new['gender'].isna()), 'gender'] = 'F'
    df_new.loc[(df_new['weight']> avg_weight) & (df_new['gender'].isna()), 'gender'] = 'M'
    df_new.loc[(df_new['weight']< avg_weight) & (df_new['gender'].isna()), 'gender'] = 'F'
    
    #any patients without height or weight information are defaulted to M
    df_new['gender'].fillna(value='M', inplace=True)
    
    #impute average height and weight based on patient gender
    df_new.loc[(df_new['gender'] == 'F') & (df_new['weight'].isna()), 'weight'] = height_weight_by_gender.loc['F']['weight']
    df_new.loc[(df_new['gender'] == 'F') & (df_new['height'].isna()), 'height'] = height_weight_by_gender.loc['F']['height']
    df_new.loc[(df_new['gender'] == 'M') & (df_new['weight'].isna()), 'weight'] = height_weight_by_gender.loc['M']['weight']
    df_new.loc[(df_new['gender'] == 'M') & (df_new['height'].isna()), 'height'] = height_weight_by_gender.loc['M']['height']
    
    #calculate bmi using weight and height (no more NAN values)
    df_new['bmi'] = df_new['weight'] / (df_new['height']/100)**2
    
    # DROP COLUMNS
    
    #drop columns with > 50% missings values
    x = ['albumin_apache', 'bilirubin_apache', 'fio2_apache', 'paco2_for_ph_apache', 'paco2_apache', 'pao2_apache',
         'ph_apache', 'urineoutput_apache', 'd1_diasbp_invasive_max', 'd1_diasbp_invasive_min', 'd1_mbp_invasive_max',
         'd1_mbp_invasive_min', 'd1_sysbp_invasive_max','d1_sysbp_invasive_min', 'h1_diasbp_invasive_max',
         'h1_diasbp_noninvasive_min', 'h1_mbp_invasive_max', 'h1_mbp_invasive_min', 'h1_mbp_noninvasive_max',
         'h1_mbp_noninvasive_min', 'h1_sysbp_invasive_max', 'h1_sysbp_invasive_min', 'h1_sysbp_noninvasive_max',
         'h1_sysbp_noninvasive_min', 'd1_albumin_max', 'd1_albumin_min', 'd1_bilirubin_max', 'd1_bilirubin_min',
         'd1_inr_max', 'd1_inr_min', 'd1_lactate_max', 'd1_lactate_min', 'h1_albumin_max', 'h1_albumin_min', 
         'h1_bilirubin_max', 'h1_bilirubin_min', 'h1_bun_max', 'h1_bun_min', 'h1_calcium_max', 'h1_calcium_min', 
         'h1_creatinine_max', 'h1_creatinine_min', 'h1_glucose_max', 'h1_glucose_min', 'h1_hco3_max', 'h1_hco3_min', 
         'h1_hemaglobin_max', 'h1_hemaglobin_min', 'h1_hematocrit_max', 'h1_hematocrit_min', 'h1_inr_max', 'h1_inr_min', 
         'h1_lactate_max', 'h1_lactate_min', 'h1_platelets_max','h1_platelets_min', 'h1_potassium_max', 'h1_potassium_min', 
         'h1_sodium_max', 'h1_sodium_min', 'h1_wbc_max', 'h1_wbc_min', 'd1_arterial_pco2_max', 'd1_arterial_pco2_min', 
         'd1_arterial_ph_max', 'd1_arterial_ph_min', 'd1_arterial_po2_max', 'd1_arterial_po2_min', 'd1_pao2fio2ratio_max', 
         'd1_pao2fio2ratio_min', 'h1_arterial_pco2_max', 'h1_arterial_pco2_min', 'h1_arterial_ph_max', 'h1_arterial_ph_min', 
         'h1_arterial_po2_max', 'h1_arterial_po2_min', 'h1_pao2fio2ratio_max', 'h1_pao2fio2ratio_min', 'h1_diasbp_invasive_min']
    
   
    # Columns that don't provide additional information
    # icu_id = Has 241 unique ids. Used hospital_id instead
    # readmission_status = Always 0
    # hospital_admit_source = Similar to icu_admit_source, but has more null values. Used icu_admit_source instead
    # encounter_id = All unique values
    # patient_id = All unique values
    y = ['icu_id','readmission_status','hospital_admit_source','encounter_id','patient_id']
    
    
    # Columns that have a high correlation with other columns
    # #drop due to multicollinearity or duplicate
    z = ['d1_diasbp_max', 'd1_diasbp_noninvasive_max', 'd1_mbp_noninvasive_max', 'd1_diasbp_min', 'd1_diasbp_noninvasive_min',
         'd1_mbp_noninvasive_min', 'gcs_verbal_apache', 'bun_apache', 'creatinine_apache', 'hematocrit_apache',
         'sodium_apache', 'wbc_apache', 'glucose_apache', 'd1_hemaglobin_max', 'd1_hemaglobin_min', 'heart_rate_apache',
         'apache_2_bodysystem', 'temp_apache', 'h1_temp_max', 'h1_temp_min', 'h1_diasbp_noninvasive_max']
    
    df_new.drop(x, axis=1, inplace=True)
    df_new.drop(y,axis=1, inplace=True)
    df_new.drop(z,axis=1, inplace=True)
    
    # IMPUTE NAN VALUES
    
    # Fill NaN with mean
    a = ['age', 'd1_heartrate_max', 'd1_heartrate_min', 'd1_mbp_max', 'd1_mbp_min', 'd1_resprate_max', 'd1_resprate_min',
         'd1_spo2_max', 'd1_spo2_min', 'd1_sysbp_max', 'd1_sysbp_min']
    df_new = impute_by_agg(df_new, cols=a, agg_type='mean')
    
    #impute median for the following column
    b = ['d1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min', 'd1_temp_max', 'd1_temp_min', 'h1_diasbp_max', 
         'h1_diasbp_min', 'h1_heartrate_max', 'h1_heartrate_min', 'h1_mbp_max', 'h1_mbp_min', 'h1_resprate_max',
         'h1_resprate_min', 'h1_spo2_max', 'h1_spo2_min', 'h1_sysbp_max', 'h1_sysbp_min', 'd1_bun_max', 'd1_bun_min', 
         'd1_calcium_max', 'd1_calcium_min', 'd1_creatinine_max', 'd1_creatinine_min', 'd1_glucose_max', 'd1_glucose_min', 
         'd1_hco3_max', 'd1_hco3_min', 'd1_hematocrit_max', 'd1_hematocrit_min', 'd1_platelets_max', 'd1_platelets_min', 
         'd1_potassium_max', 'd1_potassium_min', 'd1_sodium_max', 'd1_sodium_min', 'd1_wbc_max', 'd1_wbc_min', 
         'map_apache', 'resprate_apache', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob' 'apache_2_diagnosis']
    df_new = impute_by_agg(df_new, cols=b, agg_type='median')
    
    
    # Fill categorical NaN with Other
    df_new['ethnicity'].fillna('Other/Unknown', inplace=True)
    df_new['apache_3j_bodysystem'].fillna('Other', inplace=True)
    df['icu_admit_source'].fillna('Other', inplace=True)
    
    # Fill categorical columns with most common category
    df_new.gcs_eyes_apache.fillna(4.0, inplace=True)
    df_new.gcs_motor_apache.fillna(6.0, inplace=True)
    c = ['gcs_unable_apache', 'intubated_apache', 'arf_apache', 'ventilated_apache', 'aids', 'cirrhosis', 'diabetes_mellitus',
        'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']
    df_new = impute_with_value(df_new, cols=c, value = 0.0)
    
    # Bin hospital_id based on number of deaths
    h = df_new.groupby('hospital_id').sum()['hospital_death']/df.groupby('hospital_id').count()['hospital_death']
    h = list(zip(h.index, list(h)))
    dic = {i:'less than 10%' if j < 0.1 else '10-20%' if j < 0.2 else 'greater than 20%' for i,j in h}
    df_new['hospital_death_rate'] = df_new['hospital_id'].apply(lambda x: dic[x])
    # Drop hospital_id
    df_new.drop('hospital_id', axis=1, inplace=True)

    return df_new
    
    
    def impute_by_agg(df, cols: List[str], agg_type: str = 'median'):
        """
        """
        if agg_type == 'median':
            for col in cols:
                df[col].fillna(df[col].median(), inplace=True)
        elif agg_type == 'mean':
            for col in cols:
                df[col].fillna(df[col].mean(), inplace=True)
        return df
    
    def impute_with_value(df, cols: List[str], value: float = 0.0):
        """
        """
        for col in cols:
            df[col].fillna(value, inplace=True)
        return df
            
    def find_columns(cols: List[str], name: str) -> List[str]:
        """
        """
        return [col for col in cols if col.find(name) != -1]
    
    def change_col_type(df, cols: List[str], to_type: str = 'str'):
        """
        """
        if to_type == 'str':
            for col in cols:
                df[col] = df[col].astype(str)
        elif to_type == 'int':
            for col in cols:
                df[col] = df[col].astype(int)
        elif to_type == 'float':
            for col in cols:
                df[col] = df[col].astype(float)
        return df
            
        
    def one_hot_encode(df):
        """
        """
        df_new = df.copy()
        # Floats in categorical columns that need to be converted to ints
        float_2_int_cols = ['apache_2_diagnosis', 'arf_apache', 'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_unable_apache',
                            'intubated_apache', 'ventilated_apache', 'aids', 'cirrhosis', 'diabetes_mellitus',
                            'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']
        df_new = change_col_type(df_new, cols = float_2_int_cols, to_type = 'int')

        # hospital_death_rate = 3 unique
        # ethnicity = 6 unique
        # gender = 2 unique
        # icu_admit_source = 5 unique
        # icu_stay_type = 3 unique
        # apache_2_diagnosis = 44 unique
        # icu_type = 8 unique
        # gcs_eyes_apache = 4 unique
        # gcs_motor_apache = 6 unique
        # apache_3j_bodysystem = 12 unique
        categorical_cols = ['hospital_death_rate', 'ethnicity','gender', 'icu_admit_source', 'icu_stay_type', 'icu_type',
                            'gcs_eyes_apache', 'gcs_motor_apache', 'apache_3j_bodysystem']
        
        df_new = change_col_type(df_new, cols = categorical_cols, to_type = 'str')
        dummies = pd.get_dummies(df_new[categorical_cols], drop_first = True)
        df_dum = pd.concat([df_new, dummies], axis=1)
        df_dum.drop(categorical_cols, axis=1, inplace = True)
        return df_dum
    
    # add if name = main run 