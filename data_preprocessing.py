import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, filepath, target_column=None):
        self.filepath = filepath
        self.target_column = target_column


    def load_and_preprocess(self):
        df = self._load_file(self.filepath)

        if "adult" in self.filepath.lower(): # adult dataset
            return self.preprocess_adult_data(df)
        elif "german" in self.filepath.lower(): # german dataset
            return self.preprocess_german_data(df)
        elif "car" in self.filepath.lower(): # car dataset
            return self.preprocess_car_data(df)
        elif "hypothyroid" in self.filepath.lower(): # hypothyroid dataset
            return self.preprocess_hypothyroid_data(df)
        
        elif "house-votes" in self.filepath.lower():  # house votes dataset
            return self.preprocess_house_votes_data(df)
        
        elif "breast-cancer" in self.filepath.lower(): # breast cancer dataset
            return self.preprocess_breast_cancer_data(df)
        
        elif "agaricus-lepiota" in self.filepath.lower(): # mushroom dataset
            return self.preprocess_agaricus_data(df)
        
        elif "kr-vs-kp" in self.filepath.lower(): # kr-vs-kp dataset
            return self.preprocess_krkp_data(df)
        else:
            raise ValueError("Dataset not recognized for preprocessing")


    @staticmethod
    def _load_file(filepath):
        if filepath.endswith(".csv"):
            return pd.read_csv(filepath)
        elif filepath.endswith(".data") or filepath.endswith(".txt"):
            if "german" in filepath.lower():
                return pd.read_csv(filepath, header=None, sep=r'\s+')
            return pd.read_csv(filepath, header=None)
        else:
            raise ValueError("Unsupported file format")

    @staticmethod
    def preprocess_adult_data(df):

        df.columns = ["age", "workclass", "fnlwgt", "education", "education-num",
                      "marital-status", "occupation", "relationship", "race", "sex",
                      "capital-gain", "capital-loss", "hours-per-week",
                      "native-country", "income"]

        df = df.replace("?", np.nan).dropna()

        data = pd.DataFrame()

        data['age<30'] = (df['age'] < 30).astype(int)
        data['age>60'] = (df['age'] > 60).astype(int)
        data['private'] = (df['workclass'] == 'Private').astype(int)
        data['self-emp'] = df['workclass'].isin(['Self-emp-not-inc', 'Self-emp-inc']).astype(int)
        data['gov'] = df['workclass'].isin(['Federal-gov', 'Local-gov', 'State-gov']).astype(int)
        data['education-num>12'] = (df['education-num'] > 12).astype(int)
        data['education-num<9'] = (df['education-num'] < 9).astype(int)
        data['Prof'] = df['occupation'].isin(['Prof-specialty', 'Exec-managerial']).astype(int)
        data['white'] = (df['race'] == 'White').astype(int)
        data['male'] = (df['sex'] == 'Male').astype(int)
        data['hours>50'] = (df['hours-per-week'] > 50).astype(int)
        data['hours<30'] = (df['hours-per-week'] < 30).astype(int)
        data['US'] = (df['native-country'] == 'United-States').astype(int)
        data['>50K'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

        return data
    


    @staticmethod
    def preprocess_german_data(df):
        df.columns = [
            "status", "duration", "credit_history", "purpose", "amount",
            "savings", "employment_duration", "installment_rate", "personal_status_sex",
            "other_debtors", "residence_since", "property", "age",
            "other_installment_plans", "housing", "number_credits", "job",
            "people_liable", "telephone", "foreign_worker", "class"
        ]
        df = df.replace("?", np.nan).dropna()
        
        data = pd.DataFrame()

        # TARGET: 1 if Credit Risk is Bad (2), 0 if Good (1)
        data['bad_credit'] = (df['class'] == 2).astype(int)
        data['duration>24'] = (df['duration'] > 24).astype(int)
        data['amount>5000'] = (df['amount'] > 5000).astype(int)
        data['age>30'] = (df['age'] > 30).astype(int)
        data['age>60'] = (df['age'] > 60).astype(int)
        data['residence_since>2'] = (df['residence_since'] > 2).astype(int)
        data['installment_rate>2'] = (df['installment_rate'] > 2).astype(int)
        data['status_no_account'] = (df['status'] == 'A14').astype(int)
        data['status_negative'] = (df['status'] == 'A11').astype(int)
        data['history_critical'] = (df['credit_history'] == 'A34').astype(int)
        data['history_no_credits'] = (df['credit_history'] == 'A30').astype(int)
        data['purpose_new_car'] = (df['purpose'] == 'A40').astype(int)
        data['purpose_used_car'] = (df['purpose'] == 'A41').astype(int)
        data['purpose_radio_tv'] = (df['purpose'] == 'A43').astype(int)
        data['savings_unknown'] = (df['savings'] == 'A65').astype(int) # No savings account
        data['savings_low'] = (df['savings'] == 'A61').astype(int)     # < 100 DM
        data['unemployed'] = (df['employment_duration'] == 'A71').astype(int)
        data['employed_long'] = (df['employment_duration'] == 'A75').astype(int) # >= 7 years
        data['male_single'] = (df['personal_status_sex'] == 'A93').astype(int)
        data['female_divorced_separated_married'] = (df['personal_status_sex'].isin(['A92', 'A95'])).astype(int)
        data['no_guarantors'] = (df['other_debtors'] == 'A101').astype(int)
        data['prop_real_estate'] = (df['property'] == 'A121').astype(int)
        data['prop_none'] = (df['property'] == 'A124').astype(int)
        data['housing_rent'] = (df['housing'] == 'A151').astype(int)
        data['housing_own'] = (df['housing'] == 'A152').astype(int)
        data['housing_free'] = (df['housing'] == 'A153').astype(int)
        data['job_skilled'] = (df['job'] == 'A173').astype(int)
        data['job_unskilled'] = (df['job'] == 'A172').astype(int)
        data['job_mgmt_self_emp'] = (df['job'] == 'A174').astype(int)
        data['foreign_worker'] = (df['foreign_worker'] == 'A201').astype(int)

        return data

    @staticmethod
    def preprocess_car_data(df):
        df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

        df = df.replace("?", np.nan).dropna()
        data = pd.DataFrame()

        data['buying_high'] = df['buying'].isin(['high', 'vhigh']).astype(int)
        data['buying_low'] = df['buying'].isin(['low', 'med']).astype(int)
        data['maint_high'] = df['maint'].isin(['high', 'vhigh']).astype(int)
        data['maint_low'] = df['maint'].isin(['low', 'med']).astype(int)
        data['doors_2'] = (df['doors'] == '2').astype(int)
        data['doors_4more'] = df['doors'].isin(['4', 'more']).astype(int)
        data['persons_2'] = (df['persons'] == '2').astype(int)
        data['persons_more'] = (df['persons'] == 'more').astype(int)
        data['lug_boot_small'] = (df['lug_boot'] == 'small').astype(int)
        data['lug_boot_big'] = (df['lug_boot'] == 'big').astype(int)
        data['acceptable'] = df['safety'].apply(lambda x: 0 if x == 'unacc' else 1)

        return data

    @staticmethod
    def preprocess_agaricus_data(df):
        df.columns = ["class", "cap-shape", "cap-surface","cap-color","bruises","odor","gill-attachment",
                    "gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type",
                    "veil-color","ring-number","ring-type","spore-print-color","population","habitat"]
        df = df.replace("?", np.nan).dropna()

        data = pd.DataFrame()


        data['poisonous'] = df['class'].apply(lambda x: 1 if 'p' in x else 0)
        data['cap_bell'] = (df['cap-shape'] == 'b').astype(int)
        data['cap_surface_fibrous'] = (df['cap-surface'] == 'f').astype(int)
        data['cap_surface_scaly'] = (df['cap-surface'] == 'y').astype(int)
        data['bruises'] = (df['bruises'] == 't').astype(int)
        data['odor_foul'] = (df['odor'] == 'f').astype(int)
        data['odor_almond'] = (df['odor'] == 'a').astype(int)
        data['odor_anise'] = (df['odor'] == 'l').astype(int)
        data['odor_none'] = (df['odor'] == 'n').astype(int)
        data['gill_size_broad'] = (df['gill-size'] == 'b').astype(int)
        data['gill_spacing_close'] = (df['gill-spacing'] == 'c').astype(int)
        data['gill_color_white'] = (df['gill-color'] == 'w').astype(int)
        data['stalk_root_bulbous'] = (df['stalk-root'] == 'b').astype(int)
        data['stalk_surface_smooth'] = (df['stalk-surface-above-ring'] == 's').astype(int)
        data['stalk_color_white'] = (df['stalk-color-above-ring'] == 'w').astype(int)
        data['ring_pendant'] = (df['ring-type'] == 'p').astype(int)
        data['ring_number_one'] = (df['ring-number'] == 'o').astype(int)
        data['pop_solitary'] = (df['population'] == 'y').astype(int)
        data['pop_clustered'] = (df['population'] == 'c').astype(int)
        data['habitat_woods'] = (df['habitat'] == 'd').astype(int)
        data['habitat_grasses'] = (df['habitat'] == 'g').astype(int)
        data['habitat_paths'] = (df['habitat'] == 'p').astype(int)

        return data

    @staticmethod
    def preprocess_krkp_data(df):
        df.columns = ["bkblk", "bknwy", "bkon8", "bkona", "bkspr", "bkxbq", "bkxcr", "bkxwp",
                      "blxwp", "bxqsq", "cntxt", "dsopp", "dwipd", "hdchk", "katri", "mulch",
                      "qxmsq", "r2ar8", "reskd", "reskr", "rimmx", "rkxwp", "rxmsq", "simpl",
                      "skach", "skewr", "skrxp", "spcop", "stlmt", "thrsk", "wkcti", "wkna8",
                      "wknck", "wkovl", "wkpos", "wtoeg", "class"]

        df = df.replace("?", np.nan).dropna()
        data = pd.DataFrame()
        for col in df.columns[:-1]: # all except class
            data[col] = (df[col] == 't').astype(int)

        data['won'] = (df['class'] == 'won').astype(int)

        return data

    @staticmethod
    def preprocess_hypothyroid_data(df):
        # 1. DETECT VERSION & ASSIGN NAMES
        if df.shape[1] == 26:
            # 26-Column Version (No 'tumor', 'hypopituitary', 'psych')
            df.columns = [
                "class", "age", "sex", "on_thyroxine", "query_on_thyroxine",
                "on_antithyroid_medication", "sick", "pregnant", "thyroid_surgery",
                "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium",
                "goitre", "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured",
                "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG"
            ]
        else:
            # 30-Column Version
            df.columns = [
                "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
                "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
                "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary",
                "psych", "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured",
                "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured",
                "TBG", "referral_source", "class"
            ]

        # 2. Basic Cleaning
        df = df.replace("?", np.nan)
        
        # Convert numeric columns safely
        numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where target is missing
        df = df.dropna(subset=['class'])

        data = pd.DataFrame()

        # --- TARGET VARIABLE ---
        data['hypothyroid'] = df['class'].apply(lambda x: 0 if 'negative' in str(x).lower() else 1)

        # --- FEATURE ENGINEERING ---
        
        # Demographics
        if 'age' in df.columns:
            data['age>60'] = (df['age'] > 60).astype(int)
            data['age<30'] = (df['age'] < 30).astype(int)
            
        if 'sex' in df.columns:
            data['sex_M'] = (df['sex'] == 'M').astype(int)
            data['sex_F'] = (df['sex'] == 'F').astype(int)

        # Medical History (Binary Flags)
        # Note: 'tumor', 'hypopituitary', 'psych' might be missing
        binary_flags = [
            "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
            "sick", "pregnant", "thyroid_surgery", "I131_treatment",
            "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre", 
            "tumor", "hypopituitary", "psych"
        ]
        
        for col in binary_flags:
            # CRITICAL FIX: check if the column exists in 'df' before accessing
            if col in df.columns:
                data[col] = (df[col] == 't').astype(int)
            else:
                # Optional: Fill with 0 if you want the column to exist anyway
                data[col] = 0

        # Clinical Measurements (Binarization)
        if 'TSH' in df.columns:
            data['TSH_high'] = (df['TSH'] > 6).astype(int)
            data['TSH_low'] = (df['TSH'] < 0.05).astype(int)
        
        if 'T3' in df.columns:
            data['T3_low'] = (df['T3'] < 1.2).astype(int)
            
        if 'TT4' in df.columns:
            data['TT4_low'] = (df['TT4'] < 60).astype(int)
            
        if 'T4U' in df.columns:
            data['T4U_low'] = (df['T4U'] < 0.7).astype(int)
            
        if 'FTI' in df.columns:
            data['FTI_low'] = (df['FTI'] < 65).astype(int)
            
        if 'TBG' in df.columns:
            data['TBG_high'] = ((df['TBG'].notna()) & (df['TBG'] > 30)).astype(int)

        return data

    @staticmethod
    def preprocess_breast_cancer_data(df):
        df.columns = ["sample_code_number", "clump_thickness", "uniformity_cell_size", 
                      "uniformity_cell_shape", "marginal_adhesion", "single_epithelial_cell_size",
                      "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class"]

        df = df.replace("?", np.nan).dropna()
        data = pd.DataFrame()

        data['clump_thickness>5'] = (pd.to_numeric(df['clump_thickness'], errors='coerce') > 5).astype(int)
        data['clump_thickness<3'] = (pd.to_numeric(df['clump_thickness'], errors='coerce') < 3).astype(int)
        data['uniformity_cell_size>5'] = (pd.to_numeric(df['uniformity_cell_size'], errors='coerce') > 5).astype(int)
        data['uniformity_cell_size<3'] = (pd.to_numeric(df['uniformity_cell_size'], errors='coerce') < 3).astype(int)
        data['uniformity_cell_shape>5'] = (pd.to_numeric(df['uniformity_cell_shape'], errors='coerce') > 5).astype(int)
        data['uniformity_cell_shape<3'] = (pd.to_numeric(df['uniformity_cell_shape'], errors='coerce') < 3).astype(int)
        data['marginal_adhesion>5'] = (pd.to_numeric(df['marginal_adhesion'], errors='coerce') > 5).astype(int)
        data['marginal_adhesion<3'] = (pd.to_numeric(df['marginal_adhesion'], errors='coerce') < 3).astype(int)
        data['single_epithelial_cell_size>5'] = (pd.to_numeric(df['single_epithelial_cell_size'], errors='coerce') > 5).astype(int)
        data['bare_nuclei>5'] = (pd.to_numeric(df['bare_nuclei'], errors='coerce') > 5).astype(int)
        data['bland_chromatin>5'] = (pd.to_numeric(df['bland_chromatin'], errors='coerce') > 5).astype(int)
        data['normal_nucleoli>5'] = (pd.to_numeric(df['normal_nucleoli'], errors='coerce') > 5).astype(int)
        data['mitoses>3'] = (pd.to_numeric(df['mitoses'], errors='coerce') > 3).astype(int)
        # Returns True if Malignant (4), False if Benign (2)
        data['malignant'] = (pd.to_numeric(df['class'], errors='coerce') == 4).astype(bool)


        return data

    @staticmethod
    def preprocess_house_votes_data(df):
        df.columns = ["class_names", "handicapped_infants", "water_project_cost_sharing",
                      "adoption_of_budget_resolution", "physician_fee_freeze", "el_salvador_aid",
                      "religious_groups_in_schools", "anti_satellite_test_ban",
                      "aid_to_nicaraguan_contras", "mx_missile", "immigration",
                      "synfuels_corporation_cutback", "education_spending",
                      "superfund_right_to_sue", "crime", "duty_free_exports",
                      "export_administration_act_south_africa"]

        df = df.replace("?", np.nan).dropna()
        data = pd.DataFrame()

        data['handicapped_infants_y'] = (df['handicapped_infants'] == 'y').astype(int)
        data['water_project_cost_sharing_y'] = (df['water_project_cost_sharing'] == 'y').astype(int)
        data['adoption_of_budget_resolution_y'] = (df['adoption_of_budget_resolution'] == 'y').astype(int)
        data['physician_fee_freeze_y'] = (df['physician_fee_freeze'] == 'y').astype(int)
        data['el_salvador_aid_y'] = (df['el_salvador_aid'] == 'y').astype(int)
        data['religious_groups_in_schools_y'] = (df['religious_groups_in_schools'] == 'y').astype(int)
        data['anti_satellite_test_ban_y'] = (df['anti_satellite_test_ban'] == 'y').astype(int)
        data['aid_to_nicaraguan_contras_y'] = (df['aid_to_nicaraguan_contras'] == 'y').astype(int)
        data['mx_missile_y'] = (df['mx_missile'] == 'y').astype(int)
        data['immigration_y'] = (df['immigration'] == 'y').astype(int)
        data['synfuels_corporation_cutback_y'] = (df['synfuels_corporation_cutback'] == 'y').astype(int)
        data['education_spending_y'] = (df['education_spending'] == 'y').astype(int)
        data['superfund_right_to_sue_y'] = (df['superfund_right_to_sue'] == 'y').astype(int)
        data['crime_y'] = (df['crime'] == 'y').astype(int)
        data['duty_free_exports_y'] = (df['duty_free_exports'] == 'y').astype(int)
        data['export_administration_act_south_africa_y'] = (df['export_administration_act_south_africa'] == 'y').astype(int)
        data['republican'] = (df['class_names'] == 'republican').astype(int)

        return data



