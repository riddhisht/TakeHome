
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, TargetEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 

def preprocess(df):
    def map_country(val):
        if pd.isna(val): return 'other'
        if val == 'United-States': return 'us'
        if val in latam: return 'latam'
        if val in asia: return 'asia'
        if val in europe: return 'europe'
        return 'other'
    drop_cols = [
        "year","major industry code","major occupation code","hispanic origin",
        "citizenship","state of previous residence","member of a labor union",
        "migration code-change in reg","migration code-move within reg","veterans benefits",
        "migration code-change in msa","region of previous residence","migration prev res in sunbelt",
        "live in this house 1 year ago","reason for unemployment","enroll in edu inst last wk",
        "fill inc questionnaire for veteran's admin","family members under 18"
    ]
    household_role_map = {
        'Householder': 'householder',
        'Nonfamily householder': 'householder',
        'Secondary individual': 'nonfamily_individual',
        'RP of unrelated subfamily': 'nonfamily_individual',
        'In group quarters': 'group_quarters',
        'Spouse of householder': 'spouse_or_partner',
        'Spouse of RP of unrelated subfamily': 'spouse_or_partner',
        'Other Rel 18+ spouse of subfamily RP': 'spouse_or_partner',
        'Child 18+ spouse of subfamily RP': 'spouse_or_partner',
        'Child <18 spouse of subfamily RP': 'spouse_or_partner',
        'Other Rel <18 spouse of subfamily RP': 'spouse_or_partner',
        'Grandchild 18+ spouse of subfamily RP': 'spouse_or_partner',
        'Child 18+ ever marr Not in a subfamily': 'child_or_grandchild',
        'Child 18+ ever marr RP of subfamily': 'child_or_grandchild',
        'Child 18+ never marr Not in a subfamily': 'child_or_grandchild',
        'Child 18+ never marr RP of subfamily': 'child_or_grandchild',
        'Child <18 never marr not in subfamily': 'child_or_grandchild',
        'Child <18 ever marr RP of subfamily': 'child_or_grandchild',
        'Child <18 ever marr not in subfamily': 'child_or_grandchild',
        'Child <18 never marr RP of subfamily': 'child_or_grandchild',
        'Child under 18 of RP of unrel subfamily': 'child_or_grandchild',
        'Grandchild 18+ ever marr not in subfamily': 'child_or_grandchild',
        'Grandchild 18+ never marr not in subfamily': 'child_or_grandchild',
        'Grandchild 18+ never marr RP of subfamily': 'child_or_grandchild',
        'Grandchild 18+ ever marr RP of subfamily': 'child_or_grandchild',
        'Grandchild <18 never marr child of subfamily RP': 'child_or_grandchild',
        'Grandchild <18 never marr RP of subfamily': 'child_or_grandchild',
        'Grandchild <18 ever marr not in subfamily': 'child_or_grandchild',
        'Grandchild <18 never marr not in subfamily': 'child_or_grandchild',
        'Other Rel 18+ ever marr RP of subfamily': 'other_relative',
        'Other Rel 18+ ever marr not in subfamily': 'other_relative',
        'Other Rel 18+ never marr not in subfamily': 'other_relative',
        'Other Rel 18+ never marr RP of subfamily': 'other_relative',
        'Other Rel <18 ever marr RP of subfamily': 'other_relative',
        'Other Rel <18 ever marr not in subfamily': 'other_relative',
        'Other Rel <18 never marr child of subfamily RP': 'other_relative',
        'Other Rel <18 never marr not in subfamily': 'other_relative',
        'Other Rel <18 never married RP of subfamily': 'other_relative'
    }
    cow_map = {
        'Federal government': 'government', 'Local government': 'government', 'State government': 'government',
        'Self-employed-not incorporated': 'self employed', 'Self-employed-incorporated': 'self employed',
        'Private': 'private',
        'Not in universe': 'not employed', 'Never worked': 'not employed', 'Without pay': 'not employed'
    }
    tax_map = {
        'Joint both 65+': 'above 65',
        'Single': 'single',
        'Nonfiler': 'non filer',
        'Head of household': 'household', 'Joint both under 65': 'household', 'Joint one under 65 & one 65+': 'household'
    }
    rel_to_householder_map = {
        'Householder': 'self_householder',
        'Group Quarters- Secondary individual': 'group_quarters_or_secondary',
        'Spouse of householder': 'spouse_partner',
        'Child 18 or older': 'child',
        'Child under 18 never married': 'child',
        'Child under 18 ever married': 'child',
        'Other relative of householder': 'relative_or_nonrelative',
        'Nonrelative of householder': 'relative_or_nonrelative'
    }
    marital_mapping = {
        'Married-civilian spouse present': 'Married (spouse present)', 'Married-A F spouse present': 'Married (spouse present)',
        'Married-spouse absent': 'Married (spouse absent / disrupted)', 'Separated': 'Married (spouse absent / disrupted)',
        'Divorced': 'Previously married', 'Widowed': 'Previously married',
        'Never married': 'Never married'
    }
    ft_pt_map = {
        'Children or Armed Forces': 'children or armed',
        'Full-time schedules': 'ft',
        'Unemployed full-time': 'unemployed', 'Unemployed part- time': 'unemployed',
        'PT for non-econ reasons usually FT': 'part time', 'PT for econ reasons usually PT': 'part time', 'PT for econ reasons usually FT': 'part time',
        'Not in labor force': 'unemployed'
    }
    latam = ['Mexico', 'Columbia', 'Cuba', 'Puerto-Rico', 'Dominican-Republic', 'El-Salvador',
             'Guatemala', 'Ecuador', 'Peru', 'Nicaragua', 'Honduras', 'Panama', 'Haiti', 'Trinadad&Tobago', 'Jamaica', 'Chicano']
    asia = ['Vietnam', 'Philippines', 'Japan', 'South Korea', 'China', 'Cambodia', 'Taiwan', 'Iran', 'India', 'Hong Kong', 'Thailand', 'Laos']
    europe = ['Germany', 'Italy', 'Poland', 'England', 'Scotland', 'Ireland', 'Portugal', 'France', 'Greece', 'Holand-Netherlands', 'Yugoslavia', 'Hungary']
    categorical_features = [
        'education','detailed occupation recode','sex','detailed industry recode','class of worker',
        'full or part time employment stat','detailed household and family stat','marital stat',
        'detailed household summary in household','tax filer stat',
        'country of birth self','country of birth mother','country of birth father','race'
    ]

    df = df.drop(columns=drop_cols)
    df['detailed household and family stat'] = (df['detailed household and family stat'].map(household_role_map).fillna('other'))
    df['class of worker'] = df['class of worker'].map(cow_map).fillna(df['class of worker'])
    df['tax filer stat'] = df['tax filer stat'].map(tax_map).fillna(df['tax filer stat'])
    df['detailed household summary in household'] = (df['detailed household summary in household'].map(rel_to_householder_map).fillna('other'))
    df['marital stat'] = df['marital stat'].map(marital_mapping).fillna(df['marital stat'])
    df['full or part time employment stat'] = df['full or part time employment stat'].map(ft_pt_map).fillna(df['full or part time employment stat'])

    for c in ['country of birth father', 'country of birth mother', 'country of birth self']:
        if c in df.columns:
            df[c] = df[c].apply(map_country)
    df['label'] = df['label'].map(lambda x: 1 if str(x).startswith('50000') else 0)
    X = df.drop(columns=["label"])
    y = df["label"]

    numeric_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    X_train_enc = preprocessor.fit_transform(X_train, y_train)
    X_val_enc   = preprocessor.transform(X_val)
    X_test_enc  = preprocessor.transform(X_test)

    return X_train_enc, y_train.values, X_val_enc, y_val.values, X_test_enc, y_test.values
