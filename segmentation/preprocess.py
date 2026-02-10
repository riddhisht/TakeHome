import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def apply_mappings(df):

    def map_country(val):
        if pd.isna(val): return 'other'
        if val == 'United-States': return 'us'
        if val in latam: return 'latam'
        if val in asia: return 'asia'
        if val in europe: return 'europe'
        return 'other'

    def map_hispanic(val):
        if pd.isna(val): return 'unknown'
        if val == 'All other': return 'all other'
        if val == 'Do not know': return 'unknown'
        return 'rest'
        
    cols_to_remove = [
        "fill inc questionnaire for veteran's admin",
        'migration code-change in msa',
        'migration code-change in reg',
        'migration code-move within reg',
        'migration prev res in sunbelt',
        'reason for unemployment',
        'year',
        'detailed industry recode',
        'detailed occupation recode',
        'state of previous residence',
        'live in this house 1 year ago',
        'region of previous residence'
    ]
    for c in cols_to_remove:
        if c in df.columns:
            df = df.drop(columns=c)

    citizenship_map = {
        'Native- Born in the United States': 'native born',
        'Native- Born in Puerto Rico or U S Outlying': 'native born',
        'Native- Born abroad of American Parent(s)': 'foreign born',
        'Foreign born- Not a citizen of U S': 'foreign born',
        'Foreign born- U S citizen by naturalization': 'foreign born'
    }
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
    family_map = {
        'Both parents present': 'both',
        'Mother only present': 'single or none', 'Father only present': 'single or none',
        'Neither parent present': 'single or none', 'Not in universe': 'single or none'
    }
    ft_pt_map = {
        'Children or Armed Forces': 'children or armed',
        'Full-time schedules': 'ft',
        'Unemployed full-time': 'unemployed', 'Unemployed part- time': 'unemployed',
        'PT for non-econ reasons usually FT': 'part time', 'PT for econ reasons usually PT': 'part time', 'PT for econ reasons usually FT': 'part time',
        'Not in labor force': 'unemployed'
    }
    tax_map = {
        'Joint both 65+': 'above 65',
        'Single': 'single',
        'Nonfiler': 'non filer',
        'Head of household': 'household', 'Joint both under 65': 'household', 'Joint one under 65 & one 65+': 'household'
    }
    marital_mapping = {
        'Married-civilian spouse present': 'Married (spouse present)', 'Married-A F spouse present': 'Married (spouse present)',
        'Married-spouse absent': 'Married (spouse absent / disrupted)', 'Separated': 'Married (spouse absent / disrupted)',
        'Divorced': 'Previously married', 'Widowed': 'Previously married',
        'Never married': 'Never married'
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
    latam = ['Mexico', 'Columbia', 'Cuba', 'Puerto-Rico', 'Dominican-Republic', 'El-Salvador',
             'Guatemala', 'Ecuador', 'Peru', 'Nicaragua', 'Honduras', 'Panama', 'Haiti', 'Trinadad&Tobago', 'Jamaica', 'Chicano']
    asia = ['Vietnam', 'Philippines', 'Japan', 'South Korea', 'China', 'Cambodia', 'Taiwan', 'Iran', 'India', 'Hong Kong', 'Thailand', 'Laos']
    europe = ['Germany', 'Italy', 'Poland', 'England', 'Scotland', 'Ireland', 'Portugal', 'France', 'Greece', 'Holand-Netherlands', 'Yugoslavia', 'Hungary']

    df['citizenship'] = df['citizenship'].map(citizenship_map).fillna(df['citizenship'])
    df['dividends from stocks'] = df['dividends from stocks'].apply(lambda x: '0' if x == 0 else 'non-zero')
    df['family members under 18'] = df['family members under 18'].map(family_map).fillna(df['family members under 18'])
    df['full or part time employment stat'] = df['full or part time employment stat'].map(ft_pt_map).fillna(df['full or part time employment stat'])
    df['hispanic origin'] = df['hispanic origin'].apply(map_hispanic)
    df['tax filer stat'] = df['tax filer stat'].map(tax_map).fillna(df['tax filer stat'])
    df['marital stat'] = df['marital stat'].map(marital_mapping).fillna(df['marital stat'])
    df['class of worker'] = df['class of worker'].map(cow_map).fillna(df['class of worker'])
    df['detailed household and family stat'] = (df['detailed household and family stat'].map(household_role_map).fillna('other'))
    df['detailed household summary in household'] = (df['detailed household summary in household'].map(rel_to_householder_map).fillna('other'))



    for c in ['country of birth father', 'country of birth mother', 'country of birth self']:
        if c in df.columns:
            df[c] = df[c].apply(map_country)

    return df

def build_preprocessor(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ],
        remainder='drop'
    )
    return ct, cat_cols, num_cols
