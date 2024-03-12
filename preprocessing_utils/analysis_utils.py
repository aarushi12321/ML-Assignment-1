import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_numerical_catagorical_columns(data, catagorical_thresh):
    catagorical_cols = []
    numerical_cols = []
    id_cols = []
    misc_cols = []

    for col in data.columns:
        col_unique = data[col].nunique()
        col_type = data[col].dtype

        if (col_unique == data.shape[0] and 'float' not in str(col_type)):
            id_cols.append(col)
        elif (str(col_type) == 'object' and col_unique <= data.shape[0]/2):
            catagorical_cols.append(col)
        elif 'float' in str(col_type):
            numerical_cols.append(col)
        elif ('int' in str(col_type) and col_unique <= catagorical_thresh):
            catagorical_cols.append(col)
        else:
            misc_cols.append(col)
    
    return catagorical_cols, numerical_cols, id_cols, misc_cols

def get_drop_cols(data, catagorical_cols, id_cols, misc_cols):
    drop_cols = []
    drop_cols += id_cols
    drop_cols += misc_cols

    for col in catagorical_cols:
        if data[col].isnull().sum() > data.shape[0]/2:
            drop_cols += [col]

    return drop_cols

def get_isna_cols(data, drop_cols):
    isna_cols = []
    for col in data.columns:
        if col not in drop_cols and data[col].isnull().sum() > 0:
            isna_cols.append(col)
    
    return isna_cols

def check_na(data):
    for col in data.columns:
        assert(data[col].isnull().sum() == 0)
    
    return

def scaleDataset(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    return scaled_df