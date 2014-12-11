import datetime
import matplotlib.pyplot as plt
import pandas as pd

def load_quarter_hour_data(path='data/smb_ami_quarter_hour.tsv'):
    qh = pd.read_csv(path, sep='\t')
    qh = drop_negative_day_values(qh)
    #qh = drop_insufficient_days(qh)
    return qh.iloc[:, range(4) + range(4, len(qh.columns), 2)]  # drop usage_type columns

def load_quarter_hour_data_d(path='data/dentist_ami_01-2014.csv'):
    qh = pd.read_csv(path, sep='\t', dtype=unicode, parse_dates=['date_value'])
    qh = qh.iloc[:, range(4) + range(4, len(qh.columns), 2)] # drop usage_type columns
    qh = qh.convert_objects(convert_numeric=True)
    #qh = drop_negative_day_values(qh)
    return qh

def load_quarter_hour_data_d_july(path='data/dentist_ami_07-2014.csv'):
    qh = pd.read_csv(path, sep='\t', dtype=unicode, parse_dates=['date_value'])
    qh = qh.iloc[:, range(4) + range(4, len(qh.columns), 2)] # drop usage_type columns
    qh = qh.convert_objects(convert_numeric=True)
    #qh = drop_negative_day_values(qh)
    return qh


def load_quarter_hour_data_s(path='data/school_ami_07-2014.csv'):
    qh = pd.read_csv(path, sep='\t', dtype=unicode, parse_dates=['date_value'])
    qh = qh.iloc[:, range(4) + range(4, len(qh.columns), 2)] # drop usage_type columns
    qh = qh.convert_objects(convert_numeric=True)
    #qh = drop_negative_day_values(qh)
    return qh

def load_quarter_hour_data_s_jan(path='data/school_ami_01-2014.csv'):
    qh = pd.read_csv(path, sep='\t', dtype=unicode, parse_dates=['date_value'])
    qh = qh.iloc[:, range(4) + range(4, len(qh.columns), 2)] # drop usage_type columns
    qh = qh.convert_objects(convert_numeric=True)
    #qh = drop_negative_day_values(qh)
    return qh

def load_quarter_hour_data_r(path='data/restaurant_ami_01-2014.csv'):
    qh = pd.read_csv(path, sep='\t', dtype=unicode, parse_dates=['date_value'])
    qh = qh.iloc[:, range(4) + range(4, len(qh.columns), 2)]  # drop usage_type columns
    qh = qh.convert_objects(convert_numeric=True)
    #qh = drop_negative_day_values(qh)
    return qh

def load_quarter_hour_data_r_july(path='data/restaurant_ami_07-2014.csv'):
    qh = pd.read_csv(path, sep='\t', dtype=unicode, parse_dates=['date_value'])
    qh = qh.iloc[:, range(4) + range(4, len(qh.columns), 2)]  # drop usage_type columns
    qh = qh.convert_objects(convert_numeric=True)
    #qh = drop_negative_day_values(qh)
    return qh

def drop_negative_day_values(qh):
    negative_check = qh.groupby('util_service_point_id').mean()[3:].apply(lambda x: (x < 0).any(), axis=1)

    return qh[qh['util_service_point_id'].apply(lambda x: x not in negative_check[negative_check].index.tolist())]

def drop_insufficient_days(qh):
    days_check = qh.groupby('util_service_point_id').count()[3:].apply(lambda x: (x < 5).any(), axis=1)

    return qh[qh['util_service_point_id'].apply(lambda x: x not in days_check[days_check].index.tolist())]

def get_day_values(qh_data, util_service_point_id):
    print "Number of days: %d" % len(qh_data.loc[qh_data.util_service_point_id == util_service_point_id])
    return pd.Series(qh_data.loc[qh_data.util_service_point_id == util_service_point_id].mean()[3:].values, index=range(96))

def plot_experiments(accuracies, feature_sets):
    plt.hist(accuracies)
    plt.xticks(feature_sets)




