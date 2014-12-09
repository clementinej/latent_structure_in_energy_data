import datetime
import matplotlib.pyplot as plt
import pandas as pd

# percentage above minimum usage that triggers open and close times
THRESHOLD_PERCENTAGE = 0.1

UTIL_SERVICE_POINT_ID = 'util_service_point_id'


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

def find_open_close(day_values):
    day_min = day_values.min()
    day_max = day_values.max()
    closed_threshold = (day_max-day_min)*0.1 + day_min
    min_index = day_values[day_values == day_min].index[0]
    closed_index = find_threshold(day_values, min_index, -1, closed_threshold)
    open_index = find_threshold(day_values, min_index, 1, closed_threshold)
    return open_index, closed_index


def find_threshold(day_values, min_index, direction, closed_threshold):
    ind = min_index
    while True:
        if list(day_values)[ind] >= closed_threshold:
            if ind < 0:
                ind += len(day_values)
            return ind
        if ind + 1 == len(day_values):
            ind -= len(day_values)
        ind += direction


def get_closed_usages(day_values, open_index, closed_index):
    return (calculate_waste(day_values, open_index, closed_index),
            usage_when_closed(day_values, open_index, closed_index))

def plot_experiments(accuracies, feature_sets):
    plt.hist(accuracies)
    plt.xticks(feature_sets)

def plot_day(day_values, open_index=None, closed_index=None, suppress_y=False, subplot=None):
    # TODO: Change open/close to vertical lines & annotate
    # TODO: Add hourly markings on x-axis
    if not subplot is None:
        plt.subplot(subplot)
    plt.plot(range(0, 96), day_values)
    plt.xticks(range(0, 96, 16), times_of_day()[::16])
    ymin = plt.axis()[2]

    if open_index is not None and closed_index is not None:
        if day_or_night(open_index, closed_index) == "NIGHT":
            closed_indices = range(closed_index, open_index)
            plt.fill_between(range(closed_index, open_index), ymin, day_values[closed_indices], facecolor='cyan')
            plt.fill_between(range(closed_index, open_index), day_values.min(), day_values[closed_indices],
                             facecolor='yellow')
        else:
            closed_indices = range(0, open_index)
            plt.fill_between(closed_indices, ymin, day_values[closed_indices], facecolor='cyan')
            plt.fill_between(closed_indices, day_values.min(), day_values[closed_indices],
                             facecolor='yellow')
            closed_indices = range(closed_index, len(day_values))
            plt.fill_between(closed_indices, ymin, day_values[closed_indices], facecolor='cyan')
            plt.fill_between(closed_indices, day_values.min(), day_values[closed_indices],
                             facecolor='yellow')

    if(open_index is not None):
        plt.axvline(open_index, color='green')
        #plt.plot(open_index, day_values[open_index], 'go')
        plt.text(open_index, day_values.min() + (day_values.max() - day_values.min())*0.05, ' Open', color='green',
                 verticalalignment='baseline', size='larger', weight='bold')
    if(closed_index is not None):
        plt.axvline(closed_index, color='red')
        #plt.plot(closed_index, day_values[closed_index], 'ro')
        plt.text(closed_index, day_values.min() + (day_values.max() - day_values.min())*0.05, ' Closed', color='red',
                 verticalalignment='baseline', size='larger', weight='bold')

    if suppress_y and not subplot == None:
        plt.setp(subplot.get_yticklabels(), visible=False)
    else:
        plt.ylabel('kWh')


def day_or_night(open_index, closed_index):
    if open_index < closed_index:
        return "DAY"
    else:
        return "NIGHT"


def calculate_waste(day_values, open_index, closed_index):
    business_type = day_or_night(open_index, closed_index)
    daily_min = cal_daily_min(day_values)
    closed_usage = usage_when_closed(day_values, open_index, closed_index)
    time_closed = len(day_values) - time_open(business_type, day_values, open_index, closed_index)
    return ((closed_usage/time_closed) - daily_min) * time_closed


def time_open(business_type, day_values, open_index, closed_index):
    if business_type == "DAY":
        return closed_index - open_index
    else:
        return (len(day_values) - open_index) + closed_index


def get_closed_indices(open_index, closed_index):
    business_type = day_or_night(open_index, closed_index)
    if business_type == "DAY":
        # assumes quarter hourly resolution
        return range(0, open_index, 1) + range(closed_index, 96, 1)
    else:
        return range(closed_index, open_index, 1)


def usage_when_closed(day_values, open_index, closed_index):
    return day_values.iloc[get_closed_indices(open_index, closed_index)].sum()


def cal_daily_min(day_values):
    return day_values.min()


def mode_open_time():
    pass


def mode_close_time():
    pass


def times_of_day():
    # assumes quarter hour indices
    start_time = datetime.datetime(2014, 1, 1, 0, 0)
    times = []
    for idx in xrange(0, 96):
        times.append((start_time + datetime.timedelta(minutes=15*idx)).strftime("%H:%M"))
    return times


def get_service_points(qh_data):
    return qh_data[UTIL_SERVICE_POINT_ID].unique()


