import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
   
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    for col in car_matrix.columns:
        car_matrix.at[col, col] = 0

    return car_matrix

original_df = pd.read_csv(r'C:\Users\hp\Desktop\New folder\MapUp-Data-Assessment-F-main\datasets\dataset-1.csv')
result_matrix = generate_car_matrix(original_df)
print(result_matrix)


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here

    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                      labels=['low', 'medium', 'high'], right=False)

    type_counts = df['car_type'].value_counts().to_dict()

    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

original_df = pd.read_csv(r'dataset-1.csv')
result_counts = get_type_count(original_df)
print(result_counts)


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    mean_bus_value = df['bus'].mean()

    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    bus_indexes.sort()

    return bus_indexes

original_df = pd.read_csv(r'dataset-1.csv')
result_indexes = get_bus_indexes(original_df)

print(result_indexes)


def filter_routes(df):
    """
    Filters and returns a sorted list of values of the 'route' column
    for which the average of values of the 'truck' column is greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: Sorted list of 'route' values with average 'truck' values greater than 7.
    """

    route_avg_truck = df.groupby('route')['truck'].mean()

    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    filtered_routes.sort()
    return filtered_routes

original_df = pd.read_csv('dataset-1.csv')
result_routes = filter_routes(original_df)
print(result_routes)



def multiply_matrix(input_matrix):
    """
    Modifies each value in the input DataFrame based on specified logic.

    Args:
        input_matrix (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Modified DataFrame with values rounded to 1 decimal place.
    """
    def multiply_matrix(input_df):
        modified_df = input_df.copy() 

        for column in modified_df.columns:
            for index in modified_df.index:
                value = modified_df.at[index, column]

                if value > 20:
                    modified_df.at[index, column] = value * 0.75
                else:
                    modified_df.at[index, column] = value * 1.25

        return modified_df
    df = pd.read_csv('dataset-1.csv')

    result_df = multiply_matrix(df)
    print(result_df)



def map_day_to_date(day_name):
    """
    Map day names to actual dates for the current week (assuming today is Monday).

    Args:
        day_name (str): Day name.

    Returns:
        str: Full date in 'YYYY-MM-DD' format.
    """
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    current_day_index = days_of_week.index(day_name)
    today_index = pd.to_datetime('today').dayofweek
    offset = current_day_index - today_index
    date = pd.to_datetime('today') + pd.DateOffset(days=offset)
    return date.strftime('%Y-%m-%d')

def check_timestamps(df):
    """
    Verify the completeness of the time data.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns id, id_2, startDay, startTime, endDay, endTime.

    Returns:
        pandas.Series: Boolean series indicating if each (id, id_2) pair has incorrect timestamps.
    """
    df['startDate'] = df['startDay'].apply(map_day_to_date)
    df['endDate'] = df['endDay'].apply(map_day_to_date)

    df['startTime'] = pd.to_datetime(df['startDate'] + ' ' + df['startTime'])
    df['endTime'] = pd.to_datetime(df['endDate'] + ' ' + df['endTime'])
    incorrect_timestamps = (
        df.groupby(['id', 'id_2'])
        .apply(lambda group: not (
            group['startTime'].min() == pd.Timestamp('00:00:00') and
            group['endTime'].max() == pd.Timestamp('23:59:59') and
            set(group['startTime'].dt.dayofweek) == set(range(7))
        ))
    )
    return incorrect_timestamps

df = pd.read_csv('dataset-2.csv')
result_series = check_timestamps(df)
print(result_series)

