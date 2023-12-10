"""
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
import pandas as pd
import networkx as nx

def calculate_distance_matrix(dataset_path):
    

    df = pd.read_csv(dataset_path)

    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_edge(row['id_start'], row['id_end'], distance=row['distance'])
        G.add_edge(row['id_end'], row['id_start'], distance=row['distance'])  # Bidirectional distances

    distance_matrix = nx.floyd_warshall_numpy(G, weight='distance', nodelist=sorted(G.nodes()))

    distance_df = pd.DataFrame(distance_matrix, index=sorted(G.nodes()), columns=sorted(G.nodes()))

    distance_df.values[[range(distance_df.shape[0])]] 

    return distance_df

dataset_path = (r'C:\Users\ASUS\Downloads\dataset-3.csv')
distance_matrix = calculate_distance_matrix(r'C:\Users\ASUS\Downloads\dataset-3.csv')

print(distance_matrix)


import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Reset index to convert index values (ID values) into columns
    distance_matrix_reset = distance_matrix.reset_index()

    # Melt the DataFrame to transform it to a long format
    melted_df = pd.melt(distance_matrix_reset, id_vars='index', var_name='id_end', value_name='distance')

    # Rename columns for clarity
    melted_df.rename(columns={'index': 'id_start'}, inplace=True)

    # Filter out rows where id_start is equal to id_end
    unrolled_df = melted_df[melted_df['id_start'] != melted_df['id_end']]

    # Reset index for the final DataFrame
    unrolled_df.reset_index(drop=True, inplace=True)

    return unrolled_df

# Example usage:
# Assuming distance_matrix is the DataFrame you have displayed
result_df = unroll_distance_matrix(distance_matrix)
print(result_df.head(20))


"""
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """

def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Filter DataFrame for the reference value
    reference_df = df[df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    average_distance = reference_df['distance'].mean()

    # Calculate the threshold range (10% of the average distance)
    threshold_range = 0.1 * average_distance

    # Filter DataFrame for values within the threshold range
    within_threshold_df = df[(df['id_start'] != reference_value) &
                              (df['distance'] >= average_distance - threshold_range) &
                              (df['distance'] <= average_distance + threshold_range)]

    # Get unique sorted values of 'id_start' within the threshold
    within_threshold_ids = sorted(within_threshold_df['id_start'].unique())

    # Calculate the average distance for the sorted list of values from 'id_start'
    average_distance_threshold = df[df['id_start'].isin(within_threshold_ids)]['distance'].mean()

    return within_threshold_ids, average_distance_threshold

# Example usage:
# Assuming result_df is the DataFrame you have displayed
reference_value = 1001400  # Replace with the desired reference value
threshold_ids, average_distance_threshold = find_ids_within_ten_percentage_threshold(result_df, reference_value)

print(f"Sorted list of values from 'id_start' within 10% threshold: {threshold_ids}")
print(f"Average distance for the sorted list of values from 'id_start': {average_distance_threshold}")

def calculate_toll_rate(input_df):
        """def calculate_toll_rate(df)->pd.DataFrame():
       
        Calculate toll rates for each vehicle type based on the unrolled DataFrame.

        Args:
            df (pandas.DataFrame)

        Returns:
            pandas.DataFrame
        """
        toll_rates = {
            'moto': 0.8,
            'car': 1.2,
            'rv': 1.5,
            'bus': 2.2,
            'truck': 3.6
        }
        for vehicle_type, rate_coefficient in toll_rates.items():
            input_df[vehicle_type] = input_df['distance'] * rate_coefficient

        return input_df
df_question_2 = pd.read_csv('dataset-3.csv')

result_df_with_toll_rates = calculate_toll_rate(df_question_2)
print(result_df_with_toll_rates)

import pandas as pd
import numpy as np

def calculate_time_based_toll_rates(input_df):
    time_ranges = {
        'morning': ('00:00:00', '10:00:00'),
        'afternoon': ('10:00:00', '18:00:00'),
        'evening': ('18:00:00', '23:59:59')
    }

    # Create new columns for start_day, start_time, end_day, and end_time
    input_df['start_day'] = input_df['start_time'].dt.day_name()
    input_df['start_time'] = input_df['start_time'].dt.time
    input_df['end_day'] = input_df['end_time'].dt.day_name()
    input_df['end_time'] = input_df['end_time'].dt.time
    for time_range, (start_time, end_time) in time_ranges.items():
        start_datetime = pd.to_datetime(start_time, format='%H:%M:%S').time()
        end_datetime = pd.to_datetime(end_time, format='%H:%M:%S').time()

        mask = (
            (input_df['start_time'] >= start_datetime) &
            (input_df['start_time'] < end_datetime) &
            (input_df['start_time'].dt.weekday < 5)
        )

        discount_factor = 0.8 if 'morning' in time_range or 'evening' in time_range else 1.2
        for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
            input_df[f'{vehicle_type}_{time_range}'] = np.where(mask, input_df[vehicle_type] * discount_factor, input_df[vehicle_type])

    return input_df


df_question_3 = pd.DataFrame({
    'id_start': [1, 2, 3],
    'id_end': [2, 3, 1],
    'start_time': ['2023-01-01 08:00:00', '2023-01-02 14:30:00', '2023-01-03 20:45:00'],
    'end_time': ['2023-01-01 09:30:00', '2023-01-02 16:45:00', '2023-01-03 22:30:00'],
    'moto': [10, 15, 20],
    'car': [20, 30, 40],
    'rv': [30, 45, 60],
    'bus': [40, 60, 80],
    'truck': [50, 75, 100]
})

df_question_3['start_time'] = pd.to_datetime(df_question_3['start_time'])
df_question_3['end_time'] = pd.to_datetime(df_question_3['end_time'])

result_df_with_time_based_toll_rates = calculate_time_based_toll_rates(df_question_3)
print(result_df_with_time_based_toll_rates)
