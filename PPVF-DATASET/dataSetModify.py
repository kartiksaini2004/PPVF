import pandas as pd

# Load the CSV data into a DataFrame
data = pd.read_csv('/home/kartik/Desktop/PPVF/PPVF-DATASET/UIT_ED_25_I_10373.csv', header=None)

# Filter the data to include only rows where 'ed' is 0, 1, 2, 3, 4, or 5
filtered_data = data[data.iloc[:, 4].isin([0, 1, 2, 3, 4, 5])]

# Sample 1500-2000 rows for each 'ed' value
sampled_data = filtered_data.groupby(4).apply(lambda x: x.sample(n=min(len(x), 2000))).reset_index(drop=True)

# Save the new dataset to a CSV file
sampled_data.to_csv('/home/kartik/Desktop/PPVF/PPVF-DATASET/UIT_ED_25_I_10373_filtered.csv', header=False, index=False)

print("New dataset created with 1500-2000 values for ed 0, 1, 2, 3, 4, and 5.")