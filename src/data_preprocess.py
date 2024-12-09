import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# data_dir = '/Users/boxin/FBDQA-final/FBDQA2021A_MMP_Challenge_ver0.2/data'
data_dir = './data'
# processed_dir = '/Users/boxin/FBDQA-final/FBDQA2021A_MMP_Challenge_ver0.2/data_processed'
processed_dir = './data_processed'

if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

with open('data_processed/processed_files.txt', 'w') as f:
    for root, dirs, files in os.walk(data_dir):
        f.write(f'number of files: {len(files)}\n\n')
        for file in tqdm(files):
            # if file.endswith('.csv') and file == 'snapshot_sym1_date33_am.csv':
            if file.endswith('.csv'):
                i = 0
                k = 0
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Remove rows with NaN or inf values
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)

                # Remove rows with a large number of zeros
                zero_threshold = 15
                for index, row in df.iterrows():
                    zero_count = (row == 0.0).sum()
                    if zero_count >= zero_threshold:
                        # f.write(f"Row with large number of zeros: {row}")
                        df.drop(index, inplace=True)
                        i += 1
                # Check for rows where 'ask1' and 'n_asize1' are both 0 or 'bid1' and 'n_bsize1' are both 0
                close_values = []
                mid_values = []
                for index, row in df.iterrows():
                    if ((row['n_ask1'] == 0 and row['n_asize1'] == 0) or (row['n_bid1'] == 0 and row['n_bsize1'] == 0)):
                        zero_count = (row == 0.0).sum()
                        if zero_count >= 10:
                            close_values.append(row['n_close'])
                            mid_values.append(row['n_midprice'])
                            df.drop(index, inplace=True)
                            i += 1
                for index, row in df.iterrows():
                    for j in range(2, 6):
                        if (row[f'n_ask{j}'] == 0.0 and row[f'n_asize{j}'] == 0.0):
                            # print(row)
                            df.loc[index, f'n_ask{j}']= df.loc[index, f'n_ask{j-1}']
                            # print(f"ask_{j}: {df.loc[index, f'n_ask{j}']}")
                        if (row[f'n_bid{j}'] == 0.0 and row[f'n_bsize{j}'] == 0.0):
                            df.loc[index, f'n_bid{j}'] = df.loc[index, f'n_bid{j-1}']
                            
                    
                if i > 0: 
                    f.write(f'Processed file: {file}\n') 
                    f.write(f"Removed {i} rows with a large number of zeros in {file}\n")
                    f.write(f"Minimum 'close' value for rows with specified conditions: {min(close_values)}\n")
                    f.write(f'Minimum "midprice" value for rows with specified conditions: {min(mid_values)}\n')
                # if k > 0:
                #     f.write(f"Modify {k} rows with higher 'ask' and 'asize' or 'bid' and 'bsize' equal to 0\n\n")
                
                if len(df) != 1999:
                    f.write(f"Number of rows after cleaning: {len(df)} in {file}\n\n")
                # Save the cleaned data to the processed directory
                processed_file_path = os.path.join(processed_dir, file)
                df.to_csv(processed_file_path, index=False)