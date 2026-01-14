"""
Prepare Combined Dataset for Google Colab Training
Combines all commercial bank CSVs into one file
"""

import pandas as pd
import glob
import os

print("Combining all commercial bank data...")

# Load all bank CSV files
DATA_DIR = "commercial-banks/"
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

print(f"Found {len(csv_files)} bank CSV files")

df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    company_id = os.path.basename(file).split('.')[0]
    df['company_id'] = company_id
    df_list.append(df)
    print(f"  Loaded {company_id}: {len(df)} rows")

# Combine all banks
data = pd.concat(df_list, ignore_index=True)
data['published_date'] = pd.to_datetime(data['published_date'])
data = data.sort_values(['company_id', 'published_date']).reset_index(drop=True)

# Save combined dataset
output_file = "combined_banks_dataset.csv"
data.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Combined dataset saved: {output_file}")
print(f"  Total rows: {len(data):,}")
print(f"  Number of banks: {data['company_id'].nunique()}")
print(f"  Date range: {data['published_date'].min()} to {data['published_date'].max()}")
print(f"  Columns: {list(data.columns)}")
print(f"{'='*60}")

# Show summary statistics
print("\nDataset summary:")
print(data.describe())

print("\nSamples per bank:")
bank_counts = data.groupby('company_id').size().sort_values(ascending=False)
print(bank_counts)

print(f"\n✓ Ready to upload to Google Colab!")
 