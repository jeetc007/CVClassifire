import pandas as pd

print('=== FINAL DATA VERIFICATION ===')

# Check cleaned data quality
cleaned_df = pd.read_csv('data/processed/cleaned_customer_features.csv')
print(f'Cleaned customer data: {cleaned_df.shape}')
print(f'Columns: {list(cleaned_df.columns)}')

# Verify no missing values
print(f'Missing values: {cleaned_df.isnull().sum().sum()}')

# Verify data ranges
print(f'TotalAmount range: ${cleaned_df["TotalAmount_Sum"].min():.2f} - ${cleaned_df["TotalAmount_Sum"].max():.2f}')
print(f'TransactionCount range: {cleaned_df["TransactionCount"].min()} - {cleaned_df["TransactionCount"].max()}')
print(f'UniqueProducts range: {cleaned_df["UniqueProducts"].min()} - {cleaned_df["UniqueProducts"].max()}')

# Customer segments
segment_dist = cleaned_df['CustomerSegment_Encoded'].value_counts().sort_index()
segment_names = ['Low Value', 'Medium Value', 'High Value']
print(f'Customer segments:')
for seg, count in segment_dist.items():
    print(f'  {segment_names[seg]}: {count:,} customers')

print(f'\nData quality checks:')
print(f'  All TotalAmount_Sum > 0: {(cleaned_df["TotalAmount_Sum"] > 0).all()}')
print(f'  All TransactionCount > 0: {(cleaned_df["TransactionCount"] > 0).all()}')
print(f'  All UniqueProducts > 0: {(cleaned_df["UniqueProducts"] > 0).all()}')
print(f'  All CustomerTenureDays >= 0: {(cleaned_df["CustomerTenureDays"] >= 0).all()}')

print('\n=== SAMPLE DATA ===')
print(cleaned_df.head(3).to_string())
