import pandas as pd

# Check the cleaned data without outliers
df = pd.read_csv('data/processed/processed_customer_data.csv')
print('=== OUTLIER REMOVAL VERIFICATION ===')
print(f'Total customers: {len(df)}')

# Check TotalAmount_Sum range
print(f'TotalAmount_Sum range: ${df["TotalAmount_Sum"].min():.2f} - ${df["TotalAmount_Sum"].max():.2f}')

# Check segments
segment_dist = df['CustomerSegment_Encoded'].value_counts().sort_index()
segment_names = ['Low Value', 'Medium Value', 'High Value']
print(f'Customer segments:')
for seg, count in segment_dist.items():
    print(f'  {segment_names[seg]}: {count:,}')

# Check for outliers using IQR method
Q1 = df['TotalAmount_Sum'].quantile(0.25)
Q3 = df['TotalAmount_Sum'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

outliers = df[df['TotalAmount_Sum'] > upper_bound]
print(f'\nOutlier analysis:')
print(f'  Upper bound (IQR): ${upper_bound:.2f}')
print(f'  Customers above upper bound: {len(outliers)}')
print(f'  Max TotalAmount_Sum: ${df["TotalAmount_Sum"].max():.2f}')

# Show High Value segment stats
high_value = df[df['CustomerSegment_Encoded'] == 2]
print(f'\nHigh Value segment stats:')
print(f'  Count: {len(high_value)}')
print(f'  TotalAmount range: ${high_value["TotalAmount_Sum"].min():.2f} - ${high_value["TotalAmount_Sum"].max():.2f}')
print(f'  Mean TotalAmount: ${high_value["TotalAmount_Sum"].mean():.2f}')
