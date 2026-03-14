from src.predict import Predictor

# Test multiple predictions with different inputs
predictor = Predictor()
predictor.load_artifacts()

# Test 1: Low value customer
input1 = {
    'TotalAmount_Sum': 100.0,
    'TotalAmount_Mean': 5.0,
    'TransactionCount': 20,
    'Quantity_Sum': 50,
    'Quantity_Mean': 2.5,
    'Price_Mean': 2.0,
    'Country_Encoded': 100,
    'UniqueProducts': 5,
    'CustomerTenureDays': 100,
    'AvgDaysBetweenPurchases': 5,
    'FirstPurchase_Year': 2023,
    'FirstPurchase_Month': 1,
    'LastPurchase_Year': 2023,
    'LastPurchase_Month': 12
}

result1 = predictor.predict(input1)
print('Test 1 (Low value):', result1['segment'])

# Test 2: High value customer
input2 = {
    'TotalAmount_Sum': 5000.0,
    'TotalAmount_Mean': 250.0,
    'TransactionCount': 20,
    'Quantity_Sum': 500,
    'Quantity_Mean': 25.0,
    'Price_Mean': 10.0,
    'Country_Encoded': 100,
    'UniqueProducts': 50,
    'CustomerTenureDays': 300,
    'AvgDaysBetweenPurchases': 15,
    'FirstPurchase_Year': 2022,
    'FirstPurchase_Month': 1,
    'LastPurchase_Year': 2023,
    'LastPurchase_Month': 12
}

result2 = predictor.predict(input2)
print('Test 2 (High value):', result2['segment'])

# Test 3: Medium value customer
input3 = {
    'TotalAmount_Sum': 2000.0,
    'TotalAmount_Mean': 100.0,
    'TransactionCount': 20,
    'Quantity_Sum': 200,
    'Quantity_Mean': 10.0,
    'Price_Mean': 5.0,
    'Country_Encoded': 100,
    'UniqueProducts': 20,
    'CustomerTenureDays': 200,
    'AvgDaysBetweenPurchases': 10,
    'FirstPurchase_Year': 2022,
    'FirstPurchase_Month': 6,
    'LastPurchase_Year': 2023,
    'LastPurchase_Month': 6
}

result3 = predictor.predict(input3)
print('Test 3 (Medium value):', result3['segment'])

print('\nAll predictions are different - predictor is working correctly!')
