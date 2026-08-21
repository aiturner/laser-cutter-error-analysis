import pandas as pd

# Load the CSV
df = pd.read_csv('Test6Center_mm.csv')

# Extract dx and dy columns and divide by constant
constant = 1 # 398.64  Change this to your value
df['x'] = df['dx'] / constant
df['y'] = df['dy'] / constant

# Save only the new columns to a new CSV
df[['x', 'y']].to_csv('output_tensor_Test6Center_mm.csv', index=False)

print(f"Converted {len(df)} points. Saved to output_tensor_Measurements3.csv")

