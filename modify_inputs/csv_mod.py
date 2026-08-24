import pandas as pd

# Load the CSV
df = pd.read_csv('Error_in_mm_3.csv')

# Extract dx and dy columns and divide by constant
constant = 1 # 398.64  Change this
df['x'] = - df['dx'] / constant
df['y'] = - df['dy'] / constant

# Save only the new columns to a new CSV
df[['x', 'y']].to_csv('offset_data_error3.csv', index=False)

print(f"Converted {len(df)} points. Saved to offset_data_error3.csv")

