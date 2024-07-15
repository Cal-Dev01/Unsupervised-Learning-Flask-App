import pandas as pd
import numpy as np


# Function to generate random data
def generate_random_data(num_samples):
    np.random.seed()
    size = np.random.uniform(0.5, 10.0, num_samples)
    sound = np.random.uniform(0.1, 5.0, num_samples)
    return pd.DataFrame({'size': size, 'sound': sound})


# Generate sample data for the demo
data = generate_random_data(150)

# Save the DataFrame to a CSV file
file_path = 'unsupervised_demo_data.csv'
data.to_csv(file_path, index=False)

print(f"CSV file generated and saved as {file_path}")
