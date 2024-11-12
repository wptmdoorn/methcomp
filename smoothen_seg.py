import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# Step 1: Load the matrix from a CSV file
matrix = pd.read_csv('methcomp/static/seg.csv', header=None, delimiter=' ', dtype=str)
matrix = matrix.apply(pd.to_numeric, errors='coerce').fillna(0).values

sigmas = [1.0, 2.0, 5.0]

for sigma in sigmas:
    # Step 2: Apply a Gaussian filter to smoothen the matrix
    smoothed_matrix = gaussian_filter(matrix, sigma=sigma)

    # Step 3: Save the smoothed matrix as a new CSV file
    pd.DataFrame(smoothed_matrix).to_csv(f'methcomp/static/smoothed_seg_{sigma}.csv', index=False, header=False)

