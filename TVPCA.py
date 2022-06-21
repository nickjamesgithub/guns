import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh

# Turn off plot
make_plots = False

# Read in Guns data
guns = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Gun_events_220610.csv", index_col='Date')

# Get column names
column_names = guns.columns

# Compute rolling correlation matrix and eigenspectrum
eigenvalue_1_explanatory_variance = []
smoothing_window = 90

# Compute first difference along the columns

# Store time-varying explanatory variance exhibited by eigenvalue 1
eigenvector_1_normalised = []
eigenspectra_normalised = []
for i in range(smoothing_window, len(guns)):
    gv_slice = guns.iloc[(i-smoothing_window):i,:]
    gv_slice_diff = gv_slice.diff().replace(np.nan,0)

    # Convert back into a dataframe
    correlation = np.nan_to_num(gv_slice.corr())

    # Perform eigendecomposition and get explanatory variance
    m_vals, m_vecs = eigsh(correlation, k=10, which='LM')
    m_vecs = m_vecs[:, -1]  # Get 1st eigenvector
    m_vals_1 = m_vals[-1] / len(correlation)
    eigenvector_1_normalised.append(m_vals_1)
    normalized_eigenspectrum = m_vals / len(correlation)
    eigenspectra_normalised.append(normalized_eigenspectrum)
    print(i)

# Generate Time Axis
time_axis = pd.date_range("04-01-2018","06-09-2022",len(eigenvector_1_normalised))

# Plot explanatory variance eigenvalue 1
fig, ax = plt.subplots()
ax.plot(time_axis, eigenvector_1_normalised, color='blue')
ax.set_ylabel('Explanatory variance', color='blue')
ax.set_xlabel("Time")
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.savefig("Explanatory_variance_guns")
plt.show()