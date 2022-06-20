import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilities import mj_wasserstein, dendrogram_plot, changepoint_probabilities, dendrogram_plot_test, transitivity_test, plot_3d_mj_wasserstein
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Import Various Sectors
centralized_exchange = pd.read_csv("/Users/tassjames/Desktop/crypto_sector_data/adaptspec_results/cryptocurrency_sectors/centralized_exchange/estimates.csv")
collectibles = pd.read_csv("/Users/tassjames/Desktop/crypto_sector_data/adaptspec_results/cryptocurrency_sectors/collectibles/estimates.csv")
defi = pd.read_csv("/Users/tassjames/Desktop/crypto_sector_data/adaptspec_results/cryptocurrency_sectors/defi/estimates.csv")
platform = pd.read_csv("/Users/tassjames/Desktop/crypto_sector_data/adaptspec_results/cryptocurrency_sectors/platform/estimates.csv")
privacy = pd.read_csv("/Users/tassjames/Desktop/crypto_sector_data/adaptspec_results/cryptocurrency_sectors/privacy/estimates.csv")
smart_contracts = pd.read_csv("/Users/tassjames/Desktop/crypto_sector_data/adaptspec_results/cryptocurrency_sectors/smart_contracts/estimates.csv")
store_of_value = pd.read_csv("/Users/tassjames/Desktop/crypto_sector_data/adaptspec_results/cryptocurrency_sectors/SOV/estimates.csv")
wallet = pd.read_csv("/Users/tassjames/Desktop/crypto_sector_data/adaptspec_results/cryptocurrency_sectors/wallet/estimates.csv")

# Declare sectors
sectors = [centralized_exchange, collectibles, defi, platform, privacy, smart_contracts, store_of_value, wallet]
labels = ["C Exchange", "Collectibles", "Defi", "Platform",
                       "Privacy", "Sm Contracts", "Store Value", "Wallet"]
labels.sort()

clustering = True

if clustering:
    # Compute distance between all time-varying power spectra
    l1_distance_matrix = np.zeros((len(sectors), len(sectors))) # Initialise distance matrix
    for i in range(len(sectors)):
        for j in range(len(sectors)):
            city_tvs_i = np.array(sectors[i].iloc[:,1:])
            city_tvs_j = np.array(sectors[j].iloc[:,1:])
            # Compute distance between time-varying surface on left and right
            l1_distance = np.sum(np.abs(city_tvs_i - city_tvs_j))* (1/(100 * 1065))
            l1_distance_matrix[i,j] = l1_distance
        print("Crypto iteration", i)

    # Loop over distance lead/lag and comput country anomalies
    for i in range(len(l1_distance_matrix[0])):
        anom = np.sum(l1_distance_matrix[:,i])
        print("Sector TVS anomaly", labels[i], anom)

    # Plot Distance matrix clustered between all adaptspec surfaces
    dendrogram_plot_test(l1_distance_matrix, "_Adaptspec_surface", "_Crypto_", labels)

for i in range(len(sectors)):
    gt_spectrum = sectors[i]
    gt_spectrum = gt_spectrum.transpose()
    ts_length = len(gt_spectrum.iloc[:,0])
    gt_spectrum = np.array(gt_spectrum)[1:ts_length, :]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(25, 220)
    frequency = np.linspace(0, 0.5, 100)
    date_index = pd.date_range("2019-01-01", "2021-12-01", freq='D').strftime('%Y-%m-%d')
    # time_array = np.array(date_index)
    time_array = np.linspace(2019,2022, len(gt_spectrum))
    X, Y = np.meshgrid(frequency, time_array)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, gt_spectrum, cmap=cm.plasma, linewidth=0.25, antialiased=True)
    plt.xlabel("Frequency")
    plt.ylabel("Time")
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_zlabel("Log PSD")
    plt.savefig("3d_surface"+labels[i])
    plt.show()
