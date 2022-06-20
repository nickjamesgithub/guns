import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mj_wasserstein_utilities import mj_wasserstein, dendrogram_plot, changepoint_probabilities, dendrogram_plot_test, transitivity_test, plot_3d_mj_wasserstein

sector_returns = pd.read_csv("/Users/tassjames/Desktop/crypto_sector_data/adaptspec_results/Equity_sector_returns_label.csv", index_col='Date')

# Import States
alabama = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Alabama_cutpoints.csv")
alaska = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Alaska_cutpoints.csv")
arizona = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Arizona_cutpoints.csv")
arkansas = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Arkansas_cutpoints.csv")
california = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/California_cutpoints.csv")
colorado = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Colorado_cutpoints.csv")
connecticut = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Connecticut_cutpoints.csv")
delaware = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Delaware_cutpoints.csv")
dc = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/District.of.Columbia_cutpoints.csv")
florida = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Florida_cutpoints.csv")
georgia = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Georgia_cutpoints.csv")
hawaii = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Hawaii_cutpoints.csv")
idaho = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Idaho_cutpoints.csv")
illinois = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Illinois_cutpoints.csv")
indiana = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Indiana_cutpoints.csv")
iowa = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Iowa_cutpoints.csv")
kansas = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Kansas_cutpoints.csv")
kentucky = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Kentucky_cutpoints.csv")
louisiana = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Louisiana_cutpoints.csv")
maine = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Maine_cutpoints.csv")
maryland = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Maryland_cutpoints.csv")
massachusetts = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Massachusetts_cutpoints.csv")
michigan = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Michigan_cutpoints.csv")
minnesota = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Minnesota_cutpoints.csv")
mississippi = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Mississippi_cutpoints.csv")
missouri = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Missouri_cutpoints.csv")
montana = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Montana_cutpoints.csv")
nebraska = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Nebraska_cutpoints.csv")
nevada = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Nevada_cutpoints.csv")
new_hampshire = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/New.Hampshire_cutpoints.csv")
new_jersey = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/New.Jersey_cutpoints.csv")
new_mexico = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/New.Mexico_cutpoints.csv")
new_york = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/New.York_cutpoints.csv")
north_carolina = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/North.Carolina_cutpoints.csv")
north_dakota = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/North.Dakota_cutpoints.csv")
ohio = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Ohio_cutpoints.csv")
oklahoma = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Oklahoma_cutpoints.csv")
oregon = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Oregon_cutpoints.csv")
pennsylvania = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Pennsylvania_cutpoints.csv")
rhode_island = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Rhode.Island_cutpoints.csv")
south_carolina = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/South.Carolina_cutpoints.csv")
south_dakota = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/South.Dakota_cutpoints.csv")
tennessee = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Tennessee_cutpoints.csv")
texas = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Texas_cutpoints.csv")
utah = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Utah_cutpoints.csv")
vermont = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Vermont_cutpoints.csv")
virginia = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Virginia_cutpoints.csv")
washington = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Washington_cutpoints.csv")
west_virginia = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/West.Virginia_cutpoints.csv")
wisconsin = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Wisconsin_cutpoints.csv")
wyoming = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Wyoming_cutpoints.csv")

# Set Labels
states = [alabama, alaska, arizona, arkansas, california, colorado, connecticut, delaware, dc, florida, georgia,
          hawaii, idaho, illinois, indiana, iowa, kansas, kentucky, louisiana, maine, maryland, massachusetts,
          michigan, minnesota, mississippi, missouri, montana, nebraska, nevada, new_hampshire, new_jersey, new_mexico,
          new_york, north_carolina, north_dakota, ohio, oklahoma, oregon, pennsylvania, rhode_island, south_carolina,
          south_dakota, tennessee, texas, utah, vermont, virginia, washington, west_virginia, wisconsin, wyoming]

labels = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "DC",
          "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
          "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
          "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
          "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
          "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

# Compute MJ Wasserstein distance between states
distance = np.zeros((len(labels), len(labels)))
for i in range(len(labels)):
    for j in range(len(labels)):
        x_cutpoints, x_probabilities, x_max_prob = changepoint_probabilities(states[i])
        y_cutpoints, y_probabilities, y_max_prob = changepoint_probabilities(states[j])
        distance[i,j] = mj_wasserstein(x_cutpoints, x_probabilities, x_max_prob, y_cutpoints, y_probabilities, y_max_prob)
    print(states[i])

# # Compute sector norm
# print("L1 Distance matrix norm is ", np.sum(distance) * 1/((len(states))*(len(states)-1)))
# print("L2 Distance matrix norm is ", np.linalg.norm(distance) * 1/np.sqrt(((len(states))*(len(states)-1))))
#
# # Operator norm
# # eigenvalues and eigenvectors
# vals, vecs = np.linalg.eig(distance)
#
# # sort these based on the eigenvalues
# vecs = vecs[:, np.argsort(vals)]
# vals = vals[np.argsort(vals)]
#
# # operator norm
# print("operator norm", np.max(np.abs(vals)))

# Distance Matrix
plt.matshow(distance)
plt.title("Extreme Breaks Distance")
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.show()

# MJ-Wasserstein dendrogram on countries
dendrogram_plot_test(distance, "mj_wasserstein_", "_states_events_", labels)

# # Generate date grid
# time_returns = pd.date_range("01-01-2019","12-01-2021",len(sector_returns["Energy"]))
#
# def generate_changepoint_plot(time_returns, sector, label):
#     fig, ax = plt.subplots()
#     cuts, probs, max_prob = changepoint_probabilities(sector)
#     plt.plot(time_returns, sector_returns[label], alpha=0.75, color='orange')
#     for i in range(len(max_prob)):
#         for j in range(len(probs[i])):
#             plt.axvline(time_returns[cuts[i][j]], color='black', alpha=probs[i][j])
#     plt.xlabel("Time")
#     plt.ylabel("Log returns")
#     plt.title(label+" log returns change points")
#     ax.xaxis.set_major_locator(plt.MaxNLocator(5))
#     plt.savefig("Equity_CP_"+label+"_log_returns")
#     plt.show()
#
# # Generate plots for all sectors
# for j in range(len(sectors)):
#     generate_changepoint_plot(time_returns, sectors[j], labels_ts[j])
