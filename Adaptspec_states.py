import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilities import mj_wasserstein, dendrogram_plot, changepoint_probabilities, dendrogram_plot_test, transitivity_test, plot_3d_mj_wasserstein
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
guns = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Gun_events_220610.csv", index_col='Date')
# Get column names
column_names = guns.columns

# Import States
alabama = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Alabama_estimates.csv")
alaska = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Alaska_estimates.csv")
arizona = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Arizona_estimates.csv")
arkansas = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Arkansas_estimates.csv")
california = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/California_estimates.csv")
colorado = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Colorado_estimates.csv")
connecticut = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Connecticut_estimates.csv")
delaware = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Delaware_estimates.csv")
dc = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/District.of.Columbia_estimates.csv")
florida = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Florida_estimates.csv")
georgia = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Georgia_estimates.csv")
hawaii = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Hawaii_estimates.csv")
idaho = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Idaho_estimates.csv")
illinois = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Illinois_estimates.csv")
indiana = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Indiana_estimates.csv")
iowa = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Iowa_estimates.csv")
kansas = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Kansas_estimates.csv")
kentucky = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Kentucky_estimates.csv")
louisiana = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Louisiana_estimates.csv")
maine = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Maine_estimates.csv")
maryland = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Maryland_estimates.csv")
massachusetts = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Massachusetts_estimates.csv")
michigan = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Michigan_estimates.csv")
minnesota = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Minnesota_estimates.csv")
mississippi = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Mississippi_estimates.csv")
missouri = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Missouri_estimates.csv")
montana = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Montana_estimates.csv")
nebraska = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Nebraska_estimates.csv")
nevada = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Nevada_estimates.csv")
new_hampshire = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/New.Hampshire_estimates.csv")
new_jersey = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/New.Jersey_estimates.csv")
new_mexico = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/New.Mexico_estimates.csv")
new_york = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/New.York_estimates.csv")
north_carolina = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/North.Carolina_estimates.csv")
north_dakota = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/North.Dakota_estimates.csv")
ohio = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Ohio_estimates.csv")
oklahoma = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Oklahoma_estimates.csv")
oregon = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Oregon_estimates.csv")
pennsylvania = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Pennsylvania_estimates.csv")
rhode_island = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Rhode.Island_estimates.csv")
south_carolina = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/South.Carolina_estimates.csv")
south_dakota = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/South.Dakota_estimates.csv")
tennessee = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Tennessee_estimates.csv")
texas = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Texas_estimates.csv")
utah = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Utah_estimates.csv")
vermont = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Vermont_estimates.csv")
virginia = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Virginia_estimates.csv")
washington = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Washington_estimates.csv")
west_virginia = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/West.Virginia_estimates.csv")
wisconsin = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Wisconsin_estimates.csv")
wyoming = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Wyoming_estimates.csv")

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

clustering = True

if clustering:
    # Compute distance between all time-varying power spectra
    l1_distance_matrix = np.zeros((len(states), len(states))) # Initialise distance matrix
    for i in range(len(states)):
        for j in range(len(states)):
            city_tvs_i = np.array(states[i].iloc[:,1:])
            city_tvs_j = np.array(states[j].iloc[:,1:])
            # Compute distance between time-varying surface on left and right
            l1_distance = np.sum(np.abs(city_tvs_i - city_tvs_j))* (1/(100 * len(guns)))
            l1_distance_matrix[i,j] = l1_distance
        print("Guns iteration", i)

    # Plot Distance matrix clustered between all adaptspec surfaces
    dendrogram_plot_test(l1_distance_matrix, "_Adaptspec_surface", "_states_guns_", labels)

for i in range(len(states)):
    gt_spectrum = states[i]
    gt_spectrum = gt_spectrum.transpose()
    ts_length = len(gt_spectrum.iloc[:,0])
    gt_spectrum = np.array(gt_spectrum)[1:ts_length, :]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(35,310)
    frequency = np.linspace(0, 0.5, 100)
    date_index = pd.date_range("2018-01-01", "2022-06-09", freq='D').strftime('%Y-%m-%d')
    # time_array = np.array(date_index)
    time_array = np.linspace(2018,2022.5, len(gt_spectrum))
    X, Y = np.meshgrid(frequency, time_array)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, gt_spectrum, cmap=cm.plasma, linewidth=0.25, antialiased=True)
    plt.xlabel("Frequency")
    plt.ylabel("Time")
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_zlabel("Log PSD")
    plt.savefig("3d_surface"+labels[i])
    plt.show()
