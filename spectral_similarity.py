import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mj_wasserstein_utilities import changepoint_probabilities_plot, mj_wasserstein, dendrogram_plot, changepoint_probabilities, dendrogram_plot_test, transitivity_test, plot_3d_mj_wasserstein

make_plots = False

# Read in guns data
guns = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/Gun_events_220610.csv", index_col='Date')

# Get column names
column_names = guns.columns

# Time domain ranking
state_time_means_pre = []
state_time_means_post = []
temporal_mean_deviation = []

for i in range(len(column_names)):

    # Slice pre/post 4/20 and compute temporal means
    pre_slice = guns.iloc[:822, i]
    post_slice = guns.iloc[822:, i]
    state_time_means_pre.append(np.mean(pre_slice))
    state_time_means_post.append(np.mean(post_slice))

    # Plot temporal deviation
    if make_plots:
        fig, ax = plt.subplots()
        time_returns = pd.date_range("01-01-2018", "06-09-2022", len(guns))
        plt.plot(time_returns[:822], pre_slice, label="Events pre-4/20", color='blue', alpha = 0.4)
        plt.plot(time_returns[822:], post_slice, label="Events post 4/20", color='red', alpha = 0.4)
        ax.axhline(y=np.mean(pre_slice), xmin=0, xmax=822/1620, color='blue', alpha = 0.4)
        ax.axhline(y=np.mean(post_slice), xmin=822/1620, xmax = 1, color='red', alpha = 0.4)
        plt.title(column_names[i])
        plt.xlabel("Time")
        plt.ylabel("Events")
        plt.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.savefig("Temporal_mean_deviation_"+column_names[i])
        plt.show()

    # Temporal mean deviation
    temporal_mean_deviation.append([column_names[i], np.mean(pre_slice) - np.mean(post_slice)])

# Make difference in means an array and order
temporal_mean_deviation_array = np.array(temporal_mean_deviation)
sorted_temporal = temporal_mean_deviation_array[temporal_mean_deviation_array[:,1].argsort()]

# Import States PRE April 2020
alabama_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Alabama_pre_estimates.csv")
alaska_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Alaska_pre_estimates.csv")
arizona_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Arizona_pre_estimates.csv")
arkansas_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Arkansas_pre_estimates.csv")
california_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/California_pre_estimates.csv")
colorado_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Colorado_pre_estimates.csv")
connecticut_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Connecticut_pre_estimates.csv")
delaware_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Delaware_pre_estimates.csv")
dc_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/District of Columbia_pre_estimates.csv")
florida_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Florida_pre_estimates.csv")
georgia_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Georgia_pre_estimates.csv")
hawaii_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Hawaii_pre_estimates.csv")
idaho_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Idaho_pre_estimates.csv")
illinois_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Illinois_pre_estimates.csv")
indiana_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Indiana_pre_estimates.csv")
iowa_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Iowa_pre_estimates.csv")
kansas_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Kansas_pre_estimates.csv")
kentucky_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Kentucky_pre_estimates.csv")
louisiana_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Louisiana_pre_estimates.csv")
maine_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Maine_pre_estimates.csv")
maryland_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Maryland_pre_estimates.csv")
massachusetts_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Massachusetts_pre_estimates.csv")
michigan_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Michigan_pre_estimates.csv")
minnesota_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Minnesota_pre_estimates.csv")
mississippi_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Mississippi_pre_estimates.csv")
missouri_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Missouri_pre_estimates.csv")
montana_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Montana_pre_estimates.csv")
nebraska_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Nebraska_pre_estimates.csv")
nevada_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Nevada_pre_estimates.csv")
new_hampshire_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/New Hampshire_pre_estimates.csv")
new_jersey_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/New Jersey_pre_estimates.csv")
new_mexico_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/New Mexico_pre_estimates.csv")
new_york_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/New York_pre_estimates.csv")
north_carolina_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/North Carolina_pre_estimates.csv")
north_dakota_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/North Dakota_pre_estimates.csv")
ohio_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Ohio_pre_estimates.csv")
oklahoma_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Oklahoma_pre_estimates.csv")
oregon_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Oregon_pre_estimates.csv")
pennsylvania_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Pennsylvania_pre_estimates.csv")
rhode_island_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Rhode Island_pre_estimates.csv")
south_carolina_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/South Carolina_pre_estimates.csv")
south_dakota_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/South Dakota_pre_estimates.csv")
tennessee_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Tennessee_pre_estimates.csv")
texas_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Texas_pre_estimates.csv")
utah_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Utah_pre_estimates.csv")
vermont_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Vermont_pre_estimates.csv")
virginia_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Virginia_pre_estimates.csv")
washington_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Washington_pre_estimates.csv")
west_virginia_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/West Virginia_pre_estimates.csv")
wisconsin_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Wisconsin_pre_estimates.csv")
wyoming_pre = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Wyoming_pre_estimates.csv")

# State Spectra post April 2020
alabama_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Alabama_post_estimates.csv")
alaska_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Alaska_post_estimates.csv")
arizona_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Arizona_post_estimates.csv")
arkansas_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Arkansas_post_estimates.csv")
california_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/California_post_estimates.csv")
colorado_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Colorado_post_estimates.csv")
connecticut_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Connecticut_post_estimates.csv")
delaware_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Delaware_post_estimates.csv")
dc_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/District of Columbia_post_estimates.csv")
florida_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Florida_post_estimates.csv")
georgia_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Georgia_post_estimates.csv")
hawaii_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Hawaii_post_estimates.csv")
idaho_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Idaho_post_estimates.csv")
illinois_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Illinois_post_estimates.csv")
indiana_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Indiana_post_estimates.csv")
iowa_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Iowa_post_estimates.csv")
kansas_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Kansas_post_estimates.csv")
kentucky_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Kentucky_post_estimates.csv")
louisiana_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Louisiana_post_estimates.csv")
maine_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Maine_post_estimates.csv")
maryland_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Maryland_post_estimates.csv")
massachusetts_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Massachusetts_post_estimates.csv")
michigan_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Michigan_post_estimates.csv")
minnesota_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Minnesota_post_estimates.csv")
mississippi_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Mississippi_post_estimates.csv")
missouri_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Missouri_post_estimates.csv")
montana_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Montana_post_estimates.csv")
nebraska_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Nebraska_post_estimates.csv")
nevada_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Nevada_post_estimates.csv")
new_hampshire_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/New Hampshire_post_estimates.csv")
new_jersey_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/New Jersey_post_estimates.csv")
new_mexico_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/New Mexico_post_estimates.csv")
new_york_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/New York_post_estimates.csv")
north_carolina_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/North Carolina_post_estimates.csv")
north_dakota_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/North Dakota_post_estimates.csv")
ohio_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Ohio_post_estimates.csv")
oklahoma_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Oklahoma_post_estimates.csv")
oregon_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Oregon_post_estimates.csv")
pennsylvania_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Pennsylvania_post_estimates.csv")
rhode_island_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Rhode Island_post_estimates.csv")
south_carolina_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/South Carolina_post_estimates.csv")
south_dakota_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/South Dakota_post_estimates.csv")
tennessee_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Tennessee_post_estimates.csv")
texas_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Texas_post_estimates.csv")
utah_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Utah_post_estimates.csv")
vermont_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Vermont_post_estimates.csv")
virginia_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Virginia_post_estimates.csv")
washington_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Washington_post_estimates.csv")
west_virginia_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/West Virginia_post_estimates.csv")
wisconsin_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Wisconsin_post_estimates.csv")
wyoming_post = pd.read_csv("/Users/tassjames/Desktop/guns_chaos/guns_adaptspec-1/results/Wyoming_post_estimates.csv")

# Set Labels Pre
states_pre = [alabama_pre, alaska_pre, arizona_pre, arkansas_pre, california_pre, colorado_pre, connecticut_pre, delaware_pre, dc_pre, florida_pre, georgia_pre,
          hawaii_pre, idaho_pre, illinois_pre, indiana_pre, iowa_pre, kansas_pre, kentucky_pre, louisiana_pre, maine_pre, maryland_pre, massachusetts_pre,
          michigan_pre, minnesota_pre, mississippi_pre, missouri_pre, montana_pre, nebraska_pre, nevada_pre, new_hampshire_pre, new_jersey_pre, new_mexico_pre,
          new_york_pre, north_carolina_pre, north_dakota_pre, ohio_pre, oklahoma_pre, oregon_pre, pennsylvania_pre, rhode_island_pre, south_carolina_pre,
          south_dakota_pre, tennessee_pre, texas_pre, utah_pre, vermont_pre, virginia_pre, washington_pre, west_virginia_pre, wisconsin_pre, wyoming_pre]

# Set Labels Post
states_post = [alabama_post, alaska_post, arizona_post, arkansas_post, california_post, colorado_post, connecticut_post, delaware_post, dc_post, florida_post, georgia_post,
          hawaii_post, idaho_post, illinois_post, indiana_post, iowa_post, kansas_post, kentucky_post, louisiana_post, maine_post, maryland_post, massachusetts_post,
          michigan_post, minnesota_post, mississippi_post, missouri_post, montana_post, nebraska_post, nevada_post, new_hampshire_post, new_jersey_post, new_mexico_post,
          new_york_post, north_carolina_post, north_dakota_post, ohio_post, oklahoma_post, oregon_post, pennsylvania_post, rhode_island_post, south_carolina_post,
          south_dakota_post, tennessee_post, texas_post, utah_post, vermont_post, virginia_post, washington_post, west_virginia_post, wisconsin_post, wyoming_post]

# Store differences in Log PSD
spectral_pre_post_deviation = []
key_frequency_deviation = []

# Loop over states
for i in range(len(states_pre)):
    # Distance b/w mean-adjusted spectra
    pre_state = states_pre[i].iloc[1:,1]
    post_state = states_post[i].iloc[1:,1]
    pre_state_mean= pre_state - np.mean(pre_state)
    post_state_mean = post_state - np.mean(post_state)

    # Distance b/w key frequencies on mean-adjusted spectrum
    key_index = [0, 1, 2, 6, 14, 28, 56]
    # Convert key frequencies to array
    pre_key_freqs = np.array(pre_state_mean)
    post_key_freqs = np.array(post_state_mean)
    # Slice key amplitude
    pre_key_amplitude = np.array([pre_key_freqs[val] for val in key_index])
    post_key_amplitude = np.array([post_key_freqs[val] for val in key_index])

    # L1 distance in key frequencies
    l1_key_freq_diffs = np.sum(np.abs(pre_key_amplitude - post_key_amplitude))
    key_frequency_deviation.append([column_names[i], l1_key_freq_diffs]) # Append to list

    if make_plots:
        # Plot spectrum before and after
        plt.plot(np.linspace(0,0.5,len(pre_state_mean)), pre_state_mean, label="Pre-4/20")
        plt.plot(np.linspace(0,0.5,len(post_state_mean)), post_state_mean, label="Post-4/20")
        plt.xlabel("Frequency")
        plt.ylabel("Log PSD")
        plt.title(column_names[i])
        plt.legend()
        plt.savefig("Spectral_deviation_"+column_names[i])
        plt.show()

    print("Iteration", column_names[i])

    # Compute L^1 distance between vectors
    distance = np.sum(np.abs(pre_state_mean - post_state_mean))
    spectral_pre_post_deviation.append([column_names[i], distance])

# Make array and print sorted spectral deviation
spectral_pre_post_deviation_array = np.array(spectral_pre_post_deviation)
sorted_spectral = spectral_pre_post_deviation_array[spectral_pre_post_deviation_array[:,1].argsort()]

# Print sorted temporal mean deviation
print(sorted_temporal)
# Print sorted spectral deviation
print(sorted_spectral)

# Convert to DF and write to csv file
sorted_temporal_df = pd.DataFrame(sorted_temporal)
sorted_spectral_df = pd.DataFrame(sorted_spectral)
sorted_key_freqs_df = pd.DataFrame(key_frequency_deviation)
sorted_temporal_df.to_csv("/Users/tassjames/Desktop/guns_chaos/Ranking_states/sorted_temporal_deviation.csv")
sorted_spectral_df.to_csv("/Users/tassjames/Desktop/guns_chaos/Ranking_states/sorted_spectral_deviation.csv")
sorted_key_freqs_df.to_csv("/Users/tassjames/Desktop/guns_chaos/Ranking_states/sorted_key_freqs_deviation.csv")