import pandas as pd

# import libraries
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tabulate import tabulate

"""# Player Playoffs CSV"""

df_player_playoffs_unfiltered = pd.read_csv("DataAsCSV/player_playoffs.csv")
df_player_playoffs_unfiltered = df_player_playoffs_unfiltered.fillna(0)
# print(df_player_playoffs_unfiltered)

df_player_playoffs = df_player_playoffs_unfiltered.iloc[:, 7:23]


# print(df_player_playoffs)


def perform_PCA(df):
    x = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principal_Df = pd.DataFrame(data=principalComponents
                                , columns=['principal component 1', 'principal component 2'])
    return principal_Df


principalDf = perform_PCA(df_player_playoffs)
# create arrays
X = principalDf.values
# instantiate model
neighbours = NearestNeighbors(n_neighbors=3)  # fit model
neighbours.fit(X)
# distances and indexes of k-neighbours from model outputs
distances, indexes = neighbours.kneighbors(X)  # plot mean of k-distances of each observation
# plt.plot(distances.mean(axis =1))

# visually determine cutoff values
player_playoffs_outlier_index = np.where(distances.mean(axis=1) > 0.6)
outlier_values = principalDf.iloc[player_playoffs_outlier_index]
# plot outlier values
plt.scatter(principalDf["principal component 1"], principalDf["principal component 2"], color="c")
plt.scatter(outlier_values["principal component 1"], outlier_values["principal component 2"], color="m")
plt.title("Player Playoffs Outlier Detection")
plt.show()

print("Player Playoffs Outlier Detection")
print(tabulate(df_player_playoffs_unfiltered.iloc[
                   outlier_values.sort_values(by=['principal component 1', 'principal component 2'],
                                              ascending=False).index], headers='keys', tablefmt='psql'))

"""# Player Playoffs Career CSV

"""

df_player_playoffs_career_unfiltered = pd.read_csv("DataAsCSV/player_playoffs_career.csv")
df_player_playoffs_career_unfiltered = df_player_playoffs_career_unfiltered.fillna(0)
# print(df_player_playoffs_career_unfiltered)

df_player_playoffs_career = df_player_playoffs_career_unfiltered.iloc[:, 4:21]
# print(df_player_playoffs_career)

principalDf = perform_PCA(df_player_playoffs_career)
# create arrays
X = principalDf.values
# instantiate model
neighbours = NearestNeighbors(n_neighbors=3)  # fit model
neighbours.fit(X)
# distances and indexes of k-neighbours from model outputs
distances, indexes = neighbours.kneighbors(X)  # plot mean of k-distances of each observation
# plt.plot(distances.mean(axis =1))

# visually determine cutoff values
player_playoffs_career_outlier_index = np.where(distances.mean(axis=1) > 0.6)
outlier_values = principalDf.iloc[player_playoffs_career_outlier_index]
# plot outlier values
plt.scatter(principalDf["principal component 1"], principalDf["principal component 2"], color="c")
plt.scatter(outlier_values["principal component 1"], outlier_values["principal component 2"], color="m")
plt.title("Player Playoffs Career Outlier Detection")
plt.show()

print("Player Playoffs Career Detection")
print(tabulate(df_player_playoffs_career_unfiltered.iloc[
                   outlier_values.sort_values(by=['principal component 1', 'principal component 2'],
                                              ascending=False).index], headers='keys', tablefmt='psql'))

"""# Player Regular Season CSV"""

df_player_regular_season_unfiltered = pd.read_csv("DataAsCSV/player_regular_season.csv")
df_player_regular_season_unfiltered = df_player_regular_season_unfiltered.fillna(0)
# print(df_player_regular_season_unfiltered)

df_player_regular_season = df_player_regular_season_unfiltered.iloc[:, 7:23]
# print(df_player_regular_season)

principalDf = perform_PCA(df_player_regular_season)
# create arrays
X = principalDf.values
# instantiate model
neighbours = NearestNeighbors(n_neighbors=3)  # fit model
neighbours.fit(X)
# distances and indexes of k-neighbours from model outputs
distances, indexes = neighbours.kneighbors(X)  # plot mean of k-distances of each observation
# plt.plot(distances.mean(axis =1))

# visually determine cutoff values
player_regular_season_outlier_index = np.where(distances.mean(axis=1) > 0.5)
outlier_values = principalDf.iloc[player_regular_season_outlier_index]
# plot outlier values
plt.scatter(principalDf["principal component 1"], principalDf["principal component 2"], color="c")
plt.scatter(outlier_values["principal component 1"], outlier_values["principal component 2"], color="m")
plt.title("Player Regular Season Outlier Detection")
plt.show()

print("Player Regular Season Outlier Detection")
print(tabulate(
    df_player_regular_season_unfiltered.iloc[outlier_values.sort_values(by=['principal component 1', 'principal '
                                                                                                     'component '
                                                                                                     '2'],
                                                                        ascending=False).index], headers='keys',
    tablefmt='psql'))

"""# Player Regular Season Career CSV

"""

df_player_regular_season_career_unfiltered = pd.read_csv("DataAsCSV/player_regular_season_career.csv")
df_player_regular_season_career_unfiltered = df_player_regular_season_career_unfiltered.fillna(0)
# print(df_player_regular_season_career_unfiltered)

df_player_regular_season_career = df_player_regular_season_career_unfiltered.iloc[:, 4:21]
# print(df_player_regular_season_career)


principalDf = perform_PCA(df_player_regular_season_career)
# create arrays
X = principalDf.values
# instantiate model
neighbours = NearestNeighbors(n_neighbors=3)  # fit model
neighbours.fit(X)
# distances and indexes of k-neaighbors from model outputs
distances, indexes = neighbours.kneighbors(X)  # plot mean of k-distances of each observation
# plt.plot(distances.mean(axis =1))

# visually determine cutoff values
player_regular_season_career_outlier_index = np.where(distances.mean(axis=1) > 0.5)
outlier_values = principalDf.iloc[player_regular_season_career_outlier_index]
# plot outlier values
plt.scatter(principalDf["principal component 1"], principalDf["principal component 2"], color="c")
plt.scatter(outlier_values["principal component 1"], outlier_values["principal component 2"], color="m")
plt.title("Player Regular Season Career Outlier Detection")
plt.show()

print("Player Regular Season Career Outlier Detection")
print(tabulate(df_player_regular_season_career_unfiltered.iloc[
                   outlier_values.sort_values(by=['principal component 1', 'principal component 2'],
                                              ascending=False).index], headers='keys', tablefmt='psql'))
