'''
Plotting % of swim for each 100 is always going to yield a straight line - isn't easy way to show clustering although clustering can be done

Instead, let us plot how much the first 100 was on x-axis as we were, but on the y-axis, we should plot the percentage off the 200m pb, again using only swims within 2% of pb
- this will hopefully actually give us 2D clustering rather than effectively trying to perform clustering along 1D

Find top 20 athletes in a 200m event
- go to biog
- find top swims in that event where:
    - total time was within 2% of pb
    - if splits are available on rankings to scrape, collect them for that athlete

Example:
Top athletes in 200 Free --> Matt Richards
PB is 1:44.30 ==> 104.30
within 2% of this ==> 104.30 + 2.086 = 106.386 ==> 106.38 = 1:46.39 (round to the hundredth)

Find all swims for Matt where his time is within this threshold and where splits exist. Store:
[% of swim for 100m, % off pb for total time] 

Try different clustering like K-NN and DBSCAN and evaluate each
'''
from Scraper import *
import requests
from bs4 import BeautifulSoup
import re # for regular expressions when fetching ASA ID numbers from table
from datetime import datetime
from enum import Enum
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import threading
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import DBSCAN

session = requests.Session() # Create a session object
# Configure connection pool size
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
session.mount('http://', adapter)
session.mount('https://', adapter)

def get_pb(ID: int, event: int):
    '''
    Gets pb using ID and event, in seconds
    '''
    URL = f"https://www.swimmingresults.org/individualbest/personal_best_time_date.php?back=biogs&tiref={str(ID)}&mode=A&tstroke={str(event)}&tcourse=L"
    page = session.get(URL) 
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("table")[0]
    table = results.find("tbody")
    rows = table.find_all("tr")[1:] #exclude the first row since it is the headers of the table
    # the first row will have the pb - we need this in seconds so we can calculate a threshold and filter the remaining rows
    pb = convert_timestring_to_secs(rows[0].find_all("td")[0].text.strip())
    return pb


def get_links_for_swims_with_splits(ID: int, event: int):
    '''
    Swims within 2% of pb which have a hyperlink for splits will be found; the links for these splits will then be returned and further parsed to get the splits
    '''
    URL = f"https://www.swimmingresults.org/individualbest/personal_best_time_date.php?back=biogs&tiref={str(ID)}&mode=A&tstroke={str(event)}&tcourse=L"
    page = session.get(URL) 
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("table")[0]
    table = results.find("tbody")
    rows = table.find_all("tr")[1:] #exclude the first row since it is the headers of the table

    # the first row will have the pb - we need this in seconds so we can calculate a threshold and filter the remaining rows
    pb = convert_timestring_to_secs(rows[0].find_all("td")[0].text.strip())

    #print(pb)

    threshold_time = round(pb*1.02, 2) # rounding will be computationally easier than converting types just to truncate

    #print(threshold_time)

    fast_rows = list(filter(
        lambda x: convert_timestring_to_secs(x.find_all("td")[0].text.strip()) < threshold_time, 
        rows
    ))

    #print(fast_rows)

    links_rows = list(filter(
        lambda x: len(x.find_all("td")[0].find_all("a")) > 0,
        fast_rows
    ))

    #print(links)

    # the output needs to be the links from the rows - the end time and the splits are in the link
    return list(map(
        lambda x: x.find_all("td")[0].find_all("a")[0]['href'],
        links_rows
    )
    )
    
#print(get_links_for_swims_with_splits(921675, 12))

def get_splits_from_link(url_suffix):
    '''
    Provided a swim url for splits, such as /splits/?swimid=41110197, this function will append to the base url for the rankings site
    and extract the split times for a given swim, as well as the total time as the fifth item in the list

    If the splits are not structured correctly, nothing will be returned
    '''
    URL = f"https://www.swimmingresults.org{url_suffix}"
    page = session.get(URL) 
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("table")[0]
    table = results.find("tbody")
    rows = table.find_all("tr") # don't exclude first row as we need it to determine how to collect the split times
    # typically, a table of splits will have three columns: distance, cumulative, incremental
    # the first item in incremental will be blank so the first split will always need to be acquired from cumulative
    # after this, it is probably easier to scrape from incremental as no conversions are required
    splits = []
    if len(rows[0].findChildren()) == 3:
        try:
            # all three columns exist so use the approach mentioned
            # scrape rows 1 through 4 for the splits
            splits.append(convert_timestring_to_secs(rows[1].find_all("td")[1].text.strip()))
            for i in [2,3,4]:
                splits.append(convert_timestring_to_secs(rows[i].find_all("td")[2].text.strip()))
        except Exception:
            # if the above method is unsuccessful, the method using just incremental will be tried
            splits = []
    if splits == []:
        try:
            cum = []
            # typically, the column to be missing will be the incremental column so we will attempt to build the splits using just the cumulative column
            for i in range(1, 5):
                cum.append(convert_timestring_to_secs(rows[i].find_all("td")[1].text.strip()))
            # we have the splits as cumulative but need to calculate the differences
            splits.append(cum[0])
            splits.append(cum[1] - cum[0])
            splits.append(cum[2] - cum[1])
            splits.append(cum[3] - cum[2])
        except Exception:
            splits = []
    # we need to check that the splits add up to the total time to ensure they have been collected correctly
    if splits != []:
        pb = convert_timestring_to_secs(rows[-1].find_all("td")[1].text.strip())
        '''
        print(pb)
        print(splits)
        print(sum(splits))
        '''
        if sum(splits) == pb or round(sum(splits), 2) == pb:
            splits.append(pb)
            return splits
        else:
            return []
    else:
        return []

def get_100_splits_as_percentage_of_total(list_of_splits, pb):
    '''
    If the input is of length 5, it needs to sum the first two and last two 50s together to get the two 100 splits
    If the input is of length 3, it is assumed to be ready for use
    The last input is assumed to be the total time

    The return will be of the form [1st 100m as percentage of total time, difference of total time to pb by percentage]
    '''
    if len(list_of_splits) == 5:
        # need to sum the halves
        redone = [list_of_splits[0] + list_of_splits[1], list_of_splits[2] + list_of_splits[3], list_of_splits[-1]]
        list_of_splits = redone
    if len(list_of_splits) == 3:
        # the list is in the format of [1st 100, 2nd 100, total 200 time] so we just do a simple calculation
        ret = [
            list_of_splits[0] / list_of_splits[2], 
            (100 * (list_of_splits[2] / pb)) - 100
        ]
        return ret
    else:
        return []


def reworked_pipeline_100s(ID: int, event: int):
    '''
    Provided an ID and event code, this will get all their swims within 2% of their pb, find those with splits and get the first 100s from each swim as a percentage of that swim's
    total time for 200m. Then, this will be returned with the percentage within the swim pb
    '''
    output = [[],[]]
    pb = get_pb(ID, event)
    for link in get_links_for_swims_with_splits(ID, event):
        splits = get_splits_from_link(link)
        data = get_100_splits_as_percentage_of_total(splits, pb)
        #print(data)
        if len(data) > 0:
            if data[0] < .55:
                output[0].append(data[0])
                output[1].append(data[1])
    return output

#print(reworked_pipeline_100s(921675, 12))

men = [[],[]]
women = [[],[]]

array_lock = threading.Lock()

class Splits200ScrapeThread(threading.Thread):
    def __init__(self, IDs: list, event: int = 3, sex: int = 0,  *args, **kwargs):
        '''
        === Provided a list of IDs, the thread will find 200m splits for an athlete in the event specified where splits exist, taking times within 2% of their PB
        === The output should be of the form [1st 100m as a percentage of total swim time, Difference between pb and total swim time for the event as a percentage]
        Example: Matt Richards has a 200 free time of 1:45.77 == 105.77 ==> 51.04 + 54.73 ==> as percentages of the total time: [48.2556..., 51.7443...]
        We take the first 100m and his total time compared to pb - this is his pb hence it will be 0% diff. ==> [0.482556..., 0.0]
        '''
        super(Splits200ScrapeThread, self).__init__()
        self.IDs = IDs
        self.event = event
        self.sex = sex
        self.args = args
        self.kwargs = kwargs

    def run(self):
        global men, women
        x = [[],[]]
        for ID in self.IDs:
            percs = reworked_pipeline_100s(ID, self.event)
            if percs != [[],[]]:
                x[0].extend(percs[0])
                x[1].extend(percs[1])
        with array_lock:
            if self.sex == 0:
                #men[0] = np.append(x[0], men[0])
                men[0].extend(x[0])
                men[1].extend(x[1])
            else:
                #women = np.append(x, women)
                women[0].extend(x[0])
                women[1].extend(x[1])

for event in [16]:
    if event == 3:
        name = "200m Free"
        cols = ['b','r']
    elif event == 16:
        name = "200m Medley"
        cols = ['orange', 'green']

    ### UNCOMMENT THESE FOR THE REGRESSION MODEL
    #men = [[],[]]
    #women = [[],[]]

    # set nationality to british as this increases the likelihood that an athlete has been training in the country their whole life with records to show for it
    mensIDs = list(map(extract_ID_from_row, get_rows(set_parameters(stroke=str(event), nationality='A', records_to_view='40', year='A'))))
    womensIDs = list(map(extract_ID_from_row, get_rows(set_parameters(stroke=str(event), nationality='A', records_to_view='40', year='A', sex='F'))))
    t1 = Splits200ScrapeThread(mensIDs[:len(mensIDs)//2], event=event)
    t2 = Splits200ScrapeThread(mensIDs[len(mensIDs)//2:], event=event)
    t3 = Splits200ScrapeThread(womensIDs[:len(womensIDs)//2], event=event, sex=1)
    t4 = Splits200ScrapeThread(womensIDs[len(womensIDs)//2:], event=event, sex=1)
    threads = [t1, t2, t3, t4]
    for t in threads:
        t.start()
    for t in threads:
        t.join()



### DBSCAN MODEL ###

print(len(men[0]), len(men[1]), len(women[0]), len(women[1]))
while len(men[0]) > len(women[0]):
    men[0].pop(-1)
    men[1].pop(-1)
while len(women[0]) > len(men[0]):
    women[0].pop(-1)
    women[1].pop(-1)
print(len(men[0]), len(men[1]), len(women[0]), len(women[1]))
males = np.array(men)
females = np.array(women)

np.savetxt('Unscaled Men Scatter Points to use for DBSCAN.csv', males, delimiter=',', fmt='%.10f')
np.savetxt('Unscaled Women Scatter Points to use for DBSCAN.csv', females, delimiter=',', fmt='%.10f')

data = [men[0], men[1], women[0], women[1]]

# Extract x and y values into separate lists
x_vals = np.concatenate((data[0], data[2]))  # Combine men's and women's x values
y_vals = np.concatenate((data[1], data[3]))  # Combine men's and women's y values

# Restructured data
restructured_data = [x_vals, y_vals]

# Convert the restructured data to a 2D NumPy array
data_for_dbscan = np.transpose(np.array(restructured_data))

# Scale the data (optional but recommended)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_for_dbscan)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.06, min_samples=5)  # Adjust eps and min_samples as needed
clusters = dbscan.fit_predict(scaled_data)

# Create separate lists for male and female data
male_x = data[0]
male_y = data[1]
female_x = data[2]
female_y = data[3]


cluster_means = {}
for cluster_id in np.unique(clusters):
    if cluster_id != -1:  # Exclude outliers
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_data = data_for_dbscan[cluster_indices]
        cluster_mean = np.mean(cluster_data, axis=0)
        cluster_means[cluster_id] = cluster_mean


# Plot male points as circles
plt.scatter(male_x, male_y, c=clusters[:len(male_x)], cmap='viridis', marker='o', edgecolor='k', label='Male')

# Plot female points as crosses
plt.scatter(female_x, female_y, c=clusters[len(male_x):], cmap='viridis', marker='^', edgecolor='k', label='Female')

for cluster_id, mean_point in cluster_means.items():
    plt.scatter(mean_point[0], mean_point[1], marker='p', color='red', s=100)  # Red pentagon for mean points
    plt.text(mean_point[0], mean_point[1], f'Cluster {cluster_id}', fontsize=10, ha='center', va='bottom')

# Add labels and title
plt.xlabel('First 100m as a percentage of that swim time')
plt.ylabel("Difference between total 200m time and the swimmer's 200m PB as a percentage")
plt.title('Plot of first 100m as percentage of total swim time vs percentage off PB for the overall swim for 200m Freestyle in elite British male and female performances')

plt.colorbar(label='Clusters')  # Add a label to the colorbar

plt.legend()  # Add a legend to distinguish male and female points
plt.grid()
plt.show()

'''
data = [men[0], men[1], women[0], women[1]]

# Extract x and y values into separate lists
x_vals = np.concatenate((data[0], data[2]))  # Combine men's and women's x values
y_vals = np.concatenate((data[1], data[3]))  # Combine men's and women's y values

# Restructured data
restructured_data = [x_vals, y_vals]

# Convert the restructured data to a 2D NumPy array
data_for_dbscan = np.transpose(np.array(restructured_data))

# Scale the data (optional but recommended)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_for_dbscan)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.075, min_samples=5)  # Adjust eps and min_samples as needed
clusters = dbscan.fit_predict(scaled_data)
# Plot the original data with cluster colors
plt.scatter(x_vals, y_vals, c=clusters, cmap='viridis', edgecolor='k')  # Use 'viridis' or your preferred colormap
plt.colorbar(label='Clusters')  # Add a label to the colorbar
# Add labels and title
plt.xlabel('First 100m as a percentage of that swim time')
plt.ylabel("Difference between total 200m time and the swimmer's 200m PB as a percentage")
plt.title('Plot of first 100m as percentage of total swim time vs percentage off PB for the overall swim for 200m Medley in elite British male and female performances')

plt.legend()
plt.grid()
plt.show()
'''