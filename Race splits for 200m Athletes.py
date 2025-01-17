'''
Find top 20 athletes in a 200m event
- go to biog
- find top swims in that event where:
    - total time was within 2% of pb
    - if splits are available on rankings to scrape, collect them for that athlete

See how varies with LC vs SC, by gender, by event

Example:
Top athletes in 200 Free --> Matt Richards
PB is 1:44.30 ==> 104.30
within 2% of this ==> 104.30 + 2.086 = 106.386 ==> 106.38 = 1:46.39 (round to the hundredth)

Find all swims for Matt where his time is within this threshold and where splits exist --> add to array formatted such that:
[first 50, second 50, third 50, fourth 50] are the columns as percentages of the total 200 time

Once we have plenty of data for different strokes for top athletes and also for gender, we could do a PCA or some form of clustering.
The goal is to see what racing tactics seem to be most common for top athletes based on stroke and gender, if any. With this, a coach
could develop a training plan for a developing athlete to try and base their splits off a common strategy.


We can also divide the 200 by 100s although this offers less insight into the tactics as splitting by 50, but it does allow for the 
1st 100 to be plotted directly against the 2nd 100 with no further techniques like a PCA. This means clustering can be directly applied.
'''
from Scraper import *
import requests
from bs4 import BeautifulSoup
import re # for regular expressions when fetching ASA ID numbers from table
from datetime import datetime
from enum import Enum
import numpy as np 
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
    
#print(get_links_for_swims_with_splits(879146, 3))

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
        if sum(splits) == pb:
            splits.append(pb)
            return splits
        else:
            return []
    else:
        return []

def get_50_splits_as_percentage_of_total(list_of_splits):
    '''
    Provided a list of splits in the form [1st 50, 2nd 50, 3rd 50, 4th 50, total time] all in seconds,
    this function aims to get each constituent split as a percentage of the total time and return a list of 4 items:
    [percentage of total time for 1st 50, ... for 2nd 50, ... for 3rd 50, ... for 4th 50]
    '''
    if len(list_of_splits) != 5:
        return []
    else:
        percs = []
        for i in range(0,4):
            percs.append(100 * list_of_splits[i] / list_of_splits[-1])
        return percs

def get_100_splits_as_percentage_of_total(list_of_splits):
    '''
    If the input is of length 5, it needs to sum the first two and last two 50s together to get the two 100 splits
    If the input is of length 3, it is assumed to be ready for use
    The last input is assumed to be the total time
    '''
    if len(list_of_splits) == 5:
        # need to sum the halves
        redone = [list_of_splits[0] + list_of_splits[1], list_of_splits[2] + list_of_splits[3], list_of_splits[-1]]
        list_of_splits = redone
    if len(list_of_splits) == 3:
        # the list is in the format of [1st 100, 2nd 100, total 200 time] so we just do a simple calculation
        percs = [
            100 * list_of_splits[0] / list_of_splits[2], 
            100 * list_of_splits[1] / list_of_splits[2]
        ]
        return percs
    else:
        return []

'''
def splits_100_progression_pipeline(ID: int, event: int):
'''
    #Provided an ID and event, this function will get a list of splits for the 100s as follows:
    #[
    #[first 100 %%, second 100 %%],
    #...
    #...
    #]
'''
    output = []
    for link in get_links_for_swims_with_splits(ID, event):
        splits = get_splits_from_link(link)
        data = get_100_splits_as_percentage_of_total(splits)
        if len(data) > 0:
            output.append(data)
    return output
'''

def splits_100_progression_pipeline(ID: int, event: int):
    '''
    Provided an ID and event, this function will get a list of splits for the 100s where
    the first sublist is all the 1st 100s and the second sublist is all the 2nd 100s
    '''
    output = [[],[]]
    for link in get_links_for_swims_with_splits(ID, event):
        splits = get_splits_from_link(link)
        data = get_100_splits_as_percentage_of_total(splits)
        if len(data) > 0:
            output[0].append(data[0])
            output[1].append(data[1])
    return output



# now that we have our methods for getting the data, we just need to develop some threads to scrape lots of it ==> take top 20 men and top 20 women all-time for 200 free

men = [[],[]]
women = [[],[]]

array_lock = threading.Lock()

class Splits200ScrapeThread(threading.Thread):
    def __init__(self, IDs: list, event: int = 3, sex: int = 0,  *args, **kwargs):
        '''
        === Provided a list of IDs, the thread will find 200m splits for an athlete in the event specified where splits exist, taking times within 2% of their PB
        === The output should be a list of either 50s as percentages of the total swim time or 100s
        Example: Matt Richards has a 200 free time of 1:45.77 == 105.77 ==> 51.04 + 54.73 ==> as percentages of the total time: [48.2556..., 51.7443...]
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
            percs = splits_100_progression_pipeline(ID, self.event)
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

#print(splits_100_progression_pipeline(879146,3))

'''
# sample code using Matt Richards



for link in get_links_for_swims_with_splits(879146, 3):
    splits = get_splits_from_link(link)
    print(splits, "\n", get_50_splits_as_percentage_of_total(splits), "\n", get_100_splits_as_percentage_of_total(splits), "\n================\n")


'''
# our thread has been written but needs testing
for event in [3, 16]:
    if event == 3:
        name = "200m Free"
        cols = ['b','r']
    elif event == 16:
        name = "200m Medley"
        cols = ['orange', 'green']

    ### UNCOMMENT THESE FOR THE REGRESSION MODEL
    # men = [[],[]]
    # women = [[],[]]

    # set nationality to british as this increases the likelihood that an athlete has been training in the country their whole life with records to show for it
    mensIDs = list(map(extract_ID_from_row, get_rows(set_parameters(stroke=str(event), nationality='A', records_to_view='20', year='A'))))
    womensIDs = list(map(extract_ID_from_row, get_rows(set_parameters(stroke=str(event), nationality='A', records_to_view='20', year='A', sex='F'))))
    t1 = Splits200ScrapeThread(mensIDs[:len(mensIDs)//2], event=event)
    t2 = Splits200ScrapeThread(mensIDs[len(mensIDs)//2:], event=event)
    t3 = Splits200ScrapeThread(womensIDs[:len(womensIDs)//2], event=event, sex=1)
    t4 = Splits200ScrapeThread(womensIDs[len(womensIDs)//2:], event=event, sex=1)
    threads = [t1, t2, t3, t4]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    #print(men)
    #print(women)

    ### SCATTER PLOT AND REGRESSION ###
    '''
    x = np.array(men[0]).reshape(-1, 1)  # Reshape to a 2D array
    model = LinearRegression()
    model.fit(x, men[1])
    y_pred = model.predict(x)

    wx = np.array(women[0]).reshape(-1, 1)  # Reshape to a 2D array
    model = LinearRegression()
    model.fit(wx, women[1])
    wy_pred = model.predict(wx)

    plt.scatter(men[0], men[1], marker='o', linestyle='-', color=cols[0], alpha=0.3, label=f'Men {name}')
    plt.plot(x, y_pred, color=cols[0], alpha=0.2, label=f'Polynomial Regression Model for Mens {name}')

    plt.scatter(women[0], women[1], marker='x', linestyle='-', color=cols[1], alpha=0.3, label=f'Women {name}')
    plt.plot(wx, wy_pred, color=cols[1], alpha=0.2, label=f'Polynomial Regression Model for Womens {name}')
    '''


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

combined_data = np.concatenate((males, females), axis=0)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(combined_data.T)
plt.scatter(combined_data[0], combined_data[1], c=labels, cmap='viridis', marker='o')

plt.title('Plot of first 100 vs 2nd 100 as a percentage of total swim time for 200m Free and Medley in elite British male and female performances')
plt.xlabel('First 100m as a percentage')
plt.ylabel('Second 100m as a percentage')
plt.legend()
plt.grid()
plt.show()