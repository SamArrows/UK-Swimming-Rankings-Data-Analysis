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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


session = requests.Session() # Create a session object
# Configure connection pool size
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
session.mount('http://', adapter)
session.mount('https://', adapter)

'''
Objective: Get 200m times for all strokes except medley for top-ranking 200m Medley swimmers
- Find the ones with the greatest variance
- Plot PCA for it --> men and women separately at first

Provided an ASA ID number, we need a function which returns a four-item list containing their LC pbs


Issues with the model could be that such swimmers haven't swam certain events in a fit enough form for a long time, skewing the data
- essentially, there is a lot of potential for outliers
Despite this, we have some insightful data, such that the greatest variance is seen in the more technically demanding, short axis strokes (fly/breast)
and that this is consistent with men, women and when evaluating both genders combined.

Another model which could complement the one we have could be to take an athlete's top fina points for a stroke and find this as a percentage of their 200m Medley
fina points as this would allow them to be evaluated on their stroke strengths rather than just their 200m stroke strengths. This is beneficial as some athletes 
tend to race 100s or maybe 400 free moreso than they do 200s - some example athletes could be Duncan Scott and Tom Dean, who, although are strong in 200s, are more
commonly seen racing 100s of free and fly as opposed to 200s.

This model won't necessarily be perfect as fina points aren't always considered the most accurate metric to compare different events, however by using a metric like
them, we can get a more comprehensive view of a person's stroke strengths and cross-evaluate it with our 200-based model.
For this model, we need a function which, when provided a swimmer's ID number, finds their top fina points by stroke on their biog by filtering rows into 
groups by stroke. Take only LC strokes for this. We can evaluate SC with SC 200 IM times later.
'''

def get_stroke_best_points(ID: int):
    '''
    Provided an ID, this function will get an athlete's best fina point scores for each stroke on long course, as well as their 200 im fina point score
    in the form: [Free, Breast, Fly, Back, Medley]

    This can then be passed into other functions to get percentage of their medley point score, which can then be used for a PCA

    If a swimmer hasn't got any swims for a particular stroke on long course, the list will be less than length 5 so the function won't return anything.
    To mitigate this risk, scraping athletes whose nationality is set to British is more likely to yield athletes that have trained and raced in UK all their life.
    '''
    URL = f"https://www.swimmingresults.org/individualbest/personal_best.php?tiref={str(ID)}&mode=A&back=biogs"
    page = session.get(URL) 
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("table")[0]
    table = results.find("tbody")

    point_scores = [] # add each stroke's fina points to this as it gets collected

    rows = table.find_all("tr")[1:]
    strokes = ["Free", "Breast", "Butterfly", "Back"]
    for stroke in strokes:
        stroke_rows = list(filter(lambda x: stroke in x.find_all("td")[0].text, rows))
        #print(stroke_rows)
        try:
            point_scores.append(max(list(map(lambda x: int(x.find_all("td")[3].text.strip()), stroke_rows))))
        except ValueError:
            print(f"{stroke} is yielding errors for ID: {ID} - maybe swims don't exist?")
    im_row = list(filter(lambda x: "200 Ind" in x.find_all("td")[0].text, rows))[0]
    im_score = int(im_row.find_all("td")[3].text.strip())
    point_scores.append(im_score)
    if len(point_scores) < 5:
        return None
    else:
        return point_scores # the output can be directly fed into the function {{ get_200_times_as_percentage_of_200IM_time }}

#print(get_stroke_best_points(921675))

def get_200_LC_PBs(ID: int):
    '''
    Will only return the record if a swimmer has a time for each 200m - free, breast, fly, back, medley --> if the list isn't length 5, nothing gets returned
    '''
    URL = f"https://www.swimmingresults.org/individualbest/personal_best.php?tiref={str(ID)}&mode=A&back=biogs"
    page = session.get(URL) 
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("table")[0]
    table = results.find("tbody")
    rows = table.find_all("tr")[1:]
    rows = list(filter(lambda x: "200" in x.find_all("td")[0].text, rows))
    times = list(map(lambda x: convert_timestring_to_secs(x.find_all("td")[1].text.strip()), rows))
    if len(times) < 5:
        return None
    else:
        return times

def get_200_times_as_percentage_of_200IM_time(times: list):
    # with this function, we can find each 200 time as a percentage of their 200 medley time
    if len(times) < 5:
        return None
    else:
        times = list(map(lambda x: 100*x/times[4], times))
        return times[:4]    # exclude the medley portion as it will always be 100%




def plot_combined_pca(use_fina_points: bool = False):
    '''
    The code below combines top 100 men with top 100 women

    if use_fina_points is True then it uses them otherwise it uses 200 times as percentages of 200 medley
    '''
    sets = []
    IM_rows = get_rows(set_parameters(stroke=str(16), records_to_view=str(100), year='A'))
    IDs = list(map(int, get_IDs(IM_rows)))
    for ID in IDs:
        
        if use_fina_points:
            times = get_stroke_best_points(ID)
        else:
            times = get_200_LC_PBs(ID)
        
        
        
        if times != None:
            sets.append(get_200_times_as_percentage_of_200IM_time(times))
    male = np.array(sets)
    sets = []
    IM_rows = get_rows(set_parameters(stroke=str(16), sex="F", records_to_view=str(100), year='A'))
    IDs = list(map(int, get_IDs(IM_rows)))
    for ID in IDs:

        if use_fina_points:
            times = get_stroke_best_points(ID)
        else:
            times = get_200_LC_PBs(ID)
        
        
        if times != None:
            sets.append(get_200_times_as_percentage_of_200IM_time(times))
    fem = np.array(sets)
    data = np.vstack((male, fem))

    # Create labels for the groups
    labels = np.array(['Men'] * male.shape[0] + ['Women'] * fem.shape[0])

    pca = PCA(n_components=2) # Perform PCA
    principal_components = pca.fit_transform(data)

    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_pca['Label'] = labels

    plt.figure(figsize=(8, 6))
    
    colors = {'Men': 'blue', 'Women': 'red'} # Scatter plot for PCA results
    plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Label'].apply(lambda x: colors[x]), label=df_pca['Label'])

    plt.title('PCA of Combined Dataset for men and women - top 100  all time 200m IM LC swimmers for each respective gender as of 20/11/2024')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    pca_components = pca.components_
    loadings_df = pd.DataFrame(pca_components, columns=['Freestyle', 'Breaststroke', 'Butterfly', 'Backstroke'])
    loadings_df.index = ['PC1', 'PC2']

    print("PCA Loadings (Feature Contributions):")
    print(loadings_df)

    explained_variance = pca.explained_variance_ratio_

    '''
    plt.text(0.5, 0.5, f'Explained variance: PC1: {explained_variance[0]:.2f}, PC2: {explained_variance[1]:.2f}\nPCA Loadings (Feature Contributions):\n{loadings_df}', 
            horizontalalignment='center', verticalalignment='center', fontsize=12, transform=plt.gca().transAxes)
    '''

    plt.text(0.8, 0.8, f'Explained variance: PC1: {explained_variance[0]:.2f}, PC2: {explained_variance[1]:.2f}\nPCA Loadings (Feature Contributions):\n{loadings_df}', 
         horizontalalignment='center', verticalalignment='center', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.5), transform=plt.gca().transAxes)


    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Men', markerfacecolor='blue', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='Women', markerfacecolor='red', markersize=10)])

    plt.grid()
    plt.show()

def create_pca(data, title_of_plot, color = 'blue'):
    '''
    Passing in a numpy array of the form
    [
    [x,y,z,w],
    [a,b,c,d],
    ...    
    ]
    where columns are free, breast, fly, back as percentages of 200m IM time,
    this function will plot a pca.
    ===> color is custom for the scatter points, blue by default
    ===> title_of_plot is the title 
    '''
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)

    pca_components = pca.components_

    # Create a DataFrame to visualize the loadings
    loadings_df = pd.DataFrame(pca_components, columns=['Freestyle', 'Breaststroke', 'Butterfly', 'Backstroke'])
    loadings_df.index = ['PC1', 'PC2']

    print("PCA Loadings (Feature Contributions):")
    print(loadings_df)

    explained_variance = pca.explained_variance_ratio_ # Get the explained variance ratio
    print(f'Explained variance by components: {explained_variance}')

    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])


    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.7, color=color)
    plt.title(title_of_plot)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')



    # Show variance ratios in the plot
    explained_variance = pca.explained_variance_ratio_
    plt.text(0.5, 0.5, f'Explained variance: PC1: {explained_variance[0]:.2f}, PC2: {explained_variance[1]:.2f}\nPCA Loadings (Feature Contributions):\n{loadings_df}', 
            horizontalalignment='center', verticalalignment='center', fontsize=12, transform=plt.gca().transAxes)


    plt.grid()
    plt.show()





plot_combined_pca(True)





'''
Below is preparation of data for the function {{  create_pca  }}
'''

#times_200 = get_200_LC_PBs(149309)
#print(times_200)
#print(get_200_times_as_percentage_of_200IM_time(times_200))

# let us take times from all british athletes under the 2:00 mark in 200 medley and get a list of their times in that format
sets = []

'''
The first is for top 100 men of all time, the second is for top 100 women of all time
'''

# IM_rows = get_rows(set_parameters(stroke=str(16), records_to_view=str(100), year='A'))
IM_rows = get_rows(set_parameters(stroke=str(16), sex="F", records_to_view=str(100), year='A'))

#time_threshold = 120 # adjust this to determine which athletes to include --> 120 = athletes who are under 2:00, 124 = athletes under 2:04, etc.
#IDs = list(map(int, get_IDs(list(filter(lambda x: convert_timestring_to_secs(x.find_all("td")[6].text) < 124, IM_rows)))))

# alternatively, we can just map to the top n swimmers selected
IDs = list(map(int, get_IDs(IM_rows)))

for ID in IDs:
    # times = get_200_LC_PBs(ID) #### use this line to do the calculation based on 200 times
    times = get_stroke_best_points(ID) #### use this line to do the calculation using a swimmer's best fina point scores for a given stroke on LC
    if times != None:
        sets.append(get_200_times_as_percentage_of_200IM_time(times))
data = np.array(sets)

'''
label = 'PCA of British Female 200m LC IM Swimmers Data - Top 100 ALL-TIME in each respective gender as of 20/11/2024'
color = 'red'
create_pca(data, label, color)
'''


