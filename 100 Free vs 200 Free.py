'''
Research question: can swimmers who are top end 100 free and 200 free athletes be clustered by gender according to the ratio of their 100 time to 200 time?
Does this clustering seem accurate and is it possible for other strokes?
What about for 50/100 specialists of a given stroke?

Scraper: needs to select top 40 athletes open age in a given year for LC 100 free and see if they are also in the top 40 for LC 200 free
- if so, add their times to the array
- if not, they aren't a specialist

Plotter: needs to plot the points

Apply different clustering algorithms to see what happens and see which ones line up with the gender 

As men have more muscle mass, they are proportionally far better at sprinting, with women being proportionally closer as the distance goes up.
The hypothesis is that theoretically, men will cluster to having a better 100 time than 200 and similar for women

We can plot raw times or percentage difference or even fina point score, with 100 free on the x axis and 200 free on the y axis or vice versa

We could do logistic regression
'''

from Scraper import *
import requests
from bs4 import BeautifulSoup
import re # for regular expressions when fetching ASA ID numbers from tables
from datetime import datetime
from enum import Enum
import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

session = requests.Session() # Create a session object
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10) # Configure connection pool size
session.mount('http://', adapter)
session.mount('https://', adapter)

def get_top_n_athletes_in_event_pair(sex: str="M", threshold: int=60, year: int=2022, event_one: int=2, event_two: int=3):
    '''
    Gets a dictionary consisting of key value pairs where:
    = key = swimmer ID number as int
    = value = two-item list where one item is the first event's time and the other is the second
    
    The threshold value determines which athletes are selected. The function scrapes top n for each event and checks which
    athletes are in the top n for both events. These athletes are then put into the dictionary.
    '''
    men = {}

    men100 = get_rows(set_parameters(stroke=str(event_one), sex=sex, records_to_view=str(threshold), year=str(year)))
    for row in men100:
        men[int(extract_ID_from_row(row))] = [convert_timestring_to_secs(row.findChildren()[-1].text.strip())]

    men200 = get_rows(set_parameters(stroke=str(event_two), sex=sex, records_to_view=str(threshold), year=str(year)))
    for row in men200:
        ID = int(extract_ID_from_row(row))
        if ID in men:
            men[ID].append(convert_timestring_to_secs(row.findChildren()[-1].text.strip()))

    mentop40 = {key: value for key, value in men.items() if len(value) == 2}
    return mentop40

men_free_100_200 = get_top_n_athletes_in_event_pair()
women_free_100_200 = get_top_n_athletes_in_event_pair(sex="F")

men_fly_100_200 = get_top_n_athletes_in_event_pair(event_one=14, event_two=15)
women_fly_100_200 = get_top_n_athletes_in_event_pair(event_one=14, event_two=15, sex="F")


plt.scatter([value[0] for key, value in men_free_100_200.items()], [value[1] for key, value in men_free_100_200.items()], color='#0000FF', marker='o') # men 100vs200 free
plt.scatter([value[0] for key, value in women_free_100_200.items()], [value[1] for key, value in women_free_100_200.items()], color='#0000AA', marker='o') # women 100vs200 free
plt.scatter([value[0] for key, value in men_fly_100_200.items()], [value[1] for key, value in men_fly_100_200.items()], color='#FF0000', marker='o') # men 100vs200 fly
plt.scatter([value[0] for key, value in women_fly_100_200.items()], [value[1] for key, value in women_fly_100_200.items()], color='#AA0000', marker='o') # women 100vs200 fly

plt.title('Scatter Plot of 100m vs 200m Times')
plt.xlabel('100m Times (seconds)')
plt.ylabel('200m Times (seconds)')
plt.grid()
plt.show()