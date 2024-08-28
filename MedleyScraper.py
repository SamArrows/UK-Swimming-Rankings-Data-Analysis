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

session = requests.Session() # Create a session object
# Configure connection pool size
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
session.mount('http://', adapter)
session.mount('https://', adapter)

IDs_checked = []

class MedleyScrapeThread(threading.Thread):
    def __init__(self, start_year, pages_to_scrape, stroke, records_to_view, find_all_time_best_events: bool = False, parity: bool = False, reuse_IDs: bool = False, *args, **kwargs):
        '''
        == find_all_time_best_events: determines whether to find a medley swimmer's best events in a given season (False) or for all time (True)
        == parity: is to be used to determine a swimmer's best stroke if not counting medley; it can either be true to do this, or false which means
            the dictionary of events corresponding with a medley swimmer will just count all three of their best non-medley events
        TODO: parity hasn't been programmed into the run function yet and still needs doing
        '''
        super(MedleyScrapeThread, self).__init__()
        self.start_year = start_year
        self.pages_to_scrape = pages_to_scrape
        self.stroke = stroke
        self.records_to_view = records_to_view
        self.find_all_time_best_events = find_all_time_best_events
        self.parity = parity
        self.reuse_IDs = reuse_IDs
        self.args = args
        self.kwargs = kwargs
        self.dic = {}

    def run(self):
        #print("Started running...")
        for i in range(self.start_year, self.start_year + self.pages_to_scrape):
            page = set_parameters(stroke=str(self.stroke), sex='F', records_to_view=str(self.records_to_view), year=str(i))
            for ID in get_IDs(get_rows(page)):
                go = True
                if not self.reuse_IDs:
                    if ID in IDs_checked:
                        go = False
                    else:
                        IDs_checked.append(ID)
                if go:
                    # Doesn't currently distinguish between IDs which have already been done as the best events are taken during that year so the data could change for a given ID
                    # TODO: ADD PARAMETER TO KEEP TRACK OF IDS ALREADY USED
                    if(self.find_all_time_best_events):
                        for event in get_top_events_by_fina_points_from_biog(ID, exclude_medley=True):
                            if(event in self.dic):
                                self.dic[event] += 1
                            else:
                                self.dic[event] = 1
                    else:
                        for event_tup in get_best_events_by_fina_points_for_set_year(ID, year_to_search=str(i)[2:], exclude_medley=True):
                            if(event_tup[0] in self.dic):
                                self.dic[event_tup[0]] += 1
                            else:
                                self.dic[event_tup[0]] = 1

def combine_dicts(dict1: dict, dict2: dict):
    '''
    Combines two dictionaries by summing the counts of their common keys while simply concatenating key-value pairs which are unique to each other
    EXAMPLE:
        DICT1 = {'a': 3, 'b': 5, 'c' : 7 }
        DICT2 = {'b' : 1, 'c': 3, 'd': 3}
        combine_dicts(DICT1, DICT2) = {'a': 3, 'b': 6, 'c': 10, 'd': 3}
    '''
    new_dict = {}
    for key, value in dict1.items():
        if key in dict2:
            new_dict[key] = value + dict2[key]
        else:
            new_dict[key] = value
    for key, value in dict2.items():
        if key not in dict1:
            new_dict[key] = value
    return new_dict

def instantiate_threads(start_year: int, total_pages: int, stroke: int, records_to_view: int = 10, thread_count: int = 4, all_time_best_events: bool = True, parity: bool = False, reuse_IDs: bool = False, *args_for_target):
    '''
    Creates threads with as evenly distributed a workload as possible
    = thread_count: how many threads to create
    = total_pages: number of pages on rankings to scrape
    '''
    threads = []
    pages_per_thread = round(float(total_pages) / thread_count)
    if pages_per_thread != total_pages / thread_count:
        for i in range(0, thread_count-1):
            threads.append(MedleyScrapeThread(start_year+i*pages_per_thread, pages_per_thread, stroke, records_to_view, all_time_best_events, parity, reuse_IDs))
        pages_remaining = total_pages - (thread_count-1) * pages_per_thread
        threads.append(MedleyScrapeThread(start_year+((thread_count-1)*pages_per_thread), pages_remaining, stroke, records_to_view, all_time_best_events, parity, reuse_IDs))
    else:
        for i in range(0, thread_count):
            threads.append(MedleyScrapeThread(start_year+(i*pages_per_thread), pages_per_thread, stroke, records_to_view, all_time_best_events, parity, reuse_IDs))
    return threads

threads = instantiate_threads(2000, 25, 16, 10, 5)
for thread in threads:
    print(f"Start year: {thread.start_year}; pages to scrape: {thread.pages_to_scrape}")

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

dics = []

for thread in threads:
    dics.append(thread.dic)

data = {}

for i in range(0, len(dics)):
    data = combine_dicts(data, dics[i])

print(IDs_checked)

total_swims_in_database = sum(data.values())
print(f"Total swimmers: {total_swims_in_database/3}; Total swims per database taking top three non-medley events per swimmer: {total_swims_in_database}")

for value in data.values():
    value /= total_swims_in_database

data = dict(sorted(data.items(), key=lambda item: Events[convert_biog_event_text_to_enum_format(item[0])].value))
print(data)

colors = [] 
'''
https://matplotlib.org/stable/users/explain/colors/colormaps.html

Define colors: 
= Backstroke events (red shades), 
= Freestyle events (yellow shades), 
= Butterfly events (green shades), 
= Breaststroke events (blue shades)
'''
# Define the color maps for each swimming style
color_maps = {
    "Freestyle": cm.spring,      # Spring for Freestyle
    "Breaststroke": cm.autumn,   # Autumn for Breaststroke
    "Butterfly": cm.winter,      # Winter for Butterfly
    "Backstroke": cm.cool,       # Cool for Backstroke (can modify as needed)
}


# Function to adjust lightness of a color
def adjust_lightness(color, amount=1.0):
    # Convert RGB to HSV
    rgb = mcolors.to_rgb(color)
    h, s, v = mcolors.rgb_to_hsv(rgb)
    # Adjust value (lightness)
    v = max(0, min(1, v * amount))  # Ensure value stays between 0 and 1
    # Convert back to RGB
    return mcolors.hsv_to_rgb((h, s, v))

mylabels = list(data.keys())

# Base lightness factor
base_lightness = 1.0
lightness_step = 0.1  # Change this value to adjust the lightness increment

lightness_factors = [1,1,1,1]

# Assign colors based on the event type and reset lightness for each stroke
for label in mylabels:
    # Define a base lightness factor for the current stroke
    if "Freestyle" in label:
        base_color = color_maps["Freestyle"](0.5)  # Midpoint of spring colormap
        lightness_factors[0] -= 0.1  # Lightness for Freestyle
        lightness_factor = lightness_factors[0]
    elif "Breaststroke" in label:
        base_color = color_maps["Breaststroke"](0.5)  # Midpoint of autumn colormap
        lightness_factors[1] -= 0.1  # Lightness for Breaststroke
        lightness_factor = lightness_factors[1]
    elif "Butterfly" in label:
        base_color = color_maps["Butterfly"](0.5)  # Midpoint of winter colormap
        lightness_factors[2] -= 0.1  # Lightness for Butterfly
        lightness_factor = lightness_factors[2]
    else:
        base_color = color_maps["Backstroke"](0.5)  # Midpoint of cool colormap
        lightness_factors[3] -= 0.1 # Lightness for Backstroke
        lightness_factor = lightness_factors[3]

    # Adjust lightness for the base color
    adjusted_color = adjust_lightness(base_color, lightness_factor)
    colors.append(adjusted_color)


'''
# Assign shades based on the event type
for index, label in enumerate(mylabels):
    if "Freestyle" in label:
        base_color = "red"  # Base color for freestyle
        lightness_factor = 2#1 - (index * 0.2 / len(data))  # Decrease lightness
        colors.append(adjust_lightness(base_color, lightness_factor))
    elif "Breaststroke" in label:
        base_color = "blue"  # Base color for breaststroke
        lightness_factor = 2#1 - (index * 0.2 / len(data))  # Decrease lightness
        colors.append(adjust_lightness(base_color, lightness_factor))
    elif "Butterfly" in label:
        base_color = "green"  # Base color for butterfly
        lightness_factor = 2#1 - (index * 0.2 / len(data))  # Decrease lightness
        colors.append(adjust_lightness(base_color, lightness_factor))
    else:
        base_color = "orange"  # Base color for backstroke
        lightness_factor = 2#1 - (index * 0.2 / len(data))  # Decrease lightness
        colors.append(adjust_lightness(base_color, lightness_factor))
'''

'''
# Assign shades based on the event type
for label in mylabels:
    if "Freestyle" in label:
        colors.append(cm.Wistia(0.5 + 0.5 * (mylabels.index(label) % 2)))  # Gradation for freestyle
    elif "Breaststroke" in label:
        colors.append(cm.Blues(0.5 + 0.5 * (mylabels.index(label) % 2)))  # Gradation for breaststroke
    elif "Butterfly" in label:
        colors.append(cm.Greens(0.5 + 0.5 * (mylabels.index(label) % 2)))  # Gradation for butterfly
    else:
        colors.append(cm.Reds(0.5 + 0.5 * (mylabels.index(label) % 2)))  # Gradation for backstroke
'''

# Create pie chart
plt.pie(data.values(), labels=mylabels, colors=colors, autopct='%1.1f%%')

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')  
plt.title("Open Womens LC 200m IM Top 10 2000-2025 - each unique swimmer's top three non-medley events by FINA points")
plt.legend(title = "Events:")
plt.text(0, -1.25, 
        "The plot shows data taken from the British Swimming Event Rankings of the top three non-IM events for the best (top 10) 200 IM swimmers each year from 2000-2024.\n"
        f"Each slice represents how frequently each event came up in the total count, where we had: {int(total_swims_in_database/3)} unique swimmers, so with 3 top events picked per swimmer, there are:{total_swims_in_database} swims."
            ,
         fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
plt.show()