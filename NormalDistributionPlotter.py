'''
This file contains plotting functionality for normal distributions so that one can compare events to each other across gender, age, years, and other events
'''
from Scraper import *
import requests
from bs4 import BeautifulSoup
import re 
from datetime import datetime
from enum import Enum
import numpy as np 
import threading
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import statistics 
from scipy.stats import norm 

''' 
For our first plot, let us make a normal distribution of the top 50 mens LC 100m freestyle performers in 2024
'''

def plot_norm_for_one_gender(sex: str = 'M', stroke: int = 12, year: str='2024', records: int = 50, age: int = 0, title: str = "Normal distribution"):
    if age == 0:
        query = set_parameters(sex=sex, stroke=stroke, year=year, records_to_view=records, nationality='A')
    else:
        query = set_parameters(sex=sex, stroke=stroke, year=year, records_to_view=records, age=age, nationality='A')
    x_axis = list(map(convert_timestring_to_secs, map(extract_time_from_rankings_query_row, get_rows(query))))
    mean = statistics.mean(x_axis) 
    sd = statistics.stdev(x_axis) 
    y_axis = norm.pdf(x_axis, mean, sd)

    #ran = np.arange(115, 130, 0.01)

    #pdf_men = norm.pdf(ran, mean, sd)

    plt.plot(x_axis, y_axis, label='Normal Distribution')

    #plt.plot(ran, pdf_men) 
    # Annotate the mean, minimum, and maximum values
    plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {convert_secs_to_timestring(mean)}')  # Vertical line for mean
    plt.text(mean, max(y_axis) * 0.5, f'Mean: {convert_secs_to_timestring(mean)}', color='red', fontsize=10, ha='center')

    # Get the minimum and maximum values from x_axis
    min_value = np.min(x_axis)
    max_value = np.max(x_axis)

    print(min_value)
    print(convert_secs_to_timestring(min_value))

    # Annotate the largest and smallest values
    plt.text(min_value, max(y_axis) * 0.25, f'Min: {convert_secs_to_timestring(min_value)}', color='blue', fontsize=10, ha='center')
    plt.text(max_value, max(y_axis) * 0.25, f'Max: {convert_secs_to_timestring(max_value)}', color='blue', fontsize=10, ha='center')

    # Adding labels and title
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Probability Density Function (PDF)')
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()
    

plot_norm_for_one_gender(title='Normal Distribution with Mean, Min, and Max for top 50 open aged male 200m Butterfly performers 2024')
