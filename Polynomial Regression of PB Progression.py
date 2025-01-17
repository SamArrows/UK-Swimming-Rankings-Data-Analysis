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
from matplotlib.ticker import FuncFormatter


session = requests.Session() # Create a session object
# Configure connection pool size
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
session.mount('http://', adapter)
session.mount('https://', adapter)

def get_birthyear(ID: int):
    url = f"https://www.swimmingresults.org/biogs/biogs_details.php?tiref=921675"
    pg = session.get(url)
    soup = BeautifulSoup(pg.content, "html.parser")
    results = soup.find("table") # this selects first table in the document
    year = results.find_all("tr")[3].find_all("td")[1].text
    return int(year)

def get_pb_progression(ID: int, event: int, year_of_birth: int=0, course: str="L"):
    '''
    Starting at age 12, gets a list of pbs and the date they were set
    
    If a year of birth isn't provided, the scraper finds it however it is faster to just pass it in as a parameter.
    '''
    birthyear = 0
    if year_of_birth == 0:
        # find the year_of_birth
        birthyear = get_birthyear(ID)
    else:
        birthyear = year_of_birth
    URL = f"https://www.swimmingresults.org/individualbest/personal_best_time_date.php?back=biogs&tiref={str(ID)}&mode=A&tstroke={str(event)}&tcourse={course}"
    page = session.get(URL) 
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("table")[1]
    table = results.find("tbody")
    
    rows = table.find_all("tr")[1:] #exclude the first row since it is the headers of the table
    # we need to sort all swims in date order - easiest to convert our list of tr objects into pairs where 0 in the pair is the swim time in seconds as a float and 1 is the date
    # we already have a function for this: extract_time_and_date_from_row
    r = list(map(lambda row: extract_time_and_date_from_row(row, extract_year_only=False), rows))
    r.reverse()
    # include only swims which were PBs - first, ensure only swims from when a swimmer was 12 by the end of the year are included
    r = list(filter(lambda pair: int(pair[1][6:]) >= (birthyear%100)+12, r))
    # now we need to check if an adjacent time is less than the previous - if not, remove it
    
    pbs = [r[0]]
    for i in range(1, len(r)):
        if r[i][0] < pbs[-1][0]:
            pbs.append(r[i])
    return pbs

def convert_pb_to_percentage_progress(pb_dates):
    '''
    Given the output from get_pb_progression [(pb, date), (pb, date), ... etc.] ,
    this function converts to percentage difference since last pb 
    '''
    progression = []
    #print(pb_dates)

    # Using zip to pair each element with the next
    for i, j in zip(pb_dates, pb_dates[1:]):
        progression.append([100 * ((i[0] / j[0]) - 1), j[1]])
    return progression

def convert_to_datetime(date_str):
    # Split the date string into day, month, and year
    day, month, year = map(int, date_str.split('/'))
    
    # Adjust the year to be in the range 2000-2099
    if year < 50:  # Assuming that '00' to '49' represents 2000 to 2049
        year += 2000
    else:          # '50' to '99' represents 2050 to 2099
        year += 1900
    # Return the datetime object
    return datetime(year, month, day)

def convert_date_to_time_since_12(progression_dates, year_at_12):
    '''
    The pipeline should feed 
    = get_pb_progression 
    of pbs with dates into 
    = convert_pb_to_percentage_progress
    of percentage changes between pbs with dates into this function. 
    The function aims to convert the dates to time since the start of the year when an athlete enters the 12 age category for AGE AT END OF YEAR.
    i.e. for an athlete who is born 2003, in 2015, a pb on 1 July would be 
    '''
    year12 = datetime(year_at_12, 1, 1)
    for i in progression_dates:
        i[1] = (convert_to_datetime(i[1]) - year12).days
    return progression_dates

def pb_progression_pipeline(ID: int, event: int, birthyear: int = 0):
    yob = birthyear
    if(yob==0):
        yob = get_birthyear(ID)
    return convert_date_to_time_since_12(convert_pb_to_percentage_progress(get_pb_progression(ID=ID, event=event, year_of_birth=yob)), yob+12)


menX = np.array([])
menY = np.array([])

womenX = np.array([])
womenY = np.array([])

array_lock = threading.Lock()

class PBProgressionScrapeThread(threading.Thread):
    def __init__(self, IDs: list, event: int = 3, sex: int = 0, *args, **kwargs):
        '''
        === Provided a list of IDs, the thread will find pb progressions and corresponding days since turning 12 for an athlete
        '''
        super(PBProgressionScrapeThread, self).__init__()
        self.IDs = IDs
        self.event = event
        self.sex = sex
        self.args = args
        self.kwargs = kwargs

    def run(self):
        global menX, menY, womenX, womenY
        # we need to put all the days into a numpy array for x values and the percentage changes in pbs into a numpy array for y values
        x = []
        y = []
        for ID in self.IDs:
            prog = list(zip(*pb_progression_pipeline(ID, self.event)))
            x += list(prog[1])
            y += list(prog[0])
        with array_lock:
            if self.sex == 0:
                menX = np.append(x, menX)
                menY = np.append(y, menY)
            else:
                womenX = np.append(x, womenX)
                womenY = np.append(y, womenY)

event = 3
# set nationality to british as this increases the likelihood that an athlete has been training in the country their whole life with records to show for it
mensIDs = list(map(extract_ID_from_row, get_rows(set_parameters(stroke=str(event), nationality='A', records_to_view='40'))))
womensIDs = list(map(extract_ID_from_row, get_rows(set_parameters(stroke=str(event), nationality='A', records_to_view='40', sex='F'))))


t1 = PBProgressionScrapeThread(mensIDs[:len(mensIDs)//2], event=event)
t2 = PBProgressionScrapeThread(mensIDs[len(mensIDs)//2:], event=event)
t3 = PBProgressionScrapeThread(womensIDs[:len(womensIDs)//2], event=event, sex=1)
t4 = PBProgressionScrapeThread(womensIDs[len(womensIDs)//2:], event=event, sex=1)

threads = [t1, t2, t3, t4]
for t in threads:
    t.start()

for t in threads:
    t.join()

print(menX)
print(menY)
print(womenX)
print(womenY)

menvals = np.column_stack((menX, menY))
womenvals = np.column_stack((womenX, womenY))

np.savetxt('Men Vals for 200m FREE LINEAR REGRESSION MODEL.csv', menvals, delimiter=',', fmt='%.10f')
np.savetxt('Women Vals for 200m FREE LINEAR REGRESSION MODEL.csv', womenvals, delimiter=',', fmt='%.10f')

def calculate_bic(n, k, mse):
    return n * np.log(mse) + k * np.log(n)

degrees = range(1, 6)  # Testing polynomial degrees from 1 to 5
bics = []

''' Calculate BICS for men '''
# before fitting a model, we need that x is 2d and y is 1d
menX = menX.reshape(-1, 1)
for d in degrees:
    # Generate polynomial features
    poly = PolynomialFeatures(degree=d)
    mX_poly = poly.fit_transform(menX)

    # Fit the model
    model = LinearRegression()
    model.fit(mX_poly, menY)

    # Predict and calculate MSE
    my_pred = model.predict(mX_poly)
    mse = mean_squared_error(menY, my_pred)

    # Calculate BIC
    k = d + 1  # Number of parameters (degree + intercept)
    bic = calculate_bic(len(menY), k, mse)
    bics.append(bic)
# Display BIC values
for d, bic in zip(degrees, bics):
    print(f'Degree: {d}, BIC: {bic}')
''' End of Calculating BICS for men '''
bics = []
degrees = range(1,6)
''' Calculate BICS for women '''
# before fitting a model, we need that x is 2d and y is 1d
womenX = womenX.reshape(-1, 1)
for d in degrees:
    # Generate polynomial features
    poly = PolynomialFeatures(degree=d)
    wX_poly = poly.fit_transform(womenX)

    # Fit the model
    model = LinearRegression()
    model.fit(wX_poly, womenY)

    # Predict and calculate MSE
    wy_pred = model.predict(wX_poly)
    mse = mean_squared_error(womenY, wy_pred)

    # Calculate BIC
    k = d + 1  # Number of parameters (degree + intercept)
    bic = calculate_bic(len(womenY), k, mse)
    bics.append(bic)
# Display BIC values
for d, bic in zip(degrees, bics):
    print(f'Degree: {d}, BIC: {bic}')
''' End of Calculating BICS for women '''

'''
OUTPUT OF BIC CALCULATIONS FOR MEN ARE AS FOLLOWS:
Degree: 1, BIC: 483.53968289481224
Degree: 2, BIC: 489.8440270488691
Degree: 3, BIC: 495.7551585149811
Degree: 4, BIC: 497.66401671867595
Degree: 5, BIC: 503.9650971516672

OUTPUT OF BIC CALCULATIONS FOR WOMEN ARE AS FOLLOWS:
Degree: 1, BIC: 807.012753624151
Degree: 2, BIC: 812.5224588899924
Degree: 3, BIC: 818.3958862502363
Degree: 4, BIC: 821.4354529667614
Degree: 5, BIC: 822.2687697480876

Degree 1 yields the smallest BIC in both cases so is deemed the best model
'''


# Polynomial Regression for men
degree = 1  
poly = PolynomialFeatures(degree=degree)
menX_poly = poly.fit_transform(menX)
model = LinearRegression()
model.fit(menX_poly, menY)
# Generate a range of values for menX for plotting
x_range = np.linspace(min(menX), max(menX), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)  # Transform to polynomial features
# Predict corresponding percentages
y_range_pred = model.predict(x_range_poly)

# Polynomial Regression for women
degree = 1  
poly = PolynomialFeatures(degree=degree)
womenX_poly = poly.fit_transform(womenX)
model = LinearRegression()
model.fit(womenX_poly, womenY)
# Generate a range of values for menX for plotting
wx_range = np.linspace(min(womenX), max(womenX), 100).reshape(-1, 1)
wx_range_poly = poly.transform(wx_range)  # Transform to polynomial features
# Predict corresponding percentages
wy_range_pred = model.predict(wx_range_poly)




transparency_of_points = 0.2

plt.scatter(menX, menY, color='#0000FF', marker='o', alpha=transparency_of_points, label='Men 200m FREE') 
plt.plot(x_range, y_range_pred, color='blue', label='Polynomial Regression Model for men')

plt.scatter(womenX, womenY, color='#FF0000', marker='o', alpha=transparency_of_points, label='Women 200m FREE') 
plt.plot(wx_range, wy_range_pred, color='red', label='Polynomial Regression Model for women')

max_days = np.max(np.concatenate((x_range, wx_range)))

# Function to convert days to years (assuming 365 days per year)
def days_to_years(x, pos):
    return f"{12 + (x / 365.0)}"  # Convert days to years, rounded to whole number

# Set the x-axis formatter
formatter = FuncFormatter(days_to_years)
plt.gca().xaxis.set_major_formatter(formatter)

# Set x-ticks to only show whole years
years = np.arange(12, 12 + (max_days // 365) + 1)  # Generate years based on full years in max_days
ticks = np.arange(0, max_days + 1, 365)  # Ensure ticks go up to max_days
plt.xticks(ticks=ticks, labels=years)  # Set ticks and corresponding labels

plt.ylim(0,11)
plt.title('Scatter Plot of PB progressions of elite performers in 200m FREE, men vs women')
plt.xlabel('Age of athlete during year')
plt.ylabel('Percentage improvement from last PB (%)')
plt.grid()
plt.legend()
plt.show()






#pbs = get_pb_progression(921675, 11)
#prog = convert_pb_to_percentage_progress(pbs)
#convert_date_to_time_since_12(prog, 2015)

#print(list(zip(*pb_progression_pipeline(921675, 11, 2003))))