# https://realpython.com/beautiful-soup-web-scraper-python/

''' Outline of process
-for each year, find the top ten in an event; using their birth year
-then get their seasonal bests for each calendar year from age 12-20
    -currently, rankings isn't using a robot captcha so we can search biogs directly which is twice as fast as having to check rankings each year
-plot times progress from 12-20 for all athletes
-apply linear regression to find trends

-do for all events and see if the % of change from calendar year for events varies drastically, or compare to women progressions for the same event
-first off, we will use 100 Freestyle Mens

=========================================================================
PERFORMANCE COMPARISONS BETWEEN VERSIONS:
-done using Mens 200 FREE 2008-2023 ages 12-20 seasonal bests
    136.830219 = functional biog searcher
    215.720565 = search based on rankings for each year

    Can be sped up when we don't bother removing the axis of swimmer IDs in the numpy array, however
        the main point is that the functional version is almost twice as fast due to less searching performed;
        it is also more robust as there are no issues with duplicate and erroneous data 

THE ID AXIS IS NO LONGER ADDED, FURTHER OPTIMIZING THE PROGRAM
=========================================================================
LATEST VERSION USES CONNECTION POOLING AND A SESSION FOR HANDLING REQUESTS, WHICH LOWERS THE LIKELIHOOD OF CONNECTION TIMEOUTS AND IMPROVES PERFORMANCE
Example using Mens 200 FLY 2008-2023 ages 12-20 seasonal bests:
    -before connection pooling = 116.943366 seconds
    -after connection pooling = 108.801828 seconds


'''

import requests
from bs4 import BeautifulSoup

import re # for regular expressions when fetching ASA ID numbers from tables

from datetime import datetime
from enum import Enum

import numpy as np

class Events(Enum):
    '''
    1 - 6 = Free
    7 - 9 = Breast
    10 - 12 = Fly
    13 - 15 = Back
    16 - 200 IM
    17 - 400 IM

    == Subclassing enum for an events class allows for easy translation between rankings codes for events and event names
    == Freedom to define methods which act on the event names/codes, such as translating name with value easily
    '''
    FREE_50 = 1
    FREE_100 = 2
    FREE_200 = 3
    FREE_400 = 4
    FREE_800 = 5
    FREE_1500 = 6
    BREAST_50 = 7
    BREAST_100 = 8
    BREAST_200 = 9
    FLY_50 = 10
    FLY_100 = 11
    FLY_200 = 12
    BACK_50 = 13
    BACK_100 = 14
    BACK_200 = 15
    IM_200 = 16
    IM_400 = 17
    IM_100 = 18

def get_enum_member_name(enum_cls, value):
    for member in enum_cls:
        if member.value == value:
            return member.name
    return None

def convert_biog_event_text_to_enum_format(event_on_biog:str):
    '''
    Converts an event name from biog to the format for the enum
    '''
    comps = event_on_biog.upper().split(" ")
    if comps[1] == "INDIVIDUAL":
        word = "IM"
    elif comps[1] == "BACKSTROKE":
        word = "BACK"
    elif comps[1] == "BUTTERFLY":
        word = "FLY"
    elif comps[1] == "BREASTSTROKE":
        word = "BREAST"
    else:
        word = "FREE"
    return word + "_" + comps[0]

session = requests.Session() # Create a session object

# Configure connection pool size
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
session.mount('http://', adapter)
session.mount('https://', adapter)

def set_parameters(pool_type='L', stroke='1', sex='M', year='2022', age='OP', ageAt='D', start_num='1', records_to_view='25', level='N', nationality='X', region='P', county='XXXX', club='XXXX'):
    query = {
        'pool type' : pool_type,
        'stroke' : stroke,
        'sex' : sex,
        'year' : year,
        'age' : age,
        'age at' : ageAt,
        'start number' : start_num,
        'records to view' : records_to_view, 
        'level' : level,
        'nationality' : nationality,
        'region' : region,
        'county': county,
        'club' : club
    }
    return query

def convert_timestring_to_secs(time):
    time = time.strip() #removes whitespace
    try:
        time = float(time)
    except:
        # time is greater than 59.99 seconds so needs to be parsed
        # Split the time string
        minutes, seconds_and_hundredths = time.split(':')
        seconds, hundredths = seconds_and_hundredths.split('.')

        # Calculate total time in seconds
        total_seconds = int(minutes) * 60 + int(seconds) + int(hundredths) / 100
        time = float(total_seconds)
    return time

def extract_yob_from_row(row):
    '''
    Extracts a swimmer's year of birth from a swimmer row 
    '''
    try:
        return row.findChildren()[4].text
    except:
        return None

def get_YOBs(rows):
    '''
    Gets YOB for swimmers found in the query in the input
    '''
    return list(filter(None, map(extract_yob_from_row, rows)))

def extract_ID_from_row(row):
    try:
        hyperlink = str(row.findChildren()[1].findChildren()[0].attrs.get('href'))
        pattern = r'\?tiref=(.*?)&'  # Define your regular expression pattern
        regex = re.compile(pattern)   # Compile the regular expression
        match = regex.search(hyperlink)   # Search for the substring in the long string
        if match is not None:
            matched_string = match.group()
            numbers_only = re.findall(r'\d+', matched_string)
            result = ''.join(numbers_only)
            return result
    except:
        # sometimes a member is removed from rankings or hidden but their records remain so we need to catch this out when a search for child nodes fails
        return None

def extract_attributes_from_row(row, ID=True, YOB=True):
    '''
    Specify attributes to extract from a swimmer row - by default, id and year of birth are extracted
    '''
    swimmer = []
    if(ID):
        swimmer.append(extract_ID_from_row(row))
    if(YOB):
        swimmer.append(extract_yob_from_row(row))
    if(len(swimmer) == 1):
        return swimmer[0]
    else:
        return swimmer

def get_rows(query=set_parameters()):
    '''
    Gets the data from a query in the form of table rows so that other functions can be applied to extract certain features from the rows, such as IDs and birthyears
    '''
    age_at = ""
    if(query['age'] != 'OP'):
        age_at = f"AgeAt={query['age at']}"
    URL = f"https://www.swimmingresults.org/eventrankings/eventrankings.php?Pool={query['pool type']}&Stroke={query['stroke']}&Sex={query['sex']}&TargetYear={query['year']}&AgeGroup={query['age']}&{age_at}&StartNumber={query['start number']}&RecordsToView={query['records to view']}&Level={query['level']}&TargetNationality={query['nationality']}&TargetRegion={query['region']}&TargetCounty={query['county']}&TargetClub={query['club']}"
    #print("======================\n" + URL + "\n====================")
    #page = requests.get(URL)           use session.get to make use of connection pooling
    page = session.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    # there are different parsers that can be used, with html.parser being the most common - the link below discusses more parsers available
    # https://www.crummy.com/software/BeautifulSoup/bs4/doc/#differences-between-parsers
    results = soup.find(id="rankTable")
    table = results.find("tbody")
    rows = table.find_all("tr")[1:] #exclude the first row since it is the headers of the table
    return rows

def get_IDs(rows):
    '''
    Gets ASA ID numbers for swimmers found in the query in the input
    '''
    return list(filter(None, map(extract_ID_from_row, rows)))

def get_table_from_biog(ID="921675", event="12", course="L"):
    '''
    Gets a table of times in time order for a swimmer based on their ASA ID, the event and whether it was long or short course
    '''
    try:
        URL = f"https://www.swimmingresults.org/individualbest/personal_best_time_date.php?back=biogs&tiref={ID}&mode=A&tstroke={event}&tcourse={course}"
        page = session.get(URL) 
        soup = BeautifulSoup(page.content, "html.parser")

        results = soup.find(id="rankTable")
        table = results.find("tbody")
        rows = table.find_all("tr")[1:] #exclude the first row since it is the headers of the table
        return rows
    except Exception as e:
        #print(str(e))
        return None

def get_fina_points_from_biog_row(row):
    '''
    Returns the FINA points from a row in the biog
    '''
    return int(row.findChildren(name="td")[3].text)

def get_event_name_from_biog_row(row):
    '''
    Returns the name of an event from a row in the biog
    '''
    return row.findChild().text

def get_best_events_by_fina_points_for_set_year(ID="921675", course="L", year_to_search="16", total_events_to_scrape=3, exclude_medley: bool = False):
    '''
    Scrapes a biog page for the n best events by fina point scores for a given year, taking into account short course or long course or both courses
    - sequentially fetch each event history table based on course
    - map getting seasonal best fina points onto each event history table and store each event (key) with its fina point score (value) in some sort of dictionary or map
    - fetch the top n events based on this metric and return as list

    Exclude medley allows the user to exclude IM events when performing the search; this is so that a user could find a medley swimmer's best stroke or best distance
    and compile stats for medley performers.
    '''
    event_tables = [[],[]]  #the first list corresponds to event codes - zip the event codes with their corresponding fina points later
    if course == "L" or course == "S":
        if exclude_medley:
            upper_lim = 16
        else:
            upper_lim = 18 if course == "L" else 19
        for i in range(1, upper_lim):
            event_tables[1].append(get_table_from_biog(ID, event=i, course=course))  
            event_tables[0].append(get_enum_member_name(Events, i))
        scores = []
        for table in event_tables[1]:
            scores.append(get_seasonal_best_fina_pts_from_rows(table, year_to_search))
        
        return list(sorted(zip(event_tables[0], scores), key=lambda x: x[1], reverse=True)[:total_events_to_scrape]) # Select the top three scores by default
    else:
        #TODO
        return None

def is_medley(event_row_text: str):
    '''
    Determine whether a row in the biog is representing a medley event or not
    '''
    if "Medley" in event_row_text.split(" "):
        return True
    else:
        return False

def get_top_events_by_fina_points_from_biog(ID="921675", course="L", total_events_to_scrape=3, exclude_medley: bool = False):
    '''
    Using an ID, scrapes the top three events from either the long course table, short course table, or both, according to their FINA point scores
    '''
    URL = f"https://www.swimmingresults.org/individualbest/personal_best.php?tiref={ID}&mode=A&back=biogs"
    page = session.get(URL) 
    soup = BeautifulSoup(page.content, "html.parser")
    tables = soup.find_all('tbody')
    if course == 'L':
        rows = tables[0].find_all("tr")[1:]
    elif course == 'S':
        rows = tables[1].find_all("tr")[1:]
    else:
        rows = tables[0].find_all("tr")[1:] + tables[1].find_all("tr")[1:]
    
    if(exclude_medley):
        # filter the medley rows out from the rows --> 
        rows = list(filter(lambda x: not is_medley(x.find_all("td")[0].text), rows))

    numerical_values = list(map(get_fina_points_from_biog_row, rows)) # Extract numerical values from each tr element
    sorted_tr_elements = sorted(rows, key=lambda tr: get_fina_points_from_biog_row(tr), reverse=True) # Sort tr elements based on numerical values in descending order
    top_three_tr_elements = sorted_tr_elements[:total_events_to_scrape] # Select the top three tr elements with the highest values by default

    return list(map(get_event_name_from_biog_row, top_three_tr_elements))

#print(get_top_events_by_fina_points_from_biog("149309", exclude_medley=False))

def extract_time_and_date_from_row(row, extract_year_only=True):
    '''
    The time is contained in the first child of any given row - index 0; the date is fourth child, so index 3
    extract_year_only means only the year the swim was performed will be taken, which reduces processing down the line for finding best times in calendar years
    '''
    children = row.findChildren()
    time = convert_timestring_to_secs(children[0].string)
    if(len(children[0].findChildren()) == 1):  #some races have split times put up which are a hyperlink, hence adding another child to the row; this needs to be accounted for
        date = children[4].string
    else:
        date = children[3].string
    if(extract_year_only):
        date = date[-2:]
    return (time, date)

def get_seasonal_best_from_rows(rows_to_search=get_table_from_biog(), year_to_search='16'):
    '''
    Provided a set of rows from a swimmer's biog for an event, this function will calculate their best time in a given year
    '''
    try:
        return next(filter(lambda x : x[1] == year_to_search, map(extract_time_and_date_from_row, rows_to_search)))[0]   #since times are added in speed order, the first element will be the fastest - we then extract the time from this (time, date) pair
    except:
        return 0  

def get_seasonal_best_fina_pts_from_rows(rows_to_search=get_table_from_biog(), year_to_search='16'):
    '''
    Provided a set of rows from a swimmer's biog for an event, this function will find their best fina point score in a given year
    - since times are added in speed order, the faster swims and thus the better scored swims will be the first elements, hence one must extract the first element once the times have been filtered by year
    '''
    try:
        return next(filter(lambda x : x[1] == year_to_search, map(extract_fina_points_and_date_from_row, rows_to_search)))[0] 
    except Exception as e:
        #print(str(e))
        return 0  
    
def extract_fina_points_and_date_from_row(row):
    '''
    Extracts fina points and date from a row and returns it as a tuple (x,y)
    '''
    return (int((row.findChildren('td')[1]).string), (row.findChildren('td')[3]).string[-2:])

def extract_fina_points_from_event_history_row(row):
    '''
    Provided a table of times for an event for a swimmer, this function will be able to be mapped to it and extract the fina point score for a row
    '''
    return int((row.findChildren('td')[1]).string)
    

def year_when(year, age=12):
    '''
    Using the two-digit year of birth for a swimmer, this function calculates the equivalent two digits for the year when they were a certain age, which is defaulted at 12
    '''
    try:
        year = str((int(year) + age) % 100)
        if(len(year) == 1):
            return "0" + year
        else:
            return year
    except:
        return

def meet_name_contains_words(words: list, meet_name: str):
    '''
    Checks if a meet name contains a set of words, such as Swim England or British Summer
    '''
    meet_name = meet_name.split(" ")
    for name in words:
        if not(name in meet_name):
            return False
    return True

def get_results_page_from_british_summer_champs(year: int, page_no: int = 14):
    '''
    Returns the page number and meet code as part of a query for the page for British Summer Championships in a given year; 
    this can then be fed into a URL structure to get the results by gender, event or otherwise
    ===> https://www.swimmingresults.org/showmeetsbyevent/index.php + query
    where query could be equal to ==> ?targetyear=2015&masters=0&pgm=14&meetcode=19805
    NOTE: THE MEET CODE IS NOT THE LICENCE NUMBER, IT IS THE HYPERLINK STORED IN THE LICENCE NUMBER

    - year ==> year to search
    - page_no ==> when checking for meet results in a calendar year, all meets are organised into pages.
        Need to be able to search a page and then check a different one if it doesn't contain the desired meet
    '''
    parent = f"https://www.swimmingresults.org/showmeetsbyevent/index.php?targetyear={year}&masters=0&page={page_no}"
    page = session.get(parent) 
    soup = BeautifulSoup(page.content, "html.parser")
    tables = soup.find('tbody')
    rows = tables.find_all("tr")[1:]
    page_num = page_no
    meets_found = filter(lambda x: 
                                    x.find_all("td")[2].text == "National" 
                                    and meet_name_contains_words(["British", "Summer"], x.find_all("td")[0].text), 
                                    rows)
    try:
        meet = next(meets_found)
        return meet.find_all("td")[5].find('a')['href']
    except StopIteration:
        '''
        Look on a different page for the british summer champs of that year
        - need to check if the date range is before or after July as British Summers is always in July
        - date string is stored in index 3 of the td children
        - if the latest month is June (06) or earlier, then check the next month (+1 ==> July)
        - if the latest is July and we don't have the date, check the next page to see if it contains it (+1)
        - if the latest month is post-July (08+) then definitely check previous page (-1)
        - need to sort the rows by td index 3
        '''
        dates = sorted(list(map(int, map(lambda x: x.find_all("td")[3].text.split("/")[1], rows))), reverse=True)
        if(dates[0] < 7):
            page_num -= 1
        elif(dates[0] > 7):
            page_num += 1
        else:
            # page is in July so we need to check the page before and after - if we go two pages ahead, we can then cycle back
            page_num -= 2
        return get_results_page_from_british_summer_champs(year, page_num)



def get_200IM_performers_other_best_events_for_british_summers_by_event(query: str, event_code: int = 1):
    '''
    TODO: FINISH FUNCTIONALITY FOR THIS SO THAT WE CAN MAKE PIE CHARTS SHOWING PERCENTAGES
    Provided the meet code, page number and year as a query, we can scrape a meet (AKA British Summers) for information, such as:
    - finding finalists and medalists in an event, which can be used to
        - see what percentage of top-end IM performers have which strokes as their strongest

    Rough outline of algorithm:
    - set event to 200 IM
    - sort by the round column such that the 'F' for final records are ordered first
    - search by age group, i.e. finalists in age groups that are 16 or younger, such as 12/13, 13/14, 14/15, 15/16, etc. 
            --> age groups change every few years, hence having a blanket standard of 16/U who are top of their respective age band for that year
            --> age groups also vary by gender, but 17+ should cover mens and womens fairly well, i.e. women used to be 17+, then 18+, sometimes 19+
                    whereas men were 17/18 and 19+, with now a 17, 18, 19+
    - sort into a group of ASA numbers representing just finalists for those ages, and those who were also medalists --> maybe use dictionary
    - once this has been done, the dictionary can be passed into other functions or processed further within this one, with the aim being to 
        assign each swimmer their next best events, which can be rated using distance and stroke, i.e. on a pie chart, sprint events could be darker in colour than distance
        while variation in stroke could be shown using different colours, i.e freestyle events could be shown using gradients of red as opposed to fly using gradients of yellow
    - find best stroke excluding IM for that season using short and long course rankings and initially fina points as they are easy to scrape
    - plot percentages on a pie chart for age and gender of the makeup for best stroke, i.e. 16/U mens might have 60% best stroke being breast, 30% being back, etc.
        as opposed to girls 17+ maybe having high portions of free and fly
    - maybe also plot best race distance as well, such as being more 50 based, 100 based, 200 based, 400 based, or 800/1500
    '''
    base_url = "https://www.swimmingresults.org/showmeetsbyevent/index.php" + query
    return


def test_run_fina_pts():
    '''
    You can run these basic test functions to check out functionality ad hoc
    '''
    for i in range(10, 24):
        print(get_best_events_by_fina_points_for_set_year(ID="149309", course="L", year_to_search=str(i)))

def test_run_scrape_for_csv():
    '''
    You can run these basic test functions to check out functionality ad hoc
    '''
    start_time = datetime.now() #used to time how long the scraping takes - compare different algorithms to find faster solutions

    event = '4' # 400 FREE, see Events class for full list of pairs of events and their corresponding codes on rankings
    gender= 'F' # M = male, F = female

    filenamingDictionary = {
        "M" : 'Mens',
        "F" : 'Womens'
    }   # Used to build the file name

    #database = np.array([["ID", 12, 13, 14, 15, 16, 17, 18, 19, 20]])      # use this for including the ID in the table
    database = np.array([[12, 13, 14, 15, 16, 17, 18, 19, 20]])     # this line won't include ASA Rankings membership ID number in the table

    IDs = dict()

    for i in range(2008, 2023):
        for j in map(extract_attributes_from_row, get_rows(query=set_parameters(stroke=event, year=str(i), records_to_view=10, sex=gender))):
            IDs.update({ j[0] : j[1] })

    for ID in IDs.keys():
        record = []
        for i in range(12, 21):
            record.append(get_seasonal_best_from_rows(get_table_from_biog(ID, event), year_when(IDs[ID], i)))
        database = np.vstack((database, np.array(record)))


    #database = np.delete(database, 0, axis=1).astype(np.float)  #deletes the column of IDs - only relevant if IDs were added and later on, were not wanted


    end_time = datetime.now()
    execution_time = end_time - start_time  #used to time the code

    print(f"The code took {execution_time.total_seconds()}")


    file_path = f'CSV/Top 10 Performers {filenamingDictionary[gender]} {get_enum_member_name(Events, int(event))} 2008-2023.csv'

    # Write the numpy array to a CSV file
    np.savetxt(file_path, database, delimiter=',', fmt='%.2f')