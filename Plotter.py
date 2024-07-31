
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



filename = "Top 10 Performers Mens 200 Fly 2008-2023"

filename2 = "Top 10 Performers Mens 200 Free 2008-2023"

database = np.genfromtxt(filename + ".csv", delimiter=",")
databaseT = database.T
ages = databaseT[:, 0].astype(int)
athlete_times = databaseT[:, 1:]

women = np.genfromtxt(filename2 + ".csv", delimiter=",")


def percentage_changes(array):
    percentages = []
    for i in range(0, len(array)):
        if i != len(array)-1:
            if array[i] != 0 and array[i+1] != 0:
                percentages.append(100 * ((array[i+1] / array[i]) -1))
            else:
                percentages.append(100)
    return percentages

athlete_percentages = (np.apply_along_axis(percentage_changes, axis=1, arr=database)[1:]).T
women_percs = (np.apply_along_axis(percentage_changes, axis=1, arr=women)[1:]).T
#headers = np.array([f"{ages[i]}-{ages[i+1]}" for i in range(len(ages) - 1)])
head = np.array([13, 14, 15, 16, 17, 18, 19, 20])

def percent_scatter():
    plt.figure() 
    for i in range(athlete_percentages.shape[1]):
        perc = athlete_percentages[:, i]
        print(perc, "\n===============\n")
        #print(perc)
        mask = perc < 100 #create a mask for where a time wasn't found to prevent a plot being made for that specific point
        plt.scatter(head[mask], perc[mask])
    return plt

def add_labels(plot):
    plot.xlabel('Age groups for the percentage change being calculated from') #use 'Ages' for using actual ages
    plot.ylabel('Percentage changes in seasonal best times based on previous age group') # 'Times (seconds)'
    plot.title(filename + "\n Season's Best Progressions")
    plot.grid(True)
    return plot

#add_labels(percent_scatter()).show()

def plot_scatter():
    plt.figure() 
    for i in range(athlete_times.shape[1]):
        time = athlete_times[:, i]
        mask = time != 0 #create a mask for where a time wasn't found to prevent a plot being made for that specific point
        plt.scatter(ages[mask], time[mask])
    return plt

def plot_line_graph():
    plt.figure()
    for i in range(athlete_times.shape[1]):
        time = athlete_times[:, i]
        mask = time != 0 #create a mask for where a time wasn't found to prevent a plot being made for that specific point
        plt.plot(ages[mask], time[mask], marker='o', label=f'Athlete {i+1}')
    return plt

def plot_polynomial_regression():
    # Aggregate race times for each age
    unique_ages = np.unique(ages)
    aggregated_race_times = []
    for age in unique_ages:
        arr = athlete_times[ages == age]
        mean = np.mean(arr[arr != 0])
        aggregated_race_times.append(mean)
    #aggregated_race_times = [np.mean(athlete_times[ages == age]) for age in unique_ages]

    # Reshape the data
    X = unique_ages.reshape(-1, 1)
    y = aggregated_race_times

    # Fit polynomial features
    poly_features = PolynomialFeatures(degree=4)  # You can change the degree as needed
    X_poly = poly_features.fit_transform(X)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict the values
    y_pred = model.predict(X_poly)

    # Plot the aggregated data points
    for i in range(athlete_times.shape[1]):
        time = athlete_times[:, i]
        mask = time != 0 #create a mask for where a time wasn't found to prevent a plot being made for that specific point
        plt.scatter(ages[mask], time[mask], color='blue')
    plt.scatter(unique_ages, aggregated_race_times, color='red', label='Aggregated data')

    # Plot the polynomial regression line
    plt.plot(unique_ages, y_pred, color='green', label='Polynomial Regression')

    plt.xlabel('Age')
    plt.ylabel('Race Times')
    plt.title('Polynomial Regression on Aggregated Athlete Race Times - ' + filename)
    plt.legend()
    plt.show()

def plot_polynomial_regression_percentages():
    # Aggregate race times for each age
    unique_ages = np.unique(head)
    aggregated_percentages = []
    for age in head:
        arr = athlete_percentages[head == age]
        mean = np.mean(arr[arr != 100])
        aggregated_percentages.append(mean)
    #aggregated_race_times = [np.mean(athlete_times[ages == age]) for age in unique_ages]

    # Reshape the data
    X = unique_ages.reshape(-1, 1)
    y = aggregated_percentages

    # Fit polynomial features
    poly_features = PolynomialFeatures(degree=4)  # You can change the degree as needed
    X_poly = poly_features.fit_transform(X)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict the values
    y_pred = model.predict(X_poly)

    # Plot the aggregated data points
    for i in range(athlete_percentages.shape[1]):
        perc = athlete_percentages[:, i]
        mask = perc < 100 #create a mask for where a time wasn't found to prevent a plot being made for that specific point
        plt.scatter(head[mask], perc[mask], color='blue')
    plt.scatter(unique_ages, aggregated_percentages, color='red', label='Aggregated data')

    # Plot the polynomial regression line
    plt.plot(unique_ages, y_pred, color='green', label='Polynomial Regression')

    plt.xlabel('Age')
    plt.ylabel('Percentage changes')
    plt.title('Polynomial Regression on Aggregated Athlete Percentage Changes in Seasonal Bests - ' + filename)
    plt.legend()
    plt.show()

def plot_polyregression_percentages_MF():
    # Aggregate race times for each age
    unique_ages = np.unique(head)
    aggregated_percentages = []
    for age in head:
        arr = athlete_percentages[head == age]
        mean = np.mean(arr[arr != 100])
        aggregated_percentages.append(mean)
    #aggregated_race_times = [np.mean(athlete_times[ages == age]) for age in unique_ages]

    agg_women_percentages = []
    for age in head:
        arr = women_percs[head == age]
        mean = np.mean(arr[arr != 100])
        agg_women_percentages.append(mean)

    # MEN
    X = unique_ages.reshape(-1, 1)
    y = aggregated_percentages
    # Fit polynomial features
    poly_features = PolynomialFeatures(degree=4)  # You can change the degree as needed
    X_poly = poly_features.fit_transform(X)
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)
    # Predict the values
    y_pred = model.predict(X_poly)

    # Plot the aggregated data points
    label_added = False
    for i in range(athlete_percentages.shape[1]):
        perc = athlete_percentages[:, i]
        mask = perc < 100 #create a mask for where a time wasn't found to prevent a plot being made for that specific point
        if not label_added:
            label = 'Mens 200 Fly'
            label_added = True
        else:
            label = None
        plt.scatter(head[mask], perc[mask], color='blue', label=label)
    


    # WOMEN
    y = agg_women_percentages
    # Fit polynomial features
    poly_features = PolynomialFeatures(degree=4)  # You can change the degree as needed
    X_poly = poly_features.fit_transform(X)
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)
    # Predict the values
    y_predw = model.predict(X_poly)

    # Plot the aggregated data points
    label_added = False
    for i in range(women_percs.shape[1]):
        perc = women_percs[:, i]
        mask = perc < 100 #create a mask for where a time wasn't found to prevent a plot being made for that specific point
        if not label_added:
            label = 'Mens 200 Free'
            label_added = True
        else:
            label = None
        plt.scatter(head[mask], perc[mask], color='pink', label=label)

    plt.scatter(unique_ages, aggregated_percentages, color='red', label='Aggregated Men 200 Fly data')
    # Plot the polynomial regression line
    plt.plot(unique_ages, y_pred, color='green', label='Polynomial Regression Men 200 Fly')
    
    plt.scatter(unique_ages, agg_women_percentages, color='purple', label='Aggregated Men 200 Free data')
    # Plot the polynomial regression line
    plt.plot(unique_ages, y_predw, color='orange', label='Polynomial Regression Men 200 Free')

    plt.xlabel('Age')
    plt.ylabel('Percentage changes')
    plt.title('Polynomial Regression on Aggregated Athlete Percentage Changes in Seasonal Bests\n - ' + filename + " \n - " + filename2)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


plot_polyregression_percentages_MF()

#add_labels(plot_scatter()).show()
