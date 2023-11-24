"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import json
import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')

def load_json(filename):
    """Load a numpy array from a JSON document.

    Expected format:
    [
        {
            observations: [0, 1]
        },
        {
            observations: [0, 2]
        }
    ]

    :param filename: Filename of CSV to load
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data_as_json = json.load(file)
        return [np.array(entry['observations']) for entry in data_as_json]



def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

       :param data: Data of which to calculate the mean

       :returns: the mean of the data as a 1D numpy array
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.
    
       :param data: Data of which to calculate the maximum 

       :returns: the maximum value of the data as a 1D numpy array
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.
    
       :param data: Data of which to calculate the minimum
       :returns: the minimum value of the data as a 1D numpy array
    """
    return np.min(data, axis=0)


def standard_deviation(data):
    """Calculate the standard deviation of a 2d inflammation data array and return it as a dictionary."""
    mean_data = daily_mean(data)
    devs = []
    for entry in data:
        devs.append((entry - mean_data) * (entry - mean_data))

    s_dev2 = sum(devs) / len(data)
    return {'standard deviation': s_dev2}
