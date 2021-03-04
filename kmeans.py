import numpy as np
import pandas as pd 
from scipy.spatial import distance
from matplotlib import pyplot as plt 
from operator import add

# getting the data ready into a form that is viable for performing clustering
def gen_feature_vectors(fname):
    feature_vectors = []
    with open(fname) as feature_file:
        for line in feature_file:
            x = []
            count = 1
            for w in line.strip().split(','):
                #the following code is used to differentiate the data point and the text in the dataset ( titles of columns etc)
                if (count < 5):
                    x.append(float(w))
                else:
                    x.append(w)
                count += 1
            feature_vectors.append(x)
    return feature_vectors

# funcion to cluster the data 
def cluster(centers, data, c):
    c_length = len(centers)
    d_length = len(data)
    for i in range(0, d_length):
        for j in range(0, c_length):
            current_distance = distance.euclidean(data[i][0:4:1], centers[j][0:4:1])
            if (j == 0):
                dist = current_distance
                cluster = 0
            if(current_distance < dist):
                dist = current_distance
                cluster = j
        if(c):
            data[i].append(cluster)
        else:
            data[i][-1] = cluster
    return data

def calculate_centers(data, k):
    d_length = len(data)
    new_centers = []
    for i in range(0, k):
        count = 0
        total = [0.0, 0.0, 0.0, 0.0]
        for j in range(0, d_length):
            if(data[j][5] == i):
                count += 1
                total = list(map(add, data[j][0:4:1], total))
        for l in range(0, 4):
            total[l] /= count
        new_centers.append(total)
    return new_centers


# to count the number of items in a cluster
def count_elements(data, k):
    count = np.zeros(k)
    d_length = len(data)
    for i in range(0, k):
        for j in range(0, d_length):
            if(data[j][5] == i):
                count[i] += 1
    return count

#Function to check if the clusters are all non-zero
def check_count(count, k):
    for i in range(0, k):
        if(count[i] == 0):
            return False
    return True

#Function to compare two lists
def compare_counts(old_count, new_count, k):
    for i in range(0, k):
        if(old_count[i] != new_count[i]):
            return False
        else:
            continue
    return True


