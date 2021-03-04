import kmeans
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if __name__ == "__main__":
    k = 3
    init_flag = False
    while not (init_flag):
        print("\n loading dataset")
        feature_vectors = kmeans.gen_feature_vectors("iris.data.txt")
        print ("\n Initializing cluster centres")
        centers = random.sample(feature_vectors, k)
        clustered_data = kmeans.cluster(centers, feature_vectors, True) #not that this is only an initial clustering 
        print("\ncounting cluster members for each cluster")
        new_count = kmeans.count_elements(clustered_data, k)
        print("\n verifying that the clusters are acceptable") 
        init_flag = kmeans.check_count(new_count, k)
        if not (init_flag):
            print("\n Zero cluster is detected .. reinitializing algorithm")
        else:
            print("\n clusters are acceptable .. proceeding for optimization of cluster centers")
        converge_flag = False
        counter = 1
    
    while not bool(converge_flag):
        print("\n Iteration no :", counter)
        old_count = new_count
        centers = kmeans.calculate_centers(clustered_data, k)
        print("\n clustering data") 
        clustered_data = kmeans.cluster(centers, feature_vectors, False)
        new_count = kmeans.count_elements(clustered_data, k)
        converge_flag = kmeans.compare_counts(old_count, new_count, k)
        counter += 1

    print("\n The optimal clustering of the data has been achieved as per the kmeans algorithn")
    given_data = pd.DataFrame(clustered_data, columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Iris', 'Obtained value'])
    final_data = given_data.filter(['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])

    plt.figure(figsize = (12, 3))
    colors = np.array(['red', 'green', 'blue'])
    iris_targets_legend = np.array(colors)
    red_patch = mpatches.Patch(color = 'red', label = 'Iris-setosa')
    blue_patch = mpatches.Patch(color = 'blue', label = 'Iris-versicolor')
    green_patch = mpatches.Patch(color = 'green', label = 'Iris-virginica')
    plt.subplot(1, 2, 1)
    plt.scatter(given_data['Sepal Length'], given_data['Sepal Width'], c=colors[given_data['Obtained value']])
    plt.title('Sepal Length vs Sepal Width')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.subplot(1, 2, 2)
    plt.scatter(given_data['Petal Length'], given_data['Petal Width'], c=colors[given_data['Obtained value']])
    plt.title('Petal Length vs Petal Width')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.show()


