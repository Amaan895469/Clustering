# Clustering
We will implement some clustering algorithms from scratch and we will test on two data sets constituted by some 2-dimensional distributions of points. Then we will apply our algorithms to a real-word data set.
TODO:
Generate data sets DS1 (non overlapping blobs), DS2 (overlapping blobs) and load DS3 from the file iris.csv.
Implement K-Means, Fuzzy C- Means and Graded Possibilistic C-Means.
Implement WTA and the α−cut defuzzifiers of fuzzy partitions.
Implement RAND and Jaccard Indeces for hard partition comparison
Apply K-Means, Fuzzy C-Means and Graded Possibilistic C-Means to the 3 data sets using a multi-start approach; search for 2, 3, and 4 clusters.
Defuzzify the soft partitions of Fuzzy C- Means and Graded Possibilistic C-Means using the WTA (Winner-Takes-All) criterion.
Visualize the results on the scatter plot, highlighting the centroids and using a different color for each cluster.
Measure the accuracy of the hard partitions by comparing them with the ground-truth constituted by the targets of the data sets. For the comparison use RAND and Jaccard indeces.
For the Graded Possibilistic C-Means use a possibilistic degree  𝛽=0.8 and a value of  𝜂 (identical for each cluster) comparable with the standard_dev 2 for data sets DS1 and DS2. For DS3 (Iris data set)  𝜂 must be selected by checking the value of the accuracy (model selection - grid search).
