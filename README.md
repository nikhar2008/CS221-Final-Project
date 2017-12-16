# CS221-Final-Project
making a job recommendation tool

UserMemCFLibraryv5.py contains the user-based and job-based memory collaborative filtering algorithms (UUM and JJM)
content.py contains the user-based content-based CF algorithm (UUC). 

We use libraries pandas, numpy, and scikit-learn in our code. The Mem CF file uses pairwise_distances function and mean_average_precision from scikit-learn. The predict function creates the prediction matrix and rmse function does the error analysis. 
