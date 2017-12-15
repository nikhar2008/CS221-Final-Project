import sys
sys.path.append('/Users/nagrawal/anaconda2/lib/python2.7/site-packages')

import numpy as np
import pandas as pd
from surprise import *
from sklearn import cross_validation as cv
import sklearn.metrics.pairwise as pw
from sklearn.metrics import mean_squared_error
from math import sqrt

def predict(dataMatrix, simMatrix):
    pred = dataMatrix.dot(simMatrix) / np.array([np.abs(simMatrix).sum(axis=1)])
    return pred

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

def main():
    '''
    jobsHeader = ['JobID', 'WindowID', 'Title', 'Description', 'Requirements', 'City', 'State', 
              'Country', 'Zip5', 'StartDate', 'EndDate']
    jobs = pd.read_csv('jobs.tsv', sep='\t', names=jobsHeader)
    n_jobs = apps.JobID.unique().shape[0]
    ind = 0
    for line in jobs.itertuples():
        jobIDToIndex[int(line[0])] = ind
    '''
    
    appsHeader = ['UserID', 'WindowID', 'Split', 'ApplicationDate', 'JobID']
    apps = pd.read_csv('apps.tsv', sep='\t', names = appsHeader)
    apps = apps[apps['WindowID'] == '1']
    apps = apps[apps['Split'] == 'Test']

    n_users = apps.UserID.unique().shape[0]
    n_jobs = apps.JobID.unique().shape[0]

    users = apps.UserID.unique()
    jobs = apps.JobID.unique()
    print('u list', users)
    print('j list', jobs)
    
    print 'Num users: ' + str(n_users) + ' | Num Jobs = ' + str(n_jobs)
        
    train_data, test_data = cv.train_test_split(apps, test_size=0.25)

    train_data_matrix = np.zeros((n_users, n_jobs))
    for line in train_data.itertuples():
        #print('user id to index', np.where(users==line[1]))
        #print('job id to index', np.where(jobs==line[5]))
        train_data_matrix[np.where(users==line[1])[0], np.where(jobs==line[5])[0]] = 1
    print('done with train matrix')
    print('train matrix', train_data_matrix)
    
    test_data_matrix = np.zeros((n_users, n_jobs))
    for line in test_data.itertuples():
        test_data_matrix[np.where(users==line[1])[0], np.where(jobs==line[5])[0]] = 1
    print('done with test matrix')
    print('test matrix', test_data_matrix)

    user_similarity = pw.pairwise_distances(train_data_matrix, metric='cosine')
    print('user sim', user_similarity)

    item_similarity = pw.pairwise_distances(train_data_matrix.T, metric='cosine')
    print('item sim', item_similarity)

    item_pred = predict(train_data_matrix, item_similarity)
    print('item pred', item_pred)

    print('Item-based CF RMSE: ' + str(rmse(item_pred, test_data_matrix)))

    #user_similarity = np.zeroes((n_users, n_users))
    #for row1 in train_data_matrix:
    #   for row2 in train_data_matrix:
        
    #user_similarity = pw.cosine_similarity(train_data_matrix)
    #print('user sim')
    
    #item_similarity = pw.cosine_similarity(train_data_matrix.T)
    #print('done with similarity')
main()
