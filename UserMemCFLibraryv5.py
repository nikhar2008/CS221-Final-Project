import sys
sys.path.append('/Users/nagrawal/anaconda2/lib/python2.7/site-packages')

import numpy as np
import pandas as pd
from surprise import *
from sklearn import cross_validation as cv
import sklearn.metrics.pairwise as pw
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score
from math import sqrt

def predict(dataMatrix, simMatrix, CFtype):
##    print('predict '+CFtype)
    
    if CFtype == 'item':
        num = dataMatrix.dot(simMatrix)
##        print('num', num)
        numFirstElem = num[0, 1]
        den = np.abs(simMatrix).sum(axis=1)
        #print('num first elem', numFirstElem)

    if CFtype == 'user':
        num = simMatrix.dot(dataMatrix)
##        print('num', num)
        numFirstElem = num[0, 1]
        #print('num first elem', numFirstElem)
        den = np.abs(simMatrix).sum(axis=1)
        den = den[:, None]
        
    #den = np.array([np.abs(simMatrix).sum(axis=1)])
    #NOTE: I removed the outer array because it didn't seem to make sense.
    #print('den', den)
    denFirstElem = den[1]
    #print('den first elem', denFirstElem)
    pred = num/den
    print('pred first', pred[0, 1])
    print('calculated', numFirstElem/denFirstElem)
    return pred

def meanAvgPrecision(prediction, ground_truth, n_jobs):
    '''
    for uID in nUsers:
        if sum(ground_truth[uID])!=0:
            total = sum(ground_truth[uID])
            total_ap = 0.0
            for i in range(150):
                if
    '''
    return 0
    #prediction = prediction[ground_truth.nonzero()].flatten()
    #ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    #avg_prec = dict()
   # for j in range(n_jobs):    
#        avg_prec[j] = average_precision_score(ground_truth[:, j], prediction[:, j])

    #return avg_prec
    #return average_precision_score(ground_truth, prediction, average = 'macro')

def rmse(prediction, ground_truth):
    
    #NOTE: we only care about the results where the test matrix said someone would apply.
    #so not caring if we predict a job they didn't actually apply for

    #flatten: takes [1, 2], [3, 4] and makes it [1, 2, 3, 4]
    prediction = prediction[ground_truth.nonzero()].flatten()
    if prediction.max() != 0:
        print(prediction.max())
        prediction *= 1.0/prediction.max()
        #lowValues = np.where(prediction<0)
        #for v in lowValues:
        #    prediction[v] = 0
        #print('pred max =', prediction.max())
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
##    print('in rmse')
##    print('pred', prediction)
##    print('gt', ground_truth)
    #mse function(y_true, y_pred)! wrong way around in the sample code...
    #returns sum of squared distances between correct and incorrect,
    #divided by the total number of users
    return sqrt(mean_squared_error(ground_truth, prediction))

def worstResult(ground_truth, n_users, n_jobs):
    oneMatrix = np.ones((n_users, n_jobs))
    #print('one matrix', oneMatrix)
    #flips all the ones and zeroes in the test matrix
    worstMatrix = np.subtract(oneMatrix, ground_truth)
    #print('worst matrix', worstMatrix)
    return rmse(worstMatrix, ground_truth)
    
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

    #header is the name of each column in the file
    appsHeader = ['UserID', 'WindowID', 'Split', 'ApplicationDate', 'JobID']
    apps = pd.read_csv('apps.tsv', sep='\t', names = appsHeader)

    #only have lines in the file where window id is 1 and split is test
    #split is test part is really just to shrink dataset size
    apps = apps[apps['WindowID'] == '1']
    apps = apps[apps['Split'] == 'Test']

    #gets the number of unique users and unique jobs in the apps data
    n_users = apps.UserID.unique().shape[0]
    n_jobs = apps.JobID.unique().shape[0]

    #users and jobs are arrays containing unique users and unique jobs
    users = apps.UserID.unique()
    jobs = apps.JobID.unique()
    print('u list', users)
    print('j list', jobs)
    
    print 'Num users: ' + str(n_users) + ' | Num Jobs = ' + str(n_jobs)

    #split data into train and test, with 25% of apps data as test
    #NOTE: Does this function split the data differently in each run of the program?
    train_data, test_data = cv.train_test_split(apps, test_size=0.2)

    #Create a train data matrix of all zeroes: rows are all users, columns are all jobs
    #itertuples: Iterate over DataFrame rows as namedtuples, with index value as first element of the tuple.
    train_data_matrix = np.zeros((n_users, n_jobs))
    for line in train_data.itertuples():
        #print('user id to index', np.where(users==line[1]))
        #print('job id to index', np.where(jobs==line[5]))
        
        #if a user applied to a job set that user, job slot in train matrix to be 1
        #NOTE: where function gets the index of the corresponding userID, but have to
        #do [0] for it to not be an array.
        train_data_matrix[np.where(users==line[1])[0], np.where(jobs==line[5])[0]] = 1
    #print('done with train matrix')
    #print('train matrix', train_data_matrix)

    #Same with test matrix but only for test data
    test_data_matrix = np.zeros((n_users, n_jobs))
    for line in test_data.itertuples():
        test_data_matrix[np.where(users==line[1])[0], np.where(jobs==line[5])[0]] = 1
    #print('done with test matrix')
    #print('test matrix', test_data_matrix)

    #Outputs a distance matrix D such that D(i, j) is the distance between
    #the ith and the jth vectors of the given matrix.
    #train_data: (nusers, njobs)
    #D: (nusers, nusers)
    #NOTE: CAN EXPERIMENT WITH DIFFERENT PARAMETERS FOR METRIC
    user_similarity = pw.pairwise_distances(train_data_matrix, metric='cosine')
##    print('user sim', user_similarity)

    item_similarity = pw.pairwise_distances(train_data_matrix.T, metric='cosine')
##    print('item sim', item_similarity)

    item_pred = predict(train_data_matrix, item_similarity, 'item')
##    print('item pred', item_pred)

    user_pred = predict(train_data_matrix, user_similarity, 'user')
##    print('user pred', user_pred)

    print('Item-based CF RMSE: ' + str(rmse(item_pred, test_data_matrix)))
    print('User-base CF RMSE: ' + str(rmse(user_pred, test_data_matrix)))

##    print('item map: ', meanAvgPrecision(item_pred, test_data_matrix, n_jobs))
##    print('user map: ', meanAvgPrecision(user_pred, test_data_matrix, n_jobs))

    print('worst result rmse: ' + str(worstResult(test_data_matrix, n_users, n_jobs)))

    #user_similarity = np.zeroes((n_users, n_users))
    #for row1 in train_data_matrix:
    #   for row2 in train_data_matrix:
        
    #user_similarity = pw.cosine_similarity(train_data_matrix)
    #print('user sim')
    
    #item_similarity = pw.cosine_similarity(train_data_matrix.T)
    #print('done with similarity')
main()
