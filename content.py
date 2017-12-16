from __future__ import division

import itertools
import cPickle
import datetime
import hashlib
import locale
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as ss
from scipy.sparse.linalg import svds
import scipy.spatial.distance as ssd
#from surprise import *
from sklearn import cross_validation as cv
import sklearn.metrics.pairwise as pw
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score
from math import sqrt

from collections import defaultdict
from sklearn.preprocessing import normalize

class DataCleaner:
  """
  Common utilities for converting strings to equivalent numbers
  or number buckets.
  """
  def __init__(self):
    self.listMap = []


  def getJoinedYearMonth(self, dateString):
    dttm = datetime.datetime.strptime(dateString, "%Y-%m-%dT%H:%M:%S.%fZ")
    return "".join([str(dttm.year), str(dttm.month)])

  def getFeatureHash(self, value):
    if len(value.strip()) == 0:
      return -1
    else:
      return int(hashlib.sha224(value).hexdigest()[0:4], 16)

  def getFloatValue(self, value):
    if len(value.strip()) == 0:

      return 0.0
    else:
      try:
        return np.float(value)
      except:
        return 0.0




class Events:
  """
  Builds the event-event similarity matrix and event content-content
  similarity matrix for program events.
  """
  def __init__(self, eventIndex, users, psim=ssd.correlation, csim=ssd.cosine):
    cleaner = DataCleaner()
    fin = open("users.tsv", 'rb')
    fin.readline() # skip header
    nevents = len(eventIndex.keys())
    self.eventPropMatrix = np.zeros((nevents, 8))
    self.eventContMatrix = ss.dok_matrix((nevents, 100))
    ln = 0
    for line in fin.readlines():
  #      if ln > 10:
  #        break
      cols = line.strip().split("\t")
      eventId = cols[0]
      if eventIndex.has_key(eventId):
        i = eventIndex[eventId]
        self.eventPropMatrix[i, 0] = cleaner.getFloatValue(cols[10]) # start_time
        self.eventPropMatrix[i, 1] = cleaner.getFloatValue(cols[11]) # city
        if len(cols[12].strip()) > 0:
          if cols[12].strip() == "Yes": self.eventPropMatrix[i,2] = 1
          else :self.eventPropMatrix[i,2] = 2
        if len(cols[13].strip()) > 0:
          if cols[13].strip() == "Yes": self.eventPropMatrix[i,3] = 1
          else :self.eventPropMatrix[i,3] = 2
        self.eventPropMatrix[i,7] = cleaner.getFloatValue(cols[14]) # zip
        self.eventPropMatrix[i,4] = cleaner.getFloatValue(cols[6]) # zip
        self.eventPropMatrix[i,5] = cleaner.getFeatureHash(cols[7]) # degreeType
        self.eventPropMatrix[i,6] = cleaner.getFeatureHash(cols[8]) # 
          #for j in range(9, 109):
           # self.eventContMatrix[i, j-9] = cols[j]
        ln += 1
    fin.close()
    print "yeys"
    self.eventPropMatrix = normalize(self.eventPropMatrix,
          norm="l1", axis=0, copy=False)
      #sio.mmwrite("../Models/EV_eventPropMatrix", self.eventPropMatrix)
    self.eventContMatrix = normalize(self.eventContMatrix,
          norm="l1", axis=0, copy=False)
     # sio.mmwrite("../Models/EV_eventContMatrix", self.eventContMatrix)
      # calculate similarity between event pairs based on the two matrices    
    self.eventPropSim = np.zeros((nevents, nevents))
    self.eventContSim = ss.dok_matrix((nevents, nevents))
    print "yep"
    counter = 0
    for e1 in eventIndex.keys():
      for e2 in eventIndex.keys():
        if counter == 100000: break
        counter +=1
        i = eventIndex[e1]
        j = eventIndex[e2]
        #if not self.eventPropSim.has_key((i,j)):
        epsim = csim(self.eventPropMatrix[i],
            self.eventPropMatrix[j])
        try:
          float(epsim)
        except: epsim = 0.0
        self.eventPropSim[i, j] = epsim
        self.eventPropSim[j, i] = epsim

    print "chec"
      #  if not self.eventContSim.has_key((i,j)):
        #  ecsim = csim(self.eventContMatrix.getrow(i).todense(),
         #   self.eventContMatrix.getrow(j).todense())
          #self.eventContSim[i, j] = epsim
         # self.eventContSim[j, i] = epsim
    sio.mmwrite("eventProp1", self.eventPropSim)

  def getVal(self):
    print self.eventPropSim
    return self.eventPropSim
      #kmeans = KMeans(n_clusters = 12)
     # kmeans.fit(self.eventPropMatrix)

      #sio.mmwrite("../Models/EV_eventContSim", self.eventContSim)

class DataRewriter:
  def __init__(self, userSim, jobSim, jobIndex, eventIndex):
    self.eventIndex = eventIndex
    self.jobIndex = jobIndex
    self.userJobMatrix = jobSim
    self.eventPropSim = userSim

  def rmse(self, prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    #if prediction.max() != 0:
     # prediction *= 1.0/prediction.max()
            #lowValues = np.where(prediction<0)
            #for v in lowValues:
            #    prediction[v] = 0
     # print('pred max =', prediction.max())
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    print('in rmse')
    print('pred', prediction)
    print('gt', ground_truth)
        #mse function(y_true, y_pred)! wrong way around in the sample code...
        #returns sum of squared distances between correct and incorrect,
        #divided by the total number of users
    return sqrt(mean_squared_error(ground_truth, prediction))


  def eventReco(self, n_users, train_data_matrix):
    for userIndex in self.eventIndex.keys():
      i = self.eventIndex[userIndex]
      self.finalMatrix = np.zeros((len(self.eventIndex.keys()), len(self.jobIndex.keys())))
      for job in self.userJobMatrix.keys():
        users = self.userJobMatrix[job]
        sumVal = 0
        for user in users:
          counter = 0
          userVal = self.eventIndex[user]
          counter +=1
          #if user == userIndex:
           # print self.eventPropSim[i,userVal], "checl"
            #sumVal += 10
          sumVal+=self.eventPropSim[i,userVal]
        jobVal = self.jobIndex[job]
        if sumVal > 0.0:
          self.finalMatrix[i, jobVal] = sumVal/counter
    print self.finalMatrix

   # resFile = open("eventProp3.txt", "w")
   # resFile.write(self.finalMatrix)
    #resFile.close()
    #sio.mmwrite("eventProp3", self.finalMatrix)
    print average_precision_score(train_data_matrix, self.finalMatrix, average='macro', sample_weight=None)
    print "mean", mean_squared_error(train_data_matrix, self.finalMatrix)
    print('Item-based CF RMSE: ' + str(self.rmse(self.finalMatrix, train_data_matrix)))


class ProgramEntities:
  """
  Creates reference sets for the entity instances we care about
  for this exercise. The train and test files contain a small
  subset of the data provided in the auxillary files.
  """
  def __init__(self, train_data_matrix):
    # count how many unique uesers and events are in the training file
    uniqueJob = set()
    uniqueEvents = set()
    eventsForUser = defaultdict(set)
    usersForEvent = defaultdict(set)
    f = open("apps.tsv", 'rb')
    f.readline()
    for line in f:
      cols = line.strip().split('\t')
      uniqueEvents.add(cols[0])
      uniqueJob.add(cols[4])
      #print cols[0]
        #eventsForUser[cols[0]].add(cols[1])
       # usersForEvent[cols[1]].add(cols[0])
    f.close()
  #  self.userEventScores = ss.dok_matrix((len(uniqueUsers), len(uniqueEvents)))
    self.userIndex = dict()
    self.eventIndex = dict()
   # for i, u in enumerate(uniqueUsers):
    #  self.userIndex[u] = i
    for i, e in enumerate(uniqueEvents):
      self.eventIndex[e] = i
    for i, e in enumerate(uniqueJob):
      self.userIndex[e] = i
    self.userEventScores = train_data_matrix
       # ftrain.close()
    sio.mmwrite("eventProp2", self.userEventScores)
    # find all unique user pairs and event pairs that we should
    # look at. These should be users who are linked via an event
    # or events that are linked via a user in either the training
    # or test sets. This is to avoid useless calculations
   # self.uniqueUserPairs = set()
    self.uniqueEventPairs = set()
   # for event in uniqueEvents:
     # users = usersForEvent[event]
     # if len(users) > 2:
      #  self.uniqueUserPairs.update(itertools.combinations(users, 2))
    counter = 0
    for event in uniqueEvents:
      counter += 1
      if counter == 20000: break
      for event2 in uniqueEvents:
        if event != event2:
          events = (event, event2)
          if (event2, event) not in self.uniqueEventPairs:
            self.uniqueEventPairs.add(events)
    cPickle.dump(self.userIndex, open("dumm2.pkl", 'wb'))
    cPickle.dump(self.eventIndex, open("dumm.pkl", 'wb'))

def main():
  """
  Generate all the matrices and data structures required for further
  calculations.
  """
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

  eventIndex = dict()
  jobIndex = dict()
  for i, e in enumerate(users):
      eventIndex[e] = i
  for i, e in enumerate(jobs):
      jobIndex[e] = i
    
  print 'Num users: ' + str(n_users) + ' | Num Jobs = ' + str(n_jobs)
        
  train_data, test_data = cv.train_test_split(apps, test_size=0.25)

  jobMapp = dict()

  train_data_matrix = np.zeros((n_users, n_jobs))
  for line in train_data.itertuples():
   # print('user id to index', np.where(users==line[1]))
   # print('job id to index', np.where(jobs==line[5]))
    train_data_matrix[np.where(users==line[1])[0], np.where(jobs==line[5])[0]] = 1
    if line[5] in jobMapp.keys(): listToAdd = jobMapp[line[5]]
    else: listToAdd = []
    listToAdd.append(line[1])
    jobMapp[line[5]] = listToAdd
  print('done with train matrix')
  counter = 0
  for key in jobMapp.keys():
    if len(jobMapp[key]) > 2: 
      counter+=1
  print counter

  print "calculating program entities..."
  #pe = ProgramEntities(train_data_matrix)
  print "calculating event metrics..."
  event = Events(eventIndex, users)
  userSim = event.getVal()

  reWrite = DataRewriter(userSim, jobMapp, jobIndex, eventIndex)
  
  reWrite.eventReco(n_users, train_data_matrix)
    


if __name__ == "__main__":
  main()