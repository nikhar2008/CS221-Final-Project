from __future__ import division

import itertools
import cPickle
import datetime
import hashlib
import locale
import numpy as np
#import pycountry
import scipy.io as sio
import scipy.sparse as ss
import scipy.spatial.distance as ssd

from collections import defaultdict
from sklearn.preprocessing import normalize

class DataCleaner:
  """
  Common utilities for converting strings to equivalent numbers
  or number buckets.
  """
  def __init__(self):
    # load locales
   # self.localeIdMap = defaultdict(int)
    #for i, l in enumerate(locale.locale_alias.keys()):
    #  self.localeIdMap[l] = i + 1
    # load countries
    #self.countryIdMap = defaultdict(int)
   # ctryIdx = defaultdict(int)
   # for i, c in enumerate(pycountry.countries):
    #  self.countryIdMap[c.name.lower()] = i + 1
    #  if c.name.lower() == "usa":
     #   ctryIdx["US"] = i
     # if c.name.lower() == "canada":
      #  ctryIdx["CA"] = i
    #for cc in ctryIdx.keys():
     # for s in pycountry.subdivisions.get(country_code=cc):
      #  self.countryIdMap[s.name.lower()] = ctryIdx[cc] + 1
    # load gender id map
    self.genderIdMap = defaultdict(int, {"male":1, "female":2})

  def getLocaleId(self, locstr):
    return self.localeIdMap[locstr.lower()]

  def getGenderId(self, genderStr):
    return self.genderIdMap[genderStr]

  def getJoinedYearMonth(self, dateString):
    dttm = datetime.datetime.strptime(dateString, "%Y-%m-%dT%H:%M:%S.%fZ")
    return "".join([str(dttm.year), str(dttm.month)])

  def getCountryId(self, location):
    if (isinstance(location, str)
        and len(location.strip()) > 0
        and location.rfind("  ") > -1):
      return self.countryIdMap[location[location.rindex("  ") + 2:].lower()]
    else:
      return 0

  def getBirthYearInt(self, birthYear):
    try:
      return 0 if birthYear == "None" else int(birthYear)
    except:
      return 0

  def getTimezoneInt(self, timezone):
    try:
      return int(timezone)
    except:
      return 0

  def getFeatureHash(self, value):
    if len(value.strip()) == 0:
      return -1
    else:
      return int(hashlib.sha224(value).hexdigest()[0:4], 16)

  def getFloatValue(self, value):
    print value, "val"
    if len(value.strip()) == 0:

      return 0.0
    else:
      print "else"
      return np.float(value)
    print value, "val"




class Events:
  """
  Builds the event-event similarity matrix and event content-content
  similarity matrix for program events.
  """
  def __init__(self, programEntities, psim=ssd.correlation, csim=ssd.cosine):
    cleaner = DataCleaner()
    fin = open("users.tsv", 'rb')
    fin.readline() # skip header
    print 
    nevents = len(programEntities.eventIndex.keys())
    print nevents
    w, h = nevents, 100;
    self.eventPropMatrix = [[0 for x in range(w)] for y in range(h)] 
    self.eventContMatrix = [[0 for x in range(w)] for y in range(h)] 
    ln = 0
    for line in fin.readlines():
      if ln > 10:
        break
      cols = line.strip().split('\t')
      eventId = cols[0]
      print eventId, "id"
      print "ys"
      i = eventId
      print i, "yti"
      print cols
      self.eventPropMatrix[i, 0] = int(cols[10].strip()) # start_time
      print "why??"
      self.eventPropMatrix[i, 1] = int(cols[11].strip()) # city
      self.eventPropMatrix[i, 2] = int(cols[12].strip()) # state
      self.eventPropMatrix[i, 3] = int(cols[13].strip()) # zip
      self.eventPropMatrix[i, 7] = int(cols[14].strip()) # zip
      self.eventPropMatrix[i, 4] = int(cols[6].strip()) # zip
      self.eventPropMatrix[i, 5] = cleaner.getFeatureHash(cols[7]) # degreeType
      self.eventPropMatrix[i, 6] = cleaner.getJoinedYearMonth(cols[8]) # lon
      for j in range(9, 109):
        self.eventContMatrix[i, j-9] = cols[j]
      print self.eventContMatrix, "mst"
      print "oho"
      ln += 1
    fin.close()
    self.eventPropMatrix = normalize(self.eventPropMatrix,
        norm="l1", axis=0, copy=False)
    print self.eventPropMatrix, "matrix"
    sio.mmwrite("eventProp1", self.eventPropMatrix)
    self.eventContMatrix = normalize(self.eventContMatrix,
        norm="l1", axis=0, copy=False)
    sio.mmwrite("eventProp2", self.eventContMatrix)
    # calculate similarity between event pairs based on the two matrices    
    self.eventPropSim = ss.dok_matrix((nevents, nevents))
    self.eventContSim = ss.dok_matrix((nevents, nevents))
    for e1, e2 in programEntities.uniqueEventPairs:
      i = programEntities.eventIndex[e1]
      j = programEntities.eventIndex[e2]
      if not self.eventPropSim.has_key((i,j)):
        epsim = psim(self.eventPropMatrix.getrow(i).todense(),
          self.eventPropMatrix.getrow(j).todense())
       # self.eventPropSim[i, j] = epsim
      #  self.eventPropSim[j, i] = epsim
      if not self.eventContSim.has_key((i,j)):
        ecsim = csim(self.eventContMatrix.getrow(i).todense(),
          self.eventContMatrix.getrow(j).todense())

        self.eventContSim[i, j] = epsim
        self.eventContSim[j, i] = epsim
    sio.mmwrite("../Models/EV_eventPropSim", self.eventPropSim)
    sio.mmwrite("../Models/EV_eventContSim", self.eventContSim)


class ProgramEntities:
  """
  Creates reference sets for the entity instances we care about
  for this exercise. The train and test files contain a small
  subset of the data provided in the auxillary files.
  """
  def __init__(self):
    # count how many unique uesers and events are in the training file
    uniqueUsers = set()
    uniqueEvents = set()
    eventsForUser = defaultdict(set)
    usersForEvent = defaultdict(set)
    for filename in ["users.tsv"]:
      f = open(filename, 'rb')
      f.readline().strip().split("/t")
      for line in f:
        cols = line.strip().split("/t")
        uniqueEvents.add(cols[0])
        #eventsForUser[cols[0]].add(cols[1])
       # usersForEvent[cols[1]].add(cols[0])
      f.close()
  #  self.userEventScores = ss.dok_matrix((len(uniqueUsers), len(uniqueEvents)))
   # self.userIndex = dict()
    self.eventIndex = dict()
   # for i, u in enumerate(uniqueUsers):
    #  self.userIndex[u] = i
    for i, e in enumerate(uniqueEvents):
      self.eventIndex[i] = e
   # ftrain = open("../Data/train.csv", 'rb')
    #ftrain.readline()
   # for line in ftrain:
   #   cols = line.strip().split(",")
   #   i = self.userIndex[cols[0]]
   #   j = self.eventIndex[cols[1]]
    #  self.userEventScores[i, j] = int(cols[4]) - int(cols[5])
   # ftrain.close()
  #  sio.mmwrite("../Models/PE_userEventScores", self.userEventScores)
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
      if counter == 20: break
      for event2 in uniqueEvents:
        if event != event2:
          events = (event, event2)
          if (event2, event) not in self.uniqueEventPairs:
            self.uniqueEventPairs.add(events)
   # cPickle.dump(self.userIndex, open("../Models/PE_userIndex.pkl", 'wb'))
    cPickle.dump(self.eventIndex, open("dumm.pkl", 'wb'))

def main():
  """
  Generate all the matrices and data structures required for further
  calculations.
  """
  print "calculating program entities..."
  pe = ProgramEntities()
  print "calculating event metrics..."
  Events(pe)
  print "calculating event popularity metrics..."

if __name__ == "__main__":
  main()