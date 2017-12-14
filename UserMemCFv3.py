#Implement user-based memory-based collaborative filtering

#Evaluation metric: accuracy
#Regression: mean squared error
#normalize the truth matrix to get probabilities (if applied to two jobs, then make each have probability .5)
#in our matrix, you also normalize the scores
#take difference of the two probabilities for each score

#working model, results, analysis
import random
import csv
import numpy
from math import *
userIDToIndex = {}
jobIDToIndex = {}

userIndexAsKey = {}
jobIndexAsKey = {}

#Steps:
#Input: user with some of their applications known
#Output: set of jobs we suggest they apply to
#We can just set aside some of the data we have as test data,
#and hide half the jobs those users apply for,
#since we don't know the actual full competition data.

#User-based scoring function:
#for each user v in set of users who applied to job i, sum f(weight(u, v))

#Create matrix R: rows are unique users, columns are unique items
#First step is to create the matrix I guess: (u, i) = 1 if u applied to i.
#Matrix is like a dictionary: R[u1] = [u1i1, u1i2, ...]

#Takes a dictionary as input. Checks if any keys have the same value.
def duplicateCheck(myDict):
    rev_multidict = {}
    for key, value in myDict.items():
        rev_multidict.setdefault(value, set()).add(key)

    print([values for key, values in rev_multidict.items() if len(values) > 1])
    
def createMatrixR():
    #map matrix column indices to JobIDs
    print("Job map")
    ind = 0
    with open("jobs.tsv", "r") as infile:
        reader = csv.reader(infile, delimiter="\t", 
        quoting=csv.QUOTE_NONE, quotechar="")
        reader.next() # burn the header
        for line in reader:
            (JobID, WindowID, Title, Description, Requirements, City, State, 
            Country, Zip5, StartDate, EndDate) = line

            if WindowID == str(1):
                jobIDToIndex[int(JobID)] = ind
                jobIndexAsKey[ind] = int(JobID)
                ind += 1

    #map matrix row indices to UserIDs
    print("user map")
    ind = 0
    with open("users.tsv", "r") as infile:
        reader = csv.reader(infile, delimiter="\t", 
        quoting=csv.QUOTE_NONE, quotechar="")
        reader.next() # burn the header
        for line in reader:
            (UserID, WindowID, Split, City, State, Country,
             ZipCode, DegreeType, Major, GraduationDate,
             WorkHistoryCount, TotalYearsExperience, CurrentlyEmployed,
             ManagedOthers, ManagedHowMany) = line

            if WindowID == str(1):
                userIDToIndex[int(UserID)] = ind
                userIndexAsKey[ind] = int(UserID)
                ind += 1

    print("UID 47")
    print(userIDToIndex[47])
    print("UID 72")
    print(userIDToIndex[72])

    numJobs = len(jobIDToIndex.keys())
    numUsers = len(userIDToIndex.keys())
    print("# jobs: ", numJobs)
    print("# users: ", numUsers)

    #Create the matrix R
    R = numpy.zeros((numUsers, numJobs), dtype=int)

    #Split apps.tsv into test and training data
    with open ("apps.tsv", "rb") as infile:
        data = infile.read().split('\n')
        data = data[1:]
        random.shuffle(data)

        fifthPoint = len(data)//5
        test_data = data[:fifthPoint]
        train_data = data[fifthPoint:]

    print("split into training and test data")
    
    #Only modify matrix R for the training data
    train_file = open("trainfile.txt", "w")

    for line in train_data:
        #print('train line')
        #print(line)
        lineList = line.split("\t")
        UserID = lineList[0]
        try:
            JobID = lineList[4]
            JobID = JobID.replace("\r", "")
        except:
            print("Could not parse JobID for: ", UserID)

        try:
            WindowID = lineList[1]
            if WindowID == str(1):
                #print("edit R")
                train_file.write(line)
                R[userIDToIndex[int(UserID)], jobIDToIndex[int(JobID)]] = 1
                #print(R[userIDToIndex[int(UserID)], jobIDToIndex[int(JobID)]])

        except:
            print("Could not edit R for: ", UserID, " and ", JobID)

    train_file.close()

    test_file = open("testfile.txt", "w")
    testUsers = []
    for line in test_data:
        lineList = line.split("\t")
        UserID = lineList[0]
        try:
            WindowID = lineList[1]
            if WindowID == str(1):
                testUsers.append(UserID)
                test_file.write(line)
        except:
            print('Could not parse Window ID for user', UserID)

    print('testUsers', testUsers[:5])
    test_file.close()
        
    print("R done")
    print(R[0])
    return R, numJobs, numUsers, testUsers

def similarityMetric(uID1, uID2, R, numJobs):
    intersectLen = 0
    u1Ind = userIDToIndex[uID1]
    u2Ind = userIDToIndex[uID2]
    u1SetLen = 0
    u2SetLen = 0

    u1Row = R[u1Ind]
    u2Row = R[u2Ind]

    u1SetLen = sum(u1Row)
    u2SetLen = sum(u2Row)

    #CHECK ON SYNTAX FOR INDEXING NUMPY ARRAYS!
    u1RowOnes = numpy.where(u1Row == 1)[0]
    u2RowOnes = numpy.where(u2Row == 1)[0]

    intersectLen = len(list(set(u1RowOnes) & set(u2RowOnes)))
    
    #print('u1 row ones', u1RowOnes)
    #print('u1 set len', u1SetLen)
    #print('u2 set len', u2SetLen)
            
    try:
        weight = intersectLen/(sqrt(u1SetLen)*sqrt(u2SetLen))
    except:
        weight = 0
        
    return weight

'''
def similarityMatrix(testUserID, R, numJobs, numUsers):
    S = numpy.zeros((len(testUsers), numUsers), dtype = int)
    rowIndex = 0
    for row in R:
'''

#Function goal: what is the likelihood of newUser applying to newJob?
#If we already know the newUser applied to newJob, we wouldn't be calling this function.
def scoreFunc(newUserID, newJobID, R, numJobs):
    score = 0

    #Find out who are all the users who applied for newJob
    #Get the similarity weight between that user and newUser

    newJobIndex = jobIDToIndex[newJobID]
    newJobColumn = R[:, newJobIndex]
    userIndicesWhoApplied = numpy.where(newJobColumn == 1)[0]
    
    for uInd in userIndicesWhoApplied:
        #print('uInd in score func', uInd)
        u2ID = userIndexAsKey[uInd]
        weight = similarityMetric(newUserID, u2ID, R, numJobs)
        score += weight

    return score

def getTestScore(testUserID, R, numJobs):
    
    #Create the weight matrix W for the users in testfile.txt
    #W = numpy.zeros((len(testUsers), numJobs), dtype=int)
    print('test user id:', testUserID)
    testUserIndex = userIDToIndex[testUserID]
    
    testUserR = R[testUserIndex]
    ind = 0
    
    jobsKnown = []
    suggestedJobs = {}
    for rVal in testUserR:
        #print('r val', rVal)
        newJobID = jobIndexAsKey[ind]
        if rVal != 1:
            score = scoreFunc(testUserID, newJobID, R, numJobs)
            if score > 0:
                suggestedJobs[newJobID] = score
        else:
            jobsKnown.append(newJobID)

        ind += 1
        if ind == 100:
            break

    print('known', jobsKnown)
    print('suggested', suggestedJobs)

def getAllScores(testUsers, R, numJobs):
    for testUserID in testUsers:
        getTestScore(int(testUserID), R, numJobs)
        
    
    '''
    for testUserID in testUsers:
        testUserIndex = userIDToIndex[testUserID]
        for jobInd in range(numJobs):
            if R[testUserIndex, jobInd] != 1:
                score = scoreFuncJobIndex(testUserID, jobInd, R, numJobs)
                W[testUserIndex, jobInd] = score

        givenUserScores = W[testUserIndex]
        givenUserScores.sort(reverse = True)
        givenUserScores = givenUserScores[:10]
        print('user ', testUserID, ' should apply to jobs:')
        for j in range(numJobs):
            if W[testUserIndex, j] in givenUserScores:
                print(j)
    '''
            
                

R, numJobs, numUsers, testUsers = createMatrixR()
#print(scoreFuncJobID(47, 169528, R, numJobs))

getAllScores(testUsers, R, numJobs)
'''
c = 0
for k, v in userIDToIndex.items():
    print(k, v)
    c+=1
    if c==10:
        break
    
#predict the remaining jobs in the test data:
print(similarityMetric(47, 72, R, numJobs))
'''
