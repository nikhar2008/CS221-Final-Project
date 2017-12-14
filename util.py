import os, random, operator, sys, copy
from collections import Counter
from pyzipcode import ZipCodeDatabase
from math import radians, cos, sin, asin, sqrt



def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def getDistance(zipcode1, zipcode2):
    if zipcode2 == "" or zipcode1 == "": return None
    zcdb = ZipCodeDatabase()
    zipcodeUser2 = zcdb[zipcode2]
    zipcodeUser = zcdb[zipcode1]
    long_1 = zipcodeUser.longitude
    long_2 = zipcodeUser2.longitude
    lat_1 = zipcodeUser.latitude
    lat_2 = zipcodeUser2.latitude

    lon1, lat1, lon2, lat2 = map(radians, [long_1, lat_1, long_2, lat_2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def checkVal(userName, userName2, job):
    reachedVal = False
    jobApp = []
    for line in open(job):
        line = line.strip().split('\t')
        if line[0] == userName: 
            reachedVal = True
            jobApp.append(line[4])
        elif reachedVal: break
    reachedVal = False
    for line in open(job):
        line = line.strip().split('\t')
        if line[0] == userName2: 
            reachedVal = True
            if line[4] in jobApp: return True
        elif reachedVal: break
    return False




def readExamples(user, job):
    '''
    Reads a set of training examples.
    '''
    examples = []
    counter1 = 0
    counter2 = 0
    userArray = {}
    userList = []
    for line in open(user):
        #line = str(line)
        counter2 = 0
        line = line.strip().split('\t')
        if counter1 == 0:
            counter1+=1
            continue
        if counter1 == 50:
            break
        counter1 += 1
        for line2 in open(user): 
            if counter2 == 50: break   
            counter2+=1        
            line2 = line2.strip().split('\t')
            if line2[0]=="UserID": continue
            if line2[0]==line[0]: continue
            
            user12List = []
            if line2[7]==line[7]:
                user12List.append(1)
            else: user12List.append(0)
            if line2[12]==line[12]:
                user12List.append(1)
            else : user12List.append(0)
            if line2[13]==line[13]:
                user12List.append(1)
            else : user12List.append(0)
            managedDiff = abs(int(line2[14] )- int(line[14]))
            gradYear2 = line2[9]
            gradYear2 = gradYear2.split('-', 1)[0]
            gradYear1 = line[9]
            gradYear1 = gradYear1.split('-', 1)[0]
            gradDiff = 100
            if (gradYear2 != "" and gradYear1 != ""):
                gradDiff = abs(int(gradYear1)-int(gradYear2))
            user12List.append(managedDiff)
            distDifferance = None
            user12List.append(gradDiff)
            distDifferance = getDistance(line[6].strip(), line2[6].strip())
            user12List.append(distDifferance)
            userBool = checkVal(line[0], line2[0], job)
            if userBool:
                tupleVal = (user12List, 1)
                userList.append(tupleVal)

            else:
                tupleVal = (user12List, 0)
                userList.append(tupleVal)
        print userList

        #examples.append((xSen.strip(), int(y)))
        # Format of each line: <output label (+1 or -1)> <input sentence>
    return userList

def evaluatePredictor(examples, predictor):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0
    for x, y in examples:
        if predictor(x) != y:
            error += 1
    return 1.0 * error / len(examples)

def outputWeights(weights, path):
    print "%d weights" % len(weights)
    out = open(path, 'w')
    for f, v in sorted(weights.items(), key=lambda (f, v) : -v):
        print >>out, '\t'.join([f, str(v)])
    out.close()

def verbosePredict(phi, y, weights, out):
    yy = 1 if dotProduct(phi, weights) >= 0 else -1
    if y:
        print >>out, 'Truth: %s, Prediction: %s [%s]' % (y, yy, 'CORRECT' if y == yy else 'WRONG')
    else:
        print >>out, 'Prediction:', yy
    for f, v in sorted(phi.items(), key=lambda (f, v) : -v * weights.get(f, 0)):
        w = weights.get(f, 0)
        print >>out, "%-30s%s * %s = %s" % (f, v, w, v * w)
    return yy

def outputErrorAnalysis(examples, featureExtractor, weights, path):
    out = open('error-analysis', 'w')
    for x, y in examples:
        print >>out, '===', x
        verbosePredict(featureExtractor(x), y, weights, out)
    out.close()

def interactivePrompt(featureExtractor, weights):
    while True:
        print '> ',
        x = sys.stdin.readline()
        if not x: break
        phi = featureExtractor(x) 
        verbosePredict(phi, None, weights, sys.stdout)

############################################################

def generateClusteringExamples(numExamples, numWordsPerTopic, numFillerWords):
    '''
    Generate artificial examples inspired by sentiment for clustering.
    Each review has a hidden sentiment (positive or negative) and a topic (plot, acting, or music).
    The actual review consists of 2 sentiment words, 4 topic words and 2 filler words, for example:

        good:1 great:1 plot1:2 plot7:1 plot9:1 filler0:1 filler10:1

    numExamples: Number of examples to generate
    numWordsPerTopic: Number of words per topic (e.g., plot0, plot1, ...)
    numFillerWords: Number of words per filler (e.g., filler0, filler1, ...)
    '''
    sentiments = [['bad', 'awful', 'worst', 'terrible'], ['good', 'great', 'fantastic', 'excellent']]
    topics = ['plot', 'acting', 'music']
    def generateExample():
        x = Counter()
        # Choose 2 sentiment words according to some sentiment
        sentimentWords = random.choice(sentiments)
        x[random.choice(sentimentWords)] += 1
        x[random.choice(sentimentWords)] += 1
        # Choose 4 topic words from a fixed topic
        topic = random.choice(topics)
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        # Choose 2 filler words
        x['filler' + str(random.randint(0, numFillerWords-1))] += 1
        return x

    random.seed(42)
    examples = [generateExample() for _ in range(numExamples)]
    return examples

def outputClusters(path, examples, centers, assignments):
    '''
    Output the clusters to the given path.
    '''
    print 'Outputting clusters to %s' % path
    out = open(path, 'w')
    for j in range(len(centers)):
        print >>out, '====== Cluster %s' % j
        print >>out, '--- Centers:'
        for k, v in sorted(centers[j].items(), key = lambda (k,v) : -v):
            if v != 0:
                print >>out, '%s\t%s' % (k, v)
        print >>out, '--- Assigned points:'
        for i, z in enumerate(assignments):
            if z == j:
                print >>out, ' '.join(examples[i].keys())
    out.close()
