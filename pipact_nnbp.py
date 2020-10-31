import sys, math, random
#######FINAL VERSION######
####### LAYER STUFF ####
# command line: python3 pipact_nnbp.py pipact_data.txt
fileContents = list(line.strip() for line in open(sys.argv[1], 'r'))
# Data has been multiplied by .01 to make it easier to work with small weights
# Humdity * .01 Temperature * .01 => Mean RSSI value * .01
numinputs = 3 # humid, temp, bias
layercounts = [numinputs, 2, 1, 1]
print('Layer counts: ' + ' '.join(str(num) for num in layercounts))
########################

def Transfer(x): # logistic function
    if math.isnan(x): exit()
    if x < -700: return 0
    if x > 700: return 1 # try catch would be better
    if 1/(1 + math.exp(-x)) == math.inf:
        return 1e+300
    return 1/(1 + math.exp(-x))

def derivativeTransfer(y): #derivative of logistic function
    return y * (1 - y)

def cost(t, y): #cost function
    return (t-y)**2 * .5

def dotproduct(list1, list2): # dot product
    return sum([list1[i]*list2[i] for i in range(len(list1))])

def layer1BP(t, y, x): # equation for layer one of Back propagation
    '''print('layer 1:')
    print('error: ' + str((t - y)), 'x: ' + str(x))
    print('partial: ' + str((t - y) * x))
    print(' ')'''
    return (t - y) * x

def layer2BP(t, y1, y2, x, w): #equation for layer two of back prop
    '''print('layer 2:')
    print('error: ' + str((t - y1) * derivativeTransfer(y2) * w), 'x: ' + str(x))
    print('partial: ' + str((t - y1) * x * w * derivativeTransfer(y2)))
    print(' ')'''
    return (t - y1) * x * w * derivativeTransfer(y2)

def layer3BP(t, y1, y2, y3, w1, w2list, x): #equation for layer three of back prop
    '''print('layer 3:')
    print('error: ' + str((t - y1) * w1 * w2list * derivativeTransfer(y2) * derivativeTransfer(y3)), 'x: ' + str(x))
    print('partial: ' + str((t - y1) * w1 * w2list * x * derivativeTransfer(y2) * derivativeTransfer(y3)))
    print(' ')'''
    return (t - y1) * w1 * w2list * x * derivativeTransfer(y2) * derivativeTransfer(y3)

def makeNN(inputoutput, weights): # this does the feed forward network after weights are made
    layers = [inputoutput[0]]
    currentlayer = layers[0]
    for w in weights:
        newlayer = []
        if w == weights[len(weights) - 1]:
            newlayer = [w[i]*currentlayer[i] for i in range(len(w))]
        else:
            for n in range(0, len(w), len(currentlayer)):
                newlayer.append(Transfer(dotproduct(w[n:n + len(currentlayer)], currentlayer)))
        currentlayer = newlayer
        layers.append(newlayer)
    return layers

def putInWeightAdjustmentList(layers, weights, t, inputs, weightsAdjustinglist): # adds up all the differences in weights
    #print(layers)
    #print(weights)
    weightsAdjustinglist[weightcount - 1].append(layer1BP(t, layers[3][0], layers[2][0])) #OK
    weightsAdjustinglist[weightcount - 3].append(layer2BP(t, layers[3][0], layers[2][0], layers[1][0], weights[2][0]))
    weightsAdjustinglist[weightcount - 2].append(layer2BP(t, layers[3][0], layers[2][0], layers[1][1], weights[2][0]))
    for otherk in range(2):
        for k in range(numinputs):
            weightsAdjustinglist[numinputs * otherk + k].append(layer3BP(t, layers[3][0], layers[2][0], layers[1][otherk], weights[2][0], weights[1][otherk], inputs[k]))
    return weightsAdjustinglist

def makeRandomWeights(): #makes random weights (wow!)
    return [[random.random() * 2 - 1 for _ in range(layercounts[n] * layercounts[n + 1])] for n in
                       range(len(layercounts) - 1)]

def adjustWeights(weights, weightsAdjustinglist, alpha): # applies the adjustments to the weights
    tempw = weights[0] + weights[1] + weights[2]
    for k in range(len(weightsAdjustinglist)):
        tempw[k] = (tempw[k] + sum(weightsAdjustinglist[k]) * alpha)
    return [tempw[:2 * numinputs], tempw[2 * numinputs: 2 * numinputs + 2], [tempw[-1]]]



def processAllExamples(inputsAndOutputs, weights, weightsAdjustinglist): # uses the examples to find the adjustments
    costlist = []
    for inputAndOutput in inputsAndOutputs:
        networkResults = makeNN(inputAndOutput, weights)
        #print('NETWORK')
        #print(networkResults)
        #print(' ')
        putInWeightAdjustmentList(networkResults, weights, inputAndOutput[1], inputAndOutput[0], weightsAdjustinglist)
        weights = adjustWeights(weights, weightsAdjustinglist, .1)

        #print("MY WEIGHTS:")
        #printWeights(weights)
        weightsAdjustinglist = [[] for _ in range(weightcount)]
        costlist.append(cost(networkResults[3][0], inputAndOutput[1]))
    return weights, sum(costlist)

def printWeights(weights):
    print(str(weights[0]) + '\n' + str(weights[1]) + '\n' + str(weights[2]))

def tests_are_okay(testing_data, weights): # this is almost hte same as process all examples but it uses the entire set of dataand doesn't deal with weights
    mycost = 0
    for example in testing_data:
        networkResults = makeNN(example, weights)
        print(example, ' => ', networkResults[3][0])
        mycost += cost(networkResults[3][0], example[1])
    print("TEST COST: ", mycost)
    return mycost < .001




oldcost = 0
epochsSinceImprovement = 0
inputsOutputsSet = [([float(inp.replace(' ', '')) for inp in fileContents[x].split('=>')[0].split(' ') if inp != ''] + [1], float(fileContents[x].split('=>')[1].replace(' ', ''))) for x in range(len(fileContents))]
testingInputOutput = [([.6, .241, 1], -.719), ([.8, .25, 1], -.737), ([1, .257, 1], -.713)] + inputsOutputsSet #entire data set collected from experiment
weightslist = makeRandomWeights()
printWeights((weightslist))
weightcount = numinputs*2 + 3

for epoch in range(1000000):
    #print(' ')
    weightslist, currentcost = processAllExamples(inputsOutputsSet, weightslist, [[] for _ in range(weightcount)])
    epochsSinceImprovement += 1
    #print(' ')
    if not epoch:
        oldcost = currentcost
        continue
    if currentcost < oldcost*.9: # if better than 5-10 percentage
        oldcost = currentcost
        epochsSinceImprovement = 0
        print(currentcost)
        printWeights(weightslist)
        if tests_are_okay(testingInputOutput, weightslist): # another batch of data to avoid over fitting
            exit()
    if epochsSinceImprovement > 100000: #if the weights aren't moving forward a reset may be necessary
        print('started over')
        weightslist = makeRandomWeights()
        epochsSinceImprovement = 0



