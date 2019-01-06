import numpy as np


#Create a class to store the score and the backpointer value
class Viterbi:
    score = -np.inf
    backPointer = 0

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]


    #Create a LxM array of viterbi classes to store the score and the backpointer path
    viterbiMatrix = [[Viterbi() for j in range(N)] for k in range(L)]

    #Initializing the first column's score using start scores
    for j in xrange(L):
        viterbiMatrix[j][0].score = start_scores[j] + emission_scores[0][j]

    # For all the rows of current column check the best possible score for transitioning from previous row to current row and store it
    # Also, store the row index of the previous column which gave the best score in the backpointer
    for i in xrange(1,N):
        for j in xrange(L):
            for k in xrange(L):
                tempValue = viterbiMatrix[k][i-1].score + trans_scores[k][j]
                if(tempValue > viterbiMatrix[j][i].score):
                    viterbiMatrix[j][i].score = tempValue
                    viterbiMatrix[j][i].backPointer = k
            viterbiMatrix[j][i].score += emission_scores[i][j]
    
    # To the last column we add end_scores and check which row has the maximum score.
    # We store the score and the row index
    maximumScore = -np.inf
    maximumScoreIndex= 0
    for j in xrange(L):
        score = end_scores[j] + viterbiMatrix[j][N-1].score
        if(score > maximumScore):
            maximumScore = score
            maximumScoreIndex = j

    # Once we get the row index of last column which gave the best score
    # We start backtracking from the last column using the backpointer matrix to the path of best score
    bestSequence = []
    bestSequence.append(maximumScoreIndex)
    for i in range(N-1,0,-1):
        prevMaximumIndex = bestSequence[-1]
        bestSequence.append(viterbiMatrix[prevMaximumIndex][i].backPointer)


    # Return the maximum score and path from start to last column.
    #print "Computed: ", bestSequence[::-1],maxScore
    return (maximumScore, bestSequence[::-1])
