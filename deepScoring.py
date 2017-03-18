# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 23:09:59 2017

@author: moazi
"""
import numpy as np
import scipy.stats as stats
import sklearn.preprocessing as pre
import itertools as itter

# get data from csv
data = np.genfromtxt('l2Scores.csv',  delimiter=',')
data = np.delete(data, (0), axis=0)  # remove labels columb
actual = np.genfromtxt('Actual.csv',  delimiter=',')
actual = np.delete(actual, (0), axis=0)
actual = actual[:, 2]  # 0 or 1 results col 2, ordered by index


# input score array and actual ordered by subj trail, output precision
def precision(index, scores, true):
    numPos = 0    # number of positive cases
    for i in true:
        if i == 1:
            numPos += 1
    scores = scores.reshape(720, 1)
    index = index.reshape(720, 1)
    true = true.reshape(720, 1)
    scores = np.concatenate((index, scores), axis=1)  # add index col
    true = np.concatenate((index, true), axis=1)  # add index col
    scores = scores[scores[:, 1].argsort()[::-1]]  # sort higest to lowest
    holdIndex = []
    for i in range(numPos):  # take indexes of the top 1 to numPos of scores
        holdIndex.append(scores[i, 0])
    correct = 0
    for i in holdIndex:  # looks up the index in the true set to see if correct
        index = np.where(true[:, 0] == i)
        y = true[index, 1]
        if y == 1:
            correct += 1
    return(correct/numPos)


# input ordered scores and col # of score, output rank col same order as input
def ranking(scores, col):
    hold_same = []
    rank = []
    count = 0
    for i in range(0, len(scores)-1):
        count += 1
        if scores[i, col] != scores[i+1, col]:
            if len(hold_same) > 0:
                hold_same.append(i+1)
                for j in hold_same:
                    rank.append(sum(hold_same)/(len(hold_same)))
                del hold_same[:]
            else:
                rank.append(i+1)
        else:
            hold_same.append(i+1)
    if len(hold_same) > 0:
        hold_same.append(720)
        for j in hold_same:
            rank.append(sum(hold_same)/(len(hold_same)))
        del hold_same[:]

    else:
        rank.append(720)

    return rank
# ---------------------------------------------Layer 0, A to E raks and scores
sort_by = data

for i in range(1, 6):
    sort_by = sort_by[sort_by[:, i].argsort()[::-1]]
    rank = np.array(ranking(sort_by, i)).reshape(720, 1)
    sort_by = np.hstack([sort_by, rank])
    sort_by = sort_by[sort_by[:, 0].argsort()]

scoresNDranks = sort_by  # array with index ranks and scores
scores = data[:, 1:6]  # just scores Ato E
ranks = scoresNDranks[:, 6:11]  # just Ranks A to E
# ----------------------------------------------------------------------------
# ----------------Layer 1 or meta data input layer----------------------------

minr = []
maxs = []
avgs = []
avgr = []
mixr = []
mixs = []
for row in scores:
    maxs.append(row[np.argmax(row)])
    avgs.append(sum(row)/len(row))

for row in ranks:
    minr.append(row[np.argmin(row)])
    avgr.append(sum(row)/len(row))


# input ranks, output avg of lowest rank in every combo of A to E
def mixGroupRank(ranks):
    shape = np.shape(np.array(ranks))  # rows [0], cols [1]
    comb_cols = []  # every combination ie) A,B,C,D,E,AB,AC,...,ABCDE
    mix_grp_r = []
    for r in range(1, shape[1]+1):
        for i in itter.combinations(range(0, shape[1]), r):
            comb_cols.append(i)
    for row in ranks:
        group_r = []
        for grp in comb_cols:
            minR = 100000  # min rank, set to high value out of range
            for i in grp:
                if row[i] < minR:
                    minR = row[i]
            group_r.append(minR)
        mix_grp_r.append(sum(group_r)/len(group_r))
    return mix_grp_r


# input scores, output average of highest score in every combo of A to E
def mixGroupScore(scores):
    shape = np.shape(np.array(scores))  # rows [0], cols [1]
    comb_cols = []  # every combination ie) A,B,C,D,E,AB,AC,...,ABCDE
    mix_grp_s = []
    for r in range(1, shape[1]+1):
        for i in itter.combinations(range(0, shape[1]), r):
            comb_cols.append(i)
    for row in scores:
        group_s = []
        for grp in comb_cols:
            minS = -1  # min rank, set to high value out of range
            for i in grp:
                if row[i] > minS:
                    minS = row[i]
            group_s.append(minS)
        mix_grp_s.append(sum(group_s)/len(group_s))
    return mix_grp_s

mixr = mixGroupRank(ranks)
mixs = mixGroupScore(scores)

l1 = np.array([scoresNDranks[:, 0], maxs, avgs, mixs, minr, avgr, mixr]).T

l1inverse_ranks = np.reciprocal(l1[:, 4:7])
l1Scores = np.concatenate((l1[:, 0:4], l1inverse_ranks), axis=1)

sort_by = l1Scores
ranks = np.array(data[:, 0]).reshape(720, 1)

for i in range(1, np.shape(sort_by)[1]):
    sort_by = sort_by[sort_by[:, i].argsort()[::-1]]
    rank = np.array(ranking(sort_by, i)).reshape(720, 1)
    sort_by = np.hstack([sort_by, rank])
    sort_by = sort_by[sort_by[:, 0].argsort()]

l1Complete = sort_by
l1Scores = sort_by[:, 1:7]
l1Ranks = sort_by[:, 7:]
# --------------------------------------------------------#Pearsons Correlation

correl = np.corrcoef(l1Scores.T)  # correl coefient matrix
avgCorrel = []
for i in correl:
    avgCorrel.append(np.absolute(sum(i)/len(i)))

w1Correl = []
for each in avgCorrel:
    w1Correl.append(each**-1)

w1Correl = w1Correl/sum(w1Correl)
# -------------------------------------------------------------#Spearmans Rho
spearr = []
for x in range(0, 6):
    temp = []
    for i in range(0, 6):
        temp.append(stats.spearmanr(l1Ranks[x], l1Ranks[i])[0])
    spearr.append(temp)

spearr = pre.minmax_scale(spearr)  # SHOULD I?
avgSpearr = []
for i in spearr:
    avgSpearr.append(sum(i)/len(i))
w1Spearr = avgSpearr/sum(avgSpearr)
# --------------------------------Hsu Cognitive Diversity ---------------
orderedL1 = []
for i in range(7, 13):
    temp = l1Complete[l1Complete[:, i].argsort()]  # rank ordered low to high
    orderedL1.append(temp[:, i-6])  # takes scores ordered by rank


def cogdiversity(array1, array2):  # ordered normalized array
    size = len(array1)
    sqr_dif = 0
    for i in range(0, size):
        sqr_dif += (array1[i]-array2[i])**2
    cog_div = ((sqr_dif)**(1/2))/size
    return cog_div

orderedL1 = np.array(np.transpose(orderedL1))
cogdiv = []
for x in range(0, 6):
    temp = []
    for i in range(0, 6):
        temp.append(cogdiversity(pre.minmax_scale(orderedL1[x]),
                                 pre.minmax_scale(orderedL1[i])))
    cogdiv.append(temp)

avgcogdiv = []
for i in cogdiv:
    avgcogdiv.append(sum(i)/len(i))
w1CogDiv = avgcogdiv/sum(avgcogdiv)  # enter the average cog div normalized
# ---------------------------------------#Rank Accuracy, Score Accuracy
avgPrecision = []
for i in l1Scores.T:
    avgPrecision.append(precision(data[:, 0], i, actual))

w1Precision = np.divide(np.array(avgPrecision), sum(avgPrecision))
# -----------------------------------------WEIGHTS 1------------------------
w1Complete = np.array([w1Correl, w1Spearr, w1CogDiv, w1Precision]).T
# ---------------------------------------------------------------------------
# ---------------------------LAYER 2-------------------------------------------
l2Scores = l1Scores.dot(w1Complete)
l2Scores_accuracy = []
for each in l2Scores.T:
    l2Scores_accuracy.append(precision(data[:, 0], each, actual))
# Accuracies
originalScoreAcc = []
for each in scores.T:
    originalScoreAcc.append(precision(data[:, 0], each, actual))
originalRankAcc = []

#print("weights: correl, spearr, cogDiv, Precision:, ", w1Complete)
print("Original \n",originalScoreAcc, originalRankAcc)
print("Layer 1 \n", avgPrecision)
print("Layer 2 \n", l2Scores_accuracy)

# ------------------------------------------------------------------------------
# ________________________DEEP Scoring___________________________________
# taking A from input layer and all from layer 2 treat as layer 1 and run again
#np.savetxt('l2Scores.csv', l2Scores, delimiter = ',')