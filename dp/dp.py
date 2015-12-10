import codecs
from itertools import groupby
from operator import itemgetter

from numpy import arange

RANK_FILE = '../rankings.txt'

GK_FILE = '../fantasy_player_data/positions/goalkeepers'
DEF_FILE = '../fantasy_player_data/positions/defenders'
MID_FILE = '../fantasy_player_data/positions/midfielders'
STR_FILE = '../fantasy_player_data/positions/forwards'

COST_INDEX = 2
SCORE_INDEX = 3

# Your squad must consist of:
#
#     Two goalkeepers
#     Five defenders
#     Five midfielders
#     Three forwards
NUM_GK = 2
NUM_DEF = 5
NUM_MID = 5
NUM_STR = 3

BUDGET = 100.0
STEP = 0.5

teams = {}
with codecs.open(RANK_FILE, encoding='utf-8') as f:
    for line in f:
        team, rank = line.encode('ascii', 'xmlcharrefreplace').split('\t')
        rank = int(rank)
        teams[team] = rank

goalkeepers = []
with codecs.open(GK_FILE, encoding='utf-8') as f:
    for line in f:
        name, team, pos, cost = line.encode('ascii', 'xmlcharrefreplace').split(', ')
        assert pos == 'GK'
        cost = float(cost)
        score = 1.0 / teams[team]  # TODO: scoring function
        goalkeepers.append((name, team, cost, score))

# reduce goalkeepers
_reduced_goalkeepers = [max(g, key=itemgetter(SCORE_INDEX))
                        for k, g in groupby(goalkeepers, itemgetter(COST_INDEX))]
max_score = 0
reduced_goalkeepers = []
for goalkeeper in reversed(_reduced_goalkeepers):
    if goalkeeper[SCORE_INDEX] > max_score:
        max_score = goalkeeper[SCORE_INDEX]
        reduced_goalkeepers.append(goalkeeper)

defenders = []
with codecs.open(DEF_FILE, encoding='utf-8') as f:
    for line in f:
        name, team, pos, cost = line.encode('ascii', 'xmlcharrefreplace').split(', ')
        assert pos == 'DEF'
        cost = float(cost)
        score = 1.0 / teams[team]  # TODO: scoring function
        defenders.append((name, team, cost, score))

# reduce defenders
_reduced_defenders = [max(g, key=itemgetter(SCORE_INDEX))
                      for k, g in groupby(defenders, itemgetter(COST_INDEX))]
max_score = 0
reduced_defenders = []
for defender in reversed(_reduced_defenders):
    if defender[SCORE_INDEX] > max_score:
        max_score = defender[SCORE_INDEX]
        reduced_defenders.append(defender)

midfielders = []
with codecs.open(MID_FILE, encoding='utf-8') as f:
    for line in f:
        name, team, pos, cost = line.encode('ascii', 'xmlcharrefreplace').split(', ')
        assert pos == 'MID'
        cost = float(cost)
        score = 1.0 / teams[team]  # TODO: scoring function
        midfielders.append((name, team, cost, score))

# reduce midfielders
_reduced_midfielders = [max(g, key=itemgetter(SCORE_INDEX))
                        for k, g in groupby(midfielders, itemgetter(COST_INDEX))]
max_score = 0
reduced_midfielders = []
for midfielder in reversed(_reduced_midfielders):
    if midfielder[SCORE_INDEX] > max_score:
        max_score = midfielder[SCORE_INDEX]
        reduced_midfielders.append(midfielder)

forwards = []
with codecs.open(STR_FILE, encoding='utf-8') as f:
    for line in f:
        name, team, pos, cost = line.encode('ascii', 'xmlcharrefreplace').split(', ')
        assert pos == 'STR'
        cost = float(cost)
        score = 1.0 / teams[team]  # TODO: scoring function
        forwards.append((name, team, cost, score))

# reduce forwards
_reduced_forwards = [max(g, key=itemgetter(SCORE_INDEX))
                     for k, g in groupby(forwards, itemgetter(COST_INDEX))]
max_score = 0
reduced_forwards = []
for forward in reversed(_reduced_forwards):
    if forward[SCORE_INDEX] > max_score:
        max_score = forward[SCORE_INDEX]
        reduced_forwards.append(forward)

# goalkeepers knapsack problem
gk_m = {}
min_cost = reduced_goalkeepers[0][COST_INDEX]
for i in xrange(1, NUM_GK + 1):
    for j in arange(0.0, min_cost * i, STEP):
        gk_m[i, j] = float('-inf'), [None] * i  # unobtainable
costs = [x[COST_INDEX] for x in reduced_goalkeepers] + [BUDGET + STEP]
for i, (start, stop) in enumerate(zip(costs, costs[1:])):
    goalkeeper = reduced_goalkeepers[i]
    for j in arange(start, stop, STEP):
        gk_m[1, j] = goalkeeper[SCORE_INDEX], [goalkeeper]
for i in xrange(2, NUM_GK + 1):
    for j in arange(min_cost * i, BUDGET + STEP, STEP):
        gk_m[i, j] = float('-inf'), [None] * i
        for goalkeeper in sorted(reduced_goalkeepers, key=itemgetter(SCORE_INDEX), reverse=True):
            if j < goalkeeper[COST_INDEX]:
                break
            prev = gk_m[i - 1, j - goalkeeper[COST_INDEX]]
            if gk_m[i, j][0] < prev[0] + goalkeeper[SCORE_INDEX]:
                gk_m[i, j] = prev[0] + goalkeeper[SCORE_INDEX], prev[1] + [goalkeeper]

# defenders knapsack problem
def_m = {}
min_cost = reduced_defenders[0][COST_INDEX]
for i in xrange(1, NUM_DEF + 1):
    for j in arange(0.0, min_cost * i, STEP):
        def_m[i, j] = float('-inf'), [None] * i  # unobtainable
costs = [x[COST_INDEX] for x in reduced_defenders] + [BUDGET + STEP]
for i, (start, stop) in enumerate(zip(costs, costs[1:])):
    defender = reduced_defenders[i]
    for j in arange(start, stop, STEP):
        def_m[1, j] = defender[SCORE_INDEX], [defender]
for i in xrange(2, NUM_DEF + 1):
    for j in arange(min_cost * i, BUDGET + STEP, STEP):
        def_m[i, j] = float('-inf'), [None] * i
        for defender in sorted(reduced_defenders, key=itemgetter(SCORE_INDEX), reverse=True):
            if j < defender[COST_INDEX]:
                break
            prev = def_m[i - 1, j - defender[COST_INDEX]]
            if def_m[i, j][0] < prev[0] + defender[SCORE_INDEX]:
                def_m[i, j] = prev[0] + defender[SCORE_INDEX], prev[1] + [defender]

# midfielders knapsack problem
mid_m = {}
min_cost = reduced_midfielders[0][COST_INDEX]
for i in xrange(1, NUM_MID + 1):
    for j in arange(0.0, min_cost * i, STEP):
        mid_m[i, j] = float('-inf'), [None] * i  # unobtainable
costs = [x[COST_INDEX] for x in reduced_midfielders] + [BUDGET + STEP]
for i, (start, stop) in enumerate(zip(costs, costs[1:])):
    midfielder = reduced_midfielders[i]
    for j in arange(start, stop, STEP):
        mid_m[1, j] = midfielder[SCORE_INDEX], [midfielder]
for i in xrange(2, NUM_MID + 1):
    for j in arange(min_cost * i, BUDGET + STEP, STEP):
        mid_m[i, j] = float('-inf'), [None] * i
        for midfielder in sorted(reduced_midfielders, key=itemgetter(SCORE_INDEX), reverse=True):
            if j < midfielder[COST_INDEX]:
                break
            prev = mid_m[i - 1, j - midfielder[COST_INDEX]]
            if mid_m[i, j][0] < prev[0] + midfielder[SCORE_INDEX]:
                mid_m[i, j] = prev[0] + midfielder[SCORE_INDEX], prev[1] + [midfielder]

# forwards knapsack problem
str_m = {}
min_cost = reduced_forwards[0][COST_INDEX]
for i in xrange(1, NUM_STR + 1):
    for j in arange(0.0, min_cost * i, STEP):
        str_m[i, j] = float('-inf'), [None] * i  # unobtainable
costs = [x[COST_INDEX] for x in reduced_forwards] + [BUDGET + STEP]
for i, (start, stop) in enumerate(zip(costs, costs[1:])):
    forward = reduced_forwards[i]
    for j in arange(start, stop, STEP):
        str_m[1, j] = forward[SCORE_INDEX], [forward]
for i in xrange(2, NUM_STR + 1):
    for j in arange(min_cost * i, BUDGET + STEP, STEP):
        str_m[i, j] = float('-inf'), [None] * i
        for forward in sorted(reduced_forwards, key=itemgetter(SCORE_INDEX), reverse=True):
            if j < forward[COST_INDEX]:
                break
            prev = str_m[i - 1, j - forward[COST_INDEX]]
            if str_m[i, j][0] < prev[0] + forward[SCORE_INDEX]:
                str_m[i, j] = prev[0] + forward[SCORE_INDEX], prev[1] + [forward]
