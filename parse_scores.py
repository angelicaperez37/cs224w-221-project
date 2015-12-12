from collections import defaultdict
import os
import re
import unicodedata

allGames = ["matchday" + str(i) for i in xrange(1, 7)]

allGames.append("r-16")
allGames.append("q-finals")
allGames.append("s-finals")

folder = "passing_distributions/2014-15/"

matches = defaultdict(str)
teamNamesToMatchID = defaultdict(list)

def getTeamNameFromNetwork(network):
	teamName = re.sub("[^-]*-", "", network, count=1)
	teamName = re.sub("-edges", "", teamName)
	teamName = re.sub("_", " ", teamName)
	return teamName

def getMatchIDFromFile(network):
	matchID = re.sub("_.*", "", network)
	return matchID

def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

for matchday in allGames:
	path = folder + matchday + "/networks/"
	for network in os.listdir(path):
		if re.search("-edges", network):
			edgeFile = open(path + network, "r")
			teamName = getTeamNameFromNetwork(network)
			matchID = getMatchIDFromFile(network)

			m = matches[matchID]
			if m == "":
				matches[matchID] = teamName
			else:
				team1 = matches[matchID]
				matches[matchID] += "/" + teamName
				
				team2 = teamName
				team1 = strip_accents(team1)
				team2 = strip_accents(team2)

				teamNamesToMatchID[team1].append(matchID)
				teamNamesToMatchID[team2].append(matchID)

# for team in teamNamesToMatchID:
# 	print "%s => %s" % (team, teamNamesToMatchID[team])

allScoresFilename = "scores/2014-15_allScores.txt"
allScores = open(allScoresFilename, "r")
allMatchesWithScores = [line.rstrip() for line in allScores]

teamsToNumMatches = defaultdict(int)
for match in allMatchesWithScores:
	team1, score1, score2, team2 = match.split(", ")
	if score1 > score2:
		outcome1 = "win"
		outcome2 = "loss"
	elif score1 < score2:
		outcome1 = "loss"
		outcome2 = "win"
	else:
		outcome1 = "draw"
		outcome2 = "draw"

	str_team1 = strip_accents(team1)
	str_team2 = strip_accents(team2)

	matchNum = teamsToNumMatches[str_team1]
	# print "matchNum is: %s" % matchNum
	matchID = teamNamesToMatchID[str_team1][matchNum]
	print "%s, %s, %s" % (matchID, team1, outcome1)
	print "%s, %s, %s" % (matchID, team2, outcome2)

	teamsToNumMatches[str_team1] += 1
	teamsToNumMatches[str_team2] += 1
