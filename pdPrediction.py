import collections, util, math, random
import classes, scoring
import copy
import os
import glob
import re
from collections import defaultdict
import unicodedata
import random

class PredictPD():

	def __init__(self):

		self.weights = defaultdict(int)

		self.stepSize = 0.008

		self.matchdays = ["matchday" + str(i) for i in xrange(1, 7)]
		# uncomment if want to add round of 16 games to matchdays
		# self.matchdays.append("r-16")

		# # uncomment if want to add quarter final games to matchdays
		# self.matchdays.append("q-finals")

		self.folder = "passing_distributions/2014-15/"

		# init feature classes
		countAvgFile = "txt/avg_passes_count.txt"
		self.countAvgPassesFeature = classes.CountAvgPassesFeature(countAvgFile)

		squad_dir = "squads/2014-15/squad_list/"
		self.playerPosFeature = classes.PlayerPositionFeature(squad_dir)

		rankFile = "rankings/2013_14_rankings.txt"
		self.rankFeature = classes.RankingFeature(rankFile)

		self.meanDegreeFeature = classes.MeanDegreeFeature()

		self.betweenFeature = classes.BetweennessFeature()

		# this feature does not work with current files
		# self.passComplAttempFeature = classes.PassesComplAttempPerPlayerFeature()

		self.countPassesPosFeature = classes.CountPassesPerPosFeature("games_by_pos/perTeam/", "group")

		self.passComplAttempTeamFeature = classes.CountPassesComplAttempPerTeamFeature("group")

		self.teamStatsFeature = classes.TeamStatsFeature()

		# init data structures
		self.matches = defaultdict(str)

		self.totalPassesBetweenPos = defaultdict(lambda: defaultdict(int))
		self.totalPassesBetweenPlayers = defaultdict(lambda: defaultdict(int))
		self.totalPasses = defaultdict(int)

		self.teamNumToPos = defaultdict(lambda: defaultdict(str))
		self.initTeamNumToPos(squad_dir)

		self.passComplPerTeam = defaultdict(int)
		self.passAttemPerTeam = defaultdict(int)
		self.passPercPerTeam = defaultdict(float)

		self.teamStatsByMatch = defaultdict(lambda: defaultdict(list))
		self.matchesWithScores = defaultdict(str)

		self.teamPlayedWith = defaultdict(list)
		self.teamWonAgainst = defaultdict(list)

		self.teamNamesToMatchID = defaultdict(list)
		self.matchIDToScore = defaultdict(str)



	# Average pairwise error over all players in a team
	# given prediction and gold
	def evaluate(self, features, weight):
		score = self.computeScore(features, self.weights)
		loss = self.computeLoss(features, self.weights, float(weight))
		pred = self.predict(score)
		print "Score: %f, pred: %f, actual: %f" % (score, pred, weight)
		return (score, loss, pred)

	def computeLoss(self, features, weights, label):
		return (self.computeScore(features, weights) - label)**2

	# score is dot product of features & weights
	def computeScore(self, features, weights):
		score = 0.0
		for v in features:
			score += float(features[v]) * float(weights[v])
		return score

	# returns a vector
	# 2 * (phi(x) dot w - y) * phi(x)
	def computeGradientLoss(self, features, weights, label):
		scalar =  2 * self.computeScore(features, weights) - label
		mult = copy.deepcopy(features)
		for f in mult:
			mult[f] = float(mult[f])
			mult[f] *= scalar
		return mult

	# use SGD to update weights
	def updateWeights(self, features, weights, label):
		grad = self.computeGradientLoss(features, weights, label)
		for w in self.weights:
			self.weights[w] -= self.stepSize * grad[w]

	def getTeamNameFromNetwork(self, network):
		teamName = re.sub("[^-]*-", "", network, count=1)
		teamName = re.sub("-edges", "", teamName)
		teamName = re.sub("_", " ", teamName)
		return teamName

	def getTeamNameFromFile(self, teamFile):
		teamName = re.sub("-squad.*", "", teamFile)
		teamName = re.sub("_", " ", teamName)
		return teamName

	def initTeamNumToPos(self, squad_dir):
		for team in os.listdir(squad_dir):
			if re.search("-squad", team):
				path = squad_dir + team
				teamFile = open(squad_dir + team, "r")
				teamName = self.getTeamNameFromFile(team)
				for player in teamFile:
					num, name, pos = player.rstrip().split(", ")
					self.teamNumToPos[teamName][num] = pos

	def getMatchIDFromFile(self, network):
		matchID = re.sub("_.*", "", network)
		return matchID

	def getOppTeam(self, matchID, teamName):
		team1, team2 = self.matches[matchID].split("/")
		if team1 == teamName:
			return team2
		else: return team1

	def getMatchday(self, matchID):
		matchID = int(matchID)
		if matchID <= 2014322:
			return 0
		elif matchID >=2014323 and matchID <= 2014338:
			return 1
		elif matchID >= 2014339 and matchID <= 2014354:
			return 2
		elif matchID >= 2014355 and matchID <= 2014370:
			return 3
		elif matchID >= 2014371 and matchID <= 2014386:
			return 4
		elif matchID >= 2014387 and matchID <= 2014402:
			return 5
		elif matchID >= 2014403 and matchID <= 2014418:
			return 6
		elif matchID >= 2014419 and matchID <= 2014426:
			return 7
		elif matchID >= 2014427 and matchID <= 2014430:
			return 8

	def featureExtractor(self, teamName, p1, p2, matchID, matchNum, weight, score):

		# ------Calculations
		# avgPasses = self.countAvgPassesFeature.getCount(teamName, p1, p2)

		oppTeam = self.getOppTeam(matchID, teamName)
		diffInRank = self.rankFeature.isHigherInRank(teamName, oppTeam)

		features = defaultdict(float)
		# features["avgPasses"] = avgPasses

		pos1 = self.teamNumToPos[teamName][p1]
		pos2 = self.teamNumToPos[teamName][p2]

		# keep a running total of past passes between positions
		# how about a running average...
		# p_key = pos1 + "-" + pos2
		
		# --- Average passes per position, precomputed
		# avgPassesPerPos = self.countPassesPosFeature.getCount(teamName, p_key)
		# ---

		# features["avgPassesPerPos"] = avgPassesPerPos

		avgPassCompl = self.passComplAttempTeamFeature.getPCCount(teamName, matchNum)
		avgPassAttem = self.passComplAttempTeamFeature.getPACount(teamName, matchNum)
		avgPassPerc = self.passComplAttempTeamFeature.getPCPerc(teamName, matchNum)
		avgPassFail = avgPassCompl - avgPassAttem

		oppAvgPassCompl = self.passComplAttempTeamFeature.getPCCount(oppTeam, matchNum)
		oppAvgPassAttem = self.passComplAttempTeamFeature.getPACount(oppTeam, matchNum)
		oppAvgPassPerc = self.passComplAttempTeamFeature.getPCPerc(oppTeam, matchNum)
		oppAvgPassFail = oppAvgPassCompl - oppAvgPassAttem


		# for feature: won against a similar ranking team
		# define history that we are able to use, i.e. previous games
		matchday = self.getMatchday(matchID)
		history = self.teamPlayedWith[teamName][:matchday]

		if len(history) > 0:
			def computeSim(rank1, rank2):
				return (rank1**2 + rank2**2)**0.5

			# find most similar opponent in terms of rank
			oppTeamRank = self.rankFeature.getRank(oppTeam)
			simTeam = ""
			simTeamDistance = float('inf')
			rank1 = oppTeamRank
			for team in history:
				rank2 = self.rankFeature.getRank(team)
				sim = computeSim(rank1, rank2)
				if sim < simTeamDistance:
					simTeamDistance = sim
					simTeam = sim

		strip_teamName = self.strip_accents(teamName)
		strip_oppTeamName = self.strip_accents(oppTeam)
		teamStats = self.teamStatsFeature.getTeamStats(strip_teamName)
		# print "teamStats for %s during match %s are"% (teamName, matchID), teamStats
		onTarget = teamStats["on target"]
		woodwork = teamStats["woodwork"]
		blocked = teamStats["blocked"]
		yCards = teamStats["yellow cards"]
		rCards = teamStats["red cards"]
		intoThird = teamStats["into the attacking third"]
		keyArea = teamStats["into the key area"]
		penaltyArea = teamStats["into the penalty area"]
		foulsCommitted = teamStats["fouls committed"]
		foulsSuffered = teamStats["fouls suffered"]

		oppTeamStats = self.teamStatsFeature.getTeamStats(strip_oppTeamName)
		onTargetOpp = oppTeamStats["on target"]
		woodworkOpp = oppTeamStats["woodwork"]
		blockedOpp = oppTeamStats["blocked"]
		yCardsOpp = oppTeamStats["yellow cards"]
		rCardsOpp = oppTeamStats["red cards"]
		intoThirdOpp = oppTeamStats["into the attacking third"]
		keyAreaOpp = oppTeamStats["into the key area"]
		penaltyAreaOpp = oppTeamStats["into the penalty area"]
		foulsCommittedOpp = oppTeamStats["fouls committed"]
		foulsSufferedOpp = oppTeamStats["fouls suffered"]

		# -------End calculations

		# -------Features. TODO: Experiment!
		features["diffInRank"] = diffInRank
  		features["avgPassPerc"] = avgPassPerc
		features["higherPassPerc"] = 1 if avgPassPerc > oppAvgPassPerc else 0
		features["higherPassVol"] = 1 if avgPassCompl > oppAvgPassCompl else 0
		features["avgBC"] = self.betweenFeature.getAvgBetweenCentr(teamName)
		features["meanDegree"] = self.meanDegreeFeature.getMeanDegree(matchID, teamName)
		
		if len(history) > 0:
			features["wonAgainstSimTeam"] = self.teamWonAgainst[teamName][matchday]
		features["onTarget"] = onTarget
		features["woodwork"] = woodwork
		features["avgYellowCards"] = yCards
		features["avgRedCards"] = rCards
		features["blocked"] = blocked

		features["mostInThird"] = 1 if intoThird > keyArea and intoThird > penaltyArea else 0
		features["intoThird"] = intoThird / 100
		features["keyArea"] = keyArea / 100
		features["penaltyArea"] = penaltyArea / 100

		features["moreFoulsCommit"] = 1 if foulsCommitted > foulsCommittedOpp else 0
		features["moreFoulsSuff"] = 1 if foulsSuffered > foulsSufferedOpp else 0

		features["moreYellowCards"] = 1 if yCards > yCardsOpp else 0
		features["moreRedCards"] = 1 if rCards > rCardsOpp else 0
		features["moreAttackingThird"] = 1 if intoThird > intoThirdOpp else 0

		# -------End features

		return features	


	def strip_accents(self, text):
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

	def initMatches(self):
		# store match data for all games
		# match data including team + opponent team
		allGames = copy.deepcopy(self.matchdays)
		if "r-16" not in allGames:
			allGames.append("r-16")
		if "q-finals" not in allGames:
			allGames.append("q-finals")
		if "s-finals" not in allGames:
			allGames.append("s-finals")
		for matchday in allGames:
			print "Init matchday: %s" % matchday
			path = self.folder + matchday + "/networks/"
			for network in os.listdir(path):
				if re.search("-edges", network):
					edgeFile = open(path + network, "r")
					teamName = self.getTeamNameFromNetwork(network)
					matchID = self.getMatchIDFromFile(network)

					m = self.matches[matchID]
					if m == "":
						self.matches[matchID] = teamName
					else:
						team1 = self.matches[matchID]
						self.matches[matchID] += "/" + teamName
						
						team2 = teamName
						team1 = self.strip_accents(team1)
						team2 = self.strip_accents(team2)

						print "stripped team1: %s, team2: %s" % (team1, team2)
						self.teamNamesToMatchID[team1].append(matchID)
						self.teamNamesToMatchID[team2].append(matchID)
						# print teamNamesToMatchID[self.matches[matchID]]
						print "teamNamesToMatchID[%s] = %s" % (team1, self.teamNamesToMatchID[team1])
						print "teamNamesToMatchID[%s] = %s" % (team2, self.teamNamesToMatchID[team2])


		allScoresFilename = "scores/2014-15_allScores.txt"
		allScores = open(allScoresFilename, "r")
		self.allMatchesWithScores = [line.rstrip() for line in allScores]

		# for every team, store opponents in order by matchday
		teamsToNumMatches = defaultdict(int)
		for match in self.allMatchesWithScores:
			print "match was ", match
			team1, score1, score2, team2 = match.split(", ")
			team1Won = 0
			if score1 > score2:
				team1Won = 1

			team1 = self.strip_accents(team1)
			team2 = self.strip_accents(team2)

			matchNum = teamsToNumMatches[team1 + "/" + team2]
			matchID = self.teamNamesToMatchID[team1][matchNum]
			self.matchIDToScore[matchID] = match

			self.teamPlayedWith[team1].append(team2)
			self.teamPlayedWith[team2].append(team1)
			self.teamWonAgainst[team1].append(team1Won)
			self.teamWonAgainst[team2].append(abs(1 - team1Won))
			teamsToNumMatches[team1] += 1
			teamsToNumMatches[team2] += 1

	def predict(self, score):
		if score >= 0.5:
			return 1
		else: return 0

	def initTeamStats(self):
		for matchday in self.matchdays:
			path = self.folder + matchday + "/networks/"
			# iterate over games
			for network in os.listdir(path):
				if re.search("-team", network):
					teamName = self.getTeamNameFromNetwork(network)
					teamName = re.sub("-team", "", teamName)
					matchID = self.getMatchIDFromFile(network)

					stats_file = open(path + network, "r")
					for line in stats_file:
						stats = line.rstrip().split(", ")
					
					self.teamStatsByMatch[teamName][matchID] = stats

	# Training
	# 	have features calculate numbers based on data
	# 	learn weights for features via supervised data (group stage games) and SGD/EM
	def train(self):
		# iterate over matchdays, predicting match outcomes, performing SGD

		num_iter = 2
		self.initMatches()
		self.initTeamStats()
		
		pos = ["GK", "STR", "DEF", "MID"]
		allPosCombos = [pos1 + "-" + pos2 for pos1 in pos for pos2 in pos]

		for i in xrange(num_iter):
			totalCorrect = 0
			totalEx = 0
			avgLoss = 0
			print "Iteration %s" % i
			print "------------"
			for w in self.weights:
				print "weights[%s] = %f" % (w, float(self.weights[w]))
			# iterate over matchdays -- hold out on some matchdays
			matchNum = 0

			# # try shuffling matchdays
			# random.shuffle(self.matchdays)

			allGames = []

			for matchday in self.matchdays:
				print "On " + matchday
				path = self.folder + matchday + "/networks/"
				# iterate over games
				for network in os.listdir(path):
					if re.search("-edges", network):
						# passesBetweenPos = defaultdict(lambda: defaultdict(int))
						allGames.append((path, network))

			# try shuffling games
			# random.shuffle(allGames)

			for game in allGames:
				path, network = game
				edgeFile = open(path + network, "r")

				teamName = self.getTeamNameFromNetwork(network)
				matchID = self.getMatchIDFromFile(network)
				# print "team: %s" % teamName
				matchdayNum = self.getMatchday(matchID)
				str_teamName = self.strip_accents(teamName)
				# score = self.matchIDToScore[matchID]
				
				# print "%s is matchday %s" % (matchID, matchdayNum)
				# print "history for %s is" % teamName, self.teamWonAgainst[teamName]
				didWin = self.teamWonAgainst[str_teamName][matchdayNum]
				print "team: %s" % teamName
				# train on each team instead of each pass
				features = self.featureExtractor(teamName, "", "", matchID, matchNum, 0, didWin)
				score, loss, pred = self.evaluate(features, didWin)

				self.updateWeights(features, self.weights, int(didWin))
				totalCorrect += 1 if int(pred) == int(didWin) else 0
				avgLoss += loss
				totalEx += 1
				# for players in edgeFile:
				# 	p1, p2, weight = players.rstrip().split("\t")

				# 	features = self.featureExtractor(teamName, p1, p2, matchID, matchNum, weight, didWin)

				# 	# for f in features:
				# 	# 	print "features[%s] = %f" % (f, float(features[f]))
				# 	# for w in self.weights:
				# 	# 	print "weights[%s] = %f" % (w, float(self.weights[w]))

				# 	score, loss, pred = self.evaluate(features, didWin)
 			# 		self.updateWeights(features, self.weights, int(didWin))
 			# 		totalCorrect += 1 if int(pred) == int(didWin) else 0
 			# 		avgLoss += loss
				# 	totalEx += 1
				matchNum += 1
			print "Total correct: %f" % (totalCorrect / float(totalEx))
			print "Average loss: %f" % (avgLoss / totalEx)

	# Testing
	#	Predict, then compare with dev/test set (r-16 games)
	def test(self):
		# sum up average error

		print "Testing"
		print "-------"
		avgLoss = 0
		totalEx = 0
		matchNum = 0

		# uncomment below if testing on round of 16
		matchday = "r-16"

		# uncomment below if testing on quarter finals
		# matchday = "q-finals"

		# uncommend below if testing on semi-finals
		# matchday = "s-finals"
		print "On " + matchday
		path = self.folder + matchday + "/networks/"
		# iterate over games
		totalCorrect = 0
		for network in os.listdir(path):

			if re.search("-edges", network):
				edgeFile = open(path + network, "r")

				# predEdgeFile = open("predicted/pred-" + network, "w+")

				teamName = self.getTeamNameFromNetwork(network)
				matchID = self.getMatchIDFromFile(network)
				matchdayNum = self.getMatchday(matchID)
				str_teamName = self.strip_accents(teamName)
				
				didWin = self.teamWonAgainst[str_teamName][matchdayNum]
				print "team: %s" % teamName

				didWin = self.teamWonAgainst[str_teamName][matchdayNum]
				# train on each team instead of each pass
				features = self.featureExtractor(teamName, "", "", matchID, matchNum, 0, didWin)
				score, loss, pred = self.evaluate(features, didWin)
				self.updateWeights(features, self.weights, int(didWin))
				totalCorrect += 1 if int(pred) == int(didWin) else 0
				avgLoss += loss
				totalEx += 1
				# for players in edgeFile:
				# 	p1, p2, weight = players.rstrip().split("\t")
				# 	print "p1: %s, p2: %s, weight: %f" % (p1, p2, float(weight))

				# 	features = self.featureExtractor(teamName, p1, p2, matchID, matchNum, weight, didWin)

				# 	for f in features:
				# 		print "features[%s] = %f" % (f, float(features[f]))
				# 	for w in self.weights:
				# 		print "weights[%s] = %f" % (w, float(self.weights[w]))

				# 	score, loss, pred = self.evaluate(features, didWin)

				# 	# print out predicted so can visually compare to actual
				# 	# predEdgeFile.write(p1 + "\t" + p2 + "\t" + str(score) + "\n")

				# 	avgLoss += loss
				# 	totalCorrect += 1 if int(pred) == int(didWin) else 0
				# 	totalEx += 1
				matchNum += 1
		print "\n---Final weights---"
		for w in self.weights:
			print "%s = %s" % (w, self.weights[w])
		print "-------------------"
		print "Average loss: %f" % (avgLoss / totalEx)
		print "Total correct: %f" % (totalCorrect / float(totalEx))
		print "Total average loss: %f" % avgLoss
		print "Total examples (passes): %f" % totalEx

predsys = PredictPD()
predsys.train()
predsys.test()
