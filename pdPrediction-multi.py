import collections, util, math, random
import classes, scoring
import copy
import os
import glob
import re
from collections import defaultdict
import unicodedata
import random
import sys
import ast

NUM_FEATURES = 21

class PredictPD():

	def __init__(self):

		# self.weights = defaultdict(int)
		self.loseWeights = defaultdict(int)
		self.winWeights = defaultdict(int)
		self.drawWeights = defaultdict(int)
		self.nameToWeights = {"loss" : self.loseWeights, "win" : self.winWeights,\
		"draw": self.drawWeights}

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



	# Pick the best score from [win, loss, draw]
	def evaluate(self, features, outcome):
		scoreWin = self.computeScore(features, self.winWeights)
		# lossWin = self.computeLoss(features, self.winWeights, float(weight))
		# predWin = self.predict(score)

		scoreLoss = self.computeScore(features, self.loseWeights)
		# lossLoss = self.computeLoss(features, self.loseWeights, float(weight))
		# predLoss = self.predict(score)

		scoreDraw = self.computeScore(features, self.drawWeights)
		# lossDraw = self.computeLoss(features, self.drawWeights, float(weight))
		# predDraw = self.predict(score)

		print "Win score: %f, loss score: %f, draw score: %f" % (scoreWin, scoreLoss, scoreDraw)

		if scoreWin > scoreLoss and scoreWin > scoreDraw:
			score, pred =  scoreWin,  "win"
		elif scoreLoss > scoreWin and scoreLoss > scoreDraw:
			score, pred =  scoreLoss, "loss"
		else:
			score, pred =  scoreDraw, "draw"

		print "Score: %f, pred: %s, actual: %s" % (score, pred, outcome)
		# return (score, loss, pred)
		return (score, pred)

	# hinge loss
	def computeLoss(self, features, weights, label):
		correctWeights = self.nameToWeights[label]
		maxLoss = float('-inf')
		for label in self.nameToWeights:
			if label != correctLabel:
				loss -= self.computeScore(features, correctWeights)
				loss += self.computeScore(features, self.nameToWeights[label])
				loss += 1
				if loss > maxLoss:
					maxLoss = loss

		if maxLoss < 0: maxLoss = 0

		return maxLoss

	# score is dot product of features & weights
	def computeScore(self, features, weights):
		score = 0.0
		for v in features:
			score += float(features[v]) * float(weights[v])
		return score

	# returns a vector
	# 2 * (phi(x) dot w - y) * phi(x)
	def computeGradientLoss(self, features, weights, correctLabel):
		# compute score of correct
		alpha = 0.5
		correctWeights = self.nameToWeights[correctLabel]
		maxScore = float('-inf')
		for label in self.nameToWeights:
			if label != correctLabel:
				score = self.computeScore(features, self.nameToWeights[label])
				if score > maxScore:
					maxScore = score

		correctScore = self.computeScore(features, correctWeights)
		if correctScore > 1 + maxScore:
			scalar = 0
		elif correctScore < 1 + maxScore:
			scalar = 1
		else:
			scalar = alpha

		mult = copy.deepcopy(features)
		for f in mult:
			mult[f] = float(mult[f])
			mult[f] *= scalar
		return mult

	# use SGD to update weights
	def updateWeights(self, features, label):
		weights = self.nameToWeights[label]
		grad = self.computeGradientLoss(features, weights, label)
		if label == "win":
			for w in self.winWeights:
				self.winWeights[w] += self.stepSize * grad[w]
		elif label == "loss":
			for w in self.loseWeights:
				self.loseWeights[w] += self.stepSize * grad[w]
		else:
			for w in self.drawWeights:
				self.drawWeights[w] += self.stepSize * grad[w]

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

	def featureExtractor(self, teamName, matchID, matchNum, weight):

		# ------Calculations
		# avgPasses = self.countAvgPassesFeature.getCount(teamName, p1, p2)
		matchNum = self.getMatchday(matchID)
		oppTeam = self.getOppTeam(matchID, teamName)
		diffInRank = self.rankFeature.isHigherInRank(teamName, oppTeam)

		features = defaultdict(float)

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
                if len(sys.argv) == 1:
                    flags = (1 << NUM_FEATURES) - 1  # all features enabled
                else:
                    try:
                        flags = int(sys.argv[1])
                    except ValueError:
                        flag_list = ast.literal_eval(sys.argv[1])
                        flags = 0
                        for flag in flag_list:
                            flags |= 1 << flag

		if flags & 1 << 0: features["diffInRank"] = diffInRank
  		if flags & 1 << 1: features["avgPassPerc"] = avgPassPerc
		if flags & 1 << 2: features["higherPassPerc"] = 1 if avgPassPerc > oppAvgPassPerc else 0
		if flags & 1 << 3: features["higherPassVol"] = 1 if avgPassCompl > oppAvgPassCompl else 0
		if flags & 1 << 4: features["avgBC"] = self.betweenFeature.getAvgBetweenCentr(teamName)
		if flags & 1 << 5: features["meanDegree"] = self.meanDegreeFeature.getMeanDegree(matchID, teamName)
		if flags & 1 << 6: 
		    if len(history) > 0:
			features["wonAgainstSimTeam"] = self.teamWonAgainst[teamName][matchday]
		if flags & 1 << 7: features["onTarget"] = onTarget
		if flags & 1 << 8: features["woodwork"] = woodwork
		if flags & 1 << 9: features["avgYellowCards"] = yCards
		if flags & 1 << 10: features["avgRedCards"] = rCards
		if flags & 1 << 11: features["blocked"] = blocked

		if flags & 1 << 12: features["mostInThird"] = 1 if intoThird > keyArea and intoThird > penaltyArea else 0
		if flags & 1 << 13: features["intoThird"] = intoThird / 100
		if flags & 1 << 14: features["keyArea"] = keyArea / 100
		if flags & 1 << 15: features["penaltyArea"] = penaltyArea / 100

		if flags & 1 << 16: features["moreFoulsCommit"] = 1 if foulsCommitted > foulsCommittedOpp else 0
		if flags & 1 << 17: features["moreFoulsSuff"] = 1 if foulsSuffered > foulsSufferedOpp else 0

		if flags & 1 << 18: features["moreYellowCards"] = 1 if yCards > yCardsOpp else 0
		if flags & 1 << 19: features["moreRedCards"] = 1 if rCards > rCardsOpp else 0
		if flags & 1 << 20: features["moreAttackingThird"] = 1 if intoThird > intoThirdOpp else 0

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

						self.teamNamesToMatchID[team1].append(matchID)
						self.teamNamesToMatchID[team2].append(matchID)


		allScoresFilename = "scores/2014-15_allScores.txt"
		allScores = open(allScoresFilename, "r")
		self.allMatchesWithScores = [line.rstrip() for line in allScores]

		# for every team, store opponents in order by matchday
		teamsToNumMatches = defaultdict(int)
		for match in self.allMatchesWithScores:
			team1, score1, score2, team2 = match.split(", ")
			team1Won = 0
			if score1 > score2:
				team1Won = 1

			team1 = self.strip_accents(team1)
			team2 = self.strip_accents(team2)

			matchNum = teamsToNumMatches[team1]
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

	def printWeights(self):
		print "----Win weights----"
		for w in self.winWeights:
			print "weights[%s] = %f" % (w, float(self.winWeights[w]))
		print "----Lose weights----"
		for w in self.loseWeights:
			print "weights[%s] = %f" % (w, float(self.loseWeights[w]))
		print "----Draw weights----"
		for w in self.drawWeights:
			print "weights[%s] = %f" % (w, float(self.drawWeights[w]))

	# Training
	# 	have features calculate numbers based on data
	# 	learn weights for features via supervised data (group stage games) and SGD
	def train(self):
		# iterate over matchdays, predicting match outcomes, performing SGD

		num_iter = 10
		self.initMatches()
		self.initTeamStats()
		
		pos = ["GK", "STR", "DEF", "MID"]
		allPosCombos = [pos1 + "-" + pos2 for pos1 in pos for pos2 in pos]

		for i in xrange(num_iter):
			totalCorrect = 0
			totalEx = 0
			avgLoss = 0

			totalWin = 0
			totalLoss = 0
			totalDraw = 0

			correctWin = 0
			correctLoss = 0
			correctDraw = 0

			print "Iteration %s" % i
			print "------------"
			self.printWeights()

			groupGamesFilename = "data-groupGames"
			groupGames = open(groupGamesFilename, "r")
			matchNum = 0
			for match in groupGames:
				print "match is: %s" % match.rstrip()
				matchID, team, outcome = match.rstrip().split(", ")

				features = self.featureExtractor(team, matchID, matchNum, 0)
				score, pred = self.evaluate(features, outcome)

				self.updateWeights(features, outcome)
				totalCorrect += 1 if pred == outcome else 0
				if outcome == "draw":
					totalDraw += 1
					if pred == outcome:
						correctDraw += 1
				elif outcome == "win":
					totalWin += 1
					if pred == outcome:
						correctWin += 1
				else:
					totalLoss += 1
					if pred == outcome:
						correctLoss += 1

				totalEx += 1
				matchNum += 1
			print "Total correct: %f" % (totalCorrect / float(totalEx))
			print "Total correct wins: %f" % (correctWin)
			print "Total wins: %f" % (totalWin)
			print "Total correct losses: %f" % (correctLoss)
			print "Total losses: %f" % (totalLoss)
			print "Total correct draws: %f" % (correctDraw)
			print "Total draws: %f" % (totalDraw)
			print "Average loss: %f" % (avgLoss / totalEx)

		self.printWeights()

	# Testing
	def test(self):

		print "Testing"
		print "-------"

		matchNum = 0

		# iterate over games

		totalCorrect = 0
		totalEx = 0

		totalWin = 0
		totalLoss = 0
		totalDraw = 0

		correctWin = 0
		correctLoss = 0
		correctDraw = 0

		r16Filename = "data-r16Games"
		r16Games = open(r16Filename, "r")
		matchNum = 0
		for match in r16Games:
			print "match is: %s" % match.rstrip()
			matchID, team, outcome = match.rstrip().split(", ")

			features = self.featureExtractor(team, matchID, matchNum, 0)
			score, pred = self.evaluate(features, outcome)

			self.updateWeights(features, outcome)
			totalCorrect += 1 if pred == outcome else 0
			if outcome == "draw":
				totalDraw += 1
				if pred == "draw":
					correctDraw += 1
			elif outcome == "win":
				totalWin += 1
				if pred == "win":
					correctWin += 1
			else:
				totalLoss += 1
				if pred == "loss":
					correctLoss += 1
			
			totalEx += 1
			matchNum += 1

		print "\n---Final weights---"
		self.printWeights()
		print "-------------------"
		print "Total correct: %f" % (totalCorrect / float(totalEx))
		print "Total correct wins: %f" % (correctWin)
		print "Total wins: %f" % (totalWin)
		print "Total correct losses: %f" % (correctLoss)
		print "Total losses: %f" % (totalLoss)
		print "Total correct draws: %f" % (correctDraw)
		print "Total draws: %f" % (totalDraw)
		print "Total examples (games): %f" % totalEx

predsys = PredictPD()
predsys.train()
predsys.test()
