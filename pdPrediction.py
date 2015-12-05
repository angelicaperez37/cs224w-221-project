import collections, util, math, random
import classes, scoring
import copy
import os
import glob
import re
from collections import defaultdict
import unicodedata

class PredictPD():

	def __init__(self):

		# Feature: Specific pass count per match
		# This is kind of weird because it's like rote memorization
		# TODO: think more about this
		# countSpecFile = "spec_passes_count.txt"
		# countSpecificPassesFeature = classes.countSpecificPassesFeature(countSpecFile)
	
		self.weights = defaultdict(int)
		self.weights["avgPasses"] = 0
		self.weights["isSamePos"] = 0
		self.weights["isDiffPos"] = 0
		self.weights["diffInRank"] = 0
		self.weights["wonAgainstSimTeam"] = 0
		self.weights["avgPassesPerPos"] = 0
		self.weights["avgPassVol"] = 0
		self.weights["avgPassPerc"] = 0

		# TODO: can experiment with step size
		self.stepSize = 0.01

		# hold out some matchdays
		self.matchdays = ["matchday" + str(i) for i in xrange(1, 7)]

		self.folder = "passing_distributions/2014-15/"

		# Feature: Average pass count over group stage
		countAvgFile = "avg_passes_count.txt"
		self.countAvgPassesFeature = classes.CountAvgPassesFeature(countAvgFile)

		squad_dir = "squads/2014-15/squad_list/"
		self.playerPosFeature = classes.PlayerPositionFeature(squad_dir)

		rankFile = "rankings/2013_14_rankings.txt"
		self.rankFeature = classes.RankingFeature(rankFile)

		self.matches = defaultdict(str)

		self.totalPassesBetweenPos = defaultdict(lambda: defaultdict(int))
		self.totalPasses = defaultdict(int)

		self.teamNumToPos = defaultdict(lambda: defaultdict(str))
		self.initTeamNumToPos(squad_dir)

		self.passVolPerTeam = defaultdict(int)
		self.passPercPerTeam = defaultdict(float)

		self.teamStatsByMatch = defaultdict(lambda: defaultdict(list))

	# Average pairwise error over all players in a team
	# given prediction and gold
	def evaluate(self, features, weight):
		score = self.computeScore(features, self.weights)
		print "score %f vs. actual %f" % (float(score), float(weight))
		loss = self.computeLoss(features, self.weights, float(weight))
		print "Loss: %f" % loss
		return (score, loss)

	def computeLoss(self, features, weights, label):
		return (self.computeScore(features, weights) - label)**2

	# score is dot product of features & weights
	def computeScore(self, features, weights):
		score = 0.0
		for v in features:
			score += float(features[v]) * float(weights[v])
		return score

	# predict +1 if > 0, -1 otherwise
	# TODO: OR, is score == # passes
	# def predict(score):
	# 	if score > 0:
	# 		return 1
	# 	else:
	# 		return -1


	# returns a vector
	# 2 * (phi(x) dot w - y) * phi(x)
	def computeGradientLoss(self, features, weights, label):
		scalar =  2 * self.computeScore(features, weights) - label
		for f in features:
			features[f] = float(features[f])
			features[f] *= scalar
		return features

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
		elif matchID > 2014354:
			return 3
		else:
			return 4

	def featureExtractor(self, teamName, p1, p2, matchID, matchNum, weight):
		avgPasses = self.countAvgPassesFeature.getCount(teamName, p1, p2)
		isSamePos = self.playerPosFeature.isSamePos(teamName, p1, p2)
		isDiffPos = abs(1 - isSamePos)

		oppTeam = self.getOppTeam(matchID, teamName)
		diffInRank = self.rankFeature.isHigherInRank(teamName, oppTeam)

		features = defaultdict(float)
		features["avgPasses"] = avgPasses
		features["isSamePos"] = isSamePos
		features["isDiffPos"] = isDiffPos
		features["diffInRank"] = diffInRank

		pos1 = self.teamNumToPos[teamName][p1]
		pos2 = self.teamNumToPos[teamName][p2]

		# keep a running total of past passes between positions
		# how about a running average...
		p_key = pos1 + "-" + pos2
		self.totalPassesBetweenPos[teamName][p_key] += int(weight)
		self.totalPasses[teamName] += int(weight)
		print "totalPassesBetweenPos[%s][%s] = %s" % (teamName, p_key, self.totalPassesBetweenPos[teamName][p_key])
		print "totalPasses[%s] = %s" % (teamName, self.totalPasses[teamName])
		avgPassesPerPos = self.totalPassesBetweenPos[teamName][p_key] / float(self.totalPasses[teamName])
		features["avgPassesPerPos"] = avgPassesPerPos

		# TODO: avgPassVol
		avgPassVol = self.passVolPerTeam[teamName] / (matchNum + 1.0)
		avgPassPerc = self.passPercPerTeam[teamName] / (matchNum + 1.0)

		oppAvgPassVol = self.passVolPerTeam[oppTeam] / (matchNum + 1.0)
		oppAvgPassPerc = self.passPercPerTeam[oppTeam] / (matchNum + 1.0)

		print "avgPassVol: %s vs oppAvgPassVol: %s" % (avgPassVol, oppAvgPassVol)

		features["avgPassVol"] = 1 if avgPassVol > oppAvgPassVol else 0
		features["avgPassPerc"] = 1 if avgPassPerc > oppAvgPassPerc else 0

		# for feature: won against a similar ranking team
		# 1. define history that we are able to use, i.e. previous games
		matchday = self.getMatchday(matchID)
		history = self.teamPlayedWith[teamName][:matchday]
		
		if len(history) > 0:
			def computeSim(rank1, rank2):
				return (rank1**2 + rank2**2)**0.5

			# 2. find most similar opponent in terms of rank
			# TODO: similarity could be defined better?
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

			# 3. find out whether the game was won or lost
			features["wonAgainstSimTeam"] = self.teamWonAgainst[teamName][matchday]

		return features

	def initMatches(self):
		# store match data for all 6 matchdays in group stage + r-16
		# match data including team + opponent team
		for matchday in self.matchdays + ["r-16"]:
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
						self.matches[matchID] += "/" + teamName

		allScoresFilename = "matches4_groupStage_2014_15.txt"
		allScores = open(allScoresFilename, "r")
		self.matchesWithScores = [line.rstrip() for line in allScores]
		self.teamPlayedWith = defaultdict(list)
		self.teamWonAgainst = defaultdict(list)

		# for every team, store opponents in order by matchday
		for match in self.matchesWithScores:
			team1, score1, score2, team2 = match.split(", ")
			team1Won = 0
			if score1 > score2:
				team1Won = 1

			self.teamPlayedWith[team1].append(team2)
			self.teamPlayedWith[team2].append(team1)
			self.teamWonAgainst[team1].append(team1Won)
			self.teamWonAgainst[team2].append(abs(1 - team1Won))

	def initTeamStats(self):
		for matchday in self.matchdays:
			print "On " + matchday
			path = self.folder + matchday + "/networks/"
			# iterate over games
			for network in os.listdir(path):
				if re.search("-team", network):
					# store per match
					# or store per team?
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
		# iterate over matchdays, predicting passes, performing SGD, etc.

		num_iter = 2
		self.initMatches()
		self.initTeamStats()
		
		pos = ["GK", "STR", "DEF", "MID"]
		allPosCombos = [pos1 + "-" + pos2 for pos1 in pos for pos2 in pos]

		for i in xrange(num_iter):
			print "Iteration %s" % i
			print "------------"
			# iterate over matchdays -- hold out on some matchdays
			matchNum = 0
			for matchday in self.matchdays:
				print "On " + matchday
				path = self.folder + matchday + "/networks/"
				# iterate over games
				for network in os.listdir(path):
					if re.search("-edges", network):
						# passesBetweenPos = defaultdict(lambda: defaultdict(int))
						edgeFile = open(path + network, "r")

						teamName = self.getTeamNameFromNetwork(network)
						matchID = self.getMatchIDFromFile(network)
						print "team: %s" % teamName
						for players in edgeFile:
							p1, p2, weight = players.rstrip().split("\t")
							print "p1: %s, p2: %s, weight: %f" % (p1, p2, float(weight))

							teamFile = open(path + matchID + "_tpd-" + re.sub(" ", "_", teamName) + "-team", "r")
							for line in teamFile:
								stats = line.rstrip().split(", ")
							self.passVolPerTeam[teamName] += float(stats[0])
							self.passPercPerTeam[teamName] += float(stats[1])

							features = self.featureExtractor(teamName, p1, p2, matchID, matchNum, weight)
		
							for f in features:
								print "features[%s] = %f" % (f, float(features[f]))
							for w in self.weights:
								print "weights[%s] = %f" % (w, float(self.weights[w]))

							self.evaluate(features, weight)
 							self.updateWeights(features, self.weights, int(weight))
 						matchNum += 1

	# Testing
	#	Predict, then compare with dev/test set
	# "test" on r-16 games
	def test(self):
		# sum up average error

		print "Testing"
		print "-------"
		avgLoss = 0
		totalEx = 0
		matchNum = 0
		# for matchday in self.matchdays[4:]:
		matchday = "r-16"
		print "On " + matchday
		path = self.folder + matchday + "/networks/"
		# iterate over games
		for network in os.listdir(path):
			if re.search("-edges", network):
				edgeFile = open(path + network, "r")

				predEdgeFile = open("predicted/pred-" + network, "w+")

				teamName = self.getTeamNameFromNetwork(network)
				matchID = self.getMatchIDFromFile(network)
				print "team: %s" % teamName
				for players in edgeFile:
					p1, p2, weight = players.rstrip().split("\t")
					print "p1: %s, p2: %s, weight: %f" % (p1, p2, float(weight))

					features = self.featureExtractor(teamName, p1, p2, matchID, matchNum, weight)

					for f in features:
						print "features[%s] = %f" % (f, float(features[f]))
					for w in self.weights:
						print "weights[%s] = %f" % (w, float(self.weights[w]))

					score, loss = self.evaluate(features, weight)

					# print out predicted so can visually compare to actual
					predEdgeFile.write(p1 + "\t" + p2 + "\t" + str(score) + "\n")

					avgLoss += loss
					totalEx += 1
				matchNum += 1
		print "Average loss: %f" % (avgLoss / totalEx)

pred = PredictPD()
pred.train()
pred.test()

# #----------------------Store all player and team data-------------------#
# # list of all players as Players
# allPlayers = {}

# # for positions, only last names are included along with price, team, and position
# # TODO: update this with new squad list for 2014-15
# lines = [line.rstrip('\n') for line in open('fantasy_player_data/positions/defenders')]
# lines += [line.rstrip('\n') for line in open('fantasy_player_data/positions/forwards')]
# lines += [line.rstrip('\n') for line in open('fantasy_player_data/positions/goalkeepers')]
# lines += [line.rstrip('\n') for line in open('fantasy_player_data/positions/midfielders')]

# # compare ignoring accented characters
# def check_for_accented_key(k, d):
# 	for key in d:
# 		u1 = unicodedata.normalize('NFC', k.decode('utf-8'))
# 		u2 = unicodedata.normalize('NFC', key.decode('utf-8'))
# 		if u1 == u2:
# 			return d[key]
# 	raise "Couldn't find team name"

# # team_to_player_num[team][player_last_name] = player_num
# team_to_player_num = defaultdict(lambda: defaultdict(str))

# # team_to_player_name[team][player_num] = player_name
# team_to_player_name = defaultdict(lambda: defaultdict(str))
# # all_player_list includes first and last names for players, player numbers, and teams
# all_player_lines = [line.rstrip() for line in open("fantasy_player_data/all_players/all_player_list", 'r')]

# for line in all_player_lines[1:]:
# 	num, name, team = line.rstrip().split(",")

# 	# get rid of trailing whitespace
# 	name = re.sub("\s*$", "", name)
# 	if " " in name:
# 		last_name = (re.match(".* (.*)$", name)).group(1)
# 	else: last_name = name

# 	team_to_player_num[team][last_name] = num
# 	team_to_player_name[team][num] = name

# # team name as String -> Team object
# allTeams = {}

# # store basic player data
# # add players to their corresponding Teams
# for line in lines:
# 	last_name, team, position, price = line.rstrip().split(", ")
# 	price = float(price)
# 	team_dict = check_for_accented_key(team, team_to_player_num);

# 	num = team_dict[last_name]
# 	p = classes.Player(last_name, num, team, position, price)

# 	# store player_name-player_num-player_team = Player object
# 	key = last_name + "-" + num + "-" + team
# 	allPlayers[key] = p

# 	if team not in allTeams:
# 		allTeams[team] = classes.Team(team, [])

# 	allTeams[team].addPlayer(p)

# print allTeams.keys()

# #------------------END storing of all player and team data-------------------#

# def findPlayerPosition(name, team):
# 	teamObj = allTeams[team]
# 	splitName = name.split(" ")
# 	lastName = splitName[len(splitName)-1]

# 	for player in teamObj.players:
# 		if player.name == lastName:
# 			return player.position

# 	return "MID"  #returns "MID" when player not found in team
# 				#occurs only for a few players

# #store team rankings, teamName:ranking
# rankings = {}
# for line in [x.rstrip('\n') for x in open("rankings.txt")]:
# 	sl = line.split('\t')
# 	rankings[sl[0]] = sl[1]

# #matchID:classes.Match
# matches = {}

# #store matches with teams and players (just nodes files)
# for matchday in os.listdir("passing_distributions/2015-16/"):
# 	if matchday.endswith("sh") or matchday.endswith("py") or matchday.endswith("md") or matchday.endswith("Store"):
# 		continue
# 	folder = "passing_distributions/2015-16/"+matchday+"/networks/"
# 	for nodes_file in os.listdir(folder):
# 		if nodes_file.endswith("nodes"):
# 			matchID = re.search('(.+?)_tpd', nodes_file).group(1)
# 			team = re.sub("_", " ", re.search('tpd-(.+?)-nodes', nodes_file).group(1))

# 			#these two team names were showing up differently in objects
# 			if "FC Zenit" in team:
# 				team = "FC FC Zenit"
# 			elif "Maccabi" in team:
# 				team = "Maccabi Tel-Aviv FC"

# 			teamObj = classes.Team(team, [])
# 			for line in [x.rstrip('\n') for x in open(folder+nodes_file)]:
# 				elems = line.split('\t')
# 				playerName = re.sub("\s*$", "", elems[1])
# 				position = findPlayerPosition(playerName, team)
# 				player = classes.Player(playerName, elems[0], team, position, 0)
# 				teamObj.addPlayer(player)

# 			if matchID not in matches.keys():
# 				match = classes.Match(team, "")
# 				match.setHomeTeamObj(teamObj)
# 				matches[matchID] = match
# 			else:
# 				match = matches[matchID]
# 				match.setVisitingTeam(team)
# 				match.setVisitingTeamObj(teamObj)


# #store passing distributions for each match
# for matchday in os.listdir("passing_distributions/2015-16/"):
# 	if matchday.endswith("sh") or matchday.endswith("py") or matchday.endswith("md") or matchday.endswith("Store"):
# 		continue
# 	folder = "passing_distributions/2015-16/"+matchday+"/networks/"
# 	for edge_file in os.listdir(folder):
# 		if edge_file.endswith("edges"):
# 			matchID = re.search('(.+?)_tpd', edge_file).group(1)
# 			team = re.sub("_", " ", re.search('tpd-(.+?)-edges', edge_file).group(1))

# 			#these two team names were showing up differently in objects
# 			if "FC Zenit" in team:
# 				team = "FC FC Zenit"
# 			elif "Maccabi" in team:
# 				team = "Maccabi Tel-Aviv FC"

# 			match = matches[matchID]
# 			pd = match.getPD(team)

# 			for line in [x.rstrip('\n') for x in open(folder+edge_file)]:
# 				elems = line.split('\t')
				
# 				if elems[0] not in pd.keys():
# 					pd[elems[0]] = {}
# 				pd[elems[0]][elems[1]] = elems[2]


# #store team ranks in matches
# for matchID in matches.keys():
# 	match = matches[matchID]
# 	match.homeTeamObj.setRank(rankings[match.homeTeam])
# 	match.visitingTeamObj.setRank(rankings[match.visitingTeam])


# def computePositionPD(team, pd, players):
# 	num_to_pos = {}
# 	for player in players:
# 		num_to_pos[player.number] = player.position
# 	passes_to_pos = collections.Counter()
# 	for player in pd.keys():
# 		for receiver in pd[player].keys():
# 			pos = num_to_pos[receiver]
# 			passes_to_pos[pos] += int(pd[player][receiver])

# 	return passes_to_pos

# # compute PD types for each team in each match
# for id in matches.keys():
# 	match = matches[id]
# 	match.homePosPD = computePositionPD(match.homeTeam, match.homePD, match.homeTeamObj.players)
# 	match.visitorPosPD = computePositionPD(match.visitingTeam, match.visitorPD, match.visitingTeamObj.players)

# 	#to do some more analysis
# 	print "#################################################"
# 	print "MATCH ", id
# 	print match.homeTeam, " (#", rankings[match.homeTeam],") vs ", match.visitingTeam, " (#", rankings[match.visitingTeam], ")"
# 	print match.homeTeam, "'s position PD: "
# 	print match.homePosPD
# 	print match.visitingTeam, "'s position PD: "
# 	print match.visitorPosPD
# 	print "#################################################"

# #TO CHECK
# '''for id in matches.keys():
# 	print id
# 	print matches[id].homeTeam
# 	print matches[id].getPD(matches[id].homeTeam)
# 	print matches[id].visitingTeam
# 	print matches[id].getPD(matches[id].visitingTeam)
# '''	
# #	print matches[id].homeTeamObj.players[0].name
# #	print matches[id].visitingTeam
# #	print matches[id].visitingTeamObj.players[0].name
	

# '''m = re.search('AAA(.+?)ZZZ', text)
# if m:
#     found = m.group(1)
# '''
