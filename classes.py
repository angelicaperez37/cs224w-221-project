import collections
from collections import defaultdict
import os
import re
from snap import *
import unicodedata

# stores relevant team data
class Team():
	def __init__(self, name, players):
		self.name = name
		self.players = players
		self.rank = 0

	def setRank(self, rank):
		self.rank = rank

	def addPlayer(self, player):
		self.players.append(player)

	def __str__( self ):
		return self.name + "," + str(self.players) + "," + str(self.rank)

class Player():
	def __init__(self, name, number, team, position, price):
		#self.firstName = firstName
		#self.lastName = lastName
		self.name = name
		self.number = number
		self.team = team
		self.position = position
		self.price = price
		self.matches = []
		self.stats = []

	def __str__( self ):
		string = "name: " + self.name + "\nnum: " +  self.number + "\nteam: " + self.team + "\nposition: " + self.position+ \
			"\nprice: "+ str(self.price) +"\nmatches: "+str(self.matches)+"\nstats: "+str(self.stats)
		print string
		return string

class MatchPlayerStats():
	def __init__(self):
		self.minutes = 0
		self.goals = 0
		self.attemptsHittingBar = 0
		self.foulsSuffered = 0
		self.attemptsOffTarget = 0
		self.offsides = 0
		self.attemptsOnTarget = 0
		self.attemptsBlocked = 0
		self.yellowCards = 0
		self.attemptsHittingPost = 0
		self.foulsCommitted = 0
		self.redCards = 0

class Match():
	# def __init__(self, homeTeam, visitingTeam):
	def __init__(self, homeTeam):

		self.homeTeam = homeTeam
		# self.visitingTeam = visitingTeam

		self.visitingTeam = ""
		self.homeScore = 0
		self.visitorScore = 0
		self.homeTeamObj = Team("", "")
		self.visitingTeamObj = Team("", "")
		self.homePD = {}
		self.visitorPD = {}

		#stores how many passes to defenders, mids, etc.
		self.homePosPD = {}
		self.visitorPosPD = {}

		# can be one of "HEAVY_DEF", "HEAVY_MID", "HEAVY_OFF", "BALANCED", 
		# still need to add possibility of "HEAVY_RIGHT" and "HEAVY_LEFT"
		# in pdPrediction.py
		self.homePDType = ""
		self.visitorPDType = ""

	def __str__( self ):
		return "" + self.homeTeam +  "," + self.visitingTeam

	def getPD(self, team):
		if team == self.homeTeam:
			return self.homePD
		return self.visitorPD

	def winner(self):
		if self.homeScore > self.visitorScore:
			return self.homeTeam
		elif self.homeScore < self.visitorScore:
			return self.visitingTeam
		else:
			return "DRAW"

	def setHomeScore(self, score):
		self.homeScore = score

	def setVisitorScore(self, score):
		self.visitorScore = score

	def setHomeTeam(self, team):
		self.homeTeam = team

	def setVisitingTeam(self, team):
		self.visitingTeam = team

	def setHomeTeamObj(self, team):
		self.homeTeamObj = team

	def setVisitingTeamObj(self, team):
		self.visitingTeamObj = team



def getMatchIDFromFile(network):
	matchID = re.sub("_.*", "", network)
	return matchID

def getTeamNameFromNetwork(network):
	teamName = re.sub("[^-]*-", "", network, count=1)
	teamName = re.sub("-edges", "", teamName)
	teamName = re.sub("_", " ", teamName)
	return teamName

class CountAvgPassesFeature():
	def __init__(self, count_file_name):
		self.avgCounts = defaultdict(lambda: defaultdict(float))
		count_file = open(count_file_name, "r")
		for line in count_file:
			team, players, weight = line.strip().split(", ")
			self.avgCounts[team][players] = weight

	def getCount(self, team, player1, player2):
		p_key = player1 + "-" + player2
		return self.avgCounts[team][p_key]

class PlayerPositionFeature():
	def __init__(self, squad_dir):

		def getTeamNameFromFile(teamFile):
			teamName = re.sub("-squad.*", "", teamFile)
			teamName = re.sub("_", " ", teamName)
			return teamName

		self.teamNumName = defaultdict(lambda: defaultdict(str))
		self.teamNumPos = defaultdict(lambda: defaultdict(str))

		for team in os.listdir(squad_dir):
			if re.search("-squad", team):
				path = squad_dir + team
				teamFile = open(squad_dir + team, "r")
				teamName = getTeamNameFromFile(team)
				for player in teamFile:
					num, name, pos = player.rstrip().split(", ")
					self.teamNumName[teamName][num] = name
					self.teamNumPos[teamName][num] = pos

	def getPos(self, teamName, num):
		return self.teamNumPos[teamName][num]

	def getName(self, teamName, num):
		return self.teamNumName[teamName][num]

	def isSamePos(self, teamName, num1, num2):
		ret = 1
		if self.getPos(teamName, num1) != self.getPos(teamName, num2):
			ret = 0
		return ret

class RankingFeature():
	def __init__(self, rankFileName):
		self.rankings = defaultdict(int)
		rank_file = open(rankFileName, "r")
		for rank in rank_file:
			rank, team = rank.rstrip().split(", ")
			self.rankings[team] = int(rank)

	def getRank(self, team):
		return self.rankings[team]

	def isHigherInRank(self, team1, team2):
		return self.getRank(team1) > self.getRank(team2)

	def getDiffInRank(self, team1, team2):
		return self.getRank(team1) - self.getRank(team2)

# unsuccessful feature
class MeanDegreeFeature():

	def __init__(self):
		folder = "passing_distributions/2014-15/"
		allGames = ["matchday" + str(i) for i in xrange(1, 7)]
		allGames.append("r-16")
		allGames.append("q-finals")
		allGames.append("s-finals")

		self.meanDegree = defaultdict(lambda: defaultdict(float))

		for matchday in allGames:
			path = folder + matchday + "/networks/"
			for network in os.listdir(path):
				if re.search("-edges", network):
					edgeFile = open(path + network, "r")

					degreePerPlayer = defaultdict(int)
					teamName = getTeamNameFromNetwork(network)
					matchID = getMatchIDFromFile(network)
					# print "team: %s" % teamName
					totalDegree = 0

					for players in edgeFile:
						p1, p2, weight = players.rstrip().split("\t")
						# print "p1: %s, p2: %s, weight: %f" % (p1, p2, float(weight))
						degreePerPlayer[p1] += 1

					# count number of nodes to take average over team
					nodeFile = open(path + matchID + "_tpd-" + re.sub(" ", "_", teamName) + "-nodes", "r")
					players = [line.rstrip() for line in nodeFile]
					numPlayers = len(players)
					totalDegree = 0
					for player in degreePerPlayer:
						totalDegree += degreePerPlayer[player]

					avgDegree = totalDegree / numPlayers
					# print "Avg degree for %s is %f" % (teamName, avgDegree)
					self.meanDegree[matchID][teamName] = avgDegree
	
	def getMeanDegree(self, matchID, teamName):
		return self.meanDegree[matchID][teamName]

# Returns the average betweenness centrality of each player
# calculated only using group stage, like average degree
class BetweennessFeature():
	def __init__(self):
		folder = "passing_distributions/2014-15/"
		allGames = ["matchday" + str(i) for i in xrange(1, 7)]
		# allGames.append("r-16")
		# allGames.append("q-finals")
		# allGames.append("s-finals")

		self.betweenCentr = defaultdict(lambda: defaultdict(float))
		self.avgBCPerTeam = defaultdict(float)

		for matchday in allGames:
			path = folder + matchday + "/networks/"
			for network in os.listdir(path):
				if re.search("-edges", network):
					edgeFile = open(path + network, "r")

					degreePerPlayer = defaultdict(int)
					teamName = getTeamNameFromNetwork(network)
					matchID = getMatchIDFromFile(network)

					edges = [line.rstrip() for line in edgeFile]

					nodeFile = open(path + matchID + "_tpd-" + re.sub(" ", "_", teamName) + "-nodes", "r")
					players = [line.rstrip() for line in nodeFile]
					
					# build each network

					PlayerGraph = TUNGraph.New()
					for player in players:
						num, name = player.split("\t")
						PlayerGraph.AddNode(int(num))
					for edge in edges:
						src, dest, weight = edge.split("\t")
						src = int(src)
						dest = int(dest)
						PlayerGraph.AddEdge(src, dest)

					# calculate betweenness
					Nodes = TIntFltH()
					Edges = TIntPrFltH()
					GetBetweennessCentr(PlayerGraph, Nodes, Edges, 1.0)

					players_by_between = [(node, Nodes[node]) for node in Nodes]
					for player in players_by_between:
						num, betw = player
						self.betweenCentr[teamName][num] += betw

		# normalize over number of matchdays
		for teamName in self.betweenCentr:
			totalBetween = 0
			totalPlayers = 0
			for num in self.betweenCentr[teamName]:
				self.betweenCentr[teamName][num] /= 6
				totalBetween += self.betweenCentr[teamName][num]
				totalPlayers += 1
			self.avgBCPerTeam[teamName] = totalBetween / float(totalPlayers)

	def getBetweenCentr(self, matchID, teamName, player):
		return self.betweenCentr[teamName][int(player)]

	def getAvgBetweenCentr(self, teamName):
		return self.avgBCPerTeam[teamName]

# average passes completed and attempted per player feature
# averaged over all group games
class PassesComplAttempPerPlayerFeature():
	def __init__(self):
		folder = "passing_distributions/2014-15/"
		allGames = ["matchday" + str(i) for i in xrange(1, 7)]
		# allGames.append("r-16")
		# allGames.append("q-finals")
		# allGames.append("s-finals")

		self.pcPerPlayer = defaultdict(lambda: defaultdict(float))
		self.paPerPlayer = defaultdict(lambda: defaultdict(float))
		self.pcPercPerPlayer = defaultdict(lambda: defaultdict(float))

		for matchday in allGames:
			path = folder + matchday + "/networks/"
			for network in os.listdir(path):
				if "+" not in network:
					if re.search("-players", network):
						playerFile = open(path + network, "r")

						teamName = getTeamNameFromNetwork(network)
						teamName = re.sub("-players", "", teamName)
						matchID = getMatchIDFromFile(network)

						players = [line.rstrip() for line in playerFile]
						for player in players:
							print "player is ", player
							num, pc, pa, percPc = player.split(",")
							self.pcPerPlayer[teamName][num] += float(pc) / 6.0
							# print "teamName: %s, num: %s, %f" % (teamName, num, self.pcPerPlayer[teamName][num])
							self.paPerPlayer[teamName][num] += float(pa) / 6.0
							self.pcPercPerPlayer[teamName][num] += float(percPc) / 6.0

	def getPC(self, teamName, num):
		# print "teamName: ", teamName
		# print "num: ", num
		# print self.pcPerPlayer[teamName][num]
		return self.pcPerPlayer[teamName][num]

	def getPA(self, teamName, num):
		return self.pcPerPlayer[teamName][num]

	def getPCPerc(self, teamName, num):
		return self.pcPercPerPlayer[teamName][num]

# pre-load passes by position by matchID
class CountPassesPerPosFeature():
	def __init__(self, count_file_dir, train_end):
		self.countsByPos = defaultdict(lambda: defaultdict(float))

		folders = []

		if train_end == "group":
			folders.append("group/")
		elif train_end == "r-16":
			folders.append("group/")
			folders.append("r-16/")
		elif train_end == "q-finals":
			folders.append("group/")
			folders.append("r-16/")
			folders.append("q-finals/")

		# total passes per team
		self.totalCounts = defaultdict(float)
		for stage in folders:
			path = count_file_dir + stage
			for teamByGame in os.listdir(path):
				if ".DS_Store" not in teamByGame:
					teamGameFile = open(path + teamByGame, "r")
					# get teamName from filename
					teamName = re.sub(".*-", "", teamByGame)
					teamName = re.sub("_", " ", teamName)
					for line in teamGameFile:
						pos, weight = line.rstrip().split("\t")
						self.countsByPos[teamName][pos] += float(weight)
						self.totalCounts[teamName] += float(weight)

		for teamName in self.countsByPos:
			for pos in self.countsByPos[teamName]:
				self.countsByPos[teamName][pos] /= self.totalCounts[teamName]


	def getCount(self, team, pos):
		return self.countsByPos[team][pos]

# pre-load passes completed/attempted
class CountPassesComplAttempPerTeamFeature():
	def __init__(self, train_end):

		self.passComplPerTeam = defaultdict(int)
		self.passAttemPerTeam = defaultdict(int)
		self.passPercPerTeam = defaultdict(float)

		folder = "passing_distributions/2014-15/"

		allGames = ["matchday" + str(i) for i in xrange(1, 7)]

		if train_end == "r-16":
			folders.append("r-16/")
		elif train_end == "q-finals":
			folders.append("r-16/")
			folders.append("q-finals/")

		# allGames.append("r-16")
		# allGames.append("q-finals")
		# allGames.append("s-finals")

		for matchday in allGames:
			path = folder + matchday + "/networks/"
			for network in os.listdir(path):
				if re.search("-team", network):
					teamFile = open(path + network, "r")
					teamName = getTeamNameFromNetwork(network)
					teamName = re.sub("-team", "", teamName)
					matchID = getMatchIDFromFile(network)
					# print "teamName is: %s" % teamName
					# print "matchID is: %s" % matchID
					for line in teamFile:
						stats = line.rstrip().split(", ")
						self.passComplPerTeam[teamName] += float(stats[0])
						self.passAttemPerTeam[teamName] += float(stats[1])
						self.passPercPerTeam[teamName] += float(stats[2])
	
	def getPCCount(self, teamName, matchNum):
		return self.passComplPerTeam[teamName] / (matchNum + 1.0)

	def getPACount(self, teamName, matchNum):
		return self.passAttemPerTeam[teamName] / (matchNum + 1.0)

	def getPCPerc(self, teamName, matchNum):
		return self.passPercPerTeam[teamName] / (matchNum + 1.0)

	def getPassFail(self, teamName, matchNum):
		return self.getPCCount(self, teamName, matchNum) - self.getPACount(self, teamName, matchNum)

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

class TeamStatsFeature():
	def __init__(self):
		matchdays = ["matchday" + str(i) for i in xrange(1, 7)]

		folder = "team_stats_full_time/2014-15/"

		# map matchday to feature numbers to names
		features = defaultdict(lambda: defaultdict(str))
		# matchID -> team -> stats
		# self.team_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
		self.team_stats = defaultdict(lambda: defaultdict(float))
		for matchday in matchdays:
			path = folder + matchday + "/"
			feature_file = open(path + "features", "r")
			for f in feature_file:
				if "#" not in f and f != "\n":
					num, name = f.rstrip().split(", ")
					features[matchday][num] = name

			for game in os.listdir(path):
				if re.search("txt", game):
					matchID = re.sub("_.*", "", game)
					gameFile = open(path + game, "r")
					lines = [line.rstrip() for line in gameFile]
					# second line has team names

					_, _, _, team1, team2, _, _ = lines[1].split(", ")
					start = lines.index("# total") + 2
					end = lines.index("# season average") - 2

					team1 = strip_accents(team1)
					team2 = strip_accents(team2)

					team1Stats = lines[start].split(", ")[1:]
					team2Stats = lines[start+1].split(", ")[1:]
					for stat in team1Stats:
						num, feat = stat.split(":")
						feat_name = features[matchday][num]
						if "%" in feat:
							feat = "0." + feat
							feat = feat[:-1]
						if "/" in feat:
							first_num = float(re.sub("\/.*", "", feat))
							second_num = float(re.sub(".*\/", "", feat))
							feat = (first_num + second_num) / 2.0
						
						self.team_stats[team1][feat_name] += float(feat) / 6.0
					for stat in team2Stats:
						num, feat = stat.split(":")
						feat_name = features[matchday][num]
						if "%" in feat:
							feat = "0." + feat
							feat = feat[:-1]
						if "/" in feat:
							first_num = float(re.sub("\/.*", "", feat))
							second_num = float(re.sub(".*\/", "", feat))
							feat = (first_num + second_num) / 2.0

						self.team_stats[team2][feat_name] += float(feat) / 6.0


	def getTeamStats(self, team):
		return self.team_stats[team]