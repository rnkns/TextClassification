# -*- coding: utf-8; -*-
import math
import sys
from collections import defaultdict
import pyquery
import MeCab
import re

class NaiveBayes:
	def __init__(self):
		self.topics = set()     #トピックの集合
		self.vocabularies = set()   #語彙の集合
		self.wordcount = {}         #各トピックにおける単語の出現回数
		self.topiccount = {}          #トピックの出現回数
		self.denominator = {}       #P(word|topic)の分母の値

	"""ナイーブベイズ分類器の訓練"""
	def train(self, data):
		#文書集合からトピックを抽出
		for d in data:
			topic = d[0]
			self.topics.add(topic)
		#トピックごとの辞書を初期化
		for topic in self.topics:
			self.wordcount[topic] = defaultdict(int)
			self.topiccount[topic] = 0
		#文書集合からトピックと単語をカウント
		for d in data:
			topic = d[0]
			doc = d[1:]
			self.topiccount[topic] += 1
			for word in doc:
				self.vocabularies.add(word)
				self.wordcount[topic][word] += 1
		#単語の条件付き確率の分母の値をあらかじめ計算
		for topic in self.topics:
			self.denominator[topic] = sum(self.wordcount[topic].values()) + len(self.vocabularies)

	"""対数尤度最大のトピックを返す"""
	def classify(self, doc):
		best = None
		max = -sys.maxsize
		for topic in self.topiccount.keys():
			p = self.score(doc, topic)
			if p > max:
				max = p
				best = topic
		return best

	"""単語の条件付き確率 P(word|topic)"""
	def wordProb(self, word, topic):
		#ラプラススムージング
		return float(self.wordcount[topic][word] + 1) / float(self.denominator[topic])

	"""文書が与えられたときのトピックの事後確率の対数 log(P(topic|doc))"""
	def score(self, doc, topic):
		total = sum(self.topiccount.values())  #総文書数
		score = math.log(float(self.topiccount[topic]) / total)  #log P(topic)
		for word in doc:
			score += math.log(self.wordProb(word, topic))  #log P(word|topic)
			#print("P({0}|{1}): ".format(word, cat), self.wordProb(word, cat))
		return score

	'''訓練データに関する情報'''
	def __str__(self):
		total = sum(self.topiccount.values())  #総文書数
		return "%d documents, %d vocabularies, %d categories" % (total, len(self.vocabularies), len(self.topics))

'''MeCabによる名詞の抽出'''
def nounExtract(count, news_text, topic):
	nounList = []	#Bag of words用リスト
	nounList.append(topic)	#リストの先頭にカテゴリの挿入

	m = MeCab.Tagger ("-Ochasen")
	words = m.parse(news_text).split('\n')	#MeCabによる単語分割
	for word in words:
		attrs = word.split('\t')
		if len(attrs) > 3:
			w_re = re.search('.*名詞.*', attrs[3])	#名詞の判定
			if w_re:
				if len(attrs[0].strip()) > 1:	#一文字の名詞を除外
					nounList.append(attrs[0].strip())

	'''ファイルへ書き込み'''
	f = open("BagOfWords/Train/{0}_{1}.txt".format(topic, str(count)), 'w')
	f.write(str(nounList))
	f.close()

	return nounList

'''訓練データ（教師有）作成のためのWebスクレイピング'''
def scraping(topics):
	TrainDataSet = []	#訓練データを格納するリスト
	newsTitle = ""

	print("-----------------\nscraping...\n-----------------")

	'''各トピックに関するニュースのURLを格納するリストの作成'''
	for topic in topics:
		newsList = []	#ニュースのURLを格納するリスト
		html = pyquery.PyQuery("http://news.yahoo.co.jp/hl?c={0}".format(topic))
		'''各トピックから記事のURLを取得'''
		for li in html('li').items():
			for p in li('p').items():
				if p.hasClass('ttl') == True:
					for a in p('a').items():
						newsList.append(a.attr('href'))

		count = 0
		'''各ニュースから記事を取得'''
		for news in newsList:
			news_html = pyquery.PyQuery(news)
			for div in news_html('div').items():
				if div.hasClass('paragraph') == True:
					if div.hasClass('ynDetailHeading'):
						newsTitle = div.text()
					for p in div('p').items():
						if p.hasClass('ynDetailText'):
							count = count + 1
							TrainDataSet.append(nounExtract(count, p.text(), topic))	#訓練データ集合に名詞のみ抽出された1訓練データを追加
	return TrainDataSet

'''事前に取得した訓練データ（教師有）をファイルから読み込み'''
def trainLocal(topics):
	TrainDataSet = []	#訓練データを格納するリスト
	for topic in topics:
		for trainNo in range(30):
			fTest = open('BagOfWords/Train/{0}_{1}.txt'.format(topic, str(trainNo+1)), 'r')
			test = fTest.readline()
			fTest.close()

			TrainData = []
			riv1 = test.replace("[", "")
			riv2 = riv1.replace("]", "")
			elements = riv2.replace("'", "").split(",")

			for element in elements:
				if len(element.strip()) > 1:	#一文字の名詞を除外
					TrainData.append(element.strip())
			TrainDataSet.append(TrainData)
	return TrainDataSet

'''テストデータのトピックを予測'''
def testNB(topics):
	count = 0	#正解数の計算
	totalTests = 0	#テストデータの数
	for testNo in range(100):
		corTopic = ""
		fTest = open('BagOfWords/Test/{0}.txt'.format(str(testNo+1)), 'r')
		test = fTest.readline()
		fTest.close()

		testData = []
		riv1 = test.replace("[", "")
		riv2 = riv1.replace("]", "")
		elements = riv2.replace("'", "").split(",")

		for element in elements:
			if len(element.strip()) > 1:	#一文字の名詞を除外
				testData.append(element.strip())

		corTopic = testData[0]
		del testData[0]	#リストの先頭にあるトピック名の削除

		print("test{0}.txt".format(str(testNo+1)), "{ Correct Topic: ", corTopic, "}")
		for topic in topics:
			print("log P({0}|test) =".format(topic), nb.score(testData, topic))
		print(nb.classify(testData))
		totalTests = totalTests + 1
		if nb.classify(testData) == corTopic:
			count = count + 1
	print("Train: ", nb)
	print("Accuracy: ", count / totalTests, "(", count, "/", totalTests, ")")

if __name__ == "__main__":

	topics = ("base","socc","moto", "spo", "horse", "golf", "fight")	#トピック
	#訓練データ（教師有）の作成_スクレイピング
#	data = scraping(topics)
	#訓練データ（教師有）の作成_事前に取得したデータの利用
	data = trainLocal(topics)

	# ナイーブベイズ分類器を訓練
	nb = NaiveBayes()
	nb.train(data)
	print(nb)

	# テストデータのトピックを予測
	testNB(topics)
