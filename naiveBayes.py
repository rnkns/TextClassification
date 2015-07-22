# -*- coding: utf-8; -*-
import math
import sys
from collections import defaultdict
import pyquery
import MeCab
import re
from datetime import datetime


# 日付の取得(BagOfWordsファイル保存用)
date = datetime.now()
year = date.year
month = date.month
day = date.day


class NaiveBayes:
    def __init__(self):
        self.topics = set()     # トピックの集合
        self.vocabularies = set()   # 語彙の集合
        self.wordcount = {}         # 各トピックにおける単語の出現回数
        self.topiccount = {}          # トピックの出現回数
        self.denominator = {}       # P(word|topic)の分母の値

    # ナイーブベイズ分類器の訓練
    def train(self, data):
        # 文書集合からトピックを抽出
        for d in data:
            topic = d[0]
            self.topics.add(topic)
        # トピックごとの辞書を初期化
        for topic in self.topics:
            self.wordcount[topic] = defaultdict(int)
            self.topiccount[topic] = 0
        # 文書集合からトピックと単語をカウント
        for d in data:
            topic = d[0]
            doc = d[1:]
            self.topiccount[topic] += 1
            for word in doc:
                self.vocabularies.add(word)
                self.wordcount[topic][word] += 1
        # 単語の条件付き確率の分母の値をあらかじめ計算
        for topic in self.topics:
            sumCount = sum(self.wordcount[topic].values())
            lenVocas = len(self.vocabularies)
            self.denominator[topic] = sumCount + lenVocas

    # 対数尤度最大のトピックを返す
    def classify(self, doc):
        best = None
        max = -sys.maxsize
        for topic in self.topiccount.keys():
            p = self.score(doc, topic)
            if p > max:
                max = p
                best = topic
        return best

    # 単語の条件付き確率 P(word|topic)
    def word_prob(self, word, topic):
        # ラプラススムージング
        countLp = float(self.wordcount[topic][word] + 1)
        denominatorLp = float(self.denominator[topic])
        return countLp / denominatorLp

    # 文書が与えられたときのトピックの事後確率の対数 log(P(topic|doc))
    def score(self, doc, topic):
        total = sum(self.topiccount.values())  # 総文書数
        score = math.log(float(self.topiccount[topic]) / total)
        for word in doc:
            score += math.log(self.word_prob(word, topic))
            '''print("P({0}|{1}): ".format(word, cat),
                self.word_prob(word, cat))
            '''
        return score

    # 訓練データに関する情報
    def __str__(self):
        total = sum(self.topiccount.values())  # 総文書数
        train_info = "{0} documents, {1} vocabularies, {2} categories".format(
            total, len(self.vocabularies), len(self.topics))
        return train_info


# 訓練データ（教師有）をファイルへ書き込み
def make_file(count, noun_list, topic):
    f = open(
        "BagOfWords/{0}{1}{2}{3}_{4}.txt".format(
            topic, str(year), str(month), str(day), str(count)), 'w')
    f.write(str(noun_list))
    f.close()


# MeCabによる名詞の抽出
def noun_extract(count, news_text, topic):
    noun_list = []    # Bag of words用リスト
    noun_list.append(topic)    # リストの先頭にカテゴリの挿入

    m = MeCab.Tagger("-Ochasen")
    words = m.parse(news_text).split('\n')    # MeCabによる単語分割
    for word in words:
        attrs = word.split('\t')
        if len(attrs) > 3:
            w_re = re.search('.*名詞.*', attrs[3])    # 名詞の判定
            if w_re:
                if len(attrs[0].strip()) > 1:    # 一文字の名詞を除外
                    noun_list.append(attrs[0].strip())
    make_file(count, noun_list, topic)
    return noun_list


# 訓練データ（教師有）作成のためのWebスクレイピング
def scraping(topics, pages):
    train_dataset = []    # 訓練データを格納するリスト
    news_title = ""
    article_num_list = []  # 各トピックのニュース数を格納するリスト

    print("-----------------\nscraping...\n-----------------")

    # 各トピックの記事URL取得
    def html_get(topic, page):
        html = pyquery.PyQuery(
            "http://news.yahoo.co.jp/hl?c={0}&p={1}".format(topic, str(page)))
        # 各トピックからニュースのURLを取得
        for li in html('li').items():
            for p in li('p').items():
                if p.hasClass('ttl') == True:
                    for a in p('a').items():
                        news_list.append(a.attr('href'))
        return news_list

    # 各ニュースから記事本文の取得
    def article_get(news_list, topic, train_dataset, trains):
        count = 0
        train_data = ""
        for news in news_list:
            news_html = pyquery.PyQuery(news)
            # 訓練データ集合に名詞のみ抽出された1訓練データを追加
            if count <= trains and count >= 1:
                train_dataset.append(noun_extract(
                    count, train_data, topic))
            elif count > trains and count >= 1:
                noun_extract(count, train_data, topic)

            train_data = ""
            for div in news_html('div').items():
                if div.hasClass('paragraph') == True:
                    if div.hasClass('ynDetailHeading'):
                        news_title = div.text()
                    for p in div('p').items():
                        if p.hasClass('ynDetailText'):
                            train_data += p.text()
            count += 1
        return train_dataset

    # 各トピックに関するニュースの情報を格納するリストの作成
    for topic in topics:
        news_list = []    # ニュースのURLを格納するリスト
        for page in range(pages):
                news_list = html_get(topic, page)

        print(topic, ": ", str(len(news_list)), "contents")
        trains = int(len(news_list) * 0.7)
        article_num_list.append((trains, len(news_list)))
        train_dataset = article_get(news_list, topic, train_dataset, trains)
    return (train_dataset, article_num_list)


# 事前に取得した訓練データ（教師有）をファイルから読み込み
def train_local(topics):
    train_dataset = []    # 訓練データを格納するリスト
    article_num_list = []
    for topic in topics:
        for trainNo in range(108):
            f_test = open('BagOfWords/{0}{1}{2}{3}_{4}.txt'.format(
                topic, str(year), str(month), str(day), str(test_number)), 'r')
            test = f_test.readline()
            f_test.close()

            train_data = []
            riv1 = test.replace("[", "")
            riv2 = riv1.replace("]", "")
            elements = riv2.replace("'", "").split(",")

            for element in elements:
                if len(element.strip()) > 1:    # 一文字の名詞を除外
                    train_data.append(element.strip())
            train_dataset.append(train_data)
        trains = int(108 * 0.7)
        article_num_list.append((trains, 108))
    return (train_dataset, article_num_list)


# テストデータのトピックを予測
def test_nb(topics, article_num_list):
    count = 0    # 正解数の計算
    totalTests = 0    # テストデータの数
    topic_number = 0

    # 各トピックに対する尤度の出力
    def output_prob(test_number, test_data):
        for topic in topics:
            print("log P({0}|test) =".format(topic),
                  nb.score(test_data, topic))

    for topic in topics:
        topic_number += 1
        test_number = article_num_list[topic_number-1][0] + 1
        while article_num_list[topic_number-1][1]+1 > test_number:
            correct_topic = ""
            f_test = open('BagOfWords/{0}{1}{2}{3}_{4}.txt'.format(
                topic, str(year), str(month), str(day), str(test_number)), 'r')
            test = f_test.readline()
            f_test.close()
            test_number += 1

            test_data = []
            riv1 = test.replace("[", "")
            riv2 = riv1.replace("]", "")
            elements = riv2.replace("'", "").split(",")

            for element in elements:
                if len(element.strip()) > 1:    # 一文字の名詞を除外
                    test_data.append(element.strip())

            correct_topic = test_data[0]
            del test_data[0]    # リストの先頭にあるトピック名の削除

            print("{0}{1}{2}{3}_{4}.txt".format(
                topic, str(year), str(month), str(day), str(test_number-1)),
                "{ Correct Topic: ", correct_topic, "}")
            # output_prob(test_number-1, test_data)   # 各トピックに対する尤度の出力
            print(nb.classify(test_data))
            totalTests += 1
            if nb.classify(test_data) == correct_topic:
                count += 1
    print("Train: ", nb)
    print("Accuracy: ", count / totalTests, "(", count, "/", totalTests, ")")


if __name__ == "__main__":
    topics = ("base", "socc", "moto", "spo", "horse", "golf", "fight")
    pages = 4   # 1トピックに対してスクレイピングするWebページ数
    # 訓練データ（教師有）の作成_スクレイピング
    (data, article_num_list) = scraping(topics, pages)
    # 訓練データ（教師有）の作成_事前に取得したデータの利用
    # (data, article_num_list) = train_local(topics)

    # ナイーブベイズ分類器を訓練
    nb = NaiveBayes()
    nb.train(data)
    print(nb)

    print(article_num_list)
    # テストデータのトピックを予測
    test_nb(topics, article_num_list)
