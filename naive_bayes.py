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
        self.word_count = {}         # 各トピックにおける単語の出現回数
        self.topic_count = {}          # トピックの出現回数
        self.denominator = {}       # P(word|topic)の分母の値

    def train(self, data):
        """ナイーブベイズ分類器の訓練"""
        # 文書集合からトピックを抽出
        for d in data:
            topic = d[0]
            self.topics.add(topic)
        # トピックごとの辞書を初期化
        for topic in self.topics:
            self.word_count[topic] = defaultdict(int)
            self.topic_count[topic] = 0
        # 文書集合からトピックと単語をカウント
        for d in data:
            topic = d[0]
            doc = d[1:]
            self.topic_count[topic] += 1
            for word in doc:
                self.vocabularies.add(word)
                self.word_count[topic][word] += 1
        # 単語の条件付き確率の分母の値をあらかじめ計算
        for topic in self.topics:
            sum_count = sum(self.word_count[topic].values())
            length_vacabularies = len(self.vocabularies)
            self.denominator[topic] = sum_count + length_vacabularies

    def classify(self, doc):
        """対数尤度最大のトピックを返す"""
        best = None
        max_likelihood = -sys.maxsize
        for topic in self.topic_count.keys():
            p = self.score(doc, topic)
            if p > max_likelihood:
                max_likelihood = p
                best = topic
        return best

    def word_prob(self, word, topic):
        """単語の条件付き確率(ラプラススムージング) P(word|topic)"""
        count_laplace = float(self.word_count[topic][word] + 1)
        denominator_laplace = float(self.denominator[topic])
        return count_laplace / denominator_laplace

    def score(self, doc, topic):
        """文書が与えられたときのトピックの事後確率の対数 log(P(topic|doc))"""
        total = sum(self.topic_count.values())  # 総文書数
        score = math.log(float(self.topic_count[topic]) / total)
        for word in doc:
            score += math.log(self.word_prob(word, topic))
            '''print("P({0}|{1}): ".format(word, cat),
                self.word_prob(word, cat))
            '''
        return score

    def __str__(self):
        """訓練データに関する情報"""
        total = sum(self.topic_count.values())  # 総文書数
        train_info = "{0} documents, {1} vocabularies, {2} categories".format(
            total, len(self.vocabularies), len(self.topics))
        return train_info


def make_file(count, noun_list, topic):
    """訓練データ（教師有）をファイルへ書き込み"""
    f = open(
        "BagOfWords/{0}/{0}{1}{2}{3}_{4}.txt".format(
            topic, str(year), str(month), str(day), str(count)), 'w')
    f.write(str(noun_list))
    f.close()


def noun_extract(count, news_text, topic):
    """MeCabによる名詞の抽出"""
    noun_list = [topic, ]    # Bag of words用リスト

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


def scraping(topics, pages):
    """訓練データ（教師有）作成のためのWebスクレイピング"""
    train_dataset = []    # 訓練データを格納するリスト
    # news_title = ""
    article_num_list = []  # 各トピックのニュース数を格納するリスト

    print("-----------------\nscraping...\n-----------------")

    def html_get(topic, page, news_list):
        """各トピックの記事URL取得"""
        html = pyquery.PyQuery(
            "http://news.yahoo.co.jp/hl?c={0}&p={1}".format(topic, str(page)))
        # 各トピックからニュースのURLを取得
        for li in html('li').items():
            for p in li('p').items():
                if p.hasClass('ttl') is True:
                    for a in p('a').items():
                        news_list.append(a.attr('href'))
        return news_list

    def article_get(news_list, topic, get_train_dataset, trains):
        """各ニュースから記事本文の取得"""
        count = 1
        train_data = ""
        for news in news_list:
            news_html = pyquery.PyQuery(news)
            # 訓練データ集合に名詞のみ抽出された1訓練データを追加
            if 1 <= count <= trains:
                get_train_dataset.append(noun_extract(
                    count, train_data, topic))
            elif count > trains and count >= 1:
                noun_extract(count, train_data, topic)

            train_data = ""
            for div in news_html('div').items():
                if div.hasClass('paragraph') is True:
                    # if div.hasClass('ynDetailHeading'):
                        # news_title = div.text()
                    for p in div('p').items():
                        if p.hasClass('ynDetailText'):
                            train_data += p.text()
            count += 1
        return get_train_dataset

    # 各トピックに関するニュースの情報を格納するリストの作成
    for what_topic in topics:
        get_news_list = []    # ニュースのURLを格納するリスト
        for page_number in range(pages):
                get_news_list = html_get(
                    what_topic, page_number+1, get_news_list)
        print(what_topic, ": ", str(len(get_news_list)), "contents")
        trains_number = int(len(get_news_list) * 0.7)
        article_num_list.append((trains_number, len(get_news_list)))
        train_dataset = article_get(
            get_news_list, what_topic, train_dataset, trains_number)
    result_scraping = (train_dataset, article_num_list)
    return result_scraping


def train_local(topics):
    """事前に取得した訓練データ（教師有）をファイルから読み込み"""
    train_dataset = []    # 訓練データを格納するリスト
    article_num_list = []
    for topic in topics:
        for trainNo in range(108):
            f_test = open('BagOfWords/{0}/{0}{1}{2}{3}_{4}.txt'.format(
                topic, str(year), str(month), str(day), str(trainNo)), 'r')
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
    training_data = (train_dataset, article_num_list)
    return training_data


def test_nb(topics, article_num):
    """テストデータのトピックを予測"""
    count = 0    # 正解数の計算
    total_tests = 0    # テストデータの数
    topic_number = 0

    def output_prob(get_test_data):
        """各トピックに対する尤度の出力"""
        for what_topic in topics:
            print("log P({0}|test) =".format(what_topic),
                  nb.score(get_test_data, topic))

    for topic in topics:
        topic_number += 1
        test_number = article_num[topic_number-1][0] + 1
        while article_num[topic_number-1][1]+1 > test_number:
            f_test = open('BagOfWords/{0}/{0}{1}{2}{3}_{4}.txt'.format(
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

            total_tests += 1
            if nb.classify(test_data) == correct_topic:
                count += 1
            else:
                print("{0}{1}{2}{3}_{4}.txt".format(
                    topic, str(year), str(month), str(day),
                    str(test_number-1)),
                    "{ Correct Topic: ", correct_topic, "}")
                # print(nb.classify(test_data))
                output_prob(test_data)   # 各トピックに対する尤度の出力
    print("Train: ", nb)
    print("Accuracy: ", count / total_tests, "(", count, "/", total_tests, ")")


TOPICS = ("base", "socc", "moto", "spo", "horse", "golf", "fight")
PAGES = 4


if __name__ == "__main__":
    # 訓練データ（教師有）の作成_スクレイピング
    (get_data, get_article_num_list) = scraping(TOPICS, PAGES)
    # 訓練データ（教師有）の作成_事前に取得したデータの利用
    # (data, article_num_list) = train_local(topics)

    # ナイーブベイズ分類器を訓練
    nb = NaiveBayes()
    nb.train(get_data)
    print(nb)

    print(get_article_num_list)
    # テストデータのトピックを予測
    test_nb(TOPICS, get_article_num_list)
