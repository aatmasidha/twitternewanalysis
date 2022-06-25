from datetime import datetime
import json
import logging
import os
import re

from bs4 import BeautifulSoup
from jproperties import Properties
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests

from tweetdataextract import ProcessText
from classification import bbcnews
# import bbcnews
import text2emotion as te
from nltk.corpus import subjectivity
nltk.download('omw-1.4')
import configparam.readconfigparam as readConfig


logger = logging.getLogger(__name__)
# logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

configs = Properties()


def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'  


def emotion_detection_text2emotion(x):
    try:
        all_emotions_value = te.get_emotion(x)
        Keymax_value = max(zip(all_emotions_value.values(), all_emotions_value.keys()))[1]
        return Keymax_value
    except BaseException as err:
            logger.error(f"Unexpected {err=}, {type(err)=}")


def findSensitivityAndPolarity(urlList, tagToParse):

    values = {}
    # url = "https://www.bbc.co.uk/news/uk-61681066"
    
    # Sometimes there are multiple news URLs associated with the news we will consolidate all the news as one content 
    # to understand the sensitivity of the news
    for url in urlList:
       
        logger.debug("Getting news details from URL:" + url)
        res = requests.get(url)
        html_page = res.content    
        
        soup = BeautifulSoup(html_page, 'html.parser')
        text = soup.get_text()
        articleString = ''
        
        # The contents are present inside the article tag in the HTML. This is general observation for 
        # most of the news papers
        # for content in soup.find_all("article"):
        for content in soup.find_all(tagToParse):
            articleString = articleString + content.text
            articleString = articleString.strip("\n")
        
        # We remove some characters from the content like @
        articleString = ProcessText.clean_tweet(articleString)
                
        # allwords = ' '.join(articleString)
        # wordCloud = WordCloud(width = 500, height = 30, random_state = 21, max_font_size = 119).generate(allwords)
        # plt.imshow(wordCloud, interpolation = "bilinear")
        # plt.axis("off")
        # plt.show()
        
        sid = SentimentIntensityAnalyzer()
        values['articleText'] = articleString
        scores = sid.polarity_scores(articleString)
        values['score'] = scores

        article = values.get('articleText')
        
        # Initializing the sentiment analyser
        logger.debug("Start - Processing the sensitivity by ProcessText....")
        logger.debug("Article Contents are:" + article)
        article = ProcessText.clean_tweet(article)
        logger.debug("Sentiment value using SentimentIntensityAnalyzer():", scores)
        emotion = emotion_detection_text2emotion(article)
        values['emotion'] = emotion
        classifier = bbcnews.classifier(article)
        values['classifier'] = classifier
        logger.debug("Start - Processing the sensitivity by ProcessText....")
        logger.debug(ProcessText.getSubjectivity(articleString))
        values['subjectivity'] = ProcessText.getSubjectivity(articleString)
        logger.debug(getAnalysis(ProcessText.getPolarity(articleString)))  
        values['polarity'] = ProcessText.getPolarity(articleString)      
        logger.debug("End - Processing the sensitivity by ProcessText....")
        
        return values


def readFilesFromDailyPath():
    jsonPath = '.'
    logger.info('readJSONFile')
    configParam = readConfig.readConfigurationFile()
    jsonPath = configParam['jsonPath']
    
    jsonFiles = os.listdir(jsonPath)
 
    print("Files and directories in '", jsonPath, "' :")
 
    # prints all files
    print(jsonPath)
    validFileList = []
    for file in jsonFiles:
        match = re.search("\.json$", file)
        if match:
            print("The file ending with .json is:", file)
            validFileList.append(file)
         
    if(len(validFileList)):
        for file in validFileList:
            # readJSONFile( jsonFolder + "/" + file)
            readJSONFileByItems(jsonPath + "/" + file)
            

def readJSONFileByItems(fileName):
    try:
        # Read the json file by news handler
        with open(fileName) as json_file:
            data = json.load(json_file)
            
            # Read news agency details from the json file
            for newAgencyKey, newsAgencyvalue in data.items():
                newsAgencyName = ''
                captureDate = ''

                newsDetailsValue = newsAgencyvalue['newsAccountDetails']
                
                newsAgencyName = newsDetailsValue['newsaccounthandler']
                captureDate = newsDetailsValue['capturedate']
                jsonString = ''
                jsonDump = {}
                             
                cnt = 0
                # Read one by one news details                   
                for newsDetailsKey in newsAgencyvalue:
                    newsDetailsValue = newsAgencyvalue[newsDetailsKey]
                    print(newAgencyKey, newsDetailsKey)
                    if('newsId_' in  newsDetailsKey):
                        newsId = ''
                        text = ''
                        retweetCount = 0
                        likesCount = 0
                        commentCount = 0
                        positiveCommentCnt = 0 
                        negativeCommentCnt = 0
                        neutralCommentCnt = 0
                        record = {}
                        for newsKey, newsValue in newsDetailsValue.items():
                            if newsKey == 'id':
                                newsId = newsValue
                                record['newsId'] = newsId
                            elif newsKey == 'text':
                                text = ProcessText.clean_tweet(newsValue)
                                record['text'] = text
                                emotion_value = emotion_detection_text2emotion(text)
                                urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
                                print("Original string: ", text)
                                
                                if(urls != None and  len(urls) > 0):
                                    print("Urls: ", urls)
                                    # newAgencyName =      
                                    tagToParse = 'article'
                                    if(newsAgencyName == 'AJEnglish'):
                                        tagToParse = 'p'
                                    newsDetailsAndSensitivity = findSensitivityAndPolarity(urls, tagToParse)
                                    record['articleText'] = newsDetailsAndSensitivity['articleText'] 
                                    record['score'] = newsDetailsAndSensitivity['score']                                 
                                      
                            elif newsKey == 'newsRetweetDict':
                                reTweetRecords = newsValue
                                retweetCount = len(reTweetRecords)
                                record['reTweetCount'] = retweetCount 
                            elif  newsKey == 'newsLikesDict':
                                likesRecord = newsValue
                                likesCount = len(likesRecord) 
                                record['likesCount'] = likesCount
                            elif newsKey == 'newsCommentDict':
                                for commentKey, commentValue in newsValue.items():
                                    for commentTagKey, commentTagValue in commentValue.items():
                                        if commentTagKey == 'commentText':
                                            # commentTagValue = ProcessText.cleanText(commentTagValue)
                                            commentTagValue = ProcessText.clean_tweet(commentTagValue)
                                            if(not (commentTagValue and not commentTagValue.isspace())):
                                                print("Empty String")
                                                polarity = "Neutral"
                                                neutralCommentCnt = neutralCommentCnt + 1
                                            else:
                                                
                                                sid = SentimentIntensityAnalyzer()
                                                scores = sid.polarity_scores(commentTagValue)
                                                
                                                emotion = emotion_detection_text2emotion(commentTagValue)
                                                classifier = bbcnews.classifier(commentTagValue)
                                                subjectivity = ProcessText.getSubjectivity(commentTagValue)
                                                polarity = getAnalysis(ProcessText.getPolarity(commentTagValue))
                                                if(polarity == 'Positive'):
                                                    positiveCommentCnt = 1 + positiveCommentCnt
                                                elif(polarity == 'Negative'):
                                                    negativeCommentCnt = 1 + negativeCommentCnt
                                                elif(polarity == 'Neutral'):
                                                    neutralCommentCnt = 1 + neutralCommentCnt
                                    
                                commentRecord = newsValue   
                                commentCount = len(commentRecord)
                                record['commentCount'] = commentCount
                                record['positvecommentcount'] = positiveCommentCnt
                                record['negativecommentcount'] = negativeCommentCnt
                                record['neutralcommentcount'] = neutralCommentCnt
                                record['location'] = commentValue['location']
                        if "commentCount" not in record:
                            record['commentCount'] = 0
                            record['positvecommentcount'] = 0
                            record['negativecommentcount'] = 0
                            record['neutralcommentcount'] = 0
                            
                        jsonDump[newsId] = record
                        cnt = cnt + 1
                        record['newshandler'] = newsAgencyName    
                        record['capturedate'] = captureDate
            
                # jsonDump[cnt] = record
        
                        
        executionDateTime = datetime.now()
        timeStr = executionDateTime.strftime("%H_%M_%S")        
        dateStr = (executionDateTime).strftime("%Y_%m_%d")    
        with open(newsAgencyName + "_" +"output" + "_" + dateStr + "_T_" + timeStr + ".json", 'w', encoding='utf8') as json_file:
            json.dump(jsonDump, json_file, indent=4)
                        
    except json.decoder.JSONDecodeError as jsonErr:
            logger.error("Could not parse the file:" + fileName)
            logger.error(f"Unexpected {jsonErr=}, {type(jsonErr)=}")
            
    except BaseException as err:
            logger.error(f"Unexpected {err=}, {type(err)=}")

        
def main(): 
    logger.info('Started')
    readFilesFromDailyPath()
    
    
if __name__ == '__main__':
    main()
