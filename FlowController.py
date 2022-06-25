import argparse
import logging

import debugpy
from debugpy.common.json import default
from jproperties import Properties
from pipenv.patched.notpip._internal.utils import distutils_args

from tweetdataextract import tweeterdatahandler

logging.config.fileConfig('./configparam/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
configs = Properties()

#virtual colonisation are we able to proove

#This is the main entrance file which controls the flow of the program. The module takes 3 input parameters
#if use does not set any parameter then by default only the data from twitter handlers is read.  
# Data is not analysed by default. 
#it takes the parameters to understand if user wants to get data from twitter
#

def parseArguments():
    argDict = {'getdata':True, 'analysenewscategory': False, 'analysenewsemotion': False}
    try:
        parser = argparse.ArgumentParser(description="Arguments For Tweet Extraction and  Analysis",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--getdata", help="Get data from Twitter", default=True)
        parser.add_argument("--analysenewscategory", help="Get news category", default=False)
        parser.add_argument("--analysenewsemotion", help="Get news emotion", default=False)
        
        args = parser.parse_args()
        print(args.getdata)
        
        argDict['getdata'] = bool(args.getdata)
        argDict['analysenewscategory'] = bool(args.analysenewscategory)
        argDict['analysenewsemotion'] = bool(args.analysenewsemotion)
        return argDict
    
    except BaseException as err:
        logger.error(f"Unexpected {err=}, {type(err)=}")

    
def main():
    logger.debug("Hello World!")
    
    argDict = argumentsForProcessing = parseArguments()
    if(argDict['getdata'] == True):
        tweeterdatahandler.getNewsDataFromTwitterHandlers()
    # elif(argDict['analysenewscategory'] == True)
        
    
    print(argDict)


if __name__ == "__main__":
    main()
