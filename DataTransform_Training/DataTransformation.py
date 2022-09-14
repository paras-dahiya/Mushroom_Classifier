from datetime import datetime
from os import listdir
from application_logging.logger import App_logger
import pandas as pd

class dataTransform:
    '''
            This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.
    '''

    def __init__(self):
        self.goodDataPath = 'Training_Raw_files_validated/Good_Raw'
        self.logger = App_logger()

    def addQuotesToStringValuesInColumn(self):
        '''
                        Method Name: addQuotesToStringValuesInColumn
                        Description:

        '''

        log_file=open("Training_logs/addQuotesToStringValuesInColumn.txt",'a+')
        try:
            onlyfiles = [f for f in listdir(self.goodDataPath)]
            for file in onlyfiles:
                data = pd.read_csv(self.goodDataPath+"/"+file)
                for column in data.columns:
                    data[column] = data[column].apply(lambda x: "'" + str(x) + "'")
                data.to_csv(self.goodDataPath+"/"+file, index = None,header=True)
                self.logger.log(log_file, "%s: Quotes added successfully" % file)
        except Exception as e:
            self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
            log_file.close()
        log_file.close()
