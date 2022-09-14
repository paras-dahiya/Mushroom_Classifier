import sqlite3
from datetime import datetime
from os import listdir
import os
import re
import json
import shutil
import pandas as pd
from application_logging.logger import App_logger

class Prediction_Data_validation:

    '''
    This class shall be used for handling all the validation done on the Raw Prediction
    Data!!.
    '''


    def __init__(self,path):
        self.Batch_Directory = path
        self.schema_path ='schema_prediction.json'
        self.logger = App_logger()

    def valuesFromSchema(self):

       '''
                Method Name : valuesFromSchema
                Description : This method extracts all the relevent information from the predined "Schema" file.
                Output : LenghtOfDataStampInFIle , LengthofTimeStampInFile,column_names, Number of columns
                On Failure : Raise ValueError, KeyError,Exception
         '''
       try:
            with open(self.schema_path,'r') as f:
                dic = json.load(f)
                f.close()
            pattern = dic['SampleFileName']
            LengthOfDateStampInFile = dic['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dic['LengthOfTimeStampInFile']
            column_names = dic['ColName']
            NumberofColumns = dic['NumberofColumns']

            file = open("Prediction_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            message = "LengthOfDateStampInFile:: %s" %LengthOfDateStampInFile + "\t" + "LengthOfTimeStampInFile:: %s" %LengthOfTimeStampInFile + "\t" + "NumberofColumns:: %s" %NumberofColumns + "\n"
            self.logger.log(file,message)
            file.close()

       except ValueError:
            file=open("Prediction_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            self.logger.log(file,"ValueError:Value not found inside schema_training.json")
            file.close()
            raise ValueError

       except KeyError:
            file = open("Prediction_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            self.logger.log(file, "KeyError:Key value error incorrect key passed")
            file.close()
            raise KeyError

       except Exception as e:
            file = open("Prediction_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            self.logger.log(file, str(e))
            file.close()
            raise e
       return LengthOfDateStampInFile,LengthOfTimeStampInFile, column_names, NumberofColumns

    def manualRegexCreation(self):
        """
                Method Name: manualRegexCreation
                Description: This method contains a manually defined regex based on the the "FileName" given in the "Schema" file.
                             This Regex is used to validate the filename of the training data.
                Output: Regex patern
                On Failure: None
        """

        regex = "['mushroom']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    def createDirectoryForGoodBadRawData(self):
        '''
                Method Name: createDirectoryForGoodBadRawData
                Description: This method creates directories to store the Good Data and the Bad Data after validating the training data.
                Output: None
                On Failure: OSError
        '''

        try:
            path = os.path.join("Prediction_Raw_files_validated/", "Good_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)
            path = os.path.join("Prediction_Raw_files_validated/", "Bad_Raw/")
            if not os.path.isdir(path):
                os.makedirs((path))

        except OSError as ex:
            file = open("Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while creating Directory %s:" % ex)
            file.close()
            raise OSError

    def deleteExistingGoodDataTrainingFolder(self):
        '''
                        Method Name:deleteExistingGoodDataTrainingFolder
                        Description: This method deletes the directory made to store the Good Data after loading the data in the table.
                                     Once the good files are loaded in the DB, deleting the directory ensures space optimization.
                        Output: None
                        On Failure: OSError
        '''
        try:
            path = 'Prediction_Raw_files_validated/'
            # if os.path.isdir("ids/ + userName):
            # if os.path.isdir(path + 'Bad_Raw/'):
            #    shutil.rmtree(path + 'Bad_Raw/')
            if os.path.isdir(path + 'Good_Raw/'):
                shutil.rmtree(path + 'Good_Raw/')
                file = open('Prediction_Logs/GeneralLog.txt', 'a+')
                self.logger.log(file, "GoodRaw directory deleted successfully!!!")
                file.close()
        except OSError as s:
            file = open("Prediciton_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while Deleting Directory : %s" % s)
            file.close()
            raise OSError

    def deleteExistingBadDataTrainingFolder(self):

        '''
                        Method Name: deleteExistingBadDataTrainingFolder
                        Description: This method deletes the directory made to store the bad Data.
                        Output: None
                        On Failure: OSError
        '''

        try:
            path = 'Prediction_Raw_files_validated/'
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
                file = open("Prediction_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file, "BadRaw directory deleted before starting validation!!!")
                file.close()
        except OSError as s:
            file = open("Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while Deleting Directory : %s" % s)
            file.close()
            raise OSError

    def moveBadFilesTOArchiveBad(self):
        '''
                        Method Name: moveBadFilesTOArchiveBad
                        Description: This method deletes the directory made to store the Bad Data after moving the data in the archive folder.We archive the bad files to send them back to the client for invalid data issue.
                        Output: None
                        On Failure: OSError
        '''
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            source = 'Prediction_Raw_files_validated/Bad_Raw/'
            if os.path.isdir(source):
                path = 'PredictionArchiveBadData'
                if not os.path.isdir(path):
                    os.makedirs(path)
                dest = 'PredictionArchiveBadData/BadData_' + str(date) + "_" + str(time)
                if not os.path.isdir(dest):
                    os.makedirs(dest)
                files = os.listdir(source)
                for f in files:
                    if f not in os.listdir(dest):
                        shutil.move(source + f, dest)
                file = open("Prediction_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file, "Bad files moved to archive")
                path = 'Prediction_Raw_files_validated/'
                if os.path.isdir(path + 'Bad_Raw/'):
                    shutil.rmtree(path + 'Bad_Raw/')
                self.logger.log(file, "Bad Raw Data Folder Deleted successfully!!")
                file.close()

        except Exception as e:
            file = open("Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while moving bad files to archive:: %s" % e)
            file.close()
            raise e

    def validationFileNameRaw(self, regex, LengthofDateStampInFIle, LengthofTimeStampInFile):
        '''
                        Method Name: validationFileNameRaw
                        Description: This Function validates the name of the training csv file as per given name in the schema!
                                        Regex pattern is used to do the validation.If name format do not match the file is moved to Bad Raw Data folder else in Good Raw
                        Output: None
                        On Failure: Exception

        '''
        # pattern ="['mushroom']+['\_'']+[\d]+[\d]+\.csv"
        # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        self.deleteExistingBadDataTrainingFolder()
        self.deleteExistingGoodDataTrainingFolder()
        onlyfiles = [f for f in listdir(self.Batch_Directory)]
        try:
            # create new directories
            self.createDirectoryForGoodBadRawData()
            f = open("Prediction_Logs/nameValidationLog.txt", 'a+')
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = re.split('_', splitAtDot[0])
                    if len(splitAtDot[1]) == LengthofDateStampInFIle:
                        if len(splitAtDot[2]) == LengthofTimeStampInFile:
                            shutil.copy("Prediction_Batch_FIles/" + filename, "Prediction_Raw_files_validated/Good_Raw")
                            self.logger.log(f, "Valid File name!! File moved to GoodRaw Folder :: %s" % filename)

                        else:
                            shutil.copy("Prediction_Batch_Files/" + filename, "Prediction_Raw_files_validated/Bad_Raw")
                            self.logger.log(f, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)

                    else:
                        shutil.copy("Prediction_Batch_Files/" + filename, "Prediction_Raw_files_validated/Bad_Raw")
                        self.logger.log(f, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)

                else:
                    shutil.copy("Prediction_Batch_Files/" + filename, "Prediction_Raw_files_validated/Bad_Raw")
                    self.logger.log(f, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)

            f.close()

        except Exception as e:
            f = open("Prediction_Logs/nameValidationLog.txt", 'a+')
            self.logger.log(f, "Error occured while validating FileName %s" % e)
            f.close()
            raise e

    def validateColumnLength(self, NumberofColumns):
        '''
                    Method Name: validateColumnLength
                    Description: This function validates the number of columns in the csv files.
                                 It should be same as given in the schema file.
                                 If not the same file is not suitable for processing and thus is moved to Bad Raw Data folder.
                                 If the column number matches, file is kept in the Good RAw Data For processing.
                                 The csv file is missing the first column name, this function changes the missing name to "mushroom".
                    Output: None
                    On Failure: Exception
        '''

        try:
            f = open("Prediction_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Column Length Validation Started!!")
            for file in listdir('Prediction_Raw_files_validated/Good_Raw/'):
                csv = pd.read_csv("Prediction_Raw_files_validated/Good_Raw/" + file)

                if csv.shape[1] == NumberofColumns:
                    pass
                else:
                    shutil.move("Prediction_Raw_files_validated/Good_Raw/" + file, "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log(f, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
            self.logger.log(f, "Column Length Validation Completed!!")


        except OSError:
            f = open("Prediction_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
            f.close()
            raise OSError

        except Exception as e:
            f = open("Prediction_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured:: %s" % e)
            f.close()
            raise e
        f.close()

    def deletePredictionFile(self):

        if os.path.exists('Prediction_Output_File/Predictions.csv'):
            os.remove('Prediction_Output_File/Predictions.csv')

    def validateMissingValuesInWholecolumn(self):
        '''
                    Method Name: validateMissingValuesInWholecolumn
                    Description: This function validates if any column in the csv file has all values missing.
                                 If all the values are missing the file is not suitable for processing.Such files are moved to Bad Raw data.
                    Output: None
                    On Failure: Exception
        '''
        try:
            f = open("Prediction_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(f, "Missing Values Validation Started!!")

            for file in listdir('Prediction_Raw_files_validated/Good_Raw/'):
                csv = pd.read_csv("Prediction_Raw_files_validated/Good_Raw/" + file)

                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count += 1
                        shutil.move("Prediction_Raw_files_validated/Good_Raw/" + file,
                                    "Prediction_Raw_files_validated/Bad_Raw")
                        self.logger.log(f,"Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                        break

                if count == 0:
                    csv.rename(columns={"Unnamed: 0": "Mushroom"}, inplace=True)
                    csv.to_csv("Prediction_Raw_files_validated/Good_Raw/" + file, index=None, header=True)
        except OSError:
            f = open("Prediction_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
            f.close()
            raise OSError
        except Exception as e:
            f = open("Prediction_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(f, "Error Occured:: %s" % e)
            f.close()
            raise e
        f.close()