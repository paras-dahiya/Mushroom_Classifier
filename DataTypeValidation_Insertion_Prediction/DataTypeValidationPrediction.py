import shutil
import sqlite3
from os import listdir
import os
import csv
from application_logging.logger import App_logger

class dBOperation:
    '''
            This class be used for handling all the SQL operation
    '''

    def __init__(self):
        self.path = 'Prediction_Database/'
        self.badFilePath = "Prediction_Raw_files_validated/Bad_Raw"
        self.goodFilePath = "Prediction_Raw_files_validated/Good_Raw"
        self.logger = App_logger()

    def dataBaseConnection(self,DatabaseName):
        '''
                        Method Name: dataBaseConnection
                        Description: This Method creates the database with the given name and if Database already exists then opens the connection to the DB.
                        Output: Connection to the DB
                        On failure: Raise ConnectionError
        '''

        try:
            conn = sqlite3.connect(self.path+DatabaseName+'.db')

            file = open("Prediction_Logs/DatabaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Opened %s database successfully" %DatabaseName)
        except ConnectionError:
            file = open("Prediction_Logs/DatabaseConnectionLog.txt",'a+')
            self.logger.log(file, "Error while connecting to database: %s" %ConnectionError)
            file.close()
            raise ConnectionError
        return conn

    def createTableDb(self,DatabaseName,column_names):
        '''
                        Method Name: createTableDb
                        Description: This method creates a table in the given database which will be used to insert the Good data after the raw data validation>
                        Output:None
                        On Failure: Raise Exception
        '''

        try:
            conn = self.dataBaseConnection(DatabaseName)
            c = conn.cursor()
            c.execute("SELECT count(name)  FROM sqlite_master WHERE type = 'table'AND name = 'Good_Raw_Data'")
            if c.fetchone()[0] == 1:
                conn.close()
                file = open("Prediction_Logs/DbTableCreateLog.txt", 'a+')
                self.logger.log(file, "Tables created successfully!!")
                file.close()

                file = open("Prediction_Logs/DataBaseConnectionLog.txt", 'a+')
                self.logger.log(file, "Closed %s database successfully" % DatabaseName)
                file.close()

            else:

                for key in column_names.keys():
                    type = column_names[key]

                    # in try block we check if the table exists, if yes then add columns to the table
                    # else in catch block we will create the table
                    try:
                        # cur = cur.execute("SELECT name FROM {dbName} WHERE type='table' AND name='Good_Raw_Data'".format(dbName=DatabaseName))
                        conn.execute(
                            'ALTER TABLE Good_Raw_Data ADD COLUMN "{column_name}" {dataType}'.format(column_name=key,
                                                                                                     dataType=type))
                    except:
                        conn.execute('CREATE TABLE  Good_Raw_Data ({column_name} {dataType})'.format(column_name=key,
                                                                                                     dataType=type))
            '''
            conn = self.dataBaseConnection(DatabaseName)
            conn.execute('DROP TABLE IF EXISTS Good_Raw_Data;')
            
            for key in column_names.keys():
                type = column_names[key]

                # we will remove the column of string datatype before loading as it is not needed for training
                # in try block we check if the table exists, if yes then add columns to the table
                # else in catch block we create the table
                try:
                    # cur = cur.execute("SELECT name FROM {dbName} WHERE type='table' AND name='Good_Raw_Data'".format(dbName=DatabaseName))
                    conn.execute('ALTER TABLE Good_Raw_Data ADD COLUMN "{column_name}" {dataType}'.format(column_name=key,dataType=type))
                except:
                    conn.execute('CREATE TABLE  Good_Raw_Data ({column_name} {dataType})'.format(column_name=key,dataType=type))
        '''
            conn.close()

            file = open("Prediction_Logs/DbTableCreateLog.txt", 'a+')
            self.logger.log(file, "Tables created successfully!!")
            file.close()

            file = open("Prediction_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Closed %s database successfully" % DatabaseName)
            file.close()

        except Exception as e:
            file = open("Prediction_Logs/DbTableCreateLog.txt", 'a+')
            self.logger.log(file, "Error while creating table: %s " % e)
            file.close()
            conn.close()
            file = open("Prediction_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Closed %s database successfully" % DatabaseName)
            file.close()
            raise e

    def insertIntoTableGoodData(self, Database):
        '''
                        Method Name: insertIntoTableGoodData
                        Description: This method inserts the Good data files from the Good_Raw folder into the above created table.
                        Output: None
                        On failure:Raise Exception
        '''

        conn = self.dataBaseConnection(Database)
        goodFilePath = self.goodFilePath
        badFilePath = self.badFilePath
        onlyfiles = [f for f in listdir(goodFilePath)]
        log_file = open("Prediction_Logs/DBInsertLog.txt", 'a+')

        for file in onlyfiles:
            try:
                with open(goodFilePath + '/' + file, 'r') as f:
                    next(f)
                    reader = csv.reader(f, delimiter="\n")
                    for line in enumerate(reader):
                        for list in (line[1]):
                            try:
                                conn.execute('INSERT INTO Good_Raw_Data values ({values})'.format(values=(list)))
                                self.logger.log(log_file, " %s: File loaded successfully!!" % file)
                                conn.commit()
                            except Exception as e:
                                raise e
            except Exception as e:
                conn.rollback()
                self.logger.log(log_file, "Error while creating table: %s" % e)
                shutil.move(goodFilePath + '/' + file, badFilePath)
                self.logger.log(log_file, "File Moved Successfully %s" % file)
                log_file.close()
                raise e
            conn.close()
            log_file.close()


    def selectingDatafromtableintocsv(self,Database):
        '''
                            Method Name:selectingDatafromtableintocsv
                            Description: This method exports the data in the GoodData tabe as a CSV file in a given location above created
                            Output: None
                            On Failier: Raise Exception
        '''

        self.fileFromDb= 'Prediction_FileFromDb/'
        self.fileName = 'InputFIle.csv'
        log_file = open("Prediction_Logs/ExportToCsv.txt", 'a+')
        try:
            conn = self.dataBaseConnection(Database)
            sqlselect = "SELECT * FROM Good_Raw_Data"
            cursor = conn.cursor()

            cursor.execute(sqlselect)

            results= cursor.fetchall()
            # GET the headers of the csv file
            headers = [i[0] for i in cursor.description]

            #Make the csv pit directory
            if not os.path.isdir(self.fileFromDb):
                os.makedirs(self.fileFromDb)

            #open CSV file for writing
            csvFile = csv.writer(open(self.fileFromDb + self.fileName, 'w',newline=''),delimiter=',',lineterminator='\r\n',quoting=csv.QUOTE_ALL,escapechar='\\')
            #Add the header and data to the CSV file
            csvFile.writerow(headers)
            csvFile.writerows(results)

            self.logger.log(log_file, "File exported successfully!!!")
            log_file.close()

        except Exception as e:
            self.logger.log(log_file, "File exporting failed. Error : %s" % e)
            log_file.close()
            raise e

