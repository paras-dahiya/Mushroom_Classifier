
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import  tuner
from file_operations import file_methods
from application_logging import logger

#Creating the common Logging Object

class trainModel:

    def __init__(self):
        self.log_writer = logger.App_logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt",'a+')
    def trainingModel(self):
        #Logging the start of Training
        self.log_writer.log(self.file_object,'Start of Training')
        try:
            #Getting the data from the source
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()

            """Doing the data preprocessing"""
            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)

            #removing unwanted columns as discussed in the EDA part in ipynb file
            data= preprocessor.dropUnnecessaryColumns(data,['veiltype'])

            #replacing '?' values with np.nan as discussed in the EDA part
            data= preprocessor.replaceInvalidValuesWithNull(data)

            #checking if missing values are present in the dataset

            is_null_present,cols_with_missing_values=preprocessor.is_null_present(data)

            #if missing values are ther, replace them appropriately
            if(is_null_present):
                #missing values imputation
                data= preprocessor.impute_missing_values(data,cols_with_missing_values)

            # get encoded value for categorical data
            data = preprocessor.encodeCategoricalValues(data)

            #create seperate features and labels
            X, Y = preprocessor.separate_label_feature(data,label_column_name='class')

            """ Applying the clustering approach"""
            #object initialization
            kmeans=clustering.KMeansClustering(self.file_object,self.log_writer)

            #using the elbow plot to find the number of optium cluster
            number_of_clusters=kmeans.elbow_plot(X)

            #divide the data
            X=kmeans.create_clusters(X,number_of_clusters)

            #create a new column in the dataset consisting of the corresponding cluster assignments.
            X['labels']=Y

            #getting the unique clusters from our dataset
            list_of_clusters=X['Cluster'].unique()

            for i in list_of_clusters:
                cluster_data=X[X['Cluster']==i] #filter the data for one cluster

                #Prepare the feature and Label columns
                cluster_feature=cluster_data.drop(['labels','Cluster'],axis=1)
                cluster_label = cluster_data['labels']

                #spliting the data into training and test set for each cluster one by one
                x_train,x_test,y_train,y_test = train_test_split(cluster_feature,cluster_label,test_size=1/3)

                #object Initialization
                model_finder=tuner.Model_Finder(self.file_object,self.log_writer)

                #getting the best model for each of the clusters
                best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)

                #saving the best model to the directory.
                file_op = file_methods.File_Operation(self.file_object,self.log_writer)
                save_model=file_op.save_model(best_model,best_model_name+str(i))

            #logging the succesfull Training
            self.log_writer.log(self.file_object,'Successful End of Training')
            self.file_object.close()

        except Exception:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception