import pandas as pd
from sklearn import preprocessing
import numpy as np

class Collection1:
    def parse_telecom_data(filename_train,filename_test):
        '''
        Takes filename and returns X and Y after applying label encoding and OneHotEncoding

        Input:
            filename: name of CSV file to read
        Output:
            X: nparray of X data
            Y: nparray of labels
        '''
        X_train_in = pd.read_csv(filename_train)
        X_test_in = pd.read_csv(filename_test)
        ID_test = X_test_in[['customerID']].to_numpy()

        # get X with categorical data
        X_train_cat = X_train_in.drop(columns=['customerID','tenure','MonthlyCharges','TotalCharges','Discontinued'])
        X_test_cat = X_test_in.drop(columns=['customerID','tenure','MonthlyCharges','TotalCharges'])

        # get X with numeric data
        X_train_num = X_train_in[['tenure','MonthlyCharges']].to_numpy()
        X_test_num = X_test_in[['tenure','MonthlyCharges']].to_numpy()

        # get Y
        Y_train_cat = X_train_in.filter(['Discontinued'])

        # convert labels to numeric using LabelEncoder
        le = preprocessing.LabelEncoder()
        colList = X_train_cat.columns
        ncol_le = len(colList)
        m_train_rows = X_train_cat.shape[0]
        m_test_rows = X_test_cat.shape[0]
        X_train_le = np.zeros((m_train_rows,ncol_le))
        X_test_le = np.zeros((m_test_rows,ncol_le))
        for jdx in range(0,ncol_le):
            le.fit(X_train_cat[colList[jdx]])
            X_train_le[:,jdx] = le.transform(X_train_cat[colList[jdx]])
            X_test_le[:,jdx] = le.transform(X_test_cat[colList[jdx]])
        
        # get labels for training data
        Y_train = Y_train_cat.apply(le.fit_transform)
        Y_train = Y_train.to_numpy()

        # use OneHotEncoder (applied to encoded labels)
        enc = preprocessing.OneHotEncoder()
        enc.fit(X_train_le)
        X_train_ohl = enc.transform(X_train_le).toarray()
        X_test_ohl = enc.transform(X_test_le).toarray()

        # combine data
        n_ohl = X_train_ohl.shape[1]
        n_num = X_train_num.shape[1]
        X_train = np.zeros((m_train_rows,n_ohl+n_num))
        X_test = np.zeros((m_test_rows,n_ohl+n_num))
        X_train[:,0:n_ohl] = X_train_ohl
        X_test[:,0:n_ohl] = X_test_ohl
        X_train[:,n_ohl:] = X_train_num
        X_test[:,n_ohl:] = X_test_num

        return X_train,Y_train,X_test,ID_test
    

class Collection2:    
    def parse_total_charges(total_charges_arr):
        """
        Takes total charges and gets rid of NaN values
        All NaN's are associated with customers who have just started service
        Set value to zero
        Input:
            total_charges_arr: nparray of total charges
        Output:
            total_charges_arr: nparray of total charges (without NaNs)
        """
        for idx in range(0,len(total_charges_arr)):
            if np.isnan(total_charges_arr[idx]):
                total_charges_arr[idx] = 0
        return total_charges_arr

    def parse_total_internetService(internetService_arr):
        """
        Takes type of internet service and assigns a score
        Fiber optic is the highest end, score of 10
        DSL is the lower end, score of 3
        No internet gets a score of 0
        Input:
            internetService_arr: nparray of internet service type
        Output:
            internetService_arr: nparray of internet service score
        """
        for idx in range(0,len(internetService_arr)):
                if internetService_arr[idx] == "DSL":
                    thisVal = 3
                elif internetService_arr[idx] == "Fiber optic":
                    thisVal = 10
                elif internetService_arr[idx] == "No":
                    thisVal = 0
                else:
                    print("Contract length not specified properly")
                internetService_arr[idx] = thisVal
        return internetService_arr

    def parse_total_contract(contract_arr):
        """
        Takes contract length (string) and returns numerica value
        Input:
            contract_arr: nparray of contract length (string)
        Output:
            contract_arr: nparray of contract length (float)
        """
        for idx in range(0,len(contract_arr)):
                if contract_arr[idx] == "Month-to-month":
                    thisVal = 1/12
                elif contract_arr[idx] == "One year":
                    thisVal = 1
                elif contract_arr[idx] == "Two year":
                    thisVal = 2
                else:
                    print("Contract length not specified properly")
                contract_arr[idx] = thisVal
        return contract_arr

    def parse_telecom_data(filename_train,filename_test):
        '''
        Takes filename and returns data after applying label encoding and OneHotEncoding

        Input:
            filename_train: name of CSV file for train data
            filename_test: name of CSV file for test data
        Output:
            X_train: nparray of X train data
            Y_train: nparray of labels for train data
            X_train: nparray of X test data
            ID_test: nparray of customer ID for test data
        '''
        X_train_in = pd.read_csv(filename_train)
        X_test_in = pd.read_csv(filename_test)

        # get customer ID for test data
        ID_test = X_test_in[['customerID']].to_numpy()

        # get X with categorical data
        catData_list = ['customerID','tenure','InternetService','Contract','MonthlyCharges','TotalCharges']
        X_train_cat = X_train_in.drop(columns=catData_list)
        X_test_cat = X_test_in.drop(columns=catData_list)

        # get X with numeric data
        numData_list = ['tenure','MonthlyCharges','TotalCharges']
        X_train_num = X_train_in[numData_list].to_numpy()
        X_test_num = X_test_in[numData_list].to_numpy()

        # handle total charges
        X_train_num[:,2] = Collection2.parse_total_charges(X_train_num[:,2])
        X_test_num[:,2] = Collection2.parse_total_charges(X_test_num[:,2])
        
        # handle contract length
        X_train_contract = X_train_in[['Contract']].to_numpy()
        X_test_contract = X_test_in[['Contract']].to_numpy()
        X_train_contract = Collection2.parse_total_contract(X_train_contract)
        X_test_contract = Collection2.parse_total_contract(X_test_contract)
        
        # handle internet service type
        X_train_internetService = X_train_in[['InternetService']].to_numpy()
        X_test_internetService = X_test_in[['InternetService']].to_numpy()
        X_train_internetService = Collection2.parse_total_internetService(X_train_internetService)
        X_test_internetService = Collection2.parse_total_internetService(X_test_internetService)

        # augment numerical data with contract and internetService
        X_train_num = np.concatenate((X_train_num, X_train_contract,X_train_internetService), axis=1)
        X_test_num = np.concatenate((X_test_num, X_test_contract,X_test_internetService), axis=1)

        # get Y
        X_train_cat = X_train_cat.drop(columns=['Discontinued'])
        Y_train_cat = X_train_in.filter(['Discontinued'])

        # convert labels to numeric using LabelEncoder
        le = preprocessing.LabelEncoder()
        colList = X_train_cat.columns
        ncol_le = len(colList)
        m_train_rows = X_train_cat.shape[0]
        m_test_rows = X_test_cat.shape[0]
        X_train_le = np.zeros((m_train_rows,ncol_le))
        X_test_le = np.zeros((m_test_rows,ncol_le))
        for jdx in range(0,ncol_le):
            le.fit(X_train_cat[colList[jdx]])
            X_train_le[:,jdx] = le.transform(X_train_cat[colList[jdx]])
            X_test_le[:,jdx] = le.transform(X_test_cat[colList[jdx]])
        
        # get labels for training data
        Y_train = Y_train_cat.apply(le.fit_transform)
        Y_train = Y_train.to_numpy()

        # use OneHotEncoder (applied to encoded labels)
        enc = preprocessing.OneHotEncoder()
        enc.fit(X_train_le)
        X_train_ohl = enc.transform(X_train_le).toarray()
        X_test_ohl = enc.transform(X_test_le).toarray()

        # combine data
        n_ohl = X_train_ohl.shape[1]
        n_num = X_train_num.shape[1]
        X_train = np.zeros((m_train_rows,n_ohl+n_num))
        X_test = np.zeros((m_test_rows,n_ohl+n_num))
        X_train[:,0:n_ohl] = X_train_ohl
        X_test[:,0:n_ohl] = X_test_ohl
        X_train[:,n_ohl:] = X_train_num
        X_test[:,n_ohl:] = X_test_num

        return X_train,Y_train,X_test,ID_test