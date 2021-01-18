import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from sklearn.metrics import f1_score, classification_report, roc_auc_score

def final_fun_1(user_id):
    data= get_user_product_info(user_id) # this data will be saved in company's database 
    									 # for each user based on past purchase details.
    print(data, flush=True)
    if data.size ==0:
        return []
    
    model= load_model('model_lgb') # load trained model
    data_pred = (model.predict_proba(data)[:,1] >= 0.21).astype(int)
    print(data_pred)
    data['prediction']= data_pred
    
    #creating a file
    result= []
    for row in data.itertuples(): #for each row in dataframe
        if row.prediction==1: #if for an order_id and a product_id the prediction is 1.
            result.append(row.product_id)
    print(result)
    result= get_product_name(result)

    return result


def final_fun_2(X_test,y_test):
	model= load_model('model_lgb')
	 
	y_pred = (model.predict_proba(X_test)[:, 1] >= 0.21).astype('int')

	print(y_pred[:10])
	#Evaluation.
	print('F1 Score: {}'.format(f1_score(y_pred, y_test)))
	print('AUC score: {}'.format(roc_auc_score(y_test, y_pred)))
	print(classification_report(y_pred, y_test))
	return f1_score(y_pred, y_test)
    
def get_user_product_info(usr_id):
    with open('data_train.pkl', 'rb') as f:
        data_train= pkl.load(f)
    print('data_train size is {}'.format(data_train.size), flush=True)
    print('user id type is {}'.format(type(data_train['user_id'][12])), flush=True)
    with open('data_test.pkl', 'rb') as f:
        data_test= pkl.load(f)
    data_train= data_train[data_train['user_id']==usr_id]
    print(usr_id, flush=True)
    print(data_train, flush=True)
    data_test= data_test[data_test['user_id']==usr_id]
    data_train= data_train.drop('reordered', axis=1)
    user_data= pd.concat([data_train, data_test], ignore_index=True)
    return user_data


def load_model(model_name):
    model_name= model_name+'.pkl'
    with open(model_name, 'rb') as f:
        model=pkl.load(f)
    return model


def get_product_name(product_list):
	products= pd.read_csv('data/products.csv')
	product_names= set()
	for product_id in product_list:
		product_name= products[products['product_id']==product_id].iloc[0]['product_name']
		print(product_name, flush=True)
		product_names.add(str(product_name))
	return product_names
