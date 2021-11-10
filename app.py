
# coding: utf-8

# In[20]:



import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
import _pickle as pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
#from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
#import lightgbm as lgb
from sklearn.preprocessing import PowerTransformer
import copy
import re
warnings.filterwarnings('ignore')


# In[25]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv1D,MaxPool1D,Input,Dense,Flatten,Embedding,Concatenate,LSTM,Dropout,BatchNormalization,MaxPooling2D,LeakyReLU,concatenate,SpatialDropout1D
from tensorflow.keras.metrics import Accuracy,AUC
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import roc_auc_score,roc_curve,auc,f1_score
from tensorflow.keras.callbacks import Callback,ModelCheckpoint,EarlyStopping,LearningRateScheduler,TensorBoard
from tensorflow.keras.backend import backend
import tensorflow.keras.backend as k
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.regularizers import l2
from keras.models import model_from_json


# In[26]:


import tensorflow
tensorflow.keras.__version__


# In[27]:


file = open("Standard_Scaler_updated1.pickle",'rb')
Stand1 = pickle.load(file)
file.close()
file = open("customer_state_dict_response.pickle",'rb')
customer_state_dict = pickle.load(file)
file.close()

file = open("order_status_dict_response.pickle",'rb')
order_status_dict = pickle.load(file)
file.close()

file = open("payment_type_dict_response.pickle",'rb')
payment_type_dict = pickle.load(file)
file.close()

file = open("seller_city_dict_response.pickle",'rb')
seller_city_dict = pickle.load(file)
file.close()

file = open("seller_state_dict_response.pickle",'rb')
seller_state_dict = pickle.load(file)
file.close()

file = open("product_category_name_english_dict_response.pickle",'rb')
product_category_name_english_dict = pickle.load(file)
file.close()

file = open("Month_Year_Purchase_Order_dict_response.pickle",'rb')
Month_Year_Purchase_Order_dict = pickle.load(file)
file.close()

file = open("Month_year_order_deliverd_dict_response.pickle",'rb')
Month_year_order_deliverd_dict = pickle.load(file)
file.close()
file = open("yeo_johnson_transformation_updated.pickle",'rb')
yt1 = pickle.load(file)
file.close()
file = open("Outlier_Updated.pickle",'rb')
out_dict1 = pickle.load(file)
file.close()

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("ModelNo3-10-0.5664.hdf5")


# In[28]:


import keras.backend as K
def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# In[30]:


def Test_Response_1(data,feature,Train_dict):
    #x = PrettyTable()
    #x = PrettyTable([feature, 'class 5', 'class 4','class 3','class 2','class 1'])
    val=list(np.unique(data[feature]))
    train_dict_keys=list(Train_dict.keys())
    test_dict=dict()
    for i in range(len(val)):
        if(val[i] in train_dict_keys):
            test_dict[val[i]]=Train_dict[val[i]]
        else:
            test_dict[val[i]]=[0.25,0.25,0.25,0.25,0.25]
    return test_dict         


# In[31]:


def _CreateFeature_ResponseCoding(data,dict_val,feature):
    data[feature+'_'+'class_5']=np.nan
    data[feature+'_'+'class_4']=np.nan
    data[feature+'_'+'class_3']=np.nan
    data[feature+'_'+'class_2']=np.nan
    data[feature+'_'+'class_1']=np.nan
    dict_keys=list(dict_val.keys())
    for i in range(len(dict_keys)):
        val_index=data[data[feature]==dict_keys[i]][feature].index.to_list()
#         print(dict_keys[i])
#         print(dict_val[dict_keys[i]][0])
#         print(dict_val[dict_keys[i]][1])
        
        data.loc[val_index,[feature+'_'+'class_5']]=dict_val[dict_keys[i]][0]
        data.loc[val_index,[feature+'_'+'class_4']]=dict_val[dict_keys[i]][1]
        data.loc[val_index,[feature+'_'+'class_3']]=dict_val[dict_keys[i]][2]
        data.loc[val_index,[feature+'_'+'class_2']]=dict_val[dict_keys[i]][3]
        data.loc[val_index,[feature+'_'+'class_1']]=dict_val[dict_keys[i]][4]

        
    
    


# In[36]:


def RemoveChar_1(data,feature):
    special_char = "@_!#$%^&''*()<>?/\|}{~:;[]"
    for i in special_char:
        data[feature]=data[feature].str.replace(i, " ").replace(" ","_")
        data[feature]=data[feature].str.replace("  "," ")    
        data[feature]=data[feature].str.replace(" ","_")
    return np.unique(data[feature]) 


# In[35]:


#import flask
import os
from flask import Flask, render_template,request, redirect, url_for
from os.path import join, dirname, realpath

project_root = os.path.dirname('')
template_path = os.path.join(project_root, './')

app = Flask(__name__,template_folder=template_path)



app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'Datafiles'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

# project_root = os.path.dirname(__file__)
# template_path = os.path.join(project_root, '')
# app = Flask(__name__, template_folder=template_path)


def predict(x):
    
    
    #print(x)
 
  ###### Feature engineering of Price##########
    x['price_new']=x['price'].values[0]
    if (x['price_new'].values[0]<138.0):
        x['price_new']=(x['price_new'].values[0]/.01)*2
  


  ####### Feature engineering of freight_value#############
    x['freight_value_new']=x['freight_value'].values[0]
    if (x['freight_value_new'].values[0]<20.0):
        x['freight_value_new']=(x['freight_value_new'].values[0]/0.2)*35.89
  

  ########## Feature engineering of product_length_cm #########
    x['product_length_cm_new']=x['product_length_cm'].values[0]
    if (x['product_length_cm'].values[0]<37.0):
        x['product_length_cm']=(x['product_length_cm'].values[0]/.1)*2.1
        

  ####### Feature engineering of product_weight_g ####### 
    x['product_weight_g_new']=x['product_weight_g'].values[0]
    if (x['product_weight_g_new'].values[0]>1750.0):
        x['product_weight_g_new']=(x['product_weight_g_new'].values[0]/22.09)
    

  ######## Feature engineering of payment_value ########

    x['payment_value_new']=x['payment_value'].values[0]
    if (x['payment_value_new'].values[0]<180.0850):
        x['payment_value_new']=(x['payment_value_new'].values[0]/0.1)*12.2

  ##### Feature engineering of payment_installments ######## 

    x['payment_installments_new']=x['payment_installments'].values[0]
    if (x['payment_installments_new'].values[0]<4):
        x['payment_installments_new']=(x['payment_installments_new'].values[0]*35)
  

  ### Date Time Feature ########
    x['order_purchase_timestamp']=pd.to_datetime(x['order_purchase_timestamp'])
    x['order_approved_at']=pd.to_datetime(x['order_approved_at'])
    x['order_delivered_carrier_date']=pd.to_datetime(x['order_delivered_carrier_date'])
    x['order_delivered_customer_date']=pd.to_datetime(x['order_delivered_customer_date'])
    x['order_estimated_delivery_date']=pd.to_datetime(x['order_estimated_delivery_date']) 
  

  #### Feature engineering of order_delivered_customer_date
    x['Duration_delivered_Purchase_days']=((x['order_delivered_customer_date']-x['order_purchase_timestamp']).dt.days).values[0]
  
   


  ##drop in final model 
    x['Duration_Estimated_delivered_days']=((x['order_estimated_delivery_date']-x['order_purchase_timestamp']).dt.days).values[0]
  
    x['Delivered_Within_Estimated']=x['Duration_Estimated_delivered_days']-x['Duration_delivered_Purchase_days']

  #### feature engg create new feature from order_purchase_timestamp
    x['order_purchase_timestamp'+'_month']=x['order_purchase_timestamp'].dt.month
    x['order_purchase_timestamp'+'_year']=x['order_purchase_timestamp'].dt.year
    x['Month_Year_Purchase_Order']=x['order_purchase_timestamp_month'].astype(str)+'_'+x['order_purchase_timestamp_year'].astype(str)

 
  #feature engg create new feature from order_delivered_customer_date
    x['order_delivered_customer_date'+'_month']=x['order_delivered_customer_date'].dt.month
    x['order_delivered_customer_date'+'_year']=x['order_delivered_customer_date'].dt.year
    x['Month_year_order_deliverd']=x['order_delivered_customer_date_month'].astype(str)+'_'+x['order_delivered_customer_date_year'].astype(str)

  
    x.drop(columns=['customer_city','order_purchase_timestamp_month','order_purchase_timestamp_year','order_delivered_customer_date_month','order_delivered_customer_date_year'],inplace=True,axis=1)
    x.drop(columns='geolocation_city_customer',inplace=True,axis=1)
    x.drop(columns='geolocation_state_customer',inplace=True,axis=1)
    x.drop(columns=['geolocation_city_seller','geolocation_state_seller','zip_code_prefix_seller'],inplace=True,axis=1)
    x.drop(columns=['customer_id','customer_unique_id','order_id','order_item_id','product_id','seller_id','zip_code_prefix_customer'],inplace=True,axis=1)
    x.drop(columns=['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date','shipping_limit_date','Duration_Estimated_delivered_days'],axis=1,inplace=True)
  #x.drop(columns=['review_comment_title','review_comment_message','review_creation_date','review_answer_timestamp','review_id'],axis=1,inplace=True)
    x.drop(columns=['geolocation_lng_customer','geolocation_lng_seller','geolocation_lat_seller','geolocation_lat_customer','review_score'],axis=1,inplace=True)

    df_numeric = x.select_dtypes(include=np.number)
    column=df_numeric.columns.to_list()  

    for i in column:
        Outlier_index=df_numeric[(df_numeric[i]<out_dict1[i][0]) | (df_numeric[i]>out_dict1[i][1])].index.to_list()
        df_numeric.loc[Outlier_index,i]=out_dict1[i][2]

  ######### Replac value in original data #######  
    for i in range(len(column)): 
        x[column[i]]=df_numeric[column[i]]
        #print(column[i])

  ############ Transform numerical data#########  
    Col_val=list(x.select_dtypes(include='number').columns)
    X_transform_x=yt1.transform(x[Col_val])  
  
    for i in range(len(Col_val)): 
        x[Col_val[i]]=X_transform_x[:,i]

    RemoveCharval=RemoveChar_1(x,'seller_city')

    customer_state_dict_test=Test_Response_1(x,'customer_state',customer_state_dict)
    order_status_dict_test=Test_Response_1(x,'order_status',order_status_dict)
    payment_type_dict_test=Test_Response_1(x,'payment_type',payment_type_dict)
    seller_city_dict_test=Test_Response_1(x,'seller_city',seller_city_dict)
    seller_state_dict_test=Test_Response_1(x,'seller_state',seller_state_dict)
    product_category_name_english_dict_test=Test_Response_1(x,'product_category_name_english',product_category_name_english_dict)
    Month_Year_Purchase_Order_dict_test=Test_Response_1(x,'Month_Year_Purchase_Order',Month_Year_Purchase_Order_dict)
    Month_year_order_deliverd_dict_test=Test_Response_1(x,'Month_year_order_deliverd',Month_year_order_deliverd_dict)

    _CreateFeature_ResponseCoding(x,customer_state_dict_test,'customer_state')
    _CreateFeature_ResponseCoding(x,order_status_dict_test,'order_status')
    _CreateFeature_ResponseCoding(x,payment_type_dict_test,'payment_type')

    _CreateFeature_ResponseCoding(x,seller_city_dict_test,'seller_city')
    _CreateFeature_ResponseCoding(x,seller_state_dict_test,'seller_state')
    _CreateFeature_ResponseCoding(x,product_category_name_english_dict_test,'product_category_name_english')

    _CreateFeature_ResponseCoding(x,Month_Year_Purchase_Order_dict_test,'Month_Year_Purchase_Order')
    _CreateFeature_ResponseCoding(x,Month_year_order_deliverd_dict_test,'Month_year_order_deliverd')   

  ### drop categorical feature #########  
    x.drop(columns=list(x.select_dtypes(include='object').columns),axis=1,inplace=True)
  #### standarize the data ######

    Col_val=list(x.select_dtypes(include='number').columns)
    xtrain_num_stand=Stand1.transform(x[Col_val])

    val=loaded_model.predict(x)
    val=val.argmax(axis=-1)
    
    return val[0]


@app.route('/')
def index():
    return render_template('Index.html')

    #return 'This is my first API call!'

    
@app.route('/PredictReview/', methods = ['GET', 'POST'])
def uploadFiles():
    # get the uploaded file
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            # set the file path
            uploaded_file.save(file_path)
            # save the file
            #return redirect(url_for('index'))
            x=pd.read_csv(file_path)
            val=predict(x)
            Review=''
            if val in [0,1,2,3]:
                #print('Negative Review')
                Review='Negative Review'
            else:
                #print('Positive eview')
                Review='Positive eview'
            
            return Review
        
        elif(request.files['file']==''):
            return 'not select'
    

if __name__ == '__main__':
    #app.debug = True
    #app.run(host='0.0.0.0', port=8080,threaded=False)
    #from werkzeug.serving import run_simple
    #run_simple('localhost', 9000, app)
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 8080, app)

