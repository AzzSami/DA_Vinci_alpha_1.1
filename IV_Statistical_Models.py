#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:20:51 2023

@author: Sami azzoug
"""
# =============================================================================
# =============================================================================
# # Imported libraries.
# =============================================================================
# =============================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import yfinance as yf 
import plotly.graph_objects as go

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error 
from typing import Union
from itertools import product
from tqdm import tqdm
from plotly.subplots import make_subplots
from PIL import Image
    


# =============================================================================
# =============================================================================
# #  MAPE function used for our conclusions on the series forecasts results.
# =============================================================================
# =============================================================================

def mape(y_true, y_pred):
    
    return np.mean(np.abs((y_true - y_pred)/y_true ))*100

# =============================================================================
# =============================================================================
# # Defining a function for rolling forecasts on a horizon which will be used for our statistical models.
# =============================================================================
# =============================================================================
@st.cache_data
def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int, method: str) -> list:
    
    total_len = train_len + horizon
    
    if method == 'mean':
        
        pred_mean = []
        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
        
        return pred_mean
    
    elif method == 'last':
        
        pred_last_value = []
        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))
    
        return pred_last_value
    

@st.cache_data
def rolling_forecast_MA(data: pd.DataFrame, TRAIN_LEN: int, HORIZON: int, WINDOW: int,diff_order : int,MA_number: int):
    total_len = TRAIN_LEN + HORIZON

    pred_MA = []

    for i in range(TRAIN_LEN, total_len, WINDOW):
        model = SARIMAX(data['Adj Close'][:i], order=(0,diff_order,MA_number))
        res = model.fit(disp=False)
        predictions = res.get_prediction(0, i + WINDOW - 1)
        oos_pred = predictions.predicted_mean.iloc[-WINDOW:]
        pred_MA.extend(oos_pred)
        
    return pred_MA  

@st.cache_data
def rolling_forecast_AR(data: pd.DataFrame, train_len: int, horizon: int, window: int,diff_order : int, AR_number :int) :

    total_len = train_len + horizon
    
    pred_AR = []
    for i in range(train_len, total_len, window):
        model = SARIMAX(data['Adj Close'], order=(AR_number,diff_order,0)) # We specify the AR order we have noticed
        res = model.fit(disp=False)
        predictions = res.get_prediction(0, i + window - 1)
        oos_pred = predictions.predicted_mean.iloc[-window:]
        pred_AR.extend(oos_pred)
    
    return pred_AR

@st.cache_data
def rolling_forecast_ARIMA(data: pd.DataFrame, train_len: int, horizon: int, window: int, p_number_res :int,diff_order : int, q_number_res :int) :

    pred_ARIMA =[]
    
    total_len = train_len + horizon

    for i in range (train_len,total_len,window):# Windows will be equal to q_number _res as we do not want to return the mean on the MA model, so we stick to the MA order
    
        model = SARIMAX(data['Adj Close'][:i],order=(p_number_res,diff_order,q_number_res))
        res = model.fit(disp=False)
        predictions = res.get_prediction(0,i-window-1)
        oos_pred =predictions.predicted_mean.iloc[-window:]
        pred_ARIMA.extend(oos_pred)

    return pred_ARIMA

@st.cache_data  
def rolling_forecast_SARIMA(df: pd.DataFrame, train_len: int, horizon: int, window: int, p_number_res :int ,d: int, q_number_res :int,  P_number_res :int, D :int, Q_number_res :int, s: int,  method : str) :
    
    if method == 'SARIMA':
    
        pred_SARIMA =[]
        total_len = train_len + horizon
        
        for i in range (train_len,total_len,window): # Windows will be equal to q_number _res as we do not want to return the mean on the MA model, so we stick to the MA order
        
            model_SARIMA =SARIMAX (df,
                            order=(p_number_res,d,q_number_res),
                            seasonal_order=(P_number_res,D,Q_number_res,s),
                            simple_differencing=False).fit(disp =False) #Now we are adding the seasonal aspects where we have the P,D,Q seasonal order. 
            
            res_SARIMA = model_SARIMA
            predictions_SARIMA = res_SARIMA.get_prediction(0,i-window-1)
            oos_pred_SARIMA =predictions_SARIMA.predicted_mean.iloc[-window:]
            pred_SARIMA.extend(oos_pred_SARIMA)
        
    return pred_SARIMA
    


# =============================================================================
# =============================================================================
# # Defining the ARIMA function 
# =============================================================================
# =============================================================================
@st.cache_data
def optimize_ARIMA (endog : Union[pd.Series,list],order_list :list,d: int) -> pd.DataFrame: #Endog for endogenous variable
    
    results = [] # List for storing the results.
    
    for order in tqdm(order_list):
        try:
            model = SARIMAX(endog, order = (order[0],d,order[1]), simple_differencing= False).fit(disp=False) #In the ARIMA model we only use the p and q order.
        except: 
            continue 
        aic = model.aic #Recolting the AIC of the ARIMA model.
        results.append([order,aic]) #Addind it to the resulsts list defined previously.
        
    result_df= pd.DataFrame(results) #Conversion to a dataframe.
    result_df.columns = ['(p,q)','AIC'] #Rename the columns for better clearity.
    result_df= result_df.sort_values(by='AIC',ascending=True).reset_index(drop=True) #Sort in ascending order, the lower AIC is better.
    
    return result_df

# =============================================================================
# =============================================================================
# # Function to plot the seasonality on the SARIMA function
# =============================================================================
# =============================================================================
@st.cache_resource
def seasonal_plot (df,frequency,ticker): #Add the dataframe we are working with and the frequency we are supposing. Remember, for now we assume a weekly seasonality on our series.
    
    decomposition = STL(df['Adj Close'],period =frequency).fit()
    fig, (ax1,ax2,ax3,ax4) =plt.subplots(nrows= 4, ncols =1, sharex = True, figsize =(10,8))
    
    ax1.set(
           title=f"Seasonality graph of {ticker} "
           )
    ax1.plot(decomposition.seasonal)
    ax1.set_ylabel('Seasonality')
    
    ax2.plot(decomposition.observed)
    ax2.set_ylabel('Observed Data')
    
    ax3.plot(decomposition.trend)
    ax3.set_ylabel('Trend')

    ax4.plot(decomposition.resid)
    ax4.set_ylabel('Residuals')
    
    
    return st.pyplot(fig,use_container_width=True)

# =============================================================================
# =============================================================================
# # SARIMA Model
# =============================================================================
# =============================================================================

@st.cache_data(show_spinner='Loading the SARIMA model')
def optimize_SARIMA(endog :Union[pd.Series,list], order_list :list, d: int , D: int, s:int) -> pd.DataFrame:
    
    results = []
    
    for order in tqdm(order_list):
        try:
            model =SARIMAX (endog,
                            order=(order[0],d,order[1]),
                            seasonal_order=(order[2],D,order[3],s),
                            simple_differencing=False).fit(disp =False) #Now we are adding the seasonal aspects where we have the P,D,Q seasonal order. 
        except:
            continue 
        aic =model.aic
        results.append([order,aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q,P,Q)', 'AIC']
    result_df = result_df.sort_values(by ='AIC', ascending =True).reset_index(drop=True)
    
    return result_df



# =============================================================================
# =============================================================================
# # ADF test
# =============================================================================
# =============================================================================

def ad_fuller_test (df): # Adding a function to output the ADF test here as we will output it many times.

    ad_fuller_cal = adfuller(df['Adj Close']) # Taking the Adj Close price of our Dataframe.
    ad_fuller_stat_number = ad_fuller_cal[0] # Output the ADF stat. 
    ad_fuller_stat_pvalue = ad_fuller_cal[1] # Output the p value.
    
    return ad_fuller_stat_number,ad_fuller_stat_pvalue #Give ADF stat and its pvalue associated.

# =============================================================================
# =============================================================================
# # Differencing function
# =============================================================================
# =============================================================================

def func_diff_data(df,order): # Take the dataframe and look for the order of differencing we want. 

    differenced_series = np.diff(df['Adj Close'],n=order)
    
    return differenced_series

# =============================================================================
# =============================================================================
# # Seasonal differencing function
# =============================================================================
# =============================================================================

def func_seasonal_diff_data(df,order):
    
    seasonal_differenced_series = np.diff(df['Adj Close'],n=order)
    
    return seasonal_differenced_series


# =============================================================================
# =============================================================================
# # Plot the differencied series depending on the model
# =============================================================================
# =============================================================================

@st.cache_data
def funct_plot_diff_series(df : list,name_model : str):
    
    fig = px.line(df, title= f'Data differencied of the {name_model} model', width = 650)
    
    return st.write(fig)
    
    
# =============================================================================
# =============================================================================
# # Plot the differencied series and its forecasting results
# =============================================================================
# =============================================================================

@st.cache_data  
def funct_plot_diff_results_series(df : list, name_model : str):
    
    fig = px.line(df, title= f'Differencied forecasting results for the {name_model} model',width = 650)
    
    return st.write(fig)
    
# =============================================================================
# =============================================================================
# # Plot the original series and its forecasting results
# =============================================================================
# =============================================================================

@st.cache_data
def funct_plot_original_results_series(df : list, name_model : str):
    
    fig = px.line(df, title= f'Original forecasting results for the {name_model} model',width = 650, markers = True)
    
    return st.write(fig)

# =============================================================================
# =============================================================================
# # Plot the differencing series and its forecasting results
# =============================================================================
# =============================================================================

@st.cache_data  
def funct_plot_results_series(df1 : list, name_model : str):

    fig = px.line(df1, title= f'Forecast results for the {name_model} model',width =650, markers = True)
    
    # Train set layer

    fig.add_vrect(x0=st.session_state['start_date_train'], x1=st.session_state['end_date_train'], 
                annotation_text="Train", annotation_position="top left",
                annotation=dict(font_size=22),
                fillcolor="green", opacity=0.15, line_width=2)
    
    # Test set layer
    fig.add_vrect(x0=st.session_state['start_date_test'], x1=st.session_state['end_date_test'], 
                annotation_text="Test", annotation_position="top left",
                annotation=dict(font_size=22),
                fillcolor="orange", opacity=0.15, line_width=2)
    
    # Forecast set layer
    fig.add_vrect(x0 = st.session_state['end_date_forecast'], x1= st.session_state['start_date_forecast'], 
                annotation_text="Forecast", annotation_position="top left",
                annotation=dict(font_size=22),
                fillcolor="red", opacity=0.25, line_width=2)
    
    return st.write(fig)


# =============================================================================
# =============================================================================
# # Plot the future results
# =============================================================================
# =============================================================================

@st.cache_data
def funct_forecast_results_series(_model, out_of_sample_days : int, end_date, length_in_sample_prediction : int, model_name : str):
    
    
    date = pd.date_range(start= end_date, periods=(out_of_sample_days-length_in_sample_prediction))                                      
    forecast = _model.forecast(steps=out_of_sample_days)

    forecast =pd.DataFrame({
                            
                            f'{model_name}_Forecast' : forecast

                            })
    forecast=forecast[f'{model_name}_Forecast'][length_in_sample_prediction:]
    
    forecast =pd.DataFrame({
                            
                            f'{model_name}_Forecast' : forecast,
                            f'{model_name}_Forecast_Date' : date                            
                            })
    forecast = forecast.set_index(f'{model_name}_Forecast_Date')
    return forecast

# =============================================================================
# =============================================================================
# # Plot the original series and its forecasting results
# =============================================================================
# =============================================================================

@st.cache_data
def funct_plot_multiple_plot(df1 : list,ticker_1:str, df2 : list,ticker_2:str, df3 :list,ticker_3:str, df4 :list, ticker_4:str ):
    
    fig = make_subplots(rows=2, cols=2, start_cell="top-left",subplot_titles=(ticker_1, ticker_2, ticker_3, ticker_4))
    
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Adj Close'],name=f'{ticker_1} Adj close price'),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df2.index,y=df2['Adj Close'],name=f'{ticker_2} Adj close price'),
                  row=1, col=2)
    
    fig.add_trace(go.Scatter(x=df3.index,y=df3['Adj Close'],name=f'{ticker_3} Adj close price'),
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df4.index,y=df4['Adj Close'],name=f'{ticker_4} Adj close price'),
                  row=2, col=2)
    
    fig.update_layout(height=500, width=700,
                  title_text= f"Stock price comparaison with {ticker_1} ")

    return st.write(fig)

# =============================================================================
# =============================================================================
# # Get exo data function for the SARIMAX model
# =============================================================================
# =============================================================================

@st.cache_data    # Use to slow down the download of the data and make sure we use memory efficiently
def get_data (ticker,start_date,end_date):
    
    data = yf.download(ticker, start_date, end_date)
    # Removing the na values
    data = data.dropna()
    return data

# =============================================================================
# =============================================================================
# # Plot the original series and its forecasting results
# =============================================================================
# =============================================================================

@st.cache_data
def funct_MSE_SARIMA(df : list, pred_data : list):
    
    MSE = mean_squared_error(df, pred_data)

                
    # METRICS

    return st.metric("MSE for the  Forecast",value= MSE)
    
    

# =============================================================================
# =============================================================================
# # The funct stat model
# =============================================================================
# =============================================================================



def funct_stat_model():
    
    # =============================================================================
    # =============================================================================
    # # Cache Variables to use in this statistical section
    # =============================================================================
    # =============================================================================

    data = st.session_state['data'] # It is the data we have choosen in the gather data part.
    
    ticker = st.session_state['ticker'] # Get the ticker we entered in the gather data part.
    
    data_diff = st.session_state['data_diff'] # The differencied order we entered in the Baseline model part.
        
    train=st.session_state['train'] # We retreive the train set we wanted to have.
    
    test=st.session_state['test'] # Retrieving the test set as well.
    
    Forecast_horizon = st.session_state ['forecast_horizon'] # The forecast horizon we choosed on the Gather Data part
    
    d = st.session_state['chck_box_ad_fuller_diff_order'] # Add the order of integration we have choosen by differencing our series. Start using in the ARIMA,SARIMA,SARIMAX

    end_date = st.session_state['end_date']

    forecast_data_set = st.session_state ['forecast_data_set']
                    
    # =============================================================================
    # =============================================================================
    # # Statistical Model section
    # =============================================================================
    # =============================================================================

    st.write('## IV. Statistical models')
    with st.expander ('Statistical section'):

        image = Image.open('Statistical_Model.png')      
        st.image(image, caption='Statistics')
                        

    # Create tabs

    #tab_title_list = ["Stock price forecasting", "Volatility Forecasting"]
    #tab1, tab2 = st.tabs(tab_title_list)
    
    #with tab1:
    # =============================================================================
    #     #Take note that for the Statistical model section, we will follow this guidline:
        
    #         # Theoric reminder presenting the model
    #         # Plot the differencied series 
    #         # Plot the ACF, PACF 
    #         # Have the models parameters (ARIMA,SARIMA,SARIMAX)
    #         # Get the MSE on the parameters results
    #         # Plot the predictions
    # =============================================================================
    
        # =============================================================================
        # =============================================================================
        # # It is important to notice that until the ARIMA model we will have to 
          # differenciate our series by ourselves and then we will use a data_diff variable.
        # =============================================================================
        # =============================================================================

        # For the differencied series we will split our data set into a test and a train set
        # We will use the 1 week forecasting horizon for now but we will add a slider in the gathering data page to let the user choose the horizon we would like until 1 month or forcasting
        # We want then to register the train_diff and the test_diff once as we will use it many times until the ARIMA model

    train_diff = data_diff[:len(train)]
    test_diff = data_diff[-len(test):]

    # =============================================================================
    # =============================================================================
    # # Moving Average process (MA)
    # =============================================================================
    # =============================================================================    
    

    Stat_model_selection =st.selectbox("Choose a Statistic model", ('1. Moving average model (MA)','2. Autoregressive process (AR)','3. Autoregressive Integrated Moving Average Process (ARIMA)','4. Seasonal Autoregressive Integrated Moving Average Process (SARIMA)'))
    
    if Stat_model_selection == '1. Moving average model (MA)' :

        # =============================================================================
        # =============================================================================
        # # Get the MA procedure
        # =============================================================================
        # =============================================================================
        
        with st.expander('Display the MA process'):
                
            image = Image.open('MA_Process.png')
            
            st.image(image, caption='The MA steps')
                    
        # =============================================================================
        # =============================================================================
        # # MA theory reminder
        # =============================================================================
        # =============================================================================
        
        with st.expander('MA theory reminder'):
            
            st.info('''
                    A moving average (MA) model is assuming that the current values of the series are linearly dependent from the current and past error terms.
                    ''')
            
            st.latex(r'y_t = \mu + \epsilon_t + \theta_1\epsilon_1 + \theta_2\epsilon_2 + ... + \theta_t\epsilon_{t-q} ') #Formula of the MA model.
            
            st.markdown("${q} $ : Order of the MA")
            st.markdown("${\mu} $ : Trend")
            st.markdown("${\epsilon_t} $ : The present error terms")
            st.markdown("${\epsilon_{t-q}} $ : The past error terms")
            st.markdown(r"${\theta} $ : Magnitude of the impact of the past error terms")
            
            st.info('''
                        Note : The error terms are mutually independent and normally distributed, as same as white noise.
                        ''')
        # =============================================================================
        # =============================================================================
        # # There we would like to plot the differencied series        
        # =============================================================================
        # =============================================================================
        
        with st.expander('Plot the differenced series, the ACF and choose the MA number'):
        
            funct_plot_diff_series(train_diff, 'MA')
            lags_input = st.slider(" How many lags would you like to add ? ", 0,len(train['Adj Close']),5) 

            st.pyplot(plot_pacf(data_diff,title=f'Autocorrelation plot of the {ticker} close price (after differencing the series)',lags=lags_input))
            MA_number = st.number_input('What degree of autocorrelation are you seeing ?',min_value = 1, value =1)
            st.session_state['MA_number'] = MA_number
        # =============================================================================
        # =============================================================================
        # # Text explaining that, for a MA model, we would like to forecast only the next timesteps
        # =============================================================================
        # =============================================================================
        
        st.info('''
                Rolling window is a function who will repeatedly fit a model and generate
                forecasts over a certain window of time, until forecasts for the entire test set are
                obtained.
                ''')
        # =============================================================================
        # =============================================================================
        # # Forecasting results on the differencied series 
        # =============================================================================
        # =============================================================================
        with st.expander("Head the MSE for each foreacasting method on the differencied series") :
                    
            TRAIN_LEN = len(train_diff) # We stock the length of the train set
            HORIZON = len(test_diff)+Forecast_horizon # The horizon represents how many many values must be predicted and it corresponds to the length of the test_diff set
            WINDOW = 1 # Specifiying how many timesteps are predicted at a time
            
            # Have to define the data set who will take the MA forecasts and the horizon and make a dataset to contain all the length of those 2 dataset

            forecast_data_set['forecast_MA'] = pd.Series() #The horizon of forecating only
            data_MA_concat = pd.concat([data,forecast_data_set],ignore_index=False) 
            
            # We will mainly use the rolling forecast function defined previously
            # Add the mean and the last value model

            pred_MA = rolling_forecast_MA(data_MA_concat,TRAIN_LEN,HORIZON,WINDOW,st.session_state['chck_box_ad_fuller_diff_order'],MA_number)
            
            data_MA_concat['pred_MA'] = pd.Series()
            data_MA_concat['pred_MA'][len(train):-len(forecast_data_set)] = pred_MA[:len(test)]
            data_MA_concat['forecast_MA'][-len(forecast_data_set):] = pred_MA[len(test):]


            # =============================================================================
            # =============================================================================
            # # Check for the MSE of each forecasting method we have used        
            # =============================================================================
            # =============================================================================
        
            
            mse_MA = mean_squared_error(data_MA_concat['Adj Close'][len(train):-len(forecast_data_set)], data_MA_concat['pred_MA'][len(train):-len(forecast_data_set)])
            
                        
            # MSE METRICS to evaluate the results
            
            st.metric("MSE for the MA Forecast",value= mse_MA)
            # =============================================================================
            # =============================================================================
            # # MAPE
            # =============================================================================
            # =============================================================================
            
        with st.expander('Output the MAPE on the original series'):
            
            mape_MA = mape(data_MA_concat['Adj Close'][len(train):-len(forecast_data_set)], data_MA_concat['pred_MA'][len(train):-len(forecast_data_set)])

            st.metric("MAPE for the MA Forecast",value= mape_MA)
            st.write(f""" ##### Your forecasts are, on average, {mape_MA.round(2)} % below the actual adjusted close prices for the last {len(test)} days.""" )


        # =============================================================================
        # =============================================================================
        # # Plot the original series forecasting results
        # =============================================================================
        # =============================================================================

        with st.expander('Plot the predicitons on the original series'):
            
            st.warning('''Please take note that you can zoom on the series 
                        for better visibility on the in-sample prediciton you have made''')
                    
            funct_plot_results_series(data_MA_concat[['Adj Close','pred_MA','forecast_MA']],'MA')

    # =============================================================================
    # =============================================================================
    # # Autoregressive Process (AR)   
    # =============================================================================
    # =============================================================================

    if Stat_model_selection == '2. Autoregressive process (AR)': 
        
        with st.expander('AR steps'):

            image = Image.open('AR_Process.png')
            st.image(image, caption='The AR steps')

        with st.expander ('AR theory reminder') : 
            
            st.info('''
                    A autoregressive model (AR) is assuming that the outputed variable is linearly dependent on its own previous values.
                    It is an autoregression of the variable with itself.
                    ''')
            st.latex(r'''
                        y_t = C + \phi_1y_{t-1} + \phi_2y_{t-2} + ... + \phi_ty_{t-p} + \epsilon_t  
    
                        ''')
            st.markdown("${p} $ : Order of the AR")
            st.markdown("${C} $ : Constant")
            st.markdown("${\epsilon_t} $ : The present error terms (white noise)")
            st.markdown("${y_{t-p}} $ : The past values of the series")
            st.markdown(r"${\phi} $ : Magnitude of the influence of the past values on the present value")
            
        # =============================================================================
        # =============================================================================
        # # Check the autocorrelation degree of the values with theirselves      
        # =============================================================================
        # =============================================================================
        
        with st.expander('Plot the PACF function and choose the AR number') :
            
            #Lags number based on the length of the train set, for better practicality 
            pacf_lags_input = st.slider(" How many lags would you like to add for the PACF plot ? ", 0,int(len(train['Adj Close'])/2),5) 
            
            #Plot the PACF
            
            st.pyplot(plot_pacf(data_diff,title=f'Partial autocorrelation plot on the {ticker} close price (after differencing the series)', lags=pacf_lags_input ))

            # Entering the p number of the AR model and sotkcing it in th
            AR_number = st.number_input('What is the p of your AR model ', min_value=1,value=1)
            st.session_state['AR_number'] =AR_number
            
            # =============================================================================
            # =============================================================================
            # Add the AR predictions
            # =============================================================================
            # =============================================================================
            
            TRAIN_LEN = len(train_diff)
            HORIZON = len(test_diff) + Forecast_horizon
            WINDOW = 1 # Specifying how many timesteps are predicted at a time
            
            forecast_data_set['forecast_AR'] = pd.Series() #The horizon of forecating only
            data_AR_concat = pd.concat([data,forecast_data_set],ignore_index=False) 
            

            pred_AR = rolling_forecast_AR(data,TRAIN_LEN,HORIZON,WINDOW,st.session_state['chck_box_ad_fuller_diff_order'],AR_number)
            
            data_AR_concat['pred_AR'] = pd.Series()
            data_AR_concat['pred_AR'][len(train):-len(forecast_data_set)] = pred_AR[:len(test)]
            data_AR_concat['forecast_AR'][-len(forecast_data_set):] = pred_AR[len(test):]

            # =============================================================================
            # =============================================================================
            # # Heading the PACF values on the dataframe       
            # =============================================================================
            # =============================================================================
            
            pacf_AR_value_head = st.checkbox('Click here to head the significant PACF values')
            
            if pacf_AR_value_head: 
                
                df_pacf_AR=pd.DataFrame(pacf(data_diff,nlags= int(len(train['Adj Close'])/2)))
                df_pacf_AR = df_pacf_AR.abs()
                df_pacf_AR.index.names = ['Lags']
                df_pacf_AR= df_pacf_AR.rename(columns = {0:'PACF in absolute value'})
                df_pacf_AR= df_pacf_AR.sort_values(by='PACF in absolute value', ascending=False)      
                st.dataframe(df_pacf_AR,width=500)
                
            st.success(f'You are facing an AR({AR_number}) model ')
        
        # =============================================================================
        # =============================================================================
        # # MSE metrics      
        # =============================================================================
        # =============================================================================

        with st.expander ('Head the MSE ') : 
            
            mse_AR = mean_squared_error(data_AR_concat['Adj Close'][len(train):-len(forecast_data_set)], data_AR_concat['pred_AR'][len(train):-len(forecast_data_set)]) 
            
            # METRICS
            st.metric("MSE for the AR Forecast",value= mse_AR)
            
        # =============================================================================
        # =============================================================================
        # # Ploting the differencied series        
        # =============================================================================
        # =============================================================================
        
        with st.expander ('Plot the predicitons on the original series') :
            
            st.info('''
                    Since our foracasts are made on differenced values, we need to reverse the transformation 
                    in order to bring our forecasts back to the original scale of the data, otherwise it will not make sense in this context.
                    ''')

            # =============================================================================
            # =============================================================================
            # # Plot the results
            # =============================================================================
            # =============================================================================
            
            funct_plot_results_series(data_AR_concat[['Adj Close','pred_AR','forecast_AR']],'AR')
            # =============================================================================
            # =============================================================================
            # # MAPE
            # =============================================================================
            # =============================================================================
            
        with st.expander('Output the MAPE on the original series'):
            
            pred_AR = mape(data_AR_concat['Adj Close'][len(train):-len(forecast_data_set)], data_AR_concat['pred_AR'][len(train):-len(forecast_data_set)])

            st.metric("MAPE for the AR Forecast",value= pred_AR)
            st.write(f""" ##### Your forecasts are, on average, {pred_AR.round(2)} % below the actual adjusted close prices for the last {len(test)} days.""" )


    # =============================================================================
    # =============================================================================
    # # ARIMA (p,d,q)
    # =============================================================================
    # =============================================================================

    if Stat_model_selection == '3. Autoregressive Integrated Moving Average Process (ARIMA)':

        if st.checkbox('ARIMA steps'):
            st.info('''In this section, we will not have to use the ACF or PACF plot to observe the correlation order. 
                    We will simply have to : \\
                        \\
                        1.Choose a model based on the AIC, wich is a criterion of goodness of fit for a model, the lower it is the better. \\
                        \\
                        2. Perform residuals analysis based on the Q-Q plot (qualitative analysis) and Ljung-Box (quantitative analysis), we want our residuals to be independent and uncorrelated simply as white noise.
                    
                    ''')
            image = Image.open('ARIMA_Process.png')
            st.image(image, caption='The ARIMA steps')
                
        if st.checkbox('ARIMA theory reminder'):

            st.info('''
                    The Autoregressive Integrated Moving Average Process (ARIMA) is a combination of the AR(p) and 
                    the MA(q) processes, but in terms of the differencied series.
                    
                    It is denotes as ARIMA (p,d,q) where p is the order of the AR(p) process, d the order of integration and 
                    q is the order of MA(q) process.
                    
                    Integration is the reverse of differencing, the order d of integration is the number of time by which the series 
                    has been differenced to be rendered stationnary.
                    
                    The general equation of the ARIMA(p,d,q) process is :
                    ''')
                    
            st.latex(r'''
                        y_t' = C + \phi_1y'_{t-1} + \phi_2y'_{t-2} + ... + \phi_ty'_{t-p} + \mu + \epsilon_{t} + \theta_1\epsilon'_{t-1} + \theta_2\epsilon_{t-2} + ... + \theta_t\epsilon_{t-q} 
    
                        ''')
            st.markdown("""${p} $ : Order of the AR\\
                        ${C} $ : Constant\\
                        ${\epsilon_t} $ : The present error terms (white noise)\\
                        ${y_{t-p}} $ : The past values of the series\\
                        ${\phi} $ : Magnitude of the influence of the past values on the present value\\
                        ${q} $ : Order of the MA\\
                        ${\mu} $ : Trend\\
                        ${\epsilon_{t-q}} $ : The past error terms\\
                        $θ$ : Magnitude of the impact of the past error terms\\
                        """)
            
            st.info('''Note : y_t' is representing the differenced series, it may have been differenced more than once. ''')
        
        # =============================================================================
        # =============================================================================
        # # Select p and q order
        # =============================================================================
        # =============================================================================
        
        if st.checkbox("List values of p and q"):
            
            col1, col2 = st.columns([1, 1])
            with col1 :
                p_number =st.number_input('Please enter the p order',min_value=1,value =4)
            with col2:
                q_number =st.number_input('Please enter the q order',min_value=1,value =4)
                
            qs = range (0,q_number,1)
            ps = range (0,p_number,1)
            order_list = list (product(ps,qs))
            
            # Print the results on the ARIMA model
            
            result_df =optimize_ARIMA(train_diff['Adj Close'], order_list,st.session_state['chck_box_ad_fuller_diff_order'])
            
            st.info('''Note that we always want to have the simplest ARIMA model possible. 
                    So instead of having too much parameters with small AIC gains, 
                    we want to adopt the simplest.''')
            st.table(result_df)
            
        # =============================================================================
        # =============================================================================
        # # Perfom residuals analysis
            # Here we have to make sure that our residuals are resembling white noise
        # =============================================================================
        # =============================================================================
        
            if st.checkbox("Perfom residuals analysis"):
    
                st.info("""This sub section is aming to guide you to know if your residuals are likely resembling to white noise.
                        \\
                        \\
                        It is crucial for us to have gaussian white noise (e.g. normally distributed and uncorrelated) to use our ARIMA model for forecasting and the others left (ARIMA,SARIMA,SARIMAX).
                        \\
                        \\
                        Indeed, it will indicates that the model has captured all of the predictive information, it then left only random fluctuation that cannot be modeled.
                        """)
                st.info('''This sub section is aming to guide you to know if your residuals are likely resembling to white noise.
                            It is crucial for us to have guassian white noise (e.g. normally distributed and uncorrelated) to use our ARIMA model for forecasting.
                            ''')
                # =============================================================================
                # =============================================================================
                # # Q-Q plot 
                # =============================================================================
                # =============================================================================
    
                col1, col2 = st.columns([1, 1])
                with col1 :
                    p_number_res =st.number_input('Please enter the p value of the ARIMA choosen',min_value=0,value =2)
                with col2:
                    q_number_res =st.number_input('Please enter the q value of the ARIMA choosen',min_value=0,value =2)
                    
                model = SARIMAX(train['Adj Close'], order = (p_number_res,st.session_state['chck_box_ad_fuller_diff_order'],q_number_res),simple_differencing =False) # Calling the SARIMA function to get an accurate model
                model_fit  = model.fit(disp=False) #Fitting the model
                st.success (f'Residual analysis for an ARIMA({p_number_res},{q_number_res})')
                st.pyplot(model_fit.plot_diagnostics(figsize = (10,8)))
                
                # =============================================================================
                # =============================================================================
                # # Quantitative Analysis (Ljung-Box)
                # =============================================================================
                # =============================================================================
                
                residuals = model_fit.resid
                lbvalue= acorr_ljungbox(residuals, np.arange(1,11,1))
                st.table(lbvalue)
                
                # =============================================================================
                # =============================================================================
                # # Checking if the residuals seems to be gaussian white noise
                # =============================================================================
                # =============================================================================
                st.error ('The Ljung-Box p-value has to exceed 0.05 to make forecasts')
                    
                st.info (f'Your best model is an ARIMA({p_number_res},{q_number_res}), then for the rolling function you will forecast 1 step above to avoid predicting the mean')
    
                st.success ('Your ARIMA model has passed all the check. You can now go on to the forecasting part !')
                                                
                    
                # =============================================================================
                # =============================================================================
                # # MSE Metric
                # =============================================================================
                # =============================================================================
                
                if st.checkbox("Head the MSE") : 
                    
                    TRAIN_LEN = len(train_diff)
                    HORIZON = len(test_diff)+Forecast_horizon
                    WINDOW = 1 
                    
                    forecast_data_set['forecast_ARIMA'] = pd.Series() # The horizon of forecating only
                    
                    data_ARIMA_concat = pd.concat([data,forecast_data_set],ignore_index=False) 
                    
                    pred_ARIMA = rolling_forecast_ARIMA(data,TRAIN_LEN,HORIZON,WINDOW,p_number_res,st.session_state['chck_box_ad_fuller_diff_order'],q_number_res)
        
                    data_ARIMA_concat['pred_ARIMA'] = pd.Series()
                    data_ARIMA_concat['pred_ARIMA'][len(train):-len(forecast_data_set)] = pred_ARIMA[:len(test)]
                    data_ARIMA_concat['forecast_ARIMA'][-len(forecast_data_set):] = pred_ARIMA[len(test):]
                        
                    mse_ARIMA = mean_squared_error(data_ARIMA_concat['Adj Close'][len(train):-len(forecast_data_set)], data_ARIMA_concat['pred_ARIMA'][len(train):-len(forecast_data_set)]) 
                    st.metric("MSE for the ARIMA Forecast",value= mse_ARIMA)
            
                    # =============================================================================
                    # =============================================================================
                    # # Predictions on the original series
                    # =============================================================================
                    # =============================================================================
                    
                    if st.checkbox("Plot the results on the original series"):
                    
                        funct_plot_results_series(data_ARIMA_concat[['Adj Close','pred_ARIMA','forecast_ARIMA']],'ARIMA')

                        # =============================================================================
                        # =============================================================================
                        # # MAPE
                        # =============================================================================
                        # =============================================================================
                        
                        if st.checkbox('Output the MAPE on the original series'):
                            
                            mape_ARIMA = mape(data_ARIMA_concat['Adj Close'][len(train):-len(forecast_data_set)], data_ARIMA_concat['pred_ARIMA'][len(train):-len(forecast_data_set)])
                
                            st.metric("MAPE for the ARIMA Forecast",value= mape_ARIMA)
                            st.write(f""" ##### Your forecasts are, on average, {mape_ARIMA.round(2)} % below the actual adjusted close prices for the last {len(test)} days.""" )
                

    # =============================================================================
    # =============================================================================
    # # 5. Seasonal Autoregressive Integrated Moving Average Process (SARIMA)  
    # =============================================================================
    # =============================================================================

    if Stat_model_selection == '4. Seasonal Autoregressive Integrated Moving Average Process (SARIMA)'  : 
                                
        with st.expander('SARIMA steps'):

            image = Image.open('SARIMA_Process.png')
            st.image(image, caption='The SARIMA steps')

        diff_order = st.session_state['chck_box_ad_fuller_diff_order']
        
        with st.expander('SARIMA theory reminder'):
            
            st.info('''The SARIMA model is behaving exactly the same as the ARIMA model except that we will be able to take into account the seasonality of the series.
                    We will follow the same test as we did in the ARIMA model but we will differenciate the series
                    in a seasonal perspective as well.\
                    Hence we have a ''')
                    
            st.latex(''' SARIMA(p,d,q)(P,D,Q)_m ''')
            
            st.markdown('''
                        ${d} $ : Order of integration\\
                        ${P} $ : Order of the seasonal AR(P) process\\
                        ${Q} $ : Order of the seasonal MA(Q) process\\
                        ${D} $ : Seasonal order of integration\\
                        ${m} $ : Frequency (number of observation in a cycle).
                        ''')
            
            st.info('''As we are dealing with daily data here we can solely assume that we have either weekly 
                    or yearly seasonality.
                    Fot now, we will assume that we have a weekly seasonality.
                    ''')
            st.warning('''
                        Take note that to exploit the full potential of the SARIMA model you have to previously find seasonal patterns in your series.\\
                            Otherwise, using a SARIMA model will not be relevant.
                    ''')
        
        # =============================================================================
        # =============================================================================
        # # Checking the seasonality
        # =============================================================================
        # =============================================================================
        with st.expander('Plot the seasonality of the series'):

            st.info(' As we dealing with daily data here, we can only know wether we have weekly or yearly seasonal effect.')
            seasonal_plot(data,7,ticker)

        # =============================================================================
        # =============================================================================
        # # Applying a seasonal differencing 
        # =============================================================================
        # =============================================================================
            
        with st.expander('Use the ADF test for seasonal stationarity'):
        
            data_seasonal = st.session_state['data']
        
            st.metric('ADF value  : ',value = ad_fuller_test(data_seasonal)[0])
            st.metric('pvalue  : ',value =ad_fuller_test(data_seasonal)[1])
            
            if ad_fuller_test(data_seasonal)[1] < 0.05 :
                
                st.info('Seasonality : Your series is already seasonal stationary')
                
                D_SARIMA= 0
                st.session_state['D_SARIMA'] = D_SARIMA
                
            elif ad_fuller_test(data_seasonal)[1] > 0.05 :
                
                st.info('Seasonality : Your series is not seasonal stationary')
            
                data_seasonal_differencied_order = st.number_input('''Please enter the number of time you want to weekly differenciate the series in order to make it stationary 
                                                                    ( take note that for now we will solely use weekly seasonal effect)''',min_value= 1, value=1)

                # =============================================================================
                # =============================================================================
                # # Weekly differencing mulitplying by the necessary input
                # =============================================================================
                # =============================================================================
                
                data_seasonal=func_diff_data(data_seasonal, 7 * data_seasonal_differencied_order) 
                
                # =============================================================================
                # =============================================================================
                # # Render a Dataframe
                # =============================================================================
                # =============================================================================
                
                st.write(f'You have seasonal differenciate your series {data_seasonal_differencied_order} time')
                data_seasonal=pd.DataFrame(data_seasonal)
                data_seasonal = data_seasonal.rename(columns={0: "Adj Close"})
                
                # =============================================================================
                # =============================================================================
                # # Metrics
                # =============================================================================
                # =============================================================================
                
                st.metric('ADF value  : ',value = ad_fuller_test(data_seasonal)[0])
                st.metric('pvalue  : ',value =ad_fuller_test(data_seasonal)[1])
                
                
                if ad_fuller_test(data_seasonal)[1] < 0.05 :
                    
                    st.info('Seasonality : Your series is now seasonal stationary')
                    
                    # =============================================================================
                    # =============================================================================
                    # # Register the differencing order in a session state for future use instead of relying on multiple if loops
                    # =============================================================================
                    # =============================================================================
                    
                    D_SARIMA= data_seasonal_differencied_order
                    st.session_state['D_SARIMA'] =D_SARIMA

        with st.expander('Choose the p,q,P,Q order'):
            # =============================================================================
            # =============================================================================
            # # For the SARIMA stat model we simply have to make a use of s instead of m which will denote the frequency
            # =============================================================================
            # =============================================================================
            

            col1, col2,col3,col4 = st.columns([1, 1, 1, 1],gap= 'small')
            with col1 :
                p_number_SARIMA =st.number_input('Please enter a p order',min_value=1,value =2,key = ' p_number_SARIMA')
            with col2:
                q_number_SARIMA =st.number_input('Please enter a q order',min_value=1,value =2, key ='q_number_SARIMA')
                
            with col3:
                P_number_SARIMA =st.number_input('Please enter a P order',min_value=1,value =2, key ='P_number_SARIMA')
                
            with col4:
                Q_number_SARIMA =st.number_input('Please enter a Q order',min_value=1,value =2, key ='Q_number_SARIMA')
                
            qs = range (0,q_number_SARIMA,1)
            ps = range (0,p_number_SARIMA,1)
            Ps = range (0,P_number_SARIMA,1)
            Qs = range (0,Q_number_SARIMA,1)
            
            
            SARIMA_order_list  = list (product(ps,qs,Ps,Qs))
            s= 7 # Week seasonality
            
            SARIMA_result_df = optimize_SARIMA(train['Adj Close'], SARIMA_order_list, d, st.session_state['D_SARIMA'], s)
            st.table(SARIMA_result_df)
                
        # =============================================================================
        # =============================================================================
        # # Look for residuals analysis
        # =============================================================================
        # =============================================================================
        
        with st.expander("Perfom residuals analysis"):
            st.info('''This sub section is aming to guide you to know if your residuals are likely resembling to white noise.
                    It is crucial for us to have guassian white noise (e.g. normally distributed and uncorrelated) to use our SARIMA model for forecasting.
                    ''')
            

            col1, col2,col3,col4 = st.columns([1, 1, 1, 1])
            
            with col1 :
                p_number_res_SARIMA =st.number_input('Please enter the p value of the SARIMA you have choosen',min_value=0,value =2)
                st.session_state['p_number_res_SARIMA']= p_number_res_SARIMA
            with col2:
                q_number_res_SARIMA =st.number_input('Please enter the q value of the SARIMA you have choosen',min_value=0,value =2)
                st.session_state['q_number_res_SARIMA']= q_number_res_SARIMA
            with col3:
                P_number_res_SARIMA =st.number_input('Please enter the P value of the SARIMA you have choosen',min_value=0,value =2)
                st.session_state['P_number_res_SARIMA']= P_number_res_SARIMA
            with col4:
                Q_number_res_SARIMA =st.number_input('Please enter the Q value of the SARIMA you have choosen',min_value=0,value =2)
                st.session_state['Q_number_res_SARIMA']= Q_number_res_SARIMA
        # =============================================================================
        # =============================================================================
        # # Perform Residual analysis
        # =============================================================================
        # =============================================================================
                            
            # =============================================================================
            # =============================================================================
            # # Q-Q plot
            # =============================================================================
            # =============================================================================
            
            model_SARIMA = SARIMAX(train['Adj Close'], order= (p_number_res_SARIMA,diff_order,q_number_res_SARIMA),seasonal_order=(P_number_res_SARIMA,st.session_state['D_SARIMA'],Q_number_res_SARIMA,7),simple_differencing =False)
            model_fit_SARIMA  = model_SARIMA.fit(disp=False)
            st.success (f'Residual analysis for an SARIMA({p_number_res_SARIMA},{diff_order},{q_number_res_SARIMA}),({P_number_res_SARIMA},{D_SARIMA}, {Q_number_res_SARIMA})')
            st.pyplot(model_fit_SARIMA.plot_diagnostics(figsize = (10,8)))

            # =============================================================================
            # =============================================================================
            # # Ljung-Box
            # =============================================================================
            # =============================================================================

            residuals_SARIMA = model_fit_SARIMA.resid
            lbvalue_SARIMA= acorr_ljungbox(residuals_SARIMA, np.arange(1,11,1 ))
            st.table(lbvalue_SARIMA)
            
                
        # =============================================================================
        # =============================================================================
        # # METRICS
        # =============================================================================
        # =============================================================================
        with st.expander('MSE'):
            # =============================================================================
            # =============================================================================
            # # Add the SARIMA forecast
            # =============================================================================
            # =============================================================================

            TRAIN_LEN = len(train_diff)
            HORIZON = len(test_diff)+Forecast_horizon
            WINDOW = 1 
            
            forecast_data_set['forecast_SARIMA'] = pd.Series() # The horizon of forecating only
            
            data_SARIMA_concat = pd.concat([data,forecast_data_set],ignore_index=False) 
            
            pred_SARIMA = rolling_forecast_SARIMA(data['Adj Close'],TRAIN_LEN,HORIZON,WINDOW,p_number_res_SARIMA,diff_order,q_number_res_SARIMA,P_number_res_SARIMA,D_SARIMA,Q_number_res_SARIMA,7,'SARIMA')

            data_SARIMA_concat['pred_SARIMA'] = pd.Series()
            data_SARIMA_concat['pred_SARIMA'][len(train):-len(forecast_data_set)] = pred_SARIMA[:len(test)]
            data_SARIMA_concat['forecast_SARIMA'][-len(forecast_data_set):] = pred_SARIMA[len(test):]

            mse_SARIMA = mean_squared_error(data_SARIMA_concat['Adj Close'][len(train):-len(forecast_data_set)], data_SARIMA_concat['pred_SARIMA'][len(train):-len(forecast_data_set)]) 
            st.metric("MSE for the SARIMA Forecast",value= mse_SARIMA)
        
        # =============================================================================
        # =============================================================================
        # # Predictions on the original series
        # =============================================================================
        # =============================================================================
        
        with st.expander("Plot the results on the original series"):
        
            funct_plot_results_series(data_SARIMA_concat[['Adj Close','pred_SARIMA','forecast_SARIMA']],'SARIMA')

        # =============================================================================
        # =============================================================================
        # # MAPE
        # =============================================================================
        # =============================================================================
        
        with st.expander('Output the MAPE on the original series'):
            
            mape_SARIMA = mape(data_SARIMA_concat['Adj Close'][len(train):-len(forecast_data_set)], data_SARIMA_concat['pred_SARIMA'][len(train):-len(forecast_data_set)])

            st.metric("MAPE for the SARIMA Forecast",value= mape_SARIMA)
            st.write(f""" ##### Your forecasts are, on average, {mape_SARIMA.round(2)} % below the actual adjusted close prices for the last {len(test)} days.""" )

