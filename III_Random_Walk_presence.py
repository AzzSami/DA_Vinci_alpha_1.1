#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:40:33 2023

@author: morisaao
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from PIL import Image
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_squared_error 

from sklearn.metrics import mean_squared_error 
from typing import Union
from itertools import product
from tqdm import tqdm_notebook
from plotly.subplots import make_subplots
from PIL import Image

warnings.filterwarnings('ignore') #to ignore the waning errors

# =============================================================================
# =============================================================================
# # Plot the differencing series and its forecasting results
# =============================================================================
# =============================================================================
@st.cache_data  
def funct_plot_diff_results_series(df : list, name_model : str):
    
    fig = px.line(df, title= f'Differencied forecasting results for the {name_model} model',width = 650)
    
    return st.write(fig)

# =============================================================================
# =============================================================================
# # Plot the differencing series and its forecasting results
# =============================================================================
# =============================================================================
@st.cache_data  
def diff_inv(series_diff, first_value):

    series_inverted = np.r_[first_value, series_diff].cumsum().astype('float64')
                                                                      
    return st.write(series_inverted)

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

def funct_random_walk_presence ():

    # =============================================================================
    # =============================================================================
    # #  Test and Train recall
    # =============================================================================
    # =============================================================================

    train = st.session_state['train']
    
    test = st.session_state['test']
    
    # =============================================================================
    # =============================================================================
    # # Data recall
    # =============================================================================
    # =============================================================================
    
    data = st.session_state['data']
    
    # =============================================================================
    # =============================================================================
    # # Forecast dataset recall
    # =============================================================================
    # =============================================================================
    
    forecast_data_set = st.session_state ['forecast_data_set']

    # =============================================================================
    # =============================================================================
    # # Ticker
    # =============================================================================
    # =============================================================================

    ticker = st.session_state['ticker'] 

    # =============================================================================
    # # =============================================================================
    # # Random Walk checking
    # # =============================================================================
    # =============================================================================

    st.write('## III. Verification of random walk presence')

    with st.expander('Why verify random walk presence ?'):
    
        image = Image.open('/Users/morisaao/Desktop/Freelance/Da_Vinci_Forecasting/PNG/Random_walk.png')
        st.image(image, caption='Random walk presence explanation')
                 
    # =============================================================================
    # # =============================================================================
    # # Random Walk reminder
    # # =============================================================================
    # =============================================================================
    
    with st.expander ("Theory reminder"):

        st.write( " The random walk can be described as follow : ")
        st.latex(
            " y_t = C + y_{t-1} + \epsilon_t "
            )
        st.markdown("${y_t} $ : Actual Value ")
        st.markdown("${C} $ : Constant Value ")
        st.markdown("${y_{t-1}} $ : Previous Timesteps Value ")
        st.markdown("${\epsilon_t} $ : White Noise Value (random number) ~ N(0,1)")
        st.markdown("In a context of time series, a random walk is a series whose first difference is stationnary and uncorrelated.")
        st.markdown(" Note : If C ≠ 0 then we talk about random walk with drift.")    
       
    
    # =============================================================================
    # # =============================================================================
    # # Stationary Test
    # # =============================================================================
    # =============================================================================
    
    st.write (" ### 1. Stationnary test")
    unit_root_select = st.selectbox("Please select your unit root tester", ('Augmented Dickey Fuller (ADF)','Zivot-Andrews (ZA) (To be add)','Philips-Perron (PP) (To be add)','KPPS (To be add)','Variance-Ratio (VA) (To be add)'))
    
    if unit_root_select == "Augmented Dickey Fuller (ADF)" :
    
        with st.expander("Functionning of the ADF test"):

            st.markdown("Null Hypothesis (Ho) : Presence of unit root, the series is not stationary ", unsafe_allow_html=True)
            st.markdown("Alternative Hypothesis (Ha) : No presence of unit root, the series is stationary ", unsafe_allow_html=True)
        
        # ADF Inputs
        data_ad_fuller = adfuller(data['Adj Close'])
        with st.expander('Display the ADF results'):
                
            st.metric(label = 'ADF test : ', value = data_ad_fuller[0])
            st.metric(label = 'p-value : ', value = data_ad_fuller[1])
            
        # =============================================================================
        # # =============================================================================
        # # If the series is not stationary
        # # =============================================================================
        # =============================================================================
        
        if data_ad_fuller[1] > 0.05 : 
            
            st.markdown(" ###### We accept the null hypothesis (Ho) of unit root, the series is not stationary.")

            # =============================================================================
            # # =============================================================================
            # # Differencing the series
            # # =============================================================================
            # =============================================================================
        
            with st.expander('Differencing my series') : 
                
                chck_box_ad_fuller_diff_order = st.number_input('How many time do you want to differencing your series ? ( Note that in many use cases 1 order is sufficient)', min_value=1,value=1) 
                
                #Entering the ad fuller order
                st.session_state['chck_box_ad_fuller_diff_order'] = chck_box_ad_fuller_diff_order
                
                data_diff = pd.DataFrame(np.diff(data['Adj Close'], n= chck_box_ad_fuller_diff_order)) # We first set the differencing data
                data_diff = data_diff.rename(columns={0: "Adj Close"})
                
                #Entering the data diff variable in the cache

                st.session_state['data_diff'] = data_diff   

                #Test for stationary

                data_ad_fuller_diff = adfuller(data_diff)
                st.metric(label = 'ADF test : ', value = data_ad_fuller_diff[0])
                st.metric(label = 'p-value : ', value = data_ad_fuller_diff[1])
                
                # Results test

                if data_ad_fuller_diff[1] > 0.05 : 
                    
                    st.markdown(" ###### We accept the null hypothesis (Ho) of unit root, the series is not stationnary")
    
                elif data_ad_fuller_diff[1] < 0.05 :
                    
                    st.markdown("###### We reject the null hypothesis (Ho) of unit root, the series is now stationnary.")

            # =============================================================================
            # # =============================================================================
            # # ACF plot 
            # # =============================================================================
            # =============================================================================

            st.markdown("### 2. Autocorrelation function (ACF)")
            with st.expander('Plot the ACF') : 
                    
                #Lags number based on the length of the train set, for better practicality 
                lags_input = st.slider(" How many lags would you like to add ? ", 0,len(train['Adj Close']),5) 
                st.pyplot(plot_acf(data_diff,title=f'Autocorrelation plot of the {ticker} close price (after differencing the series)',lags=lags_input))
        
                #Output the higher ACF value in absolute value
                acf_value_head = st.checkbox('Click here to head the significant ACF values')
                
                #ACF value 
                
                if acf_value_head: 
                    
                    df_acf=pd.DataFrame(acf(data_diff,nlags= len(train['Adj Close'])))
                    df_acf = df_acf.abs()
                    df_acf.index.names = ['Lags']
                    df_acf = df_acf.rename(columns = {0:'ACF in absolute value'})
                    df_acf = df_acf.sort_values(by='ACF in absolute value', ascending=False)
                                                            
                    st.dataframe(df_acf,width=650)
                        
                    # =============================================================================
                    # # =============================================================================
                    # # Checking wether we still have autocorrelation 
                    # # =============================================================================
                    # =============================================================================
                    
                    verif_acf_plot= st.radio('Are you seeing autocorrelation after lag 0 ? ', ('Yes','No'))
                    if verif_acf_plot == 'Yes':

                        st.write('### 3. First conclusion on the series')                                
                        st.info('You are not facing a random walk process. Thus, it would then be relevant to go on the statistical models section for making better forecasting models by using the lags found on the ACF.')
                        
                    elif verif_acf_plot == 'No':
                        
                        st.info("You are facing a random walk process, you can only reasonably apply baseline models to make forecast. Ideally, you want to forecast random walk in the short term or the next timestep.")                
                        slct_box_naive_random_walk_forecasting = st.selectbox("Please select a naive forecasting method for your random walk :", ('Drift','Next timestep'))
                        
                        #Adding the train and the test set for the forecasting methods
                        train_diff = data_diff[:-(len(train))]
                        test_diff = data_diff[-(len(test)):]

                        # =============================================================================
                        # # =============================================================================
                        # # Naive Forecasting method
                        # # =============================================================================
                        # =============================================================================
                        
                        if slct_box_naive_random_walk_forecasting == 'Drift':
                            
                            st.info('Explanation : Calculating the slope between the first and the last value of the train set and extrapolating this straight line into the future.')
                            
                            # Forecast using the drift
                            
                            last_diff = train['Adj Close'].iloc[-1]
                            deltaX = len(train)    # Length of the training set
                            deltaY = last_diff - train['Adj Close'].iloc[0] # Remember to subtract the initial value of the training set
                            
                            drift = deltaY / deltaX
                            
                            x_vals = np.arange((len(data) - len(test)), (len(data) + len(forecast_data_set)), 1) #Creating the range of timesteps 
                            
                            pred_drift = drift * x_vals + train['Adj Close'].iloc[0] # Add the initial value back in the predictions

                            test.loc[:, 'pred_drift'] = pred_drift[:len(test)] # Only containing the test drift
                            
                            #Data diff pred_drift

                            data['pred_drift'] = test['pred_drift']

                            #Add the results to the forecast data set

                            forecast_data_set['forecast_drift'] = pd.Series()
                            forecast_data_set['forecast_drift'] = pred_drift[len(test):]

                            # =============================================================================
                            # # =============================================================================
                            # # Original series plot
                            # # =============================================================================
                            # =============================================================================
                            if st.checkbox('Output the prediction on the original series'):

                                data_merge_drift =  pd.concat([data,forecast_data_set])
                            
                                funct_plot_results_series(data_merge_drift[['Adj Close','pred_drift','forecast_drift']], 'Drift Forecasting')
                                
                            # =============================================================================
                            # # =============================================================================
                            # # Drift MSE
                            # # =============================================================================
                            # =============================================================================
                            
                            if st.checkbox('Head the MSE') :
                                # Metric
                                st.metric("Mean squarred error (MSE)", value = mean_squared_error(test['Adj Close'], test['pred_drift']))
                                st.warning('The MAPE metric could not be used here because our random walk could take the value of 0, calculate the difference from an observed value of 0 implies a division by 0 which is not allowed in mathematics.')
                            
                        # =============================================================================
                        # # =============================================================================
                        # # Next Timestep
                        # # =============================================================================
                        # =============================================================================
                        
                        if slct_box_naive_random_walk_forecasting == 'Next timestep':
                            
                            st.info('Explanation : By taking the initial observed value, we use it to predict the next timestep.')
                            
                            # Using the shifting method
                            pred_next_timestep = test['Adj Close'].shift(periods=1) 
                            pred_next_timestep = pred_next_timestep.dropna()
                            pred_next_timestep = pd.DataFrame(pred_next_timestep)
                            pred_next_timestep = pred_next_timestep.rename(columns= {'Adj Close' : 'next_timestep'})    

                            # For the data set
                            data['pred_next_timestep'] = pd.Series() # Initialize an empty column to hold our predictions
                            data['pred_next_timestep'] = pred_next_timestep['next_timestep']

                            # For the forecast set
                            forecast_data_set['forecast_next_timestep'] = pd.Series()
                            forecast_data_set['forecast_next_timestep'] = data['Adj Close'][-1]

                            data_merge_next_timestep = pd.concat([data,forecast_data_set])

                            # =============================================================================
                            # # =============================================================================
                            # # Plot
                            # # =============================================================================
                            # =============================================================================

                            if st.checkbox('Plot the results on the original series'):
                                
                                funct_plot_results_series(data_merge_next_timestep[['Adj Close','pred_next_timestep','forecast_next_timestep']],'Next timestep')
                            
                            # =============================================================================
                            # # =============================================================================
                            # # Next timestep MSE
                            # # =============================================================================
                            # =============================================================================
                            chck_box_rmse_nxt_tmstep= st.checkbox('Head the MSE') 
                            
                            if chck_box_rmse_nxt_tmstep:
                                # Metric
                                st.metric("Mean squarred error (MSE)", value = mean_squared_error(data['Adj Close'][-(len(test)-1):], pred_next_timestep['next_timestep']))
                                st.warning('The MAPE metric could not be used here because the first value of our random walk could take the value of 0, calculate the difference from an observed value of 0 implies a division by 0 which is not allowed in mathematics.')
                                st.warning('Here be careful because you are only making prediction for the next value solely by using the last past value. Hence, the MSE is not a solid indicator of performance here.')
                            
                        # =============================================================================
                        # # =============================================================================
                        # # Conclusion of our random forecasting
                        # # =============================================================================
                        # =============================================================================
                        
                        st.write(' ### 3. First conclusion on the series')
                        st.info('You did made random walk forecasting and see whom is the best baseline model with the MSE. Because of the series randomness, there is not much things to learn from. Hence, it does not make sense to use advanced models to predict the randomness of your series.')
                        st.info('The human mind sees patterns everywhere and we must be vigilant that we are not fooling ourselves and wasting time by developing elaborate models for a random walk process.')
        # =============================================================================
        # # =============================================================================
        # # Case where the series is already stationary
        # # =============================================================================
        # =============================================================================

        elif data_ad_fuller[1] < 0.05 : 
            
            st.markdown("###### We reject the null hypothesis (Ho) of unit root, the series is already stationnary.")
            chck_box_ad_fuller_diff_order = 0 # The series does not have to been stationary
            st.session_state['chck_box_ad_fuller_diff_order'] = 0   


            data_diff = pd.DataFrame(data['Adj Close']) 
            st.session_state['data_diff'] = data_diff   


            # =============================================================================
            # # =============================================================================
            # # Plot the ACF
            # # =============================================================================
            # =============================================================================
            
            st.markdown("### 2. Autocorrelation function (ACF)")
            with st.expander ('Plot the ACF') : 
                
                #Lags number based on the length of the train set, for better practicality 
                lags_input = st.slider(" How many lags would you like to add ? ", 0,len(train['Adj Close']),5) 
                st.pyplot(plot_acf(data_diff,title=f'Autocorrelation plot on the {ticker} close price (after differencing the series)', lags=lags_input))
                
                #Output the higher ACF value in absolute value
                acf_value_head = st.checkbox('Click here to head the significant ACF values')
                
                if acf_value_head: 
                    
                    df_acf=pd.DataFrame(acf(data_diff,nlags= len(train['Adj Close'])))
                    df_acf = df_acf.abs()
                    df_acf.index.names = ['Lags']
                    df_acf = df_acf.rename(columns = {0:'ACF in absolute value'})
                    df_acf = df_acf.sort_values(by='ACF in absolute value', ascending=False)
                                                            
                    st.dataframe(df_acf,width=500)
            
                #Checking wether we still have autocorrelation 
                verif_acf_plot= st.radio('Are you seeing autocorrelation after lag 0 ? ', ('Yes','No'))
                
                if verif_acf_plot == 'Yes':
    
                    st.write('### 3. First conclusion on the series')                                
                    st.info('You are not facing a random walk process. Thus, it would then be relevant to go on the statistical models section for making better forecasting models by using the lags found on the ACF.')
                
                elif verif_acf_plot == 'No':
                    
                    st.info("You are facing a random walk process, you can only reasonably apply baseline models to make forecast. Ideally, you want to forecast random walk in the short term or the next timestep.")                
                    slct_box_naive_random_walk_forecasting = st.selectbox("Please select a naive forecasting method for your random walk :", ('Drift','Next timestep'))
                    
                    # =============================================================================
                    # # =============================================================================
                    # # Naive Forecasting method
                    # # =============================================================================
                    # =============================================================================
                    
                    if slct_box_naive_random_walk_forecasting == 'Drift':
                        
                        st.info('Explanation : Calculating the slope between the first and the last value of the train set and extrapolating this straight line into the future.')
                        
                        # Forecast using the drift
                        
                        last_diff = train['Adj Close'].iloc[-1]
                        deltaX = len(train)    # Length of the training set
                        deltaY = last_diff - train['Adj Close'].iloc[0] # Remember to subtract the initial value of the training set
                        
                        drift = deltaY / deltaX
                        
                        x_vals = np.arange((len(data) - len(test)), (len(data) + len(forecast_data_set)), 1) #Creating the range of timesteps 
                        
                        pred_drift = drift * x_vals + train['Adj Close'].iloc[0] # Add the initial value back in the predictions

                        test.loc[:, 'pred_drift'] = pred_drift[:len(test)] # Only containing the test drift
                        
                        #Data diff pred_drift

                        data['pred_drift'] = test['pred_drift']

                        #Add the results to the forecast data set

                        forecast_data_set['forecast_drift'] = pd.Series()
                        forecast_data_set['forecast_drift'] = pred_drift[len(test):]

                        # =============================================================================
                        # # =============================================================================
                        # # Original series plot
                        # # =============================================================================
                        # =============================================================================
                        if st.checkbox('Output the prediction on the original series'):

                            data_merge_drift =  pd.concat([data,forecast_data_set])
                        
                            funct_plot_results_series(data_merge_drift[['Adj Close','pred_drift','forecast_drift']], 'Drift Forecasting')
                            
                        # =============================================================================
                        # # =============================================================================
                        # # Drift MSE
                        # # =============================================================================
                        # =============================================================================
                        
                        if st.checkbox('Head the MSE') :
                            # Metric
                            st.metric("Mean squarred error (MSE)", value = mean_squared_error(test['Adj Close'], test['pred_drift']))
                            st.warning('The MAPE metric could not be used here because our random walk could take the value of 0, calculate the difference from an observed value of 0 implies a division by 0 which is not allowed in mathematics.')
                        
                    # =============================================================================
                    # # =============================================================================
                    # # Next Timestep
                    # # =============================================================================
                    # =============================================================================
                    
                    if slct_box_naive_random_walk_forecasting == 'Next timestep':
                        
                        st.info('Explanation : By taking the initial observed value, we use it to predict the next timestep.')
                        
                        # Using the shifting method
                        pred_next_timestep = test['Adj Close'].shift(periods=1) 
                        pred_next_timestep = pred_next_timestep.dropna()
                        pred_next_timestep = pd.DataFrame(pred_next_timestep)
                        pred_next_timestep = pred_next_timestep.rename(columns= {'Adj Close' : 'next_timestep'})    

                        # For the data set
                        data['pred_next_timestep'] = pd.Series() # Initialize an empty column to hold our predictions
                        data['pred_next_timestep'] = pred_next_timestep['next_timestep']

                        # For the forecast set
                        forecast_data_set['forecast_next_timestep'] = pd.Series()
                        forecast_data_set['forecast_next_timestep'] = data['Adj Close'][-1]

                        data_merge_next_timestep = pd.concat([data,forecast_data_set])

                        # =============================================================================
                        # # =============================================================================
                        # # Plot
                        # # =============================================================================
                        # =============================================================================

                        if st.checkbox('Plot the results on the original series'):
                            
                            funct_plot_results_series(data_merge_next_timestep[['Adj Close','pred_next_timestep','forecast_next_timestep']],'Next timestep')
                        
                        # =============================================================================
                        # # =============================================================================
                        # # Next timestep MSE
                        # # =============================================================================
                        # =============================================================================
                        chck_box_rmse_nxt_tmstep= st.checkbox('Head the MSE') 
                        
                        if chck_box_rmse_nxt_tmstep:
                            # Metric
                            st.metric("Mean squarred error (MSE)", value = mean_squared_error(data['Adj Close'][-(len(test)-1):], pred_next_timestep['next_timestep']))
                            st.warning('The MAPE metric could not be used here because the first value of our random walk could take the value of 0, calculate the difference from an observed value of 0 implies a division by 0 which is not allowed in mathematics.')
                            st.warning('Here be careful because you are only making prediction for the next value solely by using the last past value. Hence, the MSE is not a solid indicator of performance here.')
                        
                    # =============================================================================
                    # # =============================================================================
                    # # Conclustion about the series
                    # # =============================================================================
                    # =============================================================================
                    
                    st.write(' ### 3. First conclusion on the series')
                    st.info('You did made random walk forecasting and see whom is the best baseline model with the MSE. Because of the series randomness, there is not much things to learn from. Hence, it does not make sense to use advanced models to predict the randomness of your series.')
