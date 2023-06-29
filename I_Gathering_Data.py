#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 20:36:17 2023

@author: morisaao
"""

import streamlit as st
import yfinance as yf 
import datetime
import plotly.express as px
from PIL import Image
import pandas as pd

@st.cache_data    # Use to slow down the download of the data and make sure we use memory efficiently
def get_data (ticker,start_date,end_date):
    
    data = yf.download(ticker, start_date, end_date)
    # Removing the na values
    data = data.dropna()
    return data


def split_data(df,test_split):

    n=int(len(df)*test_split)
    train,test =df[:-n],df[-n:]
    
    return train, test

# =============================================================================
# =============================================================================
# # Plot the series
# =============================================================================
# =============================================================================

@st.cache_data
def funct_plot_series(df : list, name_series : str):
    
    fig = px.line(df, title= f'The {name_series} adjusted close price',width =650)
    
    return st.write(fig)

def funct_gather_data():
    
    # =============================================================================
    # =============================================================================
    # # Title
    # =============================================================================
    # =============================================================================
    
    st.title('I / Gather the Data')
       
    # =============================================================================
    # Data processing and saving in a session state
    # =============================================================================
    
    
# =============================================================================
#     Start Date
# =============================================================================

    # Set the start and end date
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = datetime.date(2023, 1, 1)
        
    start_date = st.date_input(
        "Start Date",
        st.session_state['start_date'])
    
    st.session_state['start_date'] = start_date
    
    
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = None
    
    end_date = st.date_input(
        "End date",
        st.session_state['end_date'])
    
    st.session_state['end_date'] = end_date
    
# =============================================================================
# =============================================================================
# # Choose the ticker of our stock
# =============================================================================
# =============================================================================
    if 'ticker' not in st.session_state :
        st.session_state ['ticker'] = 'BTC-USD'
    
    # Creating ticker widget with default value from session state
    ticker = st.text_input("Enter a valid stock ticker...", st.session_state['ticker'])
    
    # Writing ticker value to session state
    st.session_state['ticker'] = ticker
    
# =============================================================================
# =============================================================================
# # Get the data registered
# =============================================================================
# =============================================================================
    # Get the data 
    if 'data' not in st.session_state :
        st.session_state ['data'] = None

    
    data = get_data (ticker,start_date,end_date)
    st.session_state['data'] = data

# =============================================================================
# =============================================================================
# # Data
# =============================================================================
# =============================================================================

    #Head the data
    with st.expander('Head the data'):
        st.dataframe(data,width=1000)
    
    #Plot the stock
    with st.expander('Plot the Adj Close price stock '):
        funct_plot_series(data['Adj Close'],ticker)


# =============================================================================
# =============================================================================
# # Get info on the Dataframe
# =============================================================================
# =============================================================================

    with st.expander('Get the summary statistic of the data'):
        st.dataframe(data.describe(include = 'all',datetime_is_numeric = True),width=700)


# =============================================================================
# =============================================================================
# # Horizon chosen
# =============================================================================
# =============================================================================

    if 'forecast_horizon' not in st.session_state :
        st.session_state ['forecast_horizon'] = None
        
    Forecast_horizon_slider = st.slider('Please choose the number of days you would like to forecast.',value= st.session_state ['forecast_horizon'], min_value =1, max_value =30)
    st.session_state ['forecast_horizon'] = Forecast_horizon_slider

    #Creating a forecast dataset according to the horizon period and adding it to the actual dataframe

    forecast_data_set = pd.date_range(st.session_state['end_date'],periods = Forecast_horizon_slider)
    forecast_data_set = pd.DataFrame({'Date': forecast_data_set })
    forecast_data_set = forecast_data_set.set_index('Date')

    #Adding the forecast set to the cache
    st.session_state['forecast_data_set'] = forecast_data_set

# =============================================================================
# =============================================================================
# # Split the data
# =============================================================================
# =============================================================================
    if 'train' not in st.session_state:
        
        st.session_state['train'] = None
        
    if 'test' not in st.session_state:
        
        st.session_state['test'] = None
        
    if 'test_percentage' not in st.session_state:
        
        st.session_state['test_percentage'] = 0.1
        

    test_percentage = st.slider('Please split your data between train and test set',0.1, 1.0,value = st.session_state ['test_percentage'])
    st.session_state ['test_percentage'] = test_percentage


    train,test=split_data(data,test_percentage)
    st.write(f'Train : {len(train)}, Test : {len(test)}')

    # Recording the train and test set to use for all of the other pages
    st.session_state['train'] = train
    st.session_state['test'] = test


# =============================================================================
# =============================================================================
# # Record in the cache the date of the train, test and forecast set
# =============================================================================
# =============================================================================

    #Start date

    st.session_state['start_date_train'] = train.index[0]

    st.session_state['start_date_test'] = test.index[0]

    st.session_state['start_date_forecast'] = forecast_data_set.index[0]

    #End date

    st.session_state['end_date_train'] = train.index[-1]

    st.session_state['end_date_test'] = test.index[-1]

    st.session_state['end_date_forecast'] = forecast_data_set.index[-1]
