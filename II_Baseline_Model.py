import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import warnings

from PIL import Image
from statsmodels.tsa.seasonal import STL
warnings.filterwarnings('ignore') #to ignore the waning errors

# =============================================================================
# =============================================================================
# # Function to plot the seasonality of the series
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
# # Plot the differentiated series and its forecasting results
# =============================================================================
# =============================================================================
@st.cache_data  
def funct_plot_results_series(df1 : list, name_model : str):

    fig = px.line(df1, title= f'Forecast results for the {name_model} model',width =650)
    
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

def add_baseline_forecast(prediction_df : list, name_model : str):

    forecast_data_set = forecast_data_set.append(prediction_df)

    return 
    
# =============================================================================
# =============================================================================
# #  MAPE function
# =============================================================================
# =============================================================================
@st.cache_data  
def mape(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred)/y_true ))*100


def funct_baseline_model():
    

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
    # # Baseline model explanation
    # # =============================================================================
    # =============================================================================
                        
    st.write(""" 
                 ## II. Choose a baseline model
                 """)
             
    with st.expander('Why using a baseline model ?'):
        
        image = Image.open('/Users/morisaao/Desktop/Freelance/Da_Vinci_Forecasting/PNG/Baseline_Model.png')
        st.image(image, caption='Baseline model explanation')

    # =============================================================================
    # # =============================================================================
    # # Chose a baseline model
    # # =============================================================================
    # =============================================================================

    chck_box_model = st.selectbox("Choose a baseline model ", ('1. Arithmetic mean','2. Last known value','3. Last year mean','4. Seasonality Forecast' ))
    
    # =============================================================================
    # # =============================================================================
    # # Arithmetic Mean
    # # =============================================================================
    # =============================================================================

    if chck_box_model == '1. Arithmetic mean':
        
        st.info('Forecasting method : Arithmetic mean of the last train week set.')
        
        # Compute the arithmetic mean and add it to the test set
        
        historical_mean = np.mean(train['Adj Close'][-7])
        
        data['pred_mean'] = pd.Series()
        data['pred_mean'][-len(test):] = historical_mean
        
        forecast_data_set['forecast_mean'] = pd.Series()
        forecast_data_set['forecast_mean'] = historical_mean

        data_arth_mean_concat = pd.concat([data,forecast_data_set],ignore_index=False)

        # =============================================================================
        # =============================================================================
        # # Plotting the differentiated series        
        # =============================================================================
        # =============================================================================
        
        with st.expander('Plot the prediction'): 

            funct_plot_results_series(data_arth_mean_concat[['Adj Close','pred_mean','forecast_mean']], 'Arithmetic Mean')
        
        # =============================================================================
        # =============================================================================
        # # Defining a MAPE (mean absolute percentage error) function who is our error metric for our naive prediction
        # =============================================================================
        # =============================================================================

        mape_hist_mean = mape(data['Adj Close'][-len(test):], data['pred_mean'])
        
        with st.expander('Display the MAPE'):

            st.metric(label="MAPE", value= mape_hist_mean)
            st.write(f"""
                ##### On the test set, your forecasts are, on average, {mape_hist_mean.round(2)} % below the actual adjusted close prices for the last {len(test)} days.
            """  )


    # =============================================================================
    # # =============================================================================
    # # Last known value
    # # =============================================================================
    # =============================================================================

    elif chck_box_model == '2. Last known value':
            
        st.info('Forecasting method : Prediction solely by using the last data of the last train week set.')
        # Separate the dataset into train and test set 
        
        last = train['Adj Close'].iloc[-1]
        
        data['pred_last'] = pd.Series()
        data['pred_last'][-len(test):] = last

        forecast_data_set['forecast_last'] = pd.Series()
        forecast_data_set['forecast_last'] = last

        data_last_concat = pd.concat([data,forecast_data_set],ignore_index=False)


        with st.expander ('Plot the results'):
            funct_plot_results_series(data_last_concat[['Adj Close','pred_last','forecast_last']], 'Last Known Value')

        mape_last_pred = mape(data['Adj Close'][-len(test):], data['pred_last'])
    
    
        with st.expander ('Display the MAPE'):
            
            st.metric(label="MAPE", value= mape_last_pred)
            st.write(f"""
                ##### On the test set, your forecasts are, on average, {mape_last_pred.round(2)} % below the actual adjusted close prices for the last {len(test)} days.
            """)  

    # =============================================================================
    # # =============================================================================
    # # Last year mean 
    # # =============================================================================
    # =============================================================================
        
    elif chck_box_model == '3. Last year mean':
        
        st.info('Forecasting method : Using the last year mean of the train set')
    
        if len(train) < 365 : 
            st.warning('Your date range must be equal or over 1 year and 1 week.')
        else :
            last_year_mean = np.mean(train['Adj Close'][-365:])
            data['pred_last_yr_mean'] = pd.Series()
            data['pred_last_yr_mean'][-len(test):] = last_year_mean
            
            forecast_data_set['forecast_last_yr'] = pd.Series()
            forecast_data_set['forecast_last_yr'] = last_year_mean

            data_last_yr_concat = pd.concat([data,forecast_data_set],ignore_index=False)

            with st.expander('Plot the results'):
                funct_plot_results_series(data_last_yr_concat [['Adj Close','pred_last_yr_mean','forecast_last_yr']], 'Last Year Mean')
            
            mape_last_year = mape(data['Adj Close'][-len(test):], data['pred_last_yr_mean'])
            
            with st.expander ('Display the MAPE'):
                st.metric(label="MAPE", value= mape_last_year)
                st.write(f"""
                    ##### On the test set, your forecasts are, on average, {mape_last_year.round(2)} % below the actual adjusted close prices for the last {len(test)} days.
                        """)    
    
    # =============================================================================
    # # =============================================================================
    # # Last season
    # # =============================================================================
    # =============================================================================

    elif chck_box_model == '4. Seasonality Forecast':
        st.info('Forecasting method : Replicating the seasonality of the last train week set.')
                    
        with st.expander('Decompose the series on a weekly basis'): 

            seasonal_plot(data,7,ticker)
        
        last_season = train['Adj Close'][-7:].values  

        data['pred_last_season'] = pd.Series()
        data['pred_last_season'][len(train):(len(train)+7)] = last_season

        forecast_data_set['forecast_last_season'] = pd.Series()
        forecast_data_set['forecast_last_season'][:7] = last_season


        data_last_season_concat = pd.concat([data,forecast_data_set],ignore_index=False)
      
        with st.expander ('Plot the results'):

            funct_plot_results_series(data_last_season_concat [['Adj Close','pred_last_season','forecast_last_season']], 'Last Season Forecast')

        # Defining a MAPE (mean absolute percentage error) function who is our error metric for our naive prediction        
        mape_seasonal = mape(data['Adj Close'][len(train):(len(train)+7)], data['pred_last_season'])
        
        with st.expander ('Display the MAPE'):

            st.metric(label="MAPE", value= mape_seasonal)
            st.write (f"""
                ##### On the test set, your forecasts are, on average, {mape_seasonal.round(2)} % below the actual adjusted close prices for the last {len(test)} days.
            """)

    