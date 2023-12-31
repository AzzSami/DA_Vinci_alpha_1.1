

import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie

def funct_intro():
    
    def local_css(file_name):
        with open(file_name) as f: 
            st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    
    #CSS sheet
    local_css("style.css")

    #Title
    st.markdown("<h1 style='justify-content: center;'> DA VINCI FORECASTING </h1>", unsafe_allow_html=True)

    # =============================================================================
    # Introduction
    # =============================================================================

    st.info('''   
            Welcome on the DA VINCI FORECASTING app, the application entirely dedicated on stock price forecasting.
            I hope that this application will perfectly respond to your expectations. 
            
            You should have at least an understanding of the basic theory of financial market and statistics, some theory reminders have been placed along the application when I thought it would be useful.
            However, it would be almost impossible to explain every step we will walk through. 
            Hence, it is for that very reason I did add a plan showing you the big lines of the process.

            When making this application, I kept in mind that not everything can be predicted. I think we have to remember that we are making predictions about the future based on past events.
            Forecasting is not a hard science, we might find encounter situations where nothing seems to work.
            
            Thus, we should always learn from failure and accept that some forecasts could fail. 
            Sometimes we win and have an edge over the others, and sometimes we learn.

            Thank you for reading this intro! Now let us use the DA VINCI FORECASTING application and see what she got in the belly, shall we ?''')
    
    st.warning('For now, on this alpha version, when first launching of this app or reloading the page with Windows + R, you have to click on the gathering data sections before going the other sections to load the dataset.')
    st.warning('Also, for the next release, the application will be more rapid, when using some models like the SARIMA you might facing slowness, I am working on to make to make the code more efficient.')
    # =============================================================================
    # CSS sidebar  
    # =============================================================================
    
    st.header("""Plan""")

    st.subheader("""I. Gathering the data""")

    st.subheader("""II. Choose a baseline model""")
    
    st.caption("""1. Arithmetic mean""")
    st.caption("""2. Last known value""")
    st.caption("""3. Last year""")
    st.caption("""4. Seasonality forecast""")


    st.subheader("""III. Verification of random walk presence""")
    
    st.caption("""1. Stationarity test""")
    st.caption("""2. ACF plot""")
    st.caption("""3. First conclusion on the series""")
        

    st.subheader("""IV. Statistical Models""") 
    
    st.caption("""1. Moving Average process (MA)""") 
    st.caption("""2. Autoregressive Process (AR)""") 
    st.caption("""3. Autoregressive Moving Average Process (ARMA)""") 
    st.caption("""4. Autoregressive Integrated Moving Average Process (ARIMA)""") 
    st.caption("""5. Seasonal Autoregressive Integrated Moving Average Process (SARIMA)""") 


    
