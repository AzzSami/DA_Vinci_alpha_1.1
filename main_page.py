
import streamlit as st
from Intro import funct_intro
from I_Gathering_Data import funct_gather_data
from II_Baseline_Model import funct_baseline_model
from III_Random_Walk_presence import funct_random_walk_presence
from IV_Statistical_Models import funct_stat_model

def main():
        
    # =============================================================================
    # # =============================================================================
    # # For styling the page
    # # =============================================================================
    # =============================================================================
    
    def local_css(file_name):
        with open(file_name) as f: 
            st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    
    #CSS sheet
    local_css("style.css")
    
    
    # Register the pages
    
    if 'page_state' not in st.session_state :
        st.session_state['page_state'] = 'Intro' #this function will make sure we go back in the main page we the app is launched/relaunched


    #Writing page selection to session page
    
    if st.sidebar.button('Intro'):
        st.session_state['page_state'] ='Intro'
    
    if st.sidebar.button('Gathering Data'):
        st.session_state['page_state'] ='Gathering Data'
    
    if st.sidebar.button('Baseline Model'):
        st.session_state['page_state'] ='Baseline Model'

    if st.sidebar.button('Random Walk'):
        st.session_state['page_state'] ='Random Walk'
    
    # =============================================================================
    # =============================================================================
    # # Statistics
    # =============================================================================
    # =============================================================================
    if st.sidebar.button('Statistical Model'):
        st.session_state['page_state'] ='Statistical Model'

    

    #Enter the page that will be displayed
    pages_main= {
        
        'Intro' : intro_page,
        'Gathering Data' : gather_data,
        'Baseline Model' : baseline_model,
        'Random Walk' : random_walk,
        'Statistical Model' : statistical_model
        }
    
    # Run the selected pages
    pages_main[st.session_state['page_state']]()
    

# We define the functions for each files to call

# Here for instance we make sure that we will call the intro function
def intro_page(): 
    
    funct_intro()
    
def gather_data():
    
    funct_gather_data()
    
def baseline_model(): 
        
    if 'Gathering Data' not in st.session_state['page_state'] :
            st.error ('Click first on the Gathering Data section to acces dataset')
    else : 
        funct_baseline_model()
    
def random_walk(): 
   
        if 'Gathering Data' not in st.session_state['page_state'] :
            st.error ('Click first on the Gathering Data section to acces dataset')
        else : 
            funct_random_walk_presence()
    
def statistical_model(): 

        if 'Gathering Data' not in st.session_state['page_state'] :
            st.error ('Click first on the Gathering Data section to acces dataset')
        else : 
            funct_stat_model()

if __name__ == '__main__':
    main()
