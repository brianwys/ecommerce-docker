#Need to convert to Flask
import os
import json
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
# import yfinance as yf
# from millify import millify
# import plotly.express as px
# from streamlit_extras.add_vertical_space import add_vertical_space

from final_hr_analytics.utils.load_data import save_department_data, get_existing_company_names, get_folder_path

#from tabs.hr_analyst.hr_Company_Insights import main_CompanyInsights
from tabs.hr_analyst.hr_department_performance import show_top_companies, show_worst_companies
# from tabs.hr_analyst.hr_Latest_News import main_LatestNews
# from tabs.hr_analyst.hr_Business_Conversation import main_BusinessConversation

import warnings
warnings.filterwarnings('ignore')

import sys
#sys.path.append("code")
#from utility import hide_github

def main():
    st.set_page_config(layout="wide")
    #hide_github()

    st.title("Human Resources Smart Assistant")

    tab_title = ["Configuration", "Department Performence", "Department Highlights"]

 
    tab_Main, tab_DepartmentPerfomance, tab_Departmenthighlights = st.tabs(tab_title)  

    #query_par = st.query_par_params.to_dict()
    #query_par = st.query_params.to_dict()
    query_par = st.session_state.to_dict()

    if "tab01" in query_par and len(query_par["tab01"]) > 0:
        index_tab = tab_title.index(query_par["tab01"])

        ## Click on that tab
        js = f"""
        <script>
            var tab = window.parent.document.getElementById('tabs-bui3-tab-{index_tab}');
            tab.click();
        </script>
        """
        st.components.v1.html(js)

    with tab_Main:
        # Main app
        st.subheader("Main")
        st.write("Please define the dataset, upload KPI Dataset analysis first before you can start analyzing")
        st.write("")

        selected_action = st.radio("Select Action", ["Select a Department", "Add New Data"])

        
        existing_name = get_existing_company_names() or ['None']
        if selected_action == "Select a Department":
            selected_Department = st.selectbox("Select Department", ['Select a Department'] + existing_name)
            if selected_Department != "Select a Department":
                st.success(f"Department '{selected_Department}' data selected")
                st.session_state['selected_department'] = selected_Department

                # Department data location
                department_data = get_folder_path(selected_Department)
                st.session_state['selected_data_folder_path'] = department_data

        else:
            st.subheader("Upload Files")
            department_name = st.text_input("Department Name", help="Type department name you want to analyze")
            today = datetime.now().date()
            default_date = (today - timedelta(days=30)).replace(day=1)
            selected_date = st.date_input("Report Date", value=default_date)
            # financial_stmt = st.file_uploader("Upload Financial Statement (XLSX)", accept_multiple_files=False, type=["xlsx"])
            # annual_report = st.file_uploader("Upload Company Review (DOCX)", accept_multiple_files=False, type=["docx"])

            if st.button("Upload and analyze"):
                # Set session state variables after process has completed
                department_folder = f"./user_files/{department_name}"

                # Save department data to JSON
                save_department_data(department_name, selected_date, department_folder)

                st.session_state.selected_Department = department_name

    with tab_DepartmentPerfomance:
        st.subheader(tab_title[1])
        try:
            selected_Department = st.session_state['selected_department'] 
            selected_file_path = st.session_state['selected_data_folder_path']
            show_top_companies(selected_file_path)
            show_worst_companies(selected_file_path)
        except:
            st.info("Please select the company first 🚨")


    with tab_Departmenthighlights:
        st.subheader(tab_title[1])

        # show_top_companies()

    # with tab_LatestNews:
    #    st.subheader(tab_title[3])

    #    main_LatestNews()

    # with tab_BusinessConversations:
    #    st.subheader(tab_title[4])

    #    main_BusinessConversation()


if __name__ == "__main__":
    main()
