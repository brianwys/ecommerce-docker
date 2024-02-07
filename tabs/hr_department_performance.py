import os
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from millify import millify
import plotly.express as px
import json


from streamlit_extras.add_vertical_space import add_vertical_space

import sys
sys.path.append('utils')

# from watsonx import WatsonXModel

import warnings
warnings.filterwarnings('ignore')


#sys.path.append("code")
#from utility import hide_github

import streamlit as st
import json
from pathlib import Path

#fungsi untuk membaca jsonfile dan dapat mengkaitkan menggunakan file_path

def read_json_file(file_path): 
    """
    Read JSON file and handle errors.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None

# Example usage with a dynamic file path

def read_worst_1(file_path):
    file_path_worst_1 = f'{file_path}/worst1.json'
    json_data = read_json_file(file_path_worst_1)
    return json_data

def read_worst_2(file_path):
    file_path_worst_2 = f'{file_path}/worst2.json'
    json_data = read_json_file(file_path_worst_2)
    return json_data


def read_rev_growth(file_path): #sales Enablement
    file_path_growth = f'{file_path}/rev_growth_percent.json'
    json_data = read_json_file(file_path_growth)
    return json_data 

# def read_rev_growth_best2(file_path): #sales Operations
#     file_path_growth = f'{file_path}/rev_growth_percent.json'
#     json_data = read_json_file(file_path_growth)
#     return json_data 

# def read_rev_growth_worst1(file_path): #Retail Sales
#     file_path_growth = f'{file_path}/rev_growth_percent.json'
#     json_data = read_json_file(file_path_growth)
#     return json_data 

# def read_rev_growth_worst1(file_path): #e-commerce Sales
#     file_path_growth = f'{file_path}/rev_growth_percent.json'
#     json_data = read_json_file(file_path_growth)
#     return json_data 

def read_best_1(file_path):
    #Read Data Best 1
    file_path_best_1 = f'{file_path}/best_1.json'
    json_data = read_json_file(file_path_best_1)
    return json_data

def read_best_2(file_path):
    file_path_best_1 = f'{file_path}/best_2.json'
    json_data = read_json_file(file_path_best_1)
    return json_data


def show_top_companies(file_path):
    # Right section for detailed information
    st.subheader('The Best 2 Department')
    col1, col2 = st.columns(2)  # Divide the layout into two columns

    # Read department data using read_best_1
    best_1_result = read_best_1(file_path)
    rev_growth_result_1 = read_rev_growth(file_path)

    # Display department information using read_best_1
    with col1: 
        st.subheader(best_1_result['department_name'])
        with st.expander(f'Detailed Information - {best_1_result["department_name"]}'):
            st.subheader('Division Forecast')
            st.image(best_1_result['graph_path'])
            st.write('------------------------------')
            st.subheader('Insight')
            st.write(best_1_result['insight']['high_kpi_achieved_ratio'])
            st.write('------------------------------')
            st.subheader('Attributes of Success')
            for attribute in best_1_result['insight']['attributes_success_to']:
                st.write(attribute)
            st.write('--------------------------')
            st.subheader('Revenue Growth')
            st.text('Predicted Next Month Revenue Growth')
            st.write(rev_growth_result_1['Sales Enablement']['start'],"-",rev_growth_result_1['Sales Enablement']['end'],"=",rev_growth_result_1['Sales Enablement']['change'])
            st.text('Revenue Change in Percentage ')
            st.write(rev_growth_result_1['Sales Enablement']['percentage'])
            st.write('------------------------------')
            st.subheader('Indicator for Success')
            for indicator in best_1_result['indicator_for_success']:
                st.write(indicator)
    
    # Read department data using read_best_2
    best_2_result = read_best_2(file_path)
    rev_growth_result_2 = read_rev_growth(file_path)
    # Display department information using read_best_2
    with col2:
        st.subheader(best_2_result['department_name'])
        with st.expander(f'Detailed Information - {best_2_result["department_name"]}'):
            st.subheader('Division Forecast')
            st.image(best_2_result['graph_path'])
            st.write('------------------------------')
            st.subheader('Insight')
            st.write(best_2_result['insight']['high_kpi_achieved_ratio'])
            st.write('------------------------------')
            st.subheader('Attributes of Success')
            for attribute in best_2_result['insight']['attributes_success_to']:
                st.write(attribute)
            st.write('--------------------------')
            st.subheader('Revenue Growth')
            st.text('Predicted Next Month Revenue Growth')
            st.write(rev_growth_result_2['Sales Operations']['start'],"-",rev_growth_result_2['Sales Operations']['end'],"=",rev_growth_result_2['Sales Operations']['change'])
            st.text('Revenue Change in Value')
            st.write(rev_growth_result_2['Sales Operations']['percentage'])
            st.write('--------------------------')
            st.subheader('Indicator for Success')
            for indicator in best_2_result['indicator_for_success']:
                st.write(indicator)

def show_worst_companies(file_path):
    st.subheader("2 Department need to Imporve")
    st.write("Revenue Growth")
    col1, col2 = st.columns(2)  # Divide the layout into two columns
    # Read department data using read_worst_1
    worst_1_result = read_worst_1(file_path)
    rev_growth_worst1=read_rev_growth(file_path)
    with col1: #bawah
        st.subheader(worst_1_result['department_name'])
        with st.expander(f'Detailed Information - {worst_1_result["department_name"]}'):
            st.subheader('Division Forecast')
            st.image(worst_1_result['graph_path'])
            st.write('------------------------------')
            st.subheader('Insight')
            st.write(worst_1_result['insight']['low_kpi_achieved_ratio'])
            st.write('------------------------------')
            st.subheader('Low Kpi Causes Possibility')
            for attribute in worst_1_result['insight']['possible_causes']:
                st.write(attribute)
            st.write('--------------------------')
            st.subheader('Revenue Growth')
            st.text('Predicted Next Month Revenue Growth')
            st.write(rev_growth_worst1['Retail Sales']['start'],"-",rev_growth_worst1['Retail Sales']['end'],"=",rev_growth_worst1['Retail Sales']['change'])
            st.text('Revenue Change in Value')
            st.write(rev_growth_worst1['Retail Sales']['percentage'])
            st.write('--------------------------')
            st.subheader('Recommendation for Improvement')
            st.write(worst_1_result['recommendations'][0])
                
    # Read department data using read_worst_2
    worst_2_result = read_worst_2(file_path)
    rev_growth_worst2 = read_rev_growth(file_path)
    # Display department information using read_worst_2
    with col2:
        st.subheader(worst_2_result['department_name'])
        with st.expander(f'Detailed Information - {worst_2_result["department_name"]}'):
            st.subheader('Division Forecast')
            st.image(worst_2_result['graph_path'])
            st.write('------------------------------')
            st.subheader('Insight')
            st.write(worst_2_result['insight']['low_kpi_achieved_ratio'])
            st.write('------------------------------')
            st.subheader('Low Kpi Causes Possibility')
            st.write(worst_2_result['insight']['possible_causes'])
            st.write('--------------------------')
            st.subheader('Revenue Growth')
            st.text('Predicted Next Month Revenue Growth')
            st.write(rev_growth_worst1['E-commerce Sales']['start'],"-",rev_growth_worst1['E-commerce Sales']['end'],"=",rev_growth_worst1['E-commerce Sales']['change'])
            st.text('Revenue Change in Value')
            st.write(rev_growth_worst1['E-commerce Sales']['percentage'])             
            st.write('--------------------------')
            st.subheader('Recommendation for Improvement')
            for attribute in worst_2_result['recommendations']:
                st.write(attribute)
            # for attribute in worst_1_result['insight']['possible_causes']:
            #     st.write(attribute)




    # Use a raw string for the file path in the error message





        # Rest of your code using best_1_data...

        
        # Do something with the loaded data
        # For example, display it using st.write

        # st.write(best_1_data)
        


# def main_DepartmentHighlights():

#     try:

#         selected_company = st.session_state["selected_company"]

#         json_file = 'resources/best_1.json'
#         with open(json_file, "r") as f:
#             company_data = json.load(f)

#         folder_path = company_data[selected_company]["folder_path"]
#         xlsx_path = company_data[selected_company]["xlsx_path"]


#         #docx_path = company_data[selected_company]["docx_path"]

#         financial_metrics = pd.read_csv(f"./{folder_path}/financial_metrics.csv")
#         financial_metrics_insights = pd.read_csv(f"./{folder_path}/financial_metrics_insights.csv")

#         #company_info_file = open("./data/company_information.json", "r")
#         #company_info = ""

#         #for line in company_info_file.readlines():
#         #    company_info = company_info + line

#         #print("name", company_info)
#         #company_info = json.loads(company_info)
#         company_name = selected_company
#         report_period = folder_path.split("/")[2]

#         #st.info(f"Analyze financial performance per {report_period} compared with prior periods, relative to top competitors, and compared to the industry. Insights are provided for key financial metrics.")
#         st.write(f"Analyze financial performance per {report_period} compared with prior periods, relative to top competitors, and compared to the industry. Insights are provided for key financial metrics.")

#         row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns(
#             (0.1, 1, 0.1, 1, 0.1)
#         )

#         with row1_1:
#             revenue_growth = financial_metrics[financial_metrics["metric_name"] == "Revenue Growth"]
#             st.subheader("Revenue Growth", help="Revenue growth measures the year-over-year percentage change in operating revenue. It is calculated as follows: (Revenue this period - Revenue last period) / Revenue last period.")
#             st.metric("Revenue Growth", millify(revenue_growth[revenue_growth["company"] == company_name]["value"].values[0], precision=2) + "%")

#             fig = px.line(
#                 revenue_growth,
#                 x="period",
#                 y="value",
#                 title=f"Comparison of Revenue Growth with Industry Benchmark",
#                 color="company",
#             )

#             fig.update_yaxes(title_text="Period")
#             fig.update_yaxes(title_text="% Revenue Growth")

#             st.plotly_chart(fig, theme="streamlit", use_container_width=True)
#             revenue_growth_insights = financial_metrics_insights[financial_metrics_insights["metric_name"] == "Revenue Growth"]["general"].values[0]
#             st.markdown(
#                 revenue_growth_insights[revenue_growth_insights.find("1. "):revenue_growth_insights.find("2. ")]
#             )
#             st.markdown(
#                 revenue_growth_insights[revenue_growth_insights.find("2. ")-1:]
#             )


#         with row1_2:
#             cogs_perc = financial_metrics[financial_metrics["metric_name"] == "Cost of Goods Sold (%)"]
#             st.subheader("% Cost of Goods Sold", help="Cost of Goods Sold (COGS) is comprised of expenses directly related to the provision of the products or services reflected in revenues. % COGS is calculated as follows: Cost of Goods Sold / Revenue")
#             st.metric("% Cost of Goods Sold", millify(cogs_perc[cogs_perc["company"] == company_name]["value"].values[0], precision=2) + "%")

#             fig = px.line(
#                 cogs_perc,
#                 x="period",
#                 y="value",
#                 title=f"Comparison of % Cost of Goods Sold (COGS) with Industry Benchmark",
#                 color="company",
#             )

#             fig.update_yaxes(title_text="Period")
#             fig.update_yaxes(title_text="% COGS")

#             st.plotly_chart(fig, theme="streamlit", use_container_width=True)
#             cogs_perc_insights = financial_metrics_insights[financial_metrics_insights["metric_name"] == "% Cost of Goods Sold"]["general"].values[0]
#             st.markdown(
#                 cogs_perc_insights[cogs_perc_insights.find("1. "):cogs_perc_insights.find("2. ")]
#             )
#             st.markdown(
#                 cogs_perc_insights[cogs_perc_insights.find("2. "):]
#             )

#         add_vertical_space()

#         row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns(
#             (0.1, 1, 0.1, 1, 0.1)
#         )

#         with row2_1:
#             operating_income_margin = financial_metrics[financial_metrics["metric_name"] == "Operating Income Margin (%)"]
#             st.subheader("% Operating Income Margin", help="Operating Income Margin measures the profitability of a company's ongoing operations and excludes non-operating income and expense. It is calculated as follows: (Revenue - Total Operating Expenses) / Revenue")
#             st.metric("Operating Income Margin", millify(operating_income_margin[operating_income_margin["company"] == company_name]["value"].values[0], precision=2) + "%")

#             fig = px.line(
#                 operating_income_margin,
#                 x="period",
#                 y="value",
#                 title=f"Comparison of % Operating Income Margin with Industry Benchmark",
#                 color="company",
#             )

#             fig.update_yaxes(title_text="Period")
#             fig.update_yaxes(title_text="% Operating Income Margin")

#             st.plotly_chart(fig, theme="streamlit", use_container_width=True)
#             operating_income_margin_insights = financial_metrics_insights[financial_metrics_insights["metric_name"] == "% Operating Income Margin"]["general"].values[0]
#             st.markdown(
#                 operating_income_margin_insights[operating_income_margin_insights.find("1. "):operating_income_margin_insights.find("2. ")]
#             )
#             st.markdown(
#                 operating_income_margin_insights[operating_income_margin_insights.find("2. "):]
#             )


#         with row2_2:
#             days_inventory = financial_metrics[financial_metrics["metric_name"] == "Days Inventory"]
#             st.subheader("Days Inventory", help="Operating Income Margin measures the profitability of a company's ongoing operations and excludes non-operating income and expense. It is calculated as follows: (Revenue - Total Operating Expenses) / Revenue")
#             st.metric("Days Inventory", millify(days_inventory[days_inventory["company"] == company_name]["value"].values[0], precision=2) + " day(s)")
#             fig = px.line(
#                 days_inventory,
#                 x="period",
#                 y="value",
#                 title=f"Comparison of Days Inventory with Industry Benchmark",
#                 color="company",
#             )

#             fig.update_yaxes(title_text="Period")
#             fig.update_yaxes(title_text="Days Inventory")

#             st.plotly_chart(fig, theme="streamlit", use_container_width=True)
#             days_inventory_insights = financial_metrics_insights[financial_metrics_insights["metric_name"] == "Days Inventory"]["general"].values[0]
#             st.markdown(
#                 days_inventory_insights[days_inventory_insights.find("1. "):days_inventory_insights.find("2. ")]
#             )
#             st.markdown(
#                 days_inventory_insights[days_inventory_insights.find("2. "):]
#             )

#         add_vertical_space()

#         row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
#             (0.1, 1, 0.1, 1, 0.1)
#         )

#         with row3_1:
#             fixed_asset_utilization = financial_metrics[financial_metrics["metric_name"] == "Fixed Asset Utilization"]
#             st.subheader("Fixed Asset Utilization", help="Fixed asset utilization measures how many dollars of revenue are generated for each dollar invested in net PP&E. It is calculated as follows: Revenue / Net Property, Plant & Equipment")
#             st.metric("Fixed Asset Utilization", millify(fixed_asset_utilization[fixed_asset_utilization["company"] == company_name]["value"].values[0], precision=2))

#             fig = px.line(
#                 fixed_asset_utilization,
#                 x="period",
#                 y="value",
#                 title=f"Comparison of % Fixed Asset Utilization with Industry Benchmark",
#                 color="company",
#             )

#             fig.update_yaxes(title_text="Period")
#             fig.update_yaxes(title_text="Fixed Asset Utilization")

#             st.plotly_chart(fig, theme="streamlit", use_container_width=True)
#             fixed_asset_utilization_insights = financial_metrics_insights[financial_metrics_insights["metric_name"] == "Fixed Asset Utilization"]["general"].values[0]
#             st.markdown(
#                 fixed_asset_utilization_insights[fixed_asset_utilization_insights.find("1. "):fixed_asset_utilization_insights.find("2. ")]
#             )
#             st.markdown(
#                 fixed_asset_utilization_insights[fixed_asset_utilization_insights.find("2. "):]
#             )
#     except Exception as e:
#         print(e)
#         st.error('Please select company name ', icon="🚨")


# if __name__ == "__main__":
#     main()
