# import streamlit as st
# import os
# import json
# from datetime import datetime
# import shutil
# from dotenv import load_dotenv


# def main():
#     st.title("The watsonx.ai Banking Solution Gallery")
#     st.write("The solution gallery is a collection of resources and examples designed to guide users in effectively utilizing watsonx.ai, an artificial intelligence (AI) platform developed by IBM, for internal IBM workloads and offers a range of capabilities and tools to support banking use cases.")
#     st.write("The purpose of the solution gallery is to provide a practical and hands-on approach to understanding and implementing watsonx.ai within the context of internal IBM workloads. The solution is tailored to meet expectation from banking perspective in certain areas which are Business, Legal and Procurement")


# if __name__ == "__main__":
#     main()

#Flask mainpage
import json
from app.utils.watsonx_hr_fin import WatsonXModel
from app.utils.kpi_predictor import *

wx  =  WatsonXModel()

#model untuk input file
#Tunggu Model dita untuk load data dan store data dari docs to json
#logic to convert from original output ke json output

# Define a function to add department data
# def add_department(data, name, recommended_job_responsibilities, reasoning):
#     department_data = {
#         "name": name,
#         "recommendedJobResponsibilities": recommended_job_responsibilities,
#         "reasoning": reasoning
#     }
#     data["departments"].append(department_data)

# # Define a function to add recommendations
# def add_recommendation(data, recommendation):
#     data["recommendation"].append(recommendation)

# # Define a function to add a note
# def add_note(data, note):
#     data["note"] = note

# def to_json(data, name, recommended_job_responsibilities, reasoning, recommendation, note):
#     # Initialize an empty dictionary for the JSON data
#     data = {
#         "departments": [],
#         "recommendation": [],
#         "note": ""
#     }
#     add_department(data, name, recommended_job_responsibilities, reasoning)
#     add_recommendation(data,recommendation)
#     add_note(data, note)
    
def genai_hr_initiate(dep_to_test):
    load_goals  = './user_files/job_resp/goal.json'
    with open(load_goals, "r") as json_file:
        com_goals= json.load(json_file)
    load_data =  f'./user_files/job_resp/{dep_to_test}/{dep_to_test}_job_output.json'
    with open(load_data, "r") as json_file:
        input_data = json.load(json_file)
    if dep_to_test == 'finance':
        run_model = wx.finance_optimized_org(load_goals, input_data)
    elif dep_to_test=='operation':
        run_model = wx.operation_optimized_org(load_goals, input_data)
    elif dep_to_test=='sales_marketing':
        run_model = wx.sales_marketing_optimized_org(load_goals, input_data)
    elif dep_to_test=='sales_platform':
        run_model = wx.sales_platform_optimized_org(load_goals, input_data)
    elif dep_to_test=='sales_support':
        run_model = wx.sales_support_optimized_org(load_goals, input_data)
    elif dep_to_test=='tech':
        run_model = wx.tech_optimized_org(load_goals, input_data)
    else:
        print('please choose right parameters')
    #logic to make it can be add to json: logic start
    #Logic end
    model_to_json = wx.json_converter(load_goals, run_model)
    json_result = json.loads(model_to_json)
    output_file_path = f"./user_files/final_result/{dep_to_test}_ideal.json"

    # Write the JSON data to the output file
    with open(output_file_path, "w") as output_file:
        json.dump(json_result, output_file, indent=4)

    #Create skill mapping
    dep_skill = wx.skill_mapping(input_data, load_goals)
    skill_result = json.loads(dep_skill)
    output_file_path_skill = f"./user_files/skill_map/{dep_to_test}_skillset.json"

    with open(output_file_path_skill, "w") as output_file:
        json.dump(skill_result, output_file, indent=4)
    
    return output_file_path

def load_avail_data(dep_to_run, div_name):
    #Load new Responsibility
    with open(f'./user_files/final_result/{dep_to_run}_ideal.json', 'r') as f:
        data = json.load(f)

    # Store each result in separate variables
    departments = []
    recommendations = data['recommendation']
    # note = data['note']


    # Extract department names and recommended job responsibilities
    for department in data["departments"]:
        if department["name"] == div_name:
            recommend_job = department["recommendedJobResponsibilities"]
            reasoning = department['reasoning']
    # Load Skill Map
    
    with open(f'./user_files/skill_map/{dep_to_run}_skillmap.json', 'r') as f:
        skill_map_json = json.load(f)
    skill_map = skill_map_json[div_name]
    try:
        return div_name, recommend_job, reasoning, recommendations, skill_map
    except:
        return recommendations

def load_data(dep_to_test, div_name, generate_new = 'no'):
    dep_to_test = dep_to_test.lower().replace(" ", "_")
    if generate_new == 'Yes':
        result = genai_hr_initiate(dep_to_test=dep_to_test)
        return result
    else:
        try:
            div_name, recommend_job, reasoning, recommendations, skill_map = load_avail_data(dep_to_run=dep_to_test, div_name = div_name)
            return div_name, recommend_job, reasoning, recommendations, skill_map
        except:
            recommendations = load_avail_data(dep_to_run=dep_to_test, div_name = div_name)
            return recommendations

def load_asis_data(dep_to_test, div_name):
    dep_to_run = dep_to_test.lower().replace(" ", "_")
    if 'sale' in dep_to_run:
        with open(f'./user_files/job_resp/sales/{dep_to_run}_jobs_output.json', 'r') as f:
            data = json.load(f)
    else:
        with open(f'./user_files/job_resp/{dep_to_run}/{dep_to_run}_jobs_output.json', 'r') as f:
            data = json.load(f)
    department_data = next((job for job in data if job['department'] == div_name), None)
    print(f'department data: {department_data}')
    return department_data['JobResponsibilities']

# #Running in Words
# div_name, recommend_job, reasoning, recommendations, note, skill_map =  load_data("finance","Legal")
# print(f"department:{div_name}")
# print(f"rec:{recommend_job}")
# print(f"note:{note}")
# print(f"skill map:{skill_map}")

#Running for plot
# plot_data(div_name)

#------------------------------------------------------------------------------------
#                                   FLASK APP
#------------------------------------------------------------------------------------
from flask import Flask, render_template, request, session, redirect, url_for
import re

app = Flask(__name__)
# Define the initial route to render the template

from flask import Flask, render_template, request

app = Flask(__name__)
app.secret_key = 'IBMECOMMERCE123132'

# Define a function to clean up the input text
def clean_text(text):
    # Remove leading/trailing whitespace and newlines
    text = text.strip()
    # Replace periods with full stops
    text = text.replace(".", ". ")
    # Capitalize the first letter of each sentence
    return text.capitalize()

# Process Category Form

@app.route('/process_category', methods=['POST'], endpoint="process_category_form")
def process_category_form():
    selected_category = request.form.get('category')
    session['selected_category'] = selected_category
    return redirect(url_for('home'))

    # Add your logic for category form processing
    print(f'\n\n\n\Category Name:{selected_category}\n\n\n\n\n\n\n')
    return render_template('index.html', selected_category=selected_category)

# Process Division Form
@app.route('/process_division', methods=['POST'], endpoint="process_division_form")
def process_division_form():
    selected_division = request.form.get('division')
    session['selected_division'] = selected_division

    try:
        div_name, recommend_job, reasoning, recommendations, skill_map =  load_data(session['selected_category'], session['selected_division'])
        skill_map = [f"{i+1}. {category}" for i, category in enumerate(skill_map)]
    except:
        recommendations = load_data(session['selected_category'], session['selected_division'])
        # Set other variables to None or appropriate default values
        div_name = None
        recommend_job = None
        reasoning = None
        skill_map = None

    asis_job = load_asis_data(session['selected_category'], session['selected_division'])
    print(session['selected_category'])
    

    graphJSON = plot_data(selected_division)
    
    return render_template('index.html', selected_division=selected_division, recommend_job=recommend_job, skill_map=skill_map, recommendations=recommendations,reasoning=reasoning, graphJSON = graphJSON, asis_job= asis_job)

@app.route('/reset_session', methods=['POST'])
def reset_session():
    session.pop('selected_category', None)
    session.pop('selected_division', None)
    return redirect(url_for('home'))

# Initial route to render the template
@app.route('/home')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

