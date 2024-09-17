<img src="https://img.shields.io/badge/Python-white?logo=Python" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/pandas-white?logo=pandas&logoColor=250458" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/NumPy-white?logo=numpy&logoColor=013243" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/Geopandas-white?logo=geopandas" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/Plotly-white?logo=plotly&logoColor=636efa" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/Seaborn-white?logo=python" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/MLforecast-white?logo=python" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/Scikit--learn-white?logo=scikit-learn" style="height: 25px; width: auto;">

# Food Carbon Footprint Analysis

## Objective

This project aims to analyze the carbon footprint of food production and consumption using data from the Food and Agriculture Organization of the United Nations (FAO). By leveraging extensive datasets and applying advanced data science techniques, I seek to uncover insights into global food emissions patterns and predict future trends.

## Project Overview

### Data Cleaning and Processing

One of the most challenging aspects of this project was the extensive data cleaning and processing required. The FAO has been gathering information since 1945, resulting in a vast and complex dataset. This historical depth provides a unique opportunity for analysis but also presents significant challenges in terms of data consistency, completeness, and interpretation.

### Forecasting Food Emissions

Using machine learning techniques, particularly the MLforecast library for automated forecasting and lag feature creation, we projected food emissions into the future. This analysis helps in understanding potential trends and identifying areas where interventions might be most effective.

### Food Item Classification

We employed hierarchical clustering to classify food items based on their emissions, consumption, and production patterns. This classification provides insights into which food categories have similar environmental impacts and how they group together.

### Transport Emissions Analysis

A unique aspect of this project was the analysis of transport emissions for various food items. This required processing a massive trading matrix, offering insights into the often-overlooked environmental costs of food transportation across global supply chains.

### Regional Agri-food Systems Emissions

In the second part of the project, we analyzed emissions from agri-food systems for different world regions. This analysis forms the basis for our future emissions forecasts and provides a comprehensive view of how different parts of the world contribute to food-related carbon emissions.

### Country Classification

Using K-means clustering, we classified countries based on their emissions profiles, considering factors such as crop production, livestock, land use, and pre/post-production activities. This classification helps in identifying patterns and similarities in how different nations contribute to global food-related emissions.

## FAO Data Insights

The FAO data used in this project is one of the most comprehensive sources of global agricultural and food-related information available. Some key points about this data that enhance the value of our analysis:

- It covers over 245 countries and territories, providing a truly global perspective.
- The data spans more than 200 primary crops and livestock products, offering a detailed view of the agricultural sector.
- FAO's datasets include information on production, trade, food balance sheets, and emissions, allowing for multifaceted analysis.
- The historical depth of the data, going back to 1961 for many indicators, enables long-term trend analysis and robust forecasting.

By leveraging this rich dataset, our project provides unique insights into the complex relationship between global food systems and climate change, offering valuable information for policymakers, researchers, and industry stakeholders.