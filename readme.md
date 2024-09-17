<img src="https://img.shields.io/badge/Python-white?logo=Python" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/pandas-white?logo=pandas&logoColor=250458" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/NumPy-white?logo=numpy&logoColor=013243" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/Geopandas-white?logo=geopandas" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/Plotly-white?logo=plotly&logoColor=636efa" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/Seaborn-white?logo=python" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/MLforecast-white?logo=python" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/Scikit--learn-white?logo=scikit-learn" style="height: 25px; width: auto;">

# Food Carbon Footprint Analysis

This document provides a comprehensive report on a data analysis project focused on the carbon footprint of food production and consumption. It covers the project's scope, methods, and findings, with an emphasis on data insights from the Food and Agriculture Organization (FAO) and various analytical approaches used.

## Objective

The objective of this project is to analyze the carbon footprint associated with global food production and consumption. Utilizing data from the FAO, the project aims to uncover insights into food emissions patterns and predict future trends through advanced data science techniques.

## Project Overview

- **Data Cleaning and Processing**: The dataset's size and historical scope presented significant challenges in ensuring consistency and completeness.

- **Transport Emissions Analysis**: This involved analyzing a large trading matrix to assess the environmental impact of food transportation.

- **Food Item Classification**: Hierarchical clustering was employed to classify food items based on emissions, consumption, and production patterns, revealing similarities and groupings among different food categories.

- **Forecasting Food Emissions**: Machine learning techniques, including the MLforecast library, were used to project future food emissions, identifying potential trends and intervention points.

- **Country Classification**: K-means clustering was applied to categorize countries based on emissions profiles, considering factors such as crop production and livestock. This analysis highlighted global patterns and similarities in food-related emissions.

## Project Structure

- **`00_raw_tables_explore/`**: Raw and preliminary data exploration.
- **`01_food_trading/`**: Notebooks and data related to the analysis of global food trading over the years.
- **`02_food_emissions/`**: Contains files for analyzing food emissions data.
- **`03_food_dataset_compiling/`**: Includes notebooks for compiling and preparing the food dataset.
- **`04_country_emissions/`**: Contains files related to analyzing and classifying country emissions profiles.


## Key Findings and Insights

### Food Trading Emissions

The analysis of food trading emissions over the years reveals several key patterns. Asia and Europe stand out as the largest importers, while North America and South America are the leading exporters. The trade flow generating the most emissions is between North America and Asia, underscoring the environmental cost of long-distance trade between these continents. However, imports into Europe from various regions around the world cumulatively contribute the most to global food transportation emissions, making Europe a major player in global food-related CO2 emissions.

<p align="center">
    <b>2022 Food Trading Associated Emissions Between Continents</b>
</p>
<p align="center">
    <img src="./readme_img/sankey.png" alt="Sankey Diagram of Food Trading Emissions" style="width: 70%;">
    <br>
    <a href="https://plotly.com/~ezemriv/7/" target="_blank">View Interactive 1986-2022 Version on Plotly</a>
</p>

A further analysis of food items' travel patterns shows that, overall, crops travel more extensively than livestock. Among them, **soya beans** stand out as the most transported food item according to the FAO data, generating **over 20,000 kilotonnes of CO2-equivalent GHG emissions per year**. This highlights the substantial environmental impact of transporting high-demand crops.

A connections map of soya trading highlights that the majority of these trade flows are directed towards Europe, further emphasizing the continent's role in driving global food transport emissions.

<p align="center">
    <b>Soya Beans Top Contaminating Trades (2021)</b>
<p align="center">
    <img src="./readme_img/map_soya.png" alt="Soya Beans CO2 Emissions" style="width: 70%;">
    <br>
    <a href="https://plotly.com/~ezemriv/18/" target="_blank">View Interactive Version on Plotly</a>
</p>

### Food Production Emissions


# Appendix

## FAO Data

The FAO dataset utilized in this project is one of the most extensive sources of global agricultural and food-related data. Key aspects include:

- Coverage of over 245 countries and territories, providing a global perspective.
- Data on more than 200 primary crops and livestock products, offering detailed insights into the agricultural sector.
- Inclusion of production, trade, food balance sheets, and emissions data, enabling multifaceted analysis.
- Historical depth extending back to 1961 for many indicators, supporting long-term trend analysis and robust forecasting.

## Data Preparation

### Food Trading

The food trading matrix from the dataset was extensive, requiring careful handling. Due to its size, I utilized Google Colab to process the data in chunks. Each row of the matrix represented a transaction of a food item between a pair of countries, including the year and transaction value.

- **Country Name Curation**: Country names were standardized to match those used in GeoPandas for geographic analyses.
- **Distance Calculation**: 
  - **International Distance**: Distances between country capitals were computed as a proxy for the distance between countries.
  - **Internal Distance**: For transactions where the origin and destination were the same country, I calculated the distance from the country centroid to various points along its borders using GeoPandas (Notebook 06).
  
  <p align="center">
    <img src="./readme_img/centroid.jpg" width="300" />
  </p>
  
  - **Geographic Classification**: Identified which countries are islands or surrounded by water to account for transportation constraints.
- **Transportation Method Assumptions**: Assumptions were made regarding transportation methods (air, land, or water) for each transaction, influencing emissions calculations. Detailed assumptions are documented in Notebook 05.

### Food Emissions

The food emissions data were processed in folders 02 and 03. The goal was to create a comprehensive dataset with relevant emissions features. The key steps involved in preparing the data included:

- **Data Processing**: Raw data containing information about various food emissions were cleaned and transformed to create a consolidated dataset.
- **Representative Emissions Features**: The final dataset includes several emission-related features, such as:
  - `"Burning_crop_residues_(Emissions_N2O)_(kt)"`
  - `"Crops_total_(Emissions_CH4)_(kt)"`
  - `"Rice_cultivation_(Emissions_CH4)_(kt)"`

- **CO2 Equivalent Transformation**: Emissions data for CH4 and N2O were converted to CO2 equivalent emissions to standardize and simplify the analysis. This transformation allows for a more consistent comparison of different types of emissions.

- **Additional Features**: Besides emissions-related features, the dataset was augmented with non-emissions related features to provide a more comprehensive view of food production and consumption. Some additional features were:
  - `"Food supply quantity (tonnes)"`
  - `"Area harvested"`
  - `"Export Quantity"`

- **Trading Emissions Features**: Emissions calculated for trading were also added to the dataset for use in analysis and clustering.

Through this extensive processing, the dataset was consolidated into two main columns of emissions:
- `'production_TOTAL_(emissions_CO2eq)_(kt)'`
- `'transport_TOTAL_(emissions_CO2eq)_(kt)'`

These columns represent the total emissions from production and transportation, respectively, and are central to the subsequent analysis and clustering.

### Region and Country Emissions

For the analysis of regional and country-specific emissions, a total emissions table was utilized. This table included CO2 equivalent transformed emissions for various features. Key steps in preparing this data included:

- **Data Processing**: The total emissions table was processed to consolidate and clean data for accurate analysis.
- **Representative Emissions Features**: The dataset included CO2 equivalent emissions for several features, such as:
  - `'Crop Residues'`
  - `'Rice Cultivation'`
  - `'Enteric Fermentation'`
  - `'Manure Management'`
  - `'Forest fires'`

- **Data Merging and Curation**: The emissions data were merged and curated with additional FAO tables containing information on:
  - Population
  - GDP
  - GDP per capita
  - Country area

- **Total and Partial Emissions Columns**: The final total emissions column, which summed all components, was obtained alongside specific partial emissions columns.
  - **Partial Columns**:
    - `'Farm_gate_Emissions_(CO2eq)_(AR5)_kt'`
    - `'Land_Use_change_Emissions_(CO2eq)_(AR5)_kt'`
    - `'Pre-_and_Post-_Production_Emissions_(CO2eq)_(AR5)_kt'`
  - **Total Column**:
    - `'Agrifood_systems_Emissions_(CO2eq)_(AR5)_kt'`

## Practical Application: Web App for Food Origin and Environmental Impact Estimation

A practical application of this analysis is demonstrated in a Dockerized Flask web app, which predicts the origin of your food and estimates the environmental impact of its transportation based on the data analyzed in this repository.

The app allows users to input their location and food item, providing insights into:
- The probable country of origin for the food.
- The CO2 emissions generated from its transportation.

You can check out the web app and its repository here:

<a href="https://fao-co2-food-travel-emissions.onrender.com/" target="_blank">
    <img src="https://img.shields.io/badge/Go%20to%20App-blue" style="height: 20px; width: auto;">
</a>  <a href="https://github.com/ezemriv/Food-CO2-app" target="_blank">
    <img src="https://img.shields.io/badge/Go%20to%20Repo-black" style="height: 20px; width: auto;">
</a>