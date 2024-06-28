# DICCIONARIO DE VARIABLES

## Índice

[Crops & Livestock Emissions](#emissions1)

[Trade Matrix](#trade-matrix)

[Table 58 - population](#tabla-58)

---------------------------------------------------------------------------------------------------------------------------

## Crops & Livestock Emissions <a name="emissions1"></a>

 0   Area Code                                          252987 non-null  int64  
 1   Area                                               252987 non-null  object 
 2   Year                                               252987 non-null  int64  
 3   Item Code                                          252987 non-null  int64  
 4   Item                                               252987 non-null  object 
 5   Stocks_(An)                                        252987 non-null  float64 --> Animal stocks
 6   crops_or_livestock                                 252987 non-null  object --> Category, crops or livestock item
 7   Burning_crop_residues_(Emissions_CH4_CO2eq)_(kt)   252987 non-null  float64
 8   Burning_crop_residues_(Emissions_N2O_CO2eq)_(kt)   252987 non-null  float64
 9   Crop_residues_(Emissions_N2O_CO2eq)_(kt)           252987 non-null  float64
 10  Crops_total_(Emissions_CH4_CO2eq)_(kt)             252987 non-null  float64
 11  Crops_total_(Emissions_N2O_CO2eq)_(kt)             252987 non-null  float64
 12  Rice_cultivation_(Emissions_CH4_CO2eq)_(kt)        252987 non-null  float64
 13  Synthetic_fertilizers_(Emissions_N2O_CO2eq)_(kt)   252987 non-null  float64
 14  Emissions_(N2O_CO2eq)_(Manure_applied)_(kt)        252987 non-null  float64 ------> ALL THESE: transformed emissiones from each gas to CO2eq
 15  Enteric_fermentation_(Emissions_CH4_CO2eq)_(kt)    252987 non-null  float64
 16  Livestock_total_(Emissions_CH4_CO2eq)_(kt)         252987 non-null  float64
 17  Livestock_total_(Emissions_N2O_CO2eq)_(kt)         252987 non-null  float64
 18  Manure_left_on_pasture_(Emissions_N2O_CO2eq)_(kt)  252987 non-null  float64
 19  Manure_management_(Emissions_CH4_CO2eq)_(kt)       252987 non-null  float64
 20  Manure_management_(Emissions_N2O_CO2eq)_(kt)       252987 non-null  float64
 21  production_TOTAL_(emissions_CO2eq)_(kt)            252987 non-null  float64  ---> TOTAL EMISSIONS in CO2eq per item for "production/managment"

## Trade Matrix <a name="trade-matrix"></a>

Value_tons          cantidad de alimento/item en toneladas

Reporter Country Code       Codigo de pais IMPORTADOR

Partner Country Code        Codigo de pais EXPORTADOR

distance_in_km          Distancia en km entre las capitales de ambos paises

same_continent          Boolean: 1 si paises dentro del mismo continente, else 0

share_border            Boolean: 1 si paises comparten frontera, else 0

any_island_or_missing       Boolean: 1 si pais es isla o no se encuentra en Geopandas, else 0

transportation_method       land, water, air. Metodo de transporte de alimentos inferido.

kgCO2eq_tkm         kilogramos de CO2eq por tonelada por kilometro. Expresado en "food miles".

food_miles          Value_tons*distance_in_km

## Table 58 - population <a name="tabla-58"></a>

pop_tot         poblacion total (en miles)

pop_male        poblacion hombres (en miles)

pop_fem         poblacion mujeres (en miles)

pop_rural       poblacion rural (en miles)
