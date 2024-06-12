# DICCIONARIO DE VARIABLES

(las vamos copiando aca y ya luego al final las podemos ordenar alfabeticamente)

**De Trade Matrix:**

Value_tons			cantidad de alimento/item en toneladas

Reporter Country Code		Codigo de pais IMPORTADOR

Partner Country Code		Codigo de pais EXPORTADOR

distance_in_km			Distancia en km entre las capitales de ambos paises

same_continent			Boolean: 1 si paises dentro del mismo continente, else 0

share_border			Boolean: 1 si paises comparten frontera, else 0

any_island_or_missing		Boolean: 1 si pais es isla o no se encuentra en Geopandas, else 0

transportation_method		land, water, air. Metodo de transporte de alimentos inferido.

kgCO2eq_tkm			kilogramos de CO2eq por tonelada por kilometro. Expresado en "food miles".

food_miles			Value_tons*distance_in_km

**De tabla 58:**

pop_tot         poblacion total (en miles)

pop_male        poblacion hombres (en miles)

pop_fem         poblacion mujeres (en miles)

pop_rural       poblacion rural (en miles)

