import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# Load your data
df = pd.read_csv(r'..\data\continent_trade_matrix_CO2eq.csv')
#Drop antarctica
df = df[df['target']!= 'antarctica']
df = df[df['source']!= 'antarctica']

# Define colors for targets
target_colors = {
    'europe': '#add8e6',        # Light Blue
    'africa': '#90ee90',        # Light Green
    'asia': '#ffb6c1',          # Light Pink
    'north america': '#fffacd', # Lemon Chiffon
    'oceania': '#ffcccb',       # Light Coral
    'south america': '#e6e6fa',
}

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the Dash app
app.layout = html.Div([
    dcc.Graph(id='sankey-graph'),
    dcc.Slider(
        id='year-slider',
        min=df['year'].min(),
        max=df['year'].max(),
        value=df['year'].min(),
        marks={str(year): str(year) for year in df['year'].unique()},
        step=None
    )
])

# Callback to update the Sankey diagram based on the selected year
@app.callback(
    Output('sankey-graph', 'figure'),
    [Input('year-slider', 'value')]
)
def update_sankey(selected_year):
    # Filter data for the selected year
    filtered_df = df[df['year'] == selected_year]

    # Prepare data for the Sankey diagram
    nodes = list(set(filtered_df['source']).union(set(filtered_df['target'])))
    node_indices = {node: i for i, node in enumerate(nodes)}

    sankey_data = {
        'type': 'sankey',
        'node': {
            'pad': 15,
            'thickness': 20,
            'line': {'color': 'black', 'width': 0.5},
            'label': nodes,
            'color': [target_colors[target] for target in nodes],
        },
        'link': {
            'source': [node_indices[source] for source in filtered_df['source']],
            'target': [node_indices[target] for target in filtered_df['target']],
            'value': filtered_df['value'],
            'color': [target_colors[target] for target in filtered_df['target']],
        }
    }

    # Create the figure
    fig = go.Figure(data=[sankey_data])
    fig.update_layout(title_text=f'Kg of CO2eq Produced by Food Trade between Continents in {selected_year}', font_size=12)

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
