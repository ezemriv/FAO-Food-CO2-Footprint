import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# Load your data
df = pd.read_csv('continent_trade_matrix.csv')

# Define colors for targets
target_colors = {
    'europe': '#add8e6',        # Light Blue
    'africa': '#90ee90',        # Light Green
    'asia': '#ffb6c1',          # Light Pink
    'north america': '#fffacd', # Lemon Chiffon
    'oceania': '#ffcccb',       # Light Coral
    'south america': '#e6e6fa',
    'antarctica': '#cccccc'  # Lavender
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
    sources = list(set(filtered_df['source']))
    targets = list(set(filtered_df['target']))
    nodes = sources + targets
    node_indices = {node: i for i, node in enumerate(nodes)}

    # Duplicate nodes for left and right alignment
    duplicated_nodes = sources + targets
    duplicated_nodes += ['Right_' + target for target in targets]  # Add prefix to distinguish right nodes
    node_indices = {node: i for i, node in enumerate(duplicated_nodes)}

    # Assigning specific indices to sources, targets, and duplicated targets
    source_indices = [node_indices[source] for source in sources]
    target_indices = [node_indices[target] for target in targets]
    target_indices_right = [node_indices['Right_' + target] for target in targets]

    # Constructing the link data
    link_source = [source_indices[sources] for s in filtered_df['source']]
    link_target = [target_indices[targets] for t in filtered_df['target']]
    link_target_right = [target_indices_right[t] for t in filtered_df['target']]
    link_value = filtered_df['value']

    # Create the Sankey diagram
    sankey_data = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=duplicated_nodes,  # Use duplicated nodes
            color=[target_colors[node] for node in duplicated_nodes],
        ),
        link=dict(
            source=link_source,
            target=link_target + link_target_right,  # Concatenate targets and duplicated targets
            value=link_value,
            color=[target_colors[node] for node in targets] * 2,  # Duplicate colors for duplicated targets
        ),
    )

    # Create the figure
    fig = go.Figure(data=[sankey_data])
    fig.update_layout(title_text=f'Food Trade between Continents in {selected_year}', font_size=10)

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
