import datetime

import dash
from dash import dcc, html
import plotly
from dash.dependencies import Input, Output

# pip install pyorbital
from pyorbital.orbital import Orbital
satellite = Orbital('TERRA')
import plotly.graph_objects as go
import csv
import boto3
import pandas as pd


s3 = boto3.client('s3')


app = dash.Dash(__name__)
app.layout = html.Div(
    html.Div([
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000*60*60, # in milliseconds
            n_intervals=0
        )
    ])
)




# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    # obj = s3.get_object(Bucket='crwpl-prices', Key='prices.csv')
    df = pd.read_csv('s3://crwpl-prices/prices.csv')


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['dt'], y=df['prices'],
                        mode='lines',
                        name='lines'))

    return fig


if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0', port=8050)