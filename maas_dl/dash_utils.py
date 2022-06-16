import os
import json
import pandas as pd
import re
import statistics
import numpy as np
import flask
import base64

from datetime import datetime as dt
from os.path import join, exists, isfile
from tempfile import TemporaryFile
from scipy import stats
from os import environ, path

# Dash libraries
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input
from dash.dependencies import Output
import plotly.graph_objs as go
import plotly.express as px


# Helper function
def alphanum_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

INSTANCE_PATH = os.getenv('INSTANCE_PATH') if os.getenv('INSTANCE_PATH') is not None else None
# Dash app functions

image_filename = 'activeeon-logo-orange-blue-trasparent.png'

def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

# Main Dash/plotly layout page
def serve_layout():
    tab_style = {
        "background": "#E6E6FA",
        'text-transform': 'uppercase',
        'color': 'black',
        'border': 'black',
        'font-size': '13px',
        'font-weight': 800,
        'align-items': 'center',
        'justify-content': 'center',
        # 'border-radius': '20px',
        #'padding': '20px'
    }

    tab_selected_style = {
        "background": "#1C84C6",
        'text-transform': 'uppercase',
        'color': 'white',
        'border': 'black',
        'font-size': '15px',
        'font-weight': 800,
        'align-items': 'center',
        'justify-content': 'center',
        # 'border-radius': '20px',
        #'padding': '20px'
    }
    header_style = {
        'background-color': "#F3F3F4",
        'height': '80px',
        'border-bottom-width': '100%',
        #'border-bottom': '3px #f47930',
        'border-bottom-width': '100%',
        #'padding-top': '40px',
    }
    header_text_style = {
        'position': 'absolute',
        'left': '50%',
        'transform': 'translate(-50%, -50%)',
        'text-align': 'center',
        'top': '40px',
        #'font-size': '15px',
    }
    image_container_style = {
        'display': 'inline-block',
        'position': 'absolute',
        'left': '20px',
        'top': '20px'
    } 
    line_separator_style = {
        'height': '3px',
        'background': '#f47930',
        'width':'100%', 
        'display': 'table'
    } 
    # Tabs of the Dash app page
    #url = "https://news.microsoft.com/wptabs-example-content-content/uploads/prod/sites/113/2019/02/activeeon-logo-orange-blue-trasparent.png"
    return html.Div([
        html.Div(children=[
            html.Div(children=[html.Img(src=b64_image(image_filename),style={'height': "40px"})], style=image_container_style),
            html.Div(children=[
                html.H1(children='Machine Learning Model as a Service', style={'font-weight': "400",'font-size': '30px'})], style=header_text_style)], style=header_style),
        html.Div(style=line_separator_style),
        html.Div(children=[
            dcc.Tabs(id='tabs-example', value='tab-0', children=[
                dcc.Tab(label='Audit and Traceability', value='tab-0', style=tab_style,
                        selected_style=tab_selected_style),
            ]),
        html.Div(id='tabs-example-content')
        ])
    ])

# Function called by the main flask app to render the dash layout
def init_dashboard(server):
    """Create a Plotly Dash dashboard."""

    app = dash.Dash(
        server=server,
        routes_pathname_prefix="/dashapp/",
        external_stylesheets=[
            'https://codepen.io/chriddyp/pen/bWLwgP.css'
        ],
        # assets_external_path='/assets/style.css'
    )

    app.layout = serve_layout

    # All callback functions that are executed once any event takes place in the dash app components
    @app.callback(Output('tabs-example-content', 'children'), Input('tabs-example', 'value'))
    def render_content(tab):
        if tab == 'tab-0':
            return html.Div([
                dcc.Interval(
                    id='interval-component',
                    # interval=5 * 1000,  # in milliseconds
                    n_intervals=0
                ),
                html.Br(),
                html.Br(),
                html.A(html.Button("Go to Swagger UI", className='three columns'),
                       href='/api/ui/#/default', target="_blank"),
                html.Br(),
                html.Br(),
                html.Label(["Configuration Details"], style={'font-weight': 'bold', "text-align": "center", "font-size": '3rem'}),
                html.Br(),
                html.Div(id='config-table'),
                html.Br(),
                html.Br(),
                html.Label(["Traceability Details"], style={'font-weight': 'bold', "text-align": "center", "font-size": '3rem'}),
                html.Br(),
                html.Div(id='trace-table')
                #]))
            ])
        elif tab == 'tab-1':
            return html.Div([
                dcc.Interval(
                    id='interval-component',
                    # interval=60 * 1000,  # in milliseconds
                    n_intervals=0
                ),
            ])

        elif tab == 'tab-3':
            return html.Div([
                dcc.Interval(
                    id='interval-component',
                    # interval=10 * 1000,  # in milliseconds
                    n_intervals=0
                ),
            ])


    @app.callback(Output('config-table', 'children'),
                  Input('interval-component', 'n_intervals'))
    def config_table(n):
        path = INSTANCE_PATH + "/config.json"
        with open(path) as f:
            config = json.load(f)
        dataframe_config = pd.DataFrame.from_records([config])
        columns = dataframe_config.columns.tolist()
        columnss = list()
        for i in range(0, len(columns)):
            columnss.append({"name": str(columns[i]), "id": str(columns[i]), })
        data = dataframe_config.to_dict('rows')
        d = [{key: str(value) for key, value in data[0].items()}]
        return dash_table.DataTable(data=d, columns=columnss,
                                    style_cell={'padding': '20px', 'border': '1px solid #E6E6FA', 'textAlign': 'center'},
                                    style_header={
                                        'backgroundColor': '#D6D6D6',
                                        'fontWeight': 'bold'
                                    })

    @app.callback(Output('trace-table', 'children'),
                  Input('interval-component', 'n_intervals'))
    def trace_table(n):
        filename = INSTANCE_PATH + "/trace.txt"
        header = ["Date Time", "Token", "Traceability information"]
        if exists(filename) and isfile(filename):
            dataframe_trace = pd.read_csv(filename, sep="|", names=header, engine='python')
            dataframe_trace.fillna('', inplace=True)
        else:
            dataframe_trace = pd.DataFrame(np.array([['', '', '[WARN] Trace file is empty']]), columns=header)

        columns = dataframe_trace.columns.tolist()
        columnss = list()
        for i in range(0, len(columns)):
            columnss.append({"name": str(columns[i]), "id": str(columns[i]), })
        data = dataframe_trace.to_dict('rows')
        l = list()
        for d in data:
            d = {key: str(value) for key, value in d.items()}
            l.append(d)
        return dash_table.DataTable(data=l, columns=columnss,
                                    style_cell={'padding': '20px', 'border': '1px solid #E6E6FA', 'textAlign': 'center'},
                                    style_header={
                                        'backgroundColor': '#D6D6D6',
                                        'fontWeight': 'bold'
                                    })