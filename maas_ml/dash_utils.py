import os
import json
import pandas as pd
import dash_utils
import re
import statistics


from datetime import datetime as dt
from os.path import join, exists, isfile
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

# Dash app functions

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
        'padding': '20px'
    }

    tab_selected_style = {
        "background": "#4d4dff",
        'text-transform': 'uppercase',
        'color': 'white',
        'border': 'black',
        'font-size': '15px',
        'font-weight': 800,
        'align-items': 'center',
        'justify-content': 'center',
        # 'border-radius': '20px',
        'padding': '20px'
    }
    
    # Tabs of the Dash app page
    return html.Div([
        dcc.Tabs(id='tabs-example', value='tab-1', children=[
            dcc.Tab(label='Dataset Analytics', value='tab-1', style=tab_style,
                    selected_style=tab_selected_style),
            dcc.Tab(label='Data Drift Analytics', value='tab-2', style=tab_style,
                    selected_style=tab_selected_style, disabled=True),
            dcc.Tab(label='Predictions Preview', value='tab-3', style=tab_style,
                    selected_style=tab_selected_style),
        ]),
        html.Div(id='tabs-example-content'),
        dcc.Store(id='feature-value'),
        dcc.Store(id='metric-value'),
        dcc.Store(id='click-data'),
        dcc.Store(id='name-value')
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
        directory_contents = os.listdir(os.environ['MODELS_PATH'])
        if tab == 'tab-1':
            return html.Div([
                dcc.Interval(
                    id='interval-component',
                    interval=60 * 1000,  # in milliseconds
                    n_intervals=0
                ),
                html.Div([
                    html.Div([
                        html.Label(["Select Model Name"], style={'font-weight': 'bold', "text-align": "center"}),

                        dcc.Dropdown(
                            id='model-dropdown',
                            options=[{'label': idx, 'value': idx} for idx in directory_contents],
                            value=directory_contents[0],
                            placeholder='Select an item...'
                        ),

                    ],
                        style={'width': '33.3%', 'display': 'inline-block', }),

                    html.Div([
                        html.Label(["Select Feature"], style={'font-weight': 'bold', "text-align": "center"}),
                        dcc.Dropdown(
                            id='feature-dropdown',
                            multi=True,
                            placeholder='Select an item...'
                        ),
                    ],
                        style={'width': '33.3%', 'display': 'inline-block', }),

                    html.Div([
                        html.Label(["Select Metric"], style={'font-weight': 'bold', "text-align": "center"}),
                        dcc.Dropdown(
                            id='metric-dropdown',
                            options=[
                                {'label': 'Mean', 'value': 'Mean'},
                                {'label': 'Min', 'value': 'Min'},
                                {'label': 'Max', 'value': 'Max'},
                                {'label': 'Standard Deviation', 'value': 'Standard Deviation'},
                                {'label': 'Variance', 'value': 'Variance'}
                            ],
                            value='Mean'
                        )
                    ],
                        style={'width': '33.3%', 'display': 'inline-block'}),

                ]),
                html.Div([
                    html.Div(id='title1'),
                    dcc.Graph(id='feature-metric-graphic', figure={
                        'layout': {'title': 'Feature analytics with respect to data version', 'paper_bgcolor': 'white',
                                   'plot_bgcolor': 'white'}}),
                ]),
                html.Div([
                    html.Div(id='title2'),
                    dcc.Graph(id='density-graphic', figure={
                        'layout': {
                            'title': 'Feature distribution comparison of the selected feature between all versions',
                            'paper_bgcolor': 'white', 'plot_bgcolor': 'white'}}),
                ])

            ])

        elif tab == 'tab-2':
            df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
            return html.Div([
                dash_table.DataTable(
                    id='datatable-interactivity',
                    columns=[
                        {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
                    ],
                    data=df.to_dict('records'),
                    editable=True,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    column_selectable="single",
                    row_selectable="multi",
                    row_deletable=True,
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current=0,
                    page_size=10,
                ),
                html.Div(id='datatable-interactivity-container'),
                html.Br(),
                html.Label(["Drift magnitude of the selected features"],
                           style={'font-weight': 'bold', "text-align": "center"}),
                html.Br(),
                html.Br(),
                dcc.Graph(id='percentage-graphic'),

            ])
        elif tab == 'tab-3':
            return html.Div([
                dcc.Interval(
                    id='interval-component',
                    interval=10 * 1000,  # in milliseconds
                    n_intervals=0
                ),
                html.Div([
                    html.Div([
                        html.Label(["Select Model Name"],
                                   style={'font-weight': 'bold', "text-align": "center"}),

                        dcc.Dropdown(
                            id='prediction-model-dropdown',
                            placeholder='Select an item...'
                        ),

                    ],
                        style={'width': '50%', 'display': 'inline-block', }),

                    html.Div([
                        html.Label(["Select Model Version"], style={'font-weight': 'bold', "text-align": "center"}),
                        dcc.Dropdown(
                            id='version-dropdown',
                            placeholder='Select an item...'
                        ),
                    ],
                        style={'width': '50%', 'display': 'inline-block', }),

                ]),
                html.Br(),
                html.Div(id='predictions-table')
            ])

    @app.callback(Output('name-value', 'data'),
                  Input('model-dropdown', 'value'), Input('interval-component', 'n_intervals'))
    def get_model(value, n):
        return value

    @app.callback(Output('model-dropdown', 'options'),
                  Input('interval-component', 'n_intervals'))
    def update_model_dropdown(n):
        directory_contents = os.listdir(os.environ['MODELS_PATH'])
        options = [{'label': idx, 'value': idx} for idx in directory_contents]
        return options

    @app.callback(Output('prediction-model-dropdown', 'options'),
                  Input('interval-component', 'n_intervals'))
    def update_model_dropdown(n):
        directory_contents = os.listdir(os.environ['MODELS_PATH'])
        options = [{'label': idx, 'value': idx} for idx in directory_contents]
        return options

    @app.callback(Output('feature-dropdown', 'options'),
                  Input('model-dropdown', 'value'), Input('interval-component', 'n_intervals'))
    def update_feature_dropdown(name, n):
        root = os.environ['MODELS_PATH'] + "/" + name + "/"
        features = list()
        for path, subdirs, files in os.walk(root):
            for file in files:
                if ("baseline_data_" in file):
                    dataframe = pd.read_csv(os.path.join(path, file))
                    features = dataframe.columns
                    break

        return [{'label': i, 'value': i} for i in features]

    @app.callback(Output('version-dropdown', 'options'), Output('version-dropdown', 'placeholder'),
                  Input('prediction-model-dropdown', 'value'), Input('interval-component', 'n_intervals'))
    def update_version_dropdown(name, n):
        root = os.environ['MODELS_PATH'] + "/" + name + "/"
        dirs = list()
        for path, subdirs, files in os.walk(root):
            for i in subdirs:
                folder = os.path.join(root, i)
                elements = os.listdir(folder)
                for file in elements:
                    if ("predictions_data_" in file):
                        dirs.append(i)
            versions = [{'label': "Version " + i, 'value': i} for i in dirs]
            versions_sorted = sorted(versions, key=lambda k: k['value'])
            break
        if (versions):
            return versions_sorted, versions_sorted[0]['label']
        else:
            return [], 'No predictions yet'

    @app.callback(Output('predictions-table', 'children'),
                  Input('prediction-model-dropdown', 'value'), Input('version-dropdown', 'value'),
                  Input('version-dropdown', 'placeholder'),
                  Input('interval-component', 'n_intervals'))
    def predictions_table(name, version, placeholder, n):
        if (placeholder == "No predictions yet"):
            return html.Label("")
        cols = list()
        path = os.environ['MODELS_PATH'] + "/" + name + "/" + version + "/" + "predictions_data_" + version + ".csv"
        predictions_dataframe = pd.read_csv(path, header=None)
        data = predictions_dataframe.to_dict('rows')
        for i in range(0, (len(predictions_dataframe.columns))):
            cols.append(str(i))
        columns = [{"name": i, "id": i, } for i in cols]
        return dash_table.DataTable(data=data, columns=columns, style_as_list_view=True,
                                    style_cell={'padding': '20px', 'border': '1px solid #E6E6FA'},
                                    style_header={
                                        'backgroundColor': '#E6E6FA',
                                        'fontWeight': 'bold'
                                    },
                                    style_data_conditional=[
                                        {
                                            'if': {
                                                'column_id': cols[len(cols) - 1],
                                            },
                                            'backgroundColor': '#E6E6FA',
                                            'fontWeight': 'bold'
                                        }
                                    ]
                                    )

    @app.callback(Output('feature-value', 'data'),
                  Input('feature-dropdown', 'value'), Input('interval-component', 'n_intervals'))
    def get_feature(value, n):
        return value

    @app.callback(Output('metric-value', 'data'),
                  Input('metric-dropdown', 'value'), Input('interval-component', 'n_intervals'))
    def get_metric(value, n):
        return value

    @app.callback(Output('density-graphic', 'figure'),
                  Input('feature-metric-graphic', 'clickData'),
                  Input('feature-dropdown', 'value'),
                  Input('model-dropdown', 'value'))
    def display_click_data(clickData, features, name):
        fig = go.Figure()
        if type(features) == str:
            features = [features]
        if (len(features) == 0):
            fig.update_layout(barmode='group', plot_bgcolor='white')
            fig.update_xaxes(title_font_family="Arial", gridcolor='grey', gridwidth=0.1)
            fig.update_yaxes(title_font_family="Arial", gridcolor='grey', gridwidth=1)
            return fig

        # version1 = str(clickData['points'][0]['x'])
        versions = list()

        curve_number = clickData['points'][0]['curveNumber']
        feature = features[curve_number]
        root = os.environ['MODELS_PATH'] + "/" + name + "/"
        myfiles = list()
        for path, subdirs, files in os.walk(root):
            for file in files:
                if ("baseline_data_" in file):
                    versions.append(re.search(r'\d+', file).group())
                    myfiles.append(os.path.join(path, file))
        versions = sorted(versions)
        myfiles = alphanum_sort(myfiles)
        for i in range(0, len(myfiles)):
            file_path = myfiles[i]
            version = versions[i]
            dataframe = pd.read_csv(file_path)
            cols = dataframe.columns
            pdf_dataframe = pd.DataFrame(columns=cols)
            for c in cols:
                pdf_dataframe[c] = stats.norm.pdf(dataframe[c])
            fig.add_trace(go.Histogram(name="version " + version, x=dataframe[feature], y=pdf_dataframe[feature]))

        fig.update_layout(barmode='group', plot_bgcolor='white',
                          title="Feature " + feature + " distribution comparison between all versions", title_x=0.5,
                          title_font_color="black")
        fig.update_xaxes(title_font_family="Arial", gridcolor='grey', gridwidth=0.1, title=feature + ' Values')
        fig.update_yaxes(title_font_family="Arial", gridcolor='grey', gridwidth=1, title='Probability Distribution')
        return fig

    @app.callback(Output('feature-metric-graphic', 'figure'),
                  Input('feature-value', 'data'),
                  Input('metric-value', 'data'),
                  Input('model-dropdown', 'value'),
                  Input('interval-component', 'n_intervals'))
    def update_graph1(features, metric, name, k):
        if type(features) == str:
            features = [features]
        rows = list()
        for feature in features:
            root = os.environ['MODELS_PATH'] + "/" + name + "/"
            for path, subdirs, files in os.walk(root):
                for file in files:
                    if ("baseline_data_" in file):
                        created_time = os.stat(os.path.join(path, file)).st_ctime
                        version = re.search(r'\d+', file).group()
                        dataframe = pd.read_csv(os.path.join(path, file))
                        feature_values = list(dataframe[feature])
                        if (metric == "Mean"):
                            value = sum(feature_values) / len(feature_values)
                        elif (metric == "Min"):
                            value = min(feature_values)
                        elif (metric == "Max"):
                            value = max(feature_values)
                        elif (metric == "Variance"):
                            value = np.var(feature_values)
                        elif (metric == "Standard Deviation"):
                            value = statistics.stdev(feature_values)
                        new_row = {'version': version, 'value': value, 'feature': feature,
                                   'time': str(dt.fromtimestamp(int(created_time)).strftime('%Y-%m-%d %H:%M:%S'))}
                        rows.append(new_row)
        fig_df = pd.DataFrame(rows, columns=['feature', 'version', 'value', 'time'])
        fig_df = fig_df.sort_values(by="version", ascending=True)
        fig = px.line(fig_df, x='version', y='value', color='feature', markers=True,
                      title=metric + " values for the feature(s): " + str(features), hover_data=["time"],
                      labels={"feature": "Feature", "value": metric + " Value", "version": "Model Version",
                              "time": "Timestamp"})

        fig.update_layout(
            plot_bgcolor='white',
            #     font_family="Courier New",
            #     font_color="white",
            #     title_font_family="Arial",
            title_font_color="black",
            #     legend_title_font_color="green",
            #     paper_bgcolor="white",
            #     plot_bgcolor="white",
            title_x=0.5
        )

        fig.update_xaxes(title_font_family="Arial", gridcolor='grey', gridwidth=1)
        fig.update_yaxes(title_font_family="Arial", gridcolor='grey', gridwidth=1)

        return fig
