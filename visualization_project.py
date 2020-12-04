import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import xgboost as xgb
from dash.dependencies import Output, Input, State
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=np.number).columns.tolist()


def fix_nans(df: pd.DataFrame, column: str):

    print('Nan fixing for column ' + str(column))  # Represents
    df_copy = df.copy()
    df_copy[column].fillna(value=np.nan, inplace=True)
    df_copy[column].interpolate(inplace=True, method='linear')
    df_copy[column].fillna(method='bfill', inplace=True)
    return df_copy


def simple_random_forest_classifier(X: pd.DataFrame, y: pd.Series,
                                    random_forest: RandomForestClassifier = None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    if random_forest is None:

        model = RandomForestClassifier(criterion="entropy", random_state=21, n_estimators=20, max_depth=40,
                                       bootstrap=True)
    else:
        model = random_forest
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)  # Use this line to get the prediction from the model
    accuracy = model.score(X_test, y_test)
    matrix = confusion_matrix(y_test, y_predict)

    return model, accuracy, y_predict, matrix


def random_forest_regressor(X: pd.DataFrame, y: pd.Series, estimators: int = 120):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    model = RandomForestRegressor(n_estimators=estimators, max_depth=200)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    score = model.score(X_test, y_test)

    return dict(model=model, score=score, test_prediction=y_predict, logerror=mean_squared_log_error(y_test, y_predict))


def decision_tree_regressor(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    model = DecisionTreeRegressor(criterion='mse', max_depth=90, min_samples_split=20)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(r2_score(y_test, y_predict))
    return dict(model=model, score=score, test_prediction=y_predict)


def xgBoostRegressor(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    model = xgb.XGBRegressor(
        n_estimators=100,
        reg_lambda=1,
        reg_alpha=0.002,
        gamma=0.3,
        max_depth=4,
        min_child_weight=4,
        subsample=1,
        colsample_bytree=1,
    )
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    score = r2_score(y_test, y_predict)
    print(r2_score(y_test, y_predict))
    return dict(model=model, score=score, test_prediction=y_predict)


def load_data():

    hap_2015 = pd.read_csv('./archive/2015.csv')
    hap_2016 = pd.read_csv('./archive/2016.csv')
    hap_2017 = pd.read_csv('./archive/2017.csv')
    hap_2018 = pd.read_csv('./archive/2018.csv')
    hap_2019 = pd.read_csv('./archive/2019.csv')
    combined_hap = pd.read_csv('./archive/dataset.csv')

    return hap_2015, hap_2016, hap_2017, hap_2018, hap_2019, combined_hap


def clean_datasets(dfs):
    cleaned_df = []
    for each_data in dfs:
        numeric_columns = get_numeric_columns(each_data)
        for col in numeric_columns:
            each_data = fix_nans(each_data, col)
        cleaned_df.append(each_data)

    return cleaned_df


def project():
    all_data = load_data()
    all_data = clean_datasets(all_data)
    hap_2015, hap_2016, hap_2017, hap_2018, hap_2019, combined_hap = all_data

    region_2015 = hap_2015[['Country', 'Region']].set_index(['Country']).to_dict()['Region']
    new_combined_hap = combined_hap[combined_hap['Country'].map(combined_hap['Country'].value_counts()) == 5]
    new_combined_hap['Region'] = new_combined_hap['Country'].apply(lambda x: region_2015[x])
    new_combined_hap['sentiment'] = new_combined_hap['Happiness Score'].apply(
        lambda x: 'happy' if x > 6.0 else ('sad' if x < 4.8 else 'neutral'))

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    default_options = [{'label': column, 'value': column} for column in get_numeric_columns(new_combined_hap)]
    default_country_options = [{'label': column, 'value': column} for column in
                               list(new_combined_hap['Country'].unique())]
    default_country_value = default_country_options[0]['value']
    default_value = default_options[0]['value']

    default_color_options = [{'label': column, 'value': column} for column in ['Country', 'Region']]
    default_color_value = default_color_options[0]['value']

    default_year_options = [{'label': column, 'value': column} for column in ['2015', '2016', '2017', '2018', '2019']]
    default_year_value = default_year_options[0]['value']

    default_axes_options = [{'label': column, 'value': column} for column in
                            set(get_numeric_columns(new_combined_hap)) - {'Year'}]
    default_axes_value = default_axes_options[0]['value']

    app.layout = dbc.Container([
        html.H1(children='Happiness Score Visualization',
                style={'text-align': 'center', 'color': '#047BD5'}),
        html.Hr(),
        html.P(children='Below Graphs Represents how any 5 countries happiness varies over the years',
               style={'text-align': 'center', 'font-weight': 'bold', 'font-size': '25px'}),
        dbc.FormGroup([
            html.Div([
                dbc.Label("Country_1", style={'font-size': '25px', 'text-align': 'center'}),
                dcc.Dropdown(id="dropdown_country1", value=default_country_value,
                             options=default_country_options)],
                style={'width': '15%', 'display': 'inline-block', 'text-align': 'center'}),
            html.Div([
                dbc.Label("Country_2", style={'font-size': '25px', 'text-align': 'center'}),
                dcc.Dropdown(id="dropdown_country2", value=default_country_value,
                             options=default_country_options)],
                style={'width': '15%', 'display': 'inline-block', 'text-align': 'center'}),
            html.Div([
                dbc.Label("Country_3", style={'font-size': '25px', 'text-align': 'center'}),
                dcc.Dropdown(id="dropdown_country3", value=default_country_value,
                             options=default_country_options)],
                style={'width': '15%', 'display': 'inline-block', 'text-align': 'center'}),
            html.Div([
                dbc.Label("Country_4", style={'font-size': '25px', 'text-align': 'center'}),
                dcc.Dropdown(id="dropdown_country4", value=default_country_value,
                             options=default_country_options)],
                style={'width': '15%', 'display': 'inline-block', 'text-align': 'center'}),
            html.Div([
                dbc.Label("Country_5", style={'font-size': '25px', 'text-align': 'center'}),
                dcc.Dropdown(id="dropdown_country5", value=default_country_value,
                             options=default_country_options)],
                style={'width': '15%', 'display': 'inline-block', 'text-align': 'center'})
        ], style={"display": "flex",
                  "justify-content": "space-between"}),
        html.Hr(),
        dbc.FormGroup([
            html.Div([
                dbc.Label("axes", style={'font-size': '25px', 'text-align': 'center'}),
                dcc.Dropdown(id="drop_axes", value=default_axes_value,
                             options=default_axes_options)],
                style={'width': '35%', 'display': 'inline-block', 'text-align': 'center'}),
            html.Div([
                dbc.Label("Color On", style={'font-size': '25px', 'text-align': 'center'}),
                dcc.Dropdown(id="drop_color", value=default_color_value,
                             options=default_color_options)],
                style={'width': '35%', 'display': 'inline-block', 'text-align': 'center'})],
            style={"display": "flex",
                   "justify-content": "space-between"}),
        html.Hr(),
        dbc.Button('Submit', id='example-button', color='primary', style={'margin-bottom': '1em'}, block=True),
        dbc.Card(
            [
                # Component card is used for representing graph and it number of rows values
                dbc.CardBody(

                    [
                        html.H4("Bar Chart", id="bar-card-title"),
                        # The paragraph displays number of rows
                        html.P(" ", id="bar-card-text"),
                        # Graph is displayed here
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='main-graph')),
                        ]),

                    ]
                ),
                dbc.CardBody(

                    [
                        html.H4("Line Chart", id="line-card-title"),
                        # The paragraph displays number of rows
                        html.P("", id="line-card-text"),
                        # Graph is displayed here
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='line-main-graph')),
                        ]),

                    ]
                ),
            ],

        ),
        html.Hr(),
        dbc.Card(
            [
                # Component card is used for representing graph and it number of rows values
                dbc.CardBody(

                    [
                        html.H4("heat Chart", id="heat-card-title"),
                        # The paragraph displays number of rows
                        html.P("Co relation matrix for all features", id="heat-card-text"),
                        # Graph is displayed here
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='heat-main-graph')),
                        ]),

                    ]
                ),
                dbc.CardBody(

                    [
                        html.H4("heatbar Chart", id="heatbar-card-title"),
                        # The paragraph displays number of rows
                        html.P("How features depend on happiness score", id="heatbar-card-text"),
                        # Graph is displayed here
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='heatbar-main-graph')),
                        ]),

                    ]
                ),

            ]),
        html.Hr(),

        dbc.Card([

            dbc.CardBody([

                html.H4("Jointplot Chart", id="jointplot-card-title"),
                # The paragraph displays number of rows
                html.P("", id="jointplot-card-text"),

                dbc.FormGroup([
                    html.Div([
                        dbc.Label("x-axis", style={'font-size': '25px', 'text-align': 'center'}),
                        dcc.Dropdown(id="drop_xaxes", value=default_value,
                                     options=default_options)],
                        style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),
                    html.Div([
                        dbc.Label("y axis", style={'font-size': '25px', 'text-align': 'center'}),
                        dcc.Dropdown(id="drop_yaxes", value=default_value,
                                     options=default_options)],
                        style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'})],
                    style={"display": "flex",
                           "justify-content": "space-between", "padding": "25px"}),
                # Graph is displayed here
                dbc.Row([
                    dbc.Col(dcc.Graph(id='jointplot-main-graph'))
                ]),
                html.H4("3D plot Chart on happiness score", id="3d-card-title"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='3d-main-graph'))
                ]),

            ])
        ]),
        html.Hr(),
        dbc.Card([

            dbc.CardBody([
                html.H4("Map plot Chart", id="mapplot-card-title"),
                dbc.FormGroup([
                    html.Div([
                        dbc.Label("Map-axis", style={'font-size': '25px', 'text-align': 'center'}),
                        dcc.Dropdown(id="map_axes", value=default_value,
                                     options=default_options)],
                        style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),
                    html.Div([
                        dbc.Label("Year", style={'font-size': '25px', 'text-align': 'center'}),
                        dcc.Dropdown(id="year_id", value=default_year_value,
                                     options=default_year_options)],
                        style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),
                ], style={"display": "flex",
                          "justify-content": "space-between", "padding": "25px"}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='mapplot-main-graph'))
                ]),

            ]),

        ]),
        html.Hr(),
        dbc.Card([

            dbc.CardBody([
                html.H4("Region Ratio Chart", id="region-card-title"),
                dbc.FormGroup([
                    html.Div([
                        dbc.Label("column_ratio", style={'font-size': '25px', 'text-align': 'center'}),
                        dcc.Dropdown(id="region_axes", value=default_value,
                                     options=default_options)],
                        style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),
                    html.Div([
                        dbc.Label("Year", style={'font-size': '25px', 'text-align': 'center'}),
                        dcc.Dropdown(id="year_id_region", value=default_year_value,
                                     options=default_year_options)],
                        style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),
                ], style={"display": "flex",
                          "justify-content": "space-between", "padding": "25px"}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='region-main-graph'))
                ]),

            ]),

        ]),
        html.Hr(),
        dbc.Card([

            dbc.CardBody([
                html.H4("Stacked and box plot chart for region", id="stack-card-title"),
                dbc.FormGroup([
                    html.Div([
                        dbc.Label("Year", style={'font-size': '25px', 'text-align': 'center'}),
                        dcc.Dropdown(id="year_id_region_stacked", value=default_year_value,
                                     options=default_year_options)],
                        style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),
                ]),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='stacked-main-graph'))

                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='box-main-graph'))

                ]),

            ]),

        ]),
        html.Hr(),
        dbc.Card([

            dbc.CardBody([
                html.H4("Top 10 countries for features", id="top-card-title"),
                dbc.FormGroup([
                    html.Div([
                        dbc.Label("Year", style={'font-size': '25px', 'text-align': 'center'}),
                        dcc.Dropdown(id="year_id_top", value=default_year_value,
                                     options=default_year_options)],
                        style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='top1-main-graph')),
                    dbc.Col(dcc.Graph(id='top2-main-graph'))
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='top3-main-graph')),
                    dbc.Col(dcc.Graph(id='top4-main-graph'))
                ]),

            ]),

        ]),
        html.Hr(),
        dbc.Card([

            dbc.CardBody([
                html.H4("Parallel comparision on features", id="parallel-card-title"),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='parallel-main-graph'))

                ]),

            ]),

        ]),
        html.Hr(),
        dbc.Card([

            dbc.CardBody([
                html.H4("Regression and Classification", id="regression-card-title"),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='regression-main-graph'))

                ]),
                html.H4("Random forest classifier", id="classifier-card-title"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='classifier-main-graph'))

                ]),

            ]),

        ]),

    ])

    @app.callback(Output('main-graph', "figure"), Output('line-main-graph', "figure"),
                  [Input('example-button', 'n_clicks')],
                  [State('dropdown_country1', 'value'), State('dropdown_country2', 'value'),
                   State('dropdown_country3', 'value'), State('dropdown_country4', 'value'),
                   State('dropdown_country5', 'value'), State('drop_axes', 'value'),
                   State('drop_color', 'value')])
    def update_fig(clicks, country1, country2, country3, country4, country5, axes, color_on):
        new_df = new_combined_hap[new_combined_hap['Country'].isin([country1, country2, country3, country4, country5])]
        new_df = new_df.sort_values(['Country', 'Year'])
        fig = px.bar(new_df, x="Year", y=axes, color=color_on, title=f"Year vs {axes} for countries", barmode="group")
        line_fig = px.line(new_df, x="Year", y=axes, color=color_on, title=f"Year vs {axes} for countries")
        return fig, line_fig

    @app.callback(Output('heat-main-graph', "figure"), Output('heatbar-main-graph', "figure"),
                  [Input('example-button', 'n_clicks')])
    def corrs_graph(nclicks):
        # new_df_corr = new_combined_hap[new_combined_hap.columns.difference(['Happiness Rank'])]
        corrs = new_combined_hap.corr(method='pearson')
        col = list(corrs.columns)
        heat_fig = ff.create_annotated_heatmap(np.round_(corrs.values, decimals=3), x=col, y=col)
        co_relation = pd.DataFrame(new_combined_hap.corrwith(new_combined_hap["Happiness Score"]).to_dict().items(),
                                   columns=['columns', 'corr_score'])
        co_relation = co_relation[
            (co_relation['columns'] != 'Happiness Score') & (co_relation['columns'] != 'Happiness Rank')]
        heat_bar_fig = px.bar(co_relation, x="columns", y="corr_score", title="happiness score for countries",
                              barmode='group')
        return heat_fig, heat_bar_fig

    @app.callback(Output('jointplot-main-graph', "figure"), Output('3d-main-graph', "figure"),
                  [Input('drop_xaxes', 'value'), Input('drop_yaxes', 'value'),
                   Input('dropdown_country1', 'value'), Input('dropdown_country2', 'value'),
                   Input('dropdown_country3', 'value'), Input('dropdown_country4', 'value'),
                   Input('dropdown_country5', 'value')])
    def jointplot_and_3Dgraph(x_axes, y_axes, country1, country2, country3, country4, country5):
        new_df = new_combined_hap[new_combined_hap['Country'].isin([country1, country2, country3, country4, country5])]
        joint_fig = px.histogram(new_combined_hap, x=x_axes, y=y_axes, color='Region',
                                 marginal="box",  # or violin, rug
                                 hover_data=new_combined_hap.columns)
        D_plot = px.scatter_3d(new_df, x=x_axes, y=y_axes, z='Happiness Score',
                               color='Country')

        return joint_fig, D_plot

    @app.callback(Output('mapplot-main-graph', "figure"),
                  [Input('map_axes', 'value'), Input('year_id', 'value')])
    def map_graph(axes, year):
        print(year)
        print(axes)
        new_df1 = new_combined_hap[new_combined_hap['Year'] == int(year)]
        # print(new_combined_hap)
        # print(new_df1)
        fig = px.choropleth(new_df1, locations="Country",
                            color=axes,
                            locationmode='country names',
                            color_continuous_scale=px.colors.sequential.Blues
                            )

        return fig

    @app.callback(Output('region-main-graph', "figure"),
                  [Input('region_axes', 'value'), Input('year_id_region', 'value')])
    def region_graph(axes, year):
        new_df = new_combined_hap[new_combined_hap['Year'] == int(year)]
        regions = list(new_df['Region'].unique())
        axes_ratio = []
        for each in regions:
            region = new_df[new_df['Region'] == each]
            axes_rate = sum(region[axes]) / len(region)
            axes_ratio.append(axes_rate)

        data = pd.DataFrame({'region': regions, axes + '_ratio': axes_ratio})
        new_index = (data[axes + '_ratio'].sort_values(ascending=False)).index.values
        sorted_data = data.reindex(new_index)

        fig = px.bar(sorted_data, x='region', y=axes + '_ratio',
                     color=axes + '_ratio', height=400)
        return fig

    @app.callback(Output('stacked-main-graph', "figure"), Output('box-main-graph', "figure"),
                  [Input('year_id_region_stacked', 'value')])
    def stacked_box_plot(year):

        new_df = new_combined_hap[new_combined_hap['Year'] == int(year)]
        region_lists = list(new_df['Region'].unique())
        share_economy = []
        share_family = []
        share_health = []
        share_freedom = []
        share_trust = []
        for each in region_lists:
            region = new_df[new_df['Region'] == each]
            share_economy.append(sum(region['Economy (GDP per Capita)']) / len(region))
            share_family.append(sum(region['Family']) / len(region))
            share_health.append(sum(region['Health (Life Expectancy)']) / len(region))
            share_freedom.append(sum(region['Freedom']) / len(region))
            share_trust.append(sum(region['Trust (Government Corruption)']) / len(region))

        fig = go.Figure(data=[
            go.Bar(x=region_lists, y=share_economy, name="Economy"),
            go.Bar(x=region_lists, y=share_family, name="Family"),
            go.Bar(x=region_lists, y=share_health, name="Health"),
            go.Bar(x=region_lists, y=share_freedom, name="Freedom"),
            go.Bar(x=region_lists, y=share_trust, name="Trust", marker_color='black'),
        ])
        # Change the bar mode
        fig.update_layout(barmode='stack', xaxis_title='Region', yaxis_title='Affecting Value',
                          title='Factors Affecting Happiness Score')
        print('fig done')
        box_plot = px.box(new_combined_hap, x="Year", y="Happiness Score", color="Region")
        return fig, box_plot

    @app.callback(Output('top1-main-graph', "figure"), Output('top2-main-graph', "figure"),
                  Output('top3-main-graph', "figure"), Output('top4-main-graph', "figure"),
                  [Input('year_id_top', 'value')])
    def top_plot(year):
        new_df = new_combined_hap[new_combined_hap['Year'] == int(year)]
        fig1 = px.bar(new_df.nlargest(10, 'Economy (GDP per Capita)'), x='Economy (GDP per Capita)', y='Country')
        fig1.update_traces(marker_color='Blue')
        fig2 = px.bar(new_df.nlargest(10, 'Family'), x='Family', y='Country')
        fig2.update_traces(marker_color='Red')
        fig3 = px.bar(new_df.nlargest(10, 'Health (Life Expectancy)'), x='Health (Life Expectancy)', y='Country')
        fig3.update_traces(marker_color='Green')
        fig4 = px.bar(new_df.nlargest(10, 'Freedom'), x='Freedom', y='Country')
        fig4.update_traces(marker_color='Brown')
        print('figures done')
        return fig1, fig2, fig3, fig4

    @app.callback(Output('parallel-main-graph', "figure"),
                  [Input('year_id_top', 'value')])
    def parallel_plot(year):
        fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=new_combined_hap['Year'])
            ,
            dimensions=list([
                dict(range=[0, 8],
                     constraintrange=[4, 8],
                     label='Happiness Score', values=new_combined_hap['Happiness Score']),
                dict(range=[0, 8],
                     label='Economy (GDP per Capita)', values=new_combined_hap['Economy (GDP per Capita)']),
                dict(range=[0, 8],
                     label='Family', values=new_combined_hap['Family']),
                dict(range=[0, 8],
                     label='Health (Life Expectancy)', values=new_combined_hap['Health (Life Expectancy)'])
            ])
        )
        )

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        print('parallel done')
        return fig

    @app.callback(Output('regression-main-graph', "figure"), Output('classifier-main-graph', "figure"),
                  [Input('year_id_top', 'value')])
    def reg_class(year):
        print('a')
        random_forewt = random_forest_regressor(new_combined_hap[new_combined_hap.columns.difference(
            ['Country', 'Happiness Score', 'sentiment', 'Region'])], new_combined_hap['Happiness Score'])
        dec_forewt = decision_tree_regressor(new_combined_hap[new_combined_hap.columns.difference(
            ['Country', 'Happiness Score', 'sentiment', 'Region'])], new_combined_hap['Happiness Score'])
        xg_forewt = xgBoostRegressor(new_combined_hap[new_combined_hap.columns.difference(
            ['Country', 'Happiness Score', 'sentiment', 'Region'])], new_combined_hap['Happiness Score'])
        randomscore = random_forewt['score']
        print(randomscore)
        decisiontreescore = dec_forewt['score']
        xgboostscore = xg_forewt['score']
        model_accuracy = pd.Series(data=[randomscore, decisiontreescore, xgboostscore],
                                   index=['Random Forest Regressor', 'DecisionTree Regressor', 'XGBoost Regressor'])
        reg_fig = plt.figure(figsize=(8, 8))
        reg_fig = px.bar(model_accuracy)
        model, accuracy, y_predict, matrix = simple_random_forest_classifier(
            new_combined_hap[new_combined_hap.columns.difference(['Country', 'sentiment', 'Region'])],
            new_combined_hap['sentiment'])
        classify_fig = ff.create_annotated_heatmap(matrix, x=['Happy', 'neutral', 'Sad'], y=['Happy', 'neutral', 'Sad'])
        print('reg_classify done')
        return reg_fig, classify_fig

    return app


if __name__ == '__main__':
    app = project()
    app.run_server(debug=True,host='0.0.0.0',port=3000)
