# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:24:14 2020

@author: DevleenaBanerjee
"""
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import datetime as dt
from dash.dependencies import Input, Output
from nsepy import get_history as gh
from fbprophet import Prophet
from scipy.stats import boxcox
from scipy.special import inv_boxcox


app = dash.Dash(__name__)
server = app.server

#Dash Layout with core component

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

card_graph = dbc.Card(
        dcc.Graph(id='our_graph', figure={}), body=True, color="secondary",
)

card_text = dbc.Card(
    dbc.CardBody(
        [
            html.H6(
                "This app is Predicting the Stock Prices using TimeSeries model for next 90 days of selected Ticker"
                " using Prophet from Facebook. Data is loaded dynamically from NSEPY. ",
                className="card-text",
                #style={'width': '70%', 'height': 30},
            ),

                ]
    ),
    style={"width": "30rem",'height':60},
)
app.layout = html.Div([

    dbc.Row(
        dbc.Col(html.H4(children="STOCK PREDICTION ANALYSIS", style={'text-align': 'center',
            'color': colors['text']}),
                    ),
            ),
    html.Br(),

    dbc.Row(dbc.Col(html.H5(children='Dash: A web application framework for Python.', style={
        'textAlign': 'center',
        'color': colors['text']}),
                ),
        ),

    html.Br(),

    dbc.Row(
            [
                dbc.Col(html.H6(children='Select the Ticker from the dropdown', style={
        'textAlign': 'left',
        'color': colors['text']
            })
            ),
            dbc.Col(card_text,
                             style={
                            'color': colors['text']} ),
            ],
            ),
    html.Br(),
    dbc.Row(
            [
                dbc.Col(dcc.Dropdown(id='selectstock', placeholder='last dropdown',
                                 options=[{'label': 'TCS', 'value': 'TCS'},
                                      {'label': 'WIPRO', 'value': 'WIPRO'},
                                      {'label': 'RELIANCE', 'value': 'RELIANCE'},
                                      {'label': 'L&T', 'value': 'LT'},
                                      {'label': 'ONGC', 'value': 'ONGC'}],
                                 value='TCS'),
                        width={'size': 3, "offset": 0, 'order': 3},


                        ),
                            ],no_gutters=True
        ),

     html.Br(),

     # dash_table.DataTable(id='stock_table'),
     dbc.Row(
            [
              dbc.Col(dcc.Graph(id='our_graph'),
                        width=6, md={'size': 6,  "offset": 0, 'order': 'first'}
                        ),
              dbc.Col(dcc.Graph(id='forecast_graph', figure={}),
                        width=6, md={'size':6,  "offset": 0, 'order': 'last'}
                        ),
            ]
        ),


      html.Br(),

        dbc.Row(dbc.Col(html.H6(children='By Devleena Banerjee', style={
        'textAlign': 'right',
        'color': colors['text']}),
                ),
        ),

])

#Function for getting data directly from NSEPY for the selected value
def get_data(value):
    start = dt.datetime(2010,1,1)
    end=pd.to_datetime(dt.datetime.today(),format='%Y-%m-%d', errors='coerce')
    stk_data = gh(symbol=value,start=start,end=end)
    stk_data.reset_index(inplace=True)
    return stk_data[['Date','Close']]
# Connect the Plotly graphs with Dash Components

@app.callback(
  Output('our_graph', 'figure'),
    #Output ('forecast_graph', 'figure')],
    [Input('selectstock','value')]
    )


def update_graph(input_value):


    #df=get_data(input_value)

    df=pd.read_csv("{}.csv".format(input_value))
    fig1 = go.Figure()

    fig1.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    margin={'b': 45},
    title={'text': 'Stock Prices of {} for Last 5 Years'.format(input_value), 'font': {'color': 'white'}, 'x': 0.5}
    )

    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
                        mode='lines',
                    name='Close'))
    min=df['Close'].min()
    max=df['Close'].max()
    fig1.update_xaxes(title_text="Date")
    fig1.update_yaxes(title_text="Closing Values")
    fig1.update_yaxes(range=[min,max],)

    return fig1

@app.callback(

    Output ('forecast_graph', 'figure'),
    [Input('selectstock','value')]
    )

    #def update_graph(input_value):
    #start = dt.datetime(2015,1,1)
    #end=pd.to_datetime(dt.datetime.today().strftime("%Y,%m,%d"))
    #global stk_data
    #stk_data = gh(symbol=value,start=start,end=end)
    #stk_data = pd.read_csv('{}.csv'.format(input_value))
    #stk_data.reset_index(inplace=True)

    #df=get_data(input_value)
    df=pd.read_csv("{}.csv".format(input_value))
    df['ds'] = df['Date']
    df['y'] = df['Close']
    df['y'], lam = boxcox(df['Close'])
    df.drop(['Date','Close'],axis=1,inplace=True)

    m =Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=90)
    forecast = m.predict(future)
    forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, lam))

    fig2 = go.Figure()
    #df=forecast_model(input_value)
    fig2.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],   font_color=colors['text'],
    margin={'b': 45},
    title={'text': 'Stock Prices of {} with 3 Months Forecasting'.format(input_value), 'font': {'color': 'white'}, 'x': 0.5}
    )
    fig2.add_trace(go.Scatter( x = forecast['ds'],y = forecast['yhat'],
                    name='Close'))
    min=forecast['yhat'].min()
    max=forecast['yhat'].max()
    fig2.update_xaxes(title_text="Date")
    fig2.update_yaxes(title_text="Forecasted Closing Values")
    fig2.update_yaxes(range=[min,max])
    #forecast_model(input_value)
    return fig2

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
