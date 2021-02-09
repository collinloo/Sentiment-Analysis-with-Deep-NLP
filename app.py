# import library
from myFunc_All import *
import time
import datetime 
import base64
import datetime
# import random
# random.seed(123)
import pandas as pd
import numpy as np
# from pickle import dump, load
from urllib.parse import quote as urlquote
from wordcloud import WordCloud
# from sklearn.metrics import confusion_matrix, classification_report
import io

# nlp pacakges
import spacy
# from spacy.matcher import Matcher
# from gensim.models import Word2Vec
# from nltk import word_tokenize

# dash and plot
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# nueral network pacakges
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.layers import Embedding, Flatten, LSTM
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# load the spacy eng library
# spacy.cli.download('en_core_web_lg')
nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
nlp.max_length = 24142831  

# add new stop words
new_sw = [' ', '\n', '\n\n']
for w in new_sw:
    nlp.Defaults.stop_words.add(w)
    nlp.vocab[w].is_stop = True

# remove default stop wrods because they may be meaningful for the neg rating
def_sw = ["n't", 'n’t', 'again', 'against', 'down',
          'neither', 'never', 'not', 'no']

for w in def_sw:
    nlp.Defaults.stop_words.remove(w)
    nlp.vocab[w].is_stop = False    

# place function here and not in py file due to css styling
def file_predict(contents, filename):
    '''
    Signature   file_predict(contens=None, filename=None)
    Docstring   return a dash table and plot fig
    Parameters  contents: upload file 
                filename: string, file name
    '''
    content_type, content_string = contents.split(',')
   
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        
        # get predictions
        start = time.time()
        pred = get_pred(df, nlp, tokenizer)
        # append predictions
        df_pred = append_pred(df, pred)
        
        # create download link
        csv_str = df_pred.to_csv(index=False, encoding='utf-8')
        csv_str = 'data:text/csv;charset=utf-8, ' + urlquote(csv_str)
        
        # plot precision score if label is included in the upload file
        if (df.columns == 'label').sum() > 0:
            fig = plot_dash_precision(df, pred)
        else:
            fig = {}
        
        # convert df to dash table
        dash_tbl = html.Div([
            dash_table.DataTable(
                data=df_pred.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_pred.columns],
                page_size=20,
                sort_action ='native',
                style_table={'overflowy':'auto'},
                style_header=css_tbl_header,
                style_cell=css_tbl_cell,
                style_data_conditional=css_tbl_condl
                )
            ], style={'font-family': 'Verdana', 'font-size':'1.4em'})
        
        end = time.time()
        total_time = str(datetime.timedelta(seconds = end - start))
        
        return dash_tbl, fig, csv_str, total_time
    
    except Exception as e:
        print(e)
        err_msg = html.Div([
            'There was an error processing this file.'
        ], style={'font-size': '1.4em', 'font-weight':'bold',
                  'color':'#CC4848 ', 'text-align':'center'})
        return err_msg, {}, None

# preload dataframes to save exec. time
df_500 = create_df('data/off_500.csv')
df_1000 = create_df('data/off_1000.csv')
df_1500 = create_df('data/off_1500.csv')
df_2000 = create_df('data/off_2000.csv')
df_2500 = create_df('data/off_2500.csv')
df_dict = {'df_500': df_500, 'df_1000': df_1000, 'df_1500': df_1500,
           'df_2000': df_2000, 'df_2500': df_2500}

# pre-generate word frequency for word cloud
pos_wc_dict = {}
neg_wc_dict = {}
for k, v in df_dict.items():
    pos_wc_dict[k] = get_freq_words(v, 1, nlp, single_word=True)
    neg_wc_dict[k] = get_freq_words(v, 0, nlp, single_word=True)

# parse model summary to dash table
dcc_tbl_mdl_sum = parse_summ()

# dash table styling
css_tbl_header = {'backgroundColor': '#dde7ed', 'fontWeight': 'bold'}
css_tbl_cell = {'textAlign':'left','whiteSpace':'normal', 'height':'auto'}
css_tbl_condl = [{'if': {'row_index': 'odd'}, 'backgroundColor': '#eef8fe'}]                

# create Dash app
app = dash.Dash(__name__)
server = app.server

# create dropdown dict that correspond to predefined csv files in data folder
pdt_name = ['Blami Chalk Marker', 'MIROO Ink Cartridge', 'Gel Pen',
            'FASTINK Ink Cartridge', 'Sargent Color Pencils']
drop_down = [{'label':f'{j}, {i}', 'value':f'df_{i}'}
             for i, j in list(zip(range(500, 3000, 500), pdt_name))]

# image file locatioin
path = 'img/'
img_banner = 'ai.png'
img_pred = 'pred.png'

# Dash layout    
app.layout = html.Div(id='main-div', children=[
    dcc.Tabs([
        # tab one
        dcc.Tab(label='MODEL SHOWCASE ', children=[
            # top banner 
            html.Img(src=encode_image(path+img_banner), className='css_banner'),
            # level 1 container
            html.Div([
                # div containing df select and table
                html.Div([
                    html.Div(id='df-tbl-top', children=[
                        html.Div('Select Set of Product Reviews:',
                            className='css_sel_lab'),  
                        html.Div(dcc.Dropdown(id='df-picker',
                            options=drop_down,
                            value=drop_down[0]['value'], className='css_dd')
                    ),
                    html.Div('Percent of Positive', id='pos', className='css_pos_neg'),
                    html.Div('Percent of Positive', id='neg', className='css_pos_neg'),
                # df output table
                ], className='css_df_tbl_top'),
                html.Div(id='df-output', className='css_df_output'),
            ], className='css_cont_df'),
    
            # div containing wc labels and plot
            html.Div([
                html.Div([
                    html.Div('Select Most Common Words:', className='css_wc_lab'),
                    html.Div(dcc.RadioItems(id='wc-radio',
                        options=[{'label':'Positive Reviews', 'value':1},
                            {'label':'Negative Reviews', 'value':0}],
                            value=1,
                            labelStyle={'display':'block', 'color':'#FBF67C'},
                            ), className='css_rad_div'),
                ], className='css_wc_top'),
                html.Div(
                    dcc.Graph(id='word-cloud', config={'displayModeBar':False}), 
                    className='css_wc_output'),
                ], className='css_cont_wc'),
            ], className='css_div_lvl1'),
        
            # leve 2 conctainer
            html.Div([
                # model summary container
                html.Div([
                    html.Div('MODEL SUMMARY', className='css_sec_common'),
                    html.Div([
                        dcc_tbl_mdl_sum,
                        html.Button('GET PREDICTION', id='get-pred',
                        className='css_set_button')      
                ], className='css_l2_out_com'),
            ], className='css_sub_cont'),
        
                # actual vs predict container
                html.Div(id='div-clsrpt', children=[
                    html.Div('ACTUAL VS PREDICT', id='act-pred',
                        className='css_sec_common'),
                    dcc.Loading(
                        id='loading-icon',
                        children=html.Div(), #id='cls-rpt'
                        type='default',
                    ),
                ], className='css_sub_cont'),
 
                # precision figure container
                html.Div(id='div-precision', children=[
                    html.Div('F1-SCORE', className='css_sec_common'),
                    html.Div(
                        dcc.Graph(id='fig-precision', figure={}),
                        className='css_precision_output'
                    ),
                    ], className='css_sub_cont'),
            ], className='css_div_lvl2'),
        ]), # end of tab one
 
        # tab two
        dcc.Tab(label='SAND BOX', children=[
            html.Img(src=encode_image(path+img_banner), className='css_banner'),
            
            # single prediction container
            html.Div([
                # prediction result and button
                html.Div('SINGLE PREDICTION', className='css_sec_common'),
                html.Div([
                    dcc.Loading(
                        id='loading-sing-pred', 
                        children=html.Div([
                            html.Img(src=encode_image(path+img_pred),
                                style={'width':'100%','height':'150px'}),
                            html.Div(id='single-pred-result-txt',
                                className='css_slg_result_txt'),
                            html.Div(id='single-pred-result',
                                className='css_slg_result'),
                            html.Button('TRY ME!', id='get-sgl-pred',
                                className='css_sgl_pred_button')
                        ]),
                        type='default'
                        ),
                ], className='css_cont_result'),
                # input text section
                html.Div(
                    dcc.Textarea(
                        id='rev-input',
                        value='Enter your texts here.',
                        className = 'css_input_box',
                    ),
                    className='css_cont_input_text'
                ),
            ], className='css_div_lvl3'),
            
                # user upload container
                html.Div([
                    # dcc upload button container
                    html.Div([
                        html.Div('UPLOAD FILE', className='css_sec_common'),
                            html.Div([
                                html.P('Review texts must have a column ' +
                                    'header named review.', className='css_p'),
                                dcc.Upload(id='upload-data', children=
                                    html.Button('SELECT A CSV FILE',
                                        id='button-upload',
                                        style={'margin-left':'25px'}
                                    )
                                ),
                            ], className='css_dcc_upload'),
                    # precision score container
                    html.Div([
                        html.Div('F1-SCORE (if label is provided)',
                            className='css_user_prec_lab'),
                        html.Div(
                            dcc.Graph(id='user-prec-fig', figure={}),
                                className='css_user_prec_fig'
                        ),
                    ]),
                ], className='css_cont_upload'),
                    
                # user data & prediction table container
                html.Div([
                    html.Div([
                        html.Div('Upload File Prediction Results',
                                 id='pred-lab', className='css_sec_common'),
                        html.A('DOWNLOAD PREDICTIONS', id='download-link',
                               href='', target='_blank',
                               download='download_pred.csv', className='css_download')
                        ]),
                        dcc.Loading(
                            id='upload-icon',
                            children = html.Div(id='output-data-upload',
                                         children=dash_table.DataTable(
                                             id='user-table'),
                                            className='css_upload_output'), 
                            type='default'
                        ),
                ], className='css_cont_upload_output'),
            ], className='css_div_lvl4'),
        ]), # end of second tab
    ]), # end of dash Tabs

], className='css_main_div')    
print('at callback section')       
# display selected dataframe
@app.callback(
    [Output('df-output', 'children'),
     Output('get-pred', 'n_clicks'),
     Output('pos', 'children'),
     Output('neg', 'children')],
    [Input('df-picker', 'value')]
)
def show_df(selected_df):
    df = df_dict.get(selected_df)
    # parse df to dash table
    html_div = html.Div([
        dash_table.DataTable(
            data = df.to_dict('records'),
            columns = [{'name': i, 'id':i} for i in df.columns],
            page_size=20,
            sort_action ='native',
            style_table={'overflowy':'auto'},
            style_header=css_tbl_header,
            style_cell=css_tbl_cell,
            style_data_conditional=css_tbl_condl
            )
        ], style={'font-family': 'Verdana', 'font-size':'1.4em'})

    p_n_ratio = round(df['label'].value_counts(normalize=True), 2)
    pos = f'% Positive Review: {p_n_ratio[1]}'
    neg = f'% Negative Review: {p_n_ratio[0]}'
    return html_div, None, pos, neg

# plot word cloud
@app.callback(
    Output('word-cloud', 'figure'),
    [Input('df-picker', 'value'),
     Input('wc-radio', 'value')],
)
def show_wc(selected_df, label):
    df = df_dict.get(selected_df)
    fig = go.Figure()
    if label == 0:
        wc_df = neg_wc_dict.get(selected_df)
        # create WordCoud object
        wc = WordCloud(background_color="white")
        # generate word cloud
        wc.generate_from_frequencies(dict(wc_df))
        title = '<b>Top 50 Words from Negative Reviews'
    else:
        wc_df = pos_wc_dict.get(selected_df)
        # create WordCoud object
        wc = WordCloud(background_color="white")
        # generate word cloud
        wc.generate_from_frequencies(dict(wc_df))
        title = '<b>Top 50 Words from Positive Reviews</b>'
        
    fig.add_trace(go.Image(z=wc))
  
    fig.update_layout(
        title={'text':title, 'x':0.5, 'y':0.85, 'xanchor':'center'},
        hovermode=False, autosize=True, margin=dict(l=5, r=5, b=0, t=0),
        font=dict(size=15, color='#c5a654')
    )
    fig.update_xaxes(showticklabels=False, showline=True,
        linewidth=1, linecolor='#88bccb', mirror=True)
    fig.update_yaxes(showticklabels=False, showline=True,
        linewidth=1, linecolor='#e4c02e', mirror=True)
   
    return fig
   
# display actual vs predicted
@app.callback(
    [Output('loading-icon', 'children'),
     Output('get-pred', 'disabled'),
     Output('act-pred', 'children')],
    [Input('df-picker', 'value'),
    Input('get-pred', 'n_clicks')]
    )
def act_vs_pred(selected_df, n_clicks):
    if n_clicks is not None and n_clicks > 0:
        start = time.time()
        df = df_dict.get(selected_df)
        # get prediction
        pred = get_pred(df, nlp, tokenizer)
        # append predictions
        df_comp = append_pred(df, pred)
        
        # convert df to dash table
        act_pred = html.Div([
            dash_table.DataTable(
                data = df_comp.to_dict('records'),
                columns = [{'name': i, 'id': i} for i in df_comp.columns],
                page_size=10,
                sort_action ='native',
                style_header=css_tbl_header,
                style_cell=css_tbl_cell,
                style_data_conditional=css_tbl_condl
            )
        ], className='css_act_pred_output')
        end = time.time()
        total_time = str(datetime.timedelta(seconds = end - start))
        ret_text = f'ACTUAL vs PREDICT: EXEC. TIME ({total_time[0:10]})'
        return act_pred, True, ret_text
    else:
        act_pred = html.Div('CLICK BUTTON LEFT TO VIEW RESULTS',
            className='css_act_pred_empty')
        ret_text = 'ACTUAL vs PREDICT'
        return act_pred, False, ret_text
    
# display precision fig
@app.callback(
    Output('fig-precision', 'figure'),
    [Input('df-picker', 'value'),
     Input('get-pred', 'n_clicks')],
)
def show_precision(selected_df, n_clicks):
    if n_clicks is not None and n_clicks > 0:
        df = df_dict.get(selected_df)
        pred = get_pred(df, nlp, tokenizer)
        fig = plot_dash_precision(df, pred)
        return fig
    else:
        return {}

# get sigle prediction
@app.callback(
    [Output('single-pred-result-txt', 'children'),
     Output('single-pred-result', 'children')],
    [Input('get-sgl-pred', 'n_clicks')],
    [State('rev-input', 'value')]
)
def get_user_rev_pred(n_clicks, value):
    if n_clicks is not None and n_clicks > 0:
        pred = get_single_pred(value, nlp, tokenizer)
        tried_pred = f'Prediction Try No. {n_clicks} Result: {pred[0]}'
        return None, tried_pred 
    else:
        return 'Click button to get prediciton', None

# make prediction from upload file
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('user-prec-fig', 'figure'),
     Output('download-link', 'href'),
     Output('pred-lab', 'children')],
     Input('upload-data', 'contents'),
     State('upload-data', 'filename')
)
def update_output(contents, file_name):
    if contents is not None:
        children, fig, csv_str, total_time = file_predict(contents, file_name)
        predict_lab = f'UPLOAD FILE PREDICTION RESULTS: EXEC. TIME ({total_time[0:10]})'
        return children, fig, csv_str, predict_lab
    else:
        predict_lab = 'UPLOAD FILE PREDICTION RESULT'
        return None, {}, None, predict_lab

print('app.run_server')    
if __name__ == '__main__':
    app.run_server(debug=True)