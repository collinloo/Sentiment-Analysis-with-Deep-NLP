import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


def clean_text(txt, nlp, single_word=False):
    '''
    Signature:   clean_text(txt=none, nlp=none, token=False)
    Doctring:    return cleaned texts
    Parameters:  txt: string
                 nlp: Spacy object
                 token: boolean
    '''
    # remove tag
    import re
    tag_pattern = re.compile(r'<[^>]+>')
    tmp = tag_pattern.sub('', str(txt))
    
    # remove stop words, punctuations, single letter, spacy entity <> 0 and 
    # number
    tmp2 = nlp(tmp)
    doc = [token for token in tmp2 if
           token.is_stop != True        # remove stop words
           and token.is_punct != True   # remove punctuations
           and token.pos_ != 'NUM'      # remove digits
           and len(token) > 1           # remove single letter
           and not token.text.isspace() # remove spaces
#            and token.ent_type == 0      # remove spacy entities
          ]  
    
    # lemmatiztion
    doc = [token.lemma_ if token.lemma_ != '-PRON-'
           else token.text for token in doc]
    
    if single_word:
        # return as individual word
        doc = [x.replace("n't", "not").lower() for x in doc]
    else:
        # concatane token text, expand contraction and convert to lower case 
        doc = (' '.join(doc).replace("n't", "not")).lower()
    
    return doc 
    

def get_freq_words(df, label, nlp, word_no=50, single_word=True):
    '''
    Signature:  get_freq_words(df=None, label=none, nlp=none,
                               word_no=50, single_word=True)
    Docstring:  Return list of tuple containing words and its frequency.
    Parameters: df: pandas dataframe,
                label: int,
                nlp: spacy object
                word_no: int, total number of most used words to return
                sigle_word: boolean, return indiviual word or concat. words
    '''
    # concat review texts as one string
    txt = (df[df['label'] == label]['review']).str.cat(sep= ' ')
    
   # clean texts
    words = clean_text(txt, nlp, single_word=single_word)
    
    # calculate word frequency
    word_freq = Counter(words)
    common_words = word_freq.most_common(word_no)
    return common_words


def plot_wcloud(lst, spt1, spt2, word_type):
    '''
    Signature:  plot_wcloud(lst=None, spt1=None, spt2=None, word_type)
    Docstring:  Return a plotly express figure of word cloud image.
    Parameters: lst: list of tuple of word and count. 
                spt1: string, subplot one title.
                spt2: string, subplot two title.
                word_type: string, subplots title word
    '''
    # create figure with 2 subplot
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.1,
        subplot_titles=(f'<i><b>{spt1}</b></i> Review Top 50 {word_type}',
                        f'<i><b>{spt2}</b></i> Review Top 50 {word_type}'))

    for i, j  in enumerate(lst):
        # create WordCoud object
        wc = WordCloud(background_color="white")
        # generate word cloud
        wc.generate_from_frequencies(dict(j))
        
        fig.add_trace(go.Image(z=wc),row=i+1, col=1)
        
    # set subplot title font size
    for annotation in fig['layout']['annotations']: 
        annotation['font']={'size':24}
    
    fig.update_layout(width=550, height=700, hovermode=False, autosize=False)
    fig.update_xaxes(showticklabels=False, showline=True, linewidth=1, linecolor='grey', mirror=True)
    fig.update_yaxes(showticklabels=False, showline=True, linewidth=1, linecolor='grey', mirror=True)
    
    fig.show(scale=10)
    

def get_clsrpt_confmat(label, pred, df=None):
    '''
    Signature:  eval_model(df=None, label=None, pred=None)
    Docstring:  Return df containing classification report and plotly figure
                , confusion matrix.
    Parameters: df: dataframe
                label: list/arrary, the label column
                pred: list/array, model prediction
    '''
    # print classificatoin report
    cls_rpt = pd.DataFrame.from_dict(classification_report(
                  label, pred, output_dict=True)).T
    display(cls_rpt)
    
    # plot confusion matrix
    z = np.round(confusion_matrix(label, pred, normalize='true'), 3)
    x = ['neg pred', 'pos pred']
    y = ['neg actual', 'pos actual']

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')

    # set font size of z values
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 16

    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                      height= 500, width=500)

    # move xaxis label to bottom
    fig.layout.xaxis.update(side='bottom')
    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=16),
                        x=0.5,
                        y=-.15,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper",
                       ))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=16),
                        x=-0.15,
                        y=0.5,
                        showarrow=False,
                        text="Actual value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"
                       ))
    
    return cls_rpt, fig
    
def eval_train_test(model, x_tr, y_tr, x_te, y_te):
    '''
    Signature:   eval_train_test(model=None, x_tr=None, y_tr=None, x_te=None, y_te=None)
    Docstring:   Return the model accuracy score of the train and test data.
    Parameters:  model: Keras Sequential object
                 x_tr: X_train, after text sequencing
                 y_tr: y_train
                 x_te: X_test
                 y_te: y_test
    '''
    score_train = np.round(model.evaluate(x_tr, y_tr), 4)
    score_test = np.round(model.evaluate(x_te, y_te), 4)
    percent_diff = np.round((score_test[1] - score_train[1]) / score_train[1], 2)
    print('\n')
    print(f'Train Accuracy: \033[1m {score_train[1]}\033[0m')
    print(f'Test Accuracy: \033[1m {score_test[1]}\033[0m')
    print('\n')
    print(f'Percent difference: \033[1m {percent_diff}\033[0m')
    
    return score_train, score_test

def print_mdl_eval(train_eval, test_eval):
    '''
    Signature    print_mdl_eval(train_eval=None, test_val=None)
    Doctring     print out train and test evaluation accuracy score from saved eval.
    Parameters   train_eval: train eval dict.
                 test_eval: test eval dict.
    '''
    percent_diff = np.round((test_eval[1] - train_eval[1]) / train_eval[1], 2)
    print(f'Train Accuracy: \033[1m {train_eval[1]}\033[0m')
    print(f'Test Accuracy: \033[1m {test_eval[1]}\033[0m')
    print('\n')
    print(f'Percent difference: \033[1m {percent_diff}\033[0m')
    
    return None
    
def plot_acc_los_epoch(history,epoch, type_=True):
    '''
    Signature:   plot_acc_los_epoch(history=None, epoch=None, type_=True)
    Docstring:   Return a plotly express plot.
    Parameters:  history: fitted model.
                 epoch: number of epoch
                 type_: boolean, True for not from saved history and vice versa
    '''
    fig = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0.1,
            subplot_titles=(f'<i><b>Accuracy Score:</b></i> Train vs Validation',
                            f'<i><b>Loss Score:</b></i> Train vs Validation')
    )
    
    # check is history is from load file or current model
    if type_:
        hist = history.history
    else:
        hist = history
    
    fig.add_trace(go.Scatter(x=list(range(0, epoch)), 
                                 y=hist['acc'], name='train acc') 
                                ,row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(0, epoch)), 
                                 y=hist['val_acc'], name='val acc') 
                                ,row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(0, epoch)), 
                                 y=hist['loss'], name='train loss') 
                                ,row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(0, epoch)), 
                                 y=hist['val_loss'], name='val loss') 
                                ,row=2, col=1)


    fig.update_layout(width=700, height=900)
    fig.update_xaxes(title='Number of Epochs')
    fig['layout']['yaxis']['title']='Accuracy'
    fig['layout']['yaxis2']['title']='Loss'

    return fig
    

def plot_precision(df, label, label_type, index, color_='red'):
    '''
    Signature   plot_precision(df=None, label=None, color=None, label_type=None)
    Docstring   return plotly bar chart
    Parameters  df: pandas df
                label: str, model label
                label_type: str, Positive or Negative
                index: int, highest precision score index
                color: str
    '''
    colors = [color_,] * 5
    colors[index] = 'gold'
    # plot precision score
    fig = go.Figure(data=[
        go.Bar(x=df['model'],
            y=df[label],
            marker_color=colors,
            opacity=0.7)
        ])
    #customize layout
    fig.update_layout(barmode='group',
        title=f'<b>True {label_type} Precision Score by Model</b>',
        title_font={'size':24},
        title_x=0.5)
    # customize x & y axes
    fig.update_xaxes(title='<b>Model</b>', title_font={'size':16})
    fig.update_yaxes(title='<b>Precisioin Score</b>', title_font={'size':16})
    
    return fig


###################################################################################
#########################  Juputer Dash Section Functions #########################
###################################################################################


# import libraries
import base64
from pickle import load
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# assign model save file to variable
model_name = 'dump/model_glv_2.h5'
# load saved tokenizer
# tokenizer = load(open('dump/tokenizer', 'rb'))

def encode_image(img_file):
    '''
    Signature:   encode_image(img_file=None)
    Docstring:   Return decoded png image file
    Parameter:   img_file: str, path/file name
    '''
    encoded = base64.b64encode(open(img_file, 'rb').read())
    return 'data:img/;base64,{}'.format(encoded.decode())


# parse dataframe into hthml table cells
def get_df(selected_df):
    '''
    Signature:   get_df(selected_df=None)
    Docstring:   Return dash table converted from dataframe
    Parameter:   selected_df: panda dataframe
    '''
    return html.Div([
        dash_table.DataTable(
            data = selected_df.to_dict('records'),
            columns = [{'name': i, 'id':i} for i in selected_df.columns],
            page_size=20,
            sort_action ='native',
            style_table={#'height': '400px', 'width': '700px',
                         'overflowY': 'auto'},
            style_header={'backgroundColor': '#C9D7BA', 'fontWeight': 'bold'},
            style_cell={'textAlign':'left','whiteSpace':'normal',
                        'height':'auto'},
            style_data_conditional=[{'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'}]
            )
        ], style={'font-family': 'Verdana', 'font-size':'1em'}) 

def create_df(selected_fn):
    '''
    Signature:   create_df(selected_fn=None)
    Doctring:    Return dataframe with new label, null values removal
    Parameter:   selected_fn: strng
    '''
    # load file
    df = pd.read_csv(selected_fn, index_col=0)
    
    # create label from overall rating & drop overall column
    rating_dict = {1:0, 2:0, 3:0, 4:1, 5:1 }
    df['label'] = df['overall'].map(rating_dict)
    df.drop(columns=['overall'], axis=1, inplace=True)
    
    # drop rows with no review texts
    df.dropna(inplace=True)
    
    # check review text column that contain no texts and only spaces
    no_texts = []
    for ind, asin, review, label in df.itertuples():
        if type(review) == str:
            if review.isspace():
                no_texts.append(ind)
    if len(no_texts) > 0:
        df.dropna(no_texts, inplace=True)
    
    return df

# parse model summary to dash table
def parse_summ():
    '''
    Signature:   parse_summ(model=None)
    Doctring:    convert model summary data into dash table
    Parameter:   
    '''
    # load model here else get tensorflow graph object error
    model = load_model(model_name)

    # convert model.summary() into string
    strlist = []
    model.summary(print_fn=lambda x: strlist.append(x))
    
    # remove summary formating, underscores and equal signs 
    tmp = []
    for i in strlist:
        if not i.startswith('_') and not i.startswith('='):
            tmp.append(i)    
        
    return html.Table(
        # body
        [html.Tr([html.Td(i[0:29]), html.Td(i[29:50]), html.Td(i[50:len(i)])])
         if not i.startswith('Non-train') else html.Tr(html.Td(i[0:35])) 
         for i in tmp[0:len(tmp)]
        ]
        )

def dash_plot_wc(selected_df, label, d_dict, pos_dict, neg_dict):
    '''
    Signature:   plot_wc(selected_df=None, label=None, d_dict=None)
    Doctring:    Return plotly figure
    Parameter:   selected_df: string, dataframe variable name
                 label: interger, 0 or 1
                 d_dict: dictionary
                 pos_dict: dictionary
                 neg_dict: dictionary
    '''
    df = d_dict.get(selected_df)
    fig = go.Figure()
    if label == 0:
        wc_df = neg_dict.get(selected_df)
        # create WordCoud object
        wc = WordCloud(background_color="white")
        # generate word cloud
        wc.generate_from_frequencies(dict(wc_df))
        title = 'Top 50 Words from Negative Reviews'
    else:
        wc_df = pos_dict.get(selected_df)
        # create WordCoud object
        wc = WordCloud(background_color="white")
        # generate word cloud
        wc.generate_from_frequencies(dict(wc_df))
        title = 'Top 50 Words from Positive Reviews'
        
    fig.add_trace(go.Image(z=wc))
  
    fig.update_layout(title={'text':title, 'x':0.5, 'xanchor':'center'},
        width=495, height=350, hovermode=False,
        autosize=False, margin=dict(l=0, r=0, b=1, t=50, pad=4))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig


def get_pred(df, nlp, tokenizer):
    '''
    Signature:   get_pred(df=None, nlp=Noned tokenzier=None)
    Docstring:   Return model predictions
    Parameter:   df:  pandas df
                 nlp: Spacy library object
                 tokenzier: tokernizer object
                
    '''
    model = load_model(model_name)
    
    # load model here else get tensorflow graph object error
    X = []

    # convert series to list
    sentences = list(df['review'])

   # remove html tags and punct. stop-words are not removed, keeping case
    for sen in sentences:
        X.append(clean_text(sen, nlp, single_word=False))
        
    # tokenizer texts
    X_ts = tokenizer.texts_to_sequences(X)

    # add padding
    X_ts_pad = pad_sequences(X_ts, padding='post', maxlen=100)

    # make prediction
    pred = model.predict(X_ts_pad)
       
    # convert pred percentage to label
    pred_label = [0 if np.round(x, 2) < 0.5 else 1 for x in pred]

    return pred_label


def get_single_pred(review, nlp, tokenizer):
    '''
    Signature:   get_single_pred(review=None, nlp=None, tokenzier=None)
    Docstring:   Return model predictions
    Parameter:   reiveiw: str
                 nlp: Spacy libraby object
                 tokenzier: tokenizer object

    '''
    # load model here else get tensorflow graph object error
    model = load_model(model_name)
    
    # remove html tags and punct.
    rev = clean_text(review, nlp, single_word=False)
    
    # tokenizer texts
    rev_ts = tokenizer.texts_to_sequences([rev])
    
    # add padding
    rev_ts_pad = pad_sequences(rev_ts, maxlen=100, padding='post')
    
    # make prediction
    pred = model.predict(rev_ts_pad)
    
    # convert pred percentage to label
    pred_label = [0 if np.round(x,0) < 0.5 else 1 for x in pred]    
            
    return (pred_label)


def plot_dash_precision(df, pred):
    '''
    Signature:   plot_precision(df=None, pred=None)
    Docstring:   Return plot bar chart with model precision score
    Parameters:  df: panda dataframe
                 pred: list/arrary, predicted vallues
    '''
    precision = pd.DataFrame.from_dict(classification_report(df['label'],
        pred, output_dict=True)).T
    precision_t = pd.DataFrame({'model':'Neural Network',
        '0':precision.iloc[0,0], '1':precision.iloc[1,0]}, index=[0])
        
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=precision_t.model,
        y=precision_t['0'],
        name="negative (0)",
        marker_color='#F47B4F',
        text=round(precision_t['0'],2),
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        x=precision_t.model,
        y=precision_t['1'],
        name="positive (1)",
        marker_color='#6DADF6',
        text=round(precision_t['1'],2),
        textposition='auto',
    ))
        
    # set the text on the bar
    fig.update_traces(textfont_size=20)
        
    # update layout
    fig.update_layout(
        height=430,
        width=340,
        margin=dict(l=40, r=5, t=20, b=20),
        legend=dict(orientation='h',
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1),
    )
        
    # updata axes
    fig.update_xaxes(title='<b>Model</b>', title_font={'size':14})
    fig.update_yaxes(title='<b>Value</b>', title_font={'size':14})
    
    return fig

def append_pred(df, pred):
    '''
    Sgnature:    plot_precision(df=None, pred=None)
    Docstring:   return panda dataframe
    Parameters:  df: panda dataframe
                 pred: list/arrary, predicted vallues
    '''
    
    # create a copy of the df to append prediction
    df_comp = df.copy()
    if (df_comp.columns =='asin').sum() > 0:
        # drop asin column
        df_comp.drop(columns=['asin'], inplace=True)
        # add pred to df
        df_comp['pred'] = pred
    else:
        # add pred to df
        df_comp['pred'] = pred
    
    return df_comp


###################################################################################
################################## Not used #######################################
###################################################################################
# def plot_confmatx(label, pred, df=None):
#     '''
#     Signature:  plot_confmatx(label=None, pred=None, df=None)
#     Docstring:  Return plotly figure of sklearn confusion matrix
#     Parameters: label: list/arrary, the label column
#                 pred: list/array, model prediction
#                 df: dataframe
#     '''
#      # plot confusion matrix
#     z = np.round(confusion_matrix(label, pred, normalize='true'), 3)
#     x = ['neg pred', 'pos pred']
#     y = ['neg actual', 'pos actual']

#     # set up figure 
#     fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')

#     # set font size of z values
#     for i in range(len(fig.layout.annotations)):
#         fig.layout.annotations[i].font.size = 16

#     fig.update_layout(margin=dict(l=25, r=5, t=2, b=45), height= 340, width=340,
#                      hovermode=False)

#     # move xaxis label to bottom
#     fig.layout.xaxis.update(side='bottom')
#     # add custom xaxis title
#     fig.add_annotation(dict(font=dict(color="black",size=13),
#                         x=0.5,
#                         y=-0.15,
#                         showarrow=False,
#                         text="Predicted Value",
#                         xref="paper",
#                         yref="paper",
#                        ))

#     # add custom yaxis title
#     fig.add_annotation(dict(font=dict(color="black",size=13),
#                         x=-0.15,
#                         y=0.5,
#                         showarrow=False,
#                         text="Actual Value",
#                         textangle=-90,
#                         xref="paper",
#                         yref="paper"
#                        ))
    
#     return fig

# def parse_contents(contents, filename):
#     content_type, content_string = contents.split(',')

#     decoded = base64.b64decode(content_string)
#     try:
#         if 'csv' in filename:
#             # Assume that the user uploaded a CSV file
#             df = pd.read_csv(
#                 io.StringIO(decoded.decode('utf-8')))
#         elif 'xls' in filename:
#             # Assume that the user uploaded an excel file
#             df = pd.read_excel(io.BytesIO(decoded))
#     except Exception as e:
#         print(e)
#         return html.Div([
#             'There was an error processing this file.'
#         ])

#     return html.Div([
#         dash_table.DataTable(id='user-table',
#             data=df.to_dict('records'),
#             columns=[{'name': i, 'id': i} for i in df.columns],
#             page_size=20,
#             sort_action ='native',
#             style_table={'overflowy':'auto'},
#             style_header=css_tbl_header,
#             style_cell=css_tbl_cell,
#             style_data_conditional=css_tbl_condl
#             )
#         ], style={'font-family': 'Verdana', 'font-size':'1.4em'})

# def conv_df_dash_tbl(df):
#     # convert df to dash table
#     act_pred = html.Div([
#         dash_table.DataTable(
#             data = df.to_dict('records'),
#             columns = [{'name': i, 'id': i} for i in df_comp.columns],
#             page_size=10,
#             sort_action ='native',
#             style_header=css_tbl_header,
#             style_cell=css_tbl_cell,
#             style_data_conditional=css_tbl_condl
#         )
#     ], style=css_act_pred_output)
    
#     return act_pred