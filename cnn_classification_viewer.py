#!/usr/bin/env python
# coding: utf-8
__author__ = "Cédric Walker"
"""
    Dash Server for visualizing the decision boundrary of a DenseNet (or general CNN with adapdation) classifier.
    Several parts regarding the DB handling are adapted from Andrew Janowczyk, github.com/choosehappy
"""
# In[5]:

import re
import matplotlib.cm
import argparse
import flask
import umap
import tables
import numpy as np
import pandas as pd
from textwrap import dedent as d
from pathlib import Path
# import jupyterlab_dash
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import DenseNet
from torch.utils.data.dataloader import default_collate

import albumentations as albmt
from albumentations.pytorch import ToTensor

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output



parser = argparse.ArgumentParser(description='Run a server for visualization of a CNN classifier')

parser.add_argument('--load_from_file', '-l', action='store_true', default=False, help='Load the embedding from a csv file. Does not compute the embedding',)
parser.add_argument('--target_class', '-t', default=None, help='Target Label, if the classifier was trained in one vs all fashion',)
parser.add_argument('--port', '-p', help='Server Port', default=8050, type = int)

parser.add_argument('database', help='Database containing image patches, labels ...',)
parser.add_argument('filename', help='Creates a csv file of the embedding')
parser.add_argument('model', help='Saved torch model dict, and architecture')

arguments = parser.parse_args()


file_name = arguments.filename
use_existing = arguments.load_from_file
target_class = arguments.target_class
use_port = arguments.port
db_path = arguments.database
model_path = arguments.model

batch_size = 32
patch_size = 224


server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)


# depending on how many colors needed taking either the tab 10 or tab20 pallete
def color_pallete(n):
    num = int(min(np.ceil(5/10), 2)*10)
    colors = matplotlib.cm.get_cmap(f'tab{num}').colors
    return ['#%02x%02x%02x' % tuple(np.array(np.array(i) * 255,dtype=np.uint8)) for i in colors]


class Dataset(object):
    "Dabase handler for torch.utils.DataLoader written by Andrew Janowczyk"

    def __init__(self, fname, img_transform=None):
        self.fname = fname
        self.img_transform = img_transform

        with tables.open_file(self.fname, 'r') as db:
            self.nitems = db.root.imgs.shape[0]

        self.imgs = None
        self.filenames = None
        self.label = None

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multithreading so doing here. need to do it everytime, otherwise hdf5 crashes

        with tables.open_file(self.fname, 'r') as db:
            self.imgs = db.root.imgs
            self.filenames = db.root.filenames
            self.label = db.root.labels
            # get the requested image and mask from the pytable
            img = self.imgs[index, :, :, :]
            fname = self.filenames[index]
            label = self.label[index]

        img_new = img
        if self.img_transform:
            img_new = self.img_transform(image=img)['image']

        return img_new, img, label, fname

    def __len__(self):
        return self.nitems


# In[7]:
def get_dataloader(batch_size, patch_size, db_path):
    # +

    def id_collate(batch):
        new_batch = []
        ids = []
        for _batch in batch:
            new_batch.append(_batch[:-1])
            ids.append(_batch[-1])
        return default_collate(new_batch), ids
    # +
    img_transform = albmt.Compose([
           albmt.RandomSizedCrop((patch_size, patch_size), patch_size, patch_size),
           ToTensor()
        ])

    if db_path[0] != '/':
        db_path = f'./{db_path}'

    # more workers do not seem no improve perfomance
    dataset = Dataset(db_path, img_transform=img_transform)
    dataLoader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=id_collate)

    print(f"dataset size:\t{len(dataset)}")
    # -
    return dataLoader, dataset


def load_model(model_path):
    device = torch.device('cuda')

    checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    # load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666

    model = DenseNet(growth_rate=checkpoint["growth_rate"],
                     block_config=checkpoint["block_config"],
                     num_init_features=checkpoint["num_init_features"],
                     bn_size=checkpoint["bn_size"],
                     drop_rate=checkpoint["drop_rate"],
                     num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_dict"])

    print(
        f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

    model.eval()
    return model, device

def load_embedding(dataLoader, model, device):

    out = {}
    def hook(module, input, output):
        out[module] = input[0]
    # works for torchvision.models.DenseNet, register_forward_hook on last layer before classifier.
    model.classifier.register_forward_hook(hook)
    # +
    # all_preds=[]
    all_last_layer = []
    all_fnames = []
    all_labels = []
    all_predictions = []
    # cmatrix = np.zeros((checkpoint['num_classes'], checkpoint['num_classes']))

    # add notification sstuff? (X, xorig, label), fname = next(iter(dataLoader[phase]))
    for (X, xorig, label), fname in dataLoader:
        X = X.to(device)
        label_pred = model(X)
        last_layer = out[model.classifier].detach().cpu().numpy()
        all_last_layer.append(last_layer)
    #    yflat = label.numpy() == target_class
        all_labels.extend(label.numpy())
        pred_class = np.argmax(label_pred.detach().cpu().numpy(), axis=1)
        all_predictions.extend(pred_class)
        all_fnames.extend([Path(fn.decode()).name for fn in fname])
    #     cmatrix = cmatrix + \
    #         confusion_matrix(yflat, pred_class, labels=range(
    #             checkpoint['num_classes']))
    # print(cmatrix)
    # acc = (cmatrix/cmatrix.sum()).trace()
    # print(acc)
    features_hists = np.vstack(all_last_layer)
    # -
    # +
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=3)
    embedding = reducer.fit_transform(features_hists)
    return embedding, all_labels, all_predictions, dataset, all_fnames


def create_confusion_map(embedding_a, target_class):
    # n_classes = len(embedding_a.Prediction.unique())
    pred = embedding_a.Prediction.values
    label = embedding_a.Label.values
    label = (label)
    label = np.array(label, dtype=np.uint8)
    conf = [f'{label[i]}{pred[i]}' for i in range(len(label))]
    return embedding_a.assign(Confusion=conf)


text_style = dict(color='#444', fontFamily='sans-serif', fontWeight=300)
dataLoader, dataset = get_dataloader(batch_size, patch_size, db_path)


if use_existing is True:
    embedding_a = pd.read_csv(file_name)
else:
    # model is not saved to variable to enable garbage collector to clean it after it is not used anymore
    embedding, all_labels, all_predictions, dataset, fnames = load_embedding(
        dataLoader, *load_model(model_path))
    embedding_a = pd.DataFrame({"x": embedding[:, 0],
                                "y": embedding[:, 1],
                                "z": embedding[:, 2],
                                "Label": all_labels,
                                "Prediction": all_predictions,
                                "index": [*range(len(all_labels))],
                                "Slide": [i[:i.find(re.findall('[A-Za-z\.\s\_]*$', i)[0])] for i in fnames]})
    embedding_a.to_csv(file_name)
embedding_a = create_confusion_map(embedding_a, target_class)


def plotly_figure(value, plot_type='2D'):
    colors = color_pallete(len(embedding_a[value].unique()))
    label_to_type = {'2D': 'scattergl', '3D': 'scatter3d'}
    type_to_size = {'2D': 15, '3D': 2.5}
    linesize = {'2D': 0.5, '3D': 0}
    return {
        'data': [dict(
            x=embedding_a[embedding_a[value] == target]['x'],
            y=embedding_a[embedding_a[value] == target]['y'],
            z=embedding_a[embedding_a[value] == target]['z'],
            text=embedding_a[embedding_a[value] == target]['index'],
            index=embedding_a[embedding_a[value] == target]['index'],
            customdata=embedding_a[embedding_a[value] == target]['index'],
            mode='markers',
            type=label_to_type[plot_type],
            name=f'{target}',
            marker={
                'size': type_to_size[plot_type],
                'opacity': 0.5,
                'color': colors[i],
                'line': {'width': linesize[plot_type], 'color': 'white'}
                }
            ) for i, target in enumerate(sorted(embedding_a[value].unique()))],
        'layout': dict(
            xaxis={
                'title': "x",
                'type': 'linear'
                },
            yaxis={
                'title': "y",
                'type': 'linear'
                },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=750,
            width=850,
            hovermode='closest',
            clickmode='event+select',

            uirevision='no reset of zoom',
            legend={'itemsizing': 'constant'}
            )

        }


app.layout = html.Div([
        html.H2('CNN Classification Viewer', style=text_style),
        # dcc.Input(id='predictor', placeholder='box', value=''),
        html.Div([
            html.Div([
                dcc.RadioItems(
                    id='color-plot1',
                    options=[{'label': i, 'value': i}
                             for i in ['Label', 'Prediction', 'Confusion', 'Slide']],
                    value='Label',
                    labelStyle={}
                ),
                dcc.RadioItems(
                    id='plot-type',
                    options=[{'label': i, 'value': i}
                             for i in ['2D', '3D']],
                    value='2D',)], style={'width': '49%', 'display': 'inline'}),  # , 'float': 'left', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='plot1', figure=plotly_figure('Label'))
                        ],  style={'float': 'left', 'display': 'inline-block'}),
            html.Div([
                html.Div([html.Img(id='image', width=patch_size, height=patch_size)], style={'display': 'inline-block'}),
                dcc.Markdown(d("""
                    **Image Properties**

                    """)),
                html.Pre(id='hover-data'),
                dcc.Markdown(d("""
                    **Frequency in selected**

                    """)),
                html.Pre(id='selected-data')
            ], style={'float': 'left', 'display': 'inline-block'}, className='three columns'),

        ])], style={'width': '65%'})


@app.callback(
    Output('selected-data', 'children'),
    [Input('plot1', 'selectedData')])
def display_selected_data(selectedData):
    text = ""
    if selectedData is not None:
        indices = pd.DataFrame.from_dict(selectedData['points'])['customdata'].values
        frame = embedding_a[embedding_a['index'].isin(indices)]
        conf_freq = frame['Confusion'].value_counts()
        df2 = pd.DataFrame({'Confusion': conf_freq._index, 'Label': conf_freq.values})
        df2['perc'] = df2['Label'] / df2['Label'].sum() *100
        df2['perc'] = df2['Percentage'].apply(lambda a: '%.1f%%' % a)
        text = "Pred:\n" + frame['Prediction'].value_counts().to_string() + \
            "\nLabel:\n" + frame['Label'].value_counts().to_string() + \
            "\nConf:\n" + df2.to_string()
    return text


@app.callback(
    Output('hover-data', 'children'),
    [Input('plot1', 'hoverData')])
def display_hover_data(hoverData):
    # ind = int(hoverData['points'][0]["pointIndex"])
    # return f"Index: {embedding_a.iloc[ind]['index']}"
    text = "\n\n\n\n\n"
    if hoverData is not None:
        ind = int(hoverData['points'][0]["customdata"])
        text = f"Index: {embedding_a.iloc[ind]['index']}\nLabel: {embedding_a.iloc[ind]['Label']}\nPrediction: {embedding_a.iloc[ind]['Prediction']}\nConfusion: {embedding_a.iloc[ind]['Confusion']}\nFile: {embedding_a.iloc[ind]['Slide']}"
    return text


@app.callback(Output('image', 'src'), [Input('plot1', 'hoverData')])
def serve_image(hoverData):
    if hoverData is not None:
        import base64
        from io import BytesIO
        from PIL import Image
        ind = hoverData['points'][0]["customdata"]
        _, img, _, _ = dataset[ind]
        pil_img = Image.fromarray(img)
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
        return 'data:image/png;base64,{}'.format(new_image_string)
    else:
        return "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="


@app.callback(Output('plot1', 'figure'),
              [Input('color-plot1', 'value'), Input('plot-type', 'value')])
def serve_plot(value, plot_type):
    print(plot_type)
    return plotly_figure(value, plot_type)


# viewer = jupyterlab_dash.AppViewer()
# viewer.show(app)
app.run_server(port=use_port)

