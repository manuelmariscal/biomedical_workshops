import os
import io
import base64
import threading

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

from src.tracking import Tracking
from src.analysis import MovementAnalysis
import corr

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Biomedical Workshops Dashboard"),
    html.Div([
        html.Label("Mode"),
        dcc.Dropdown(
            id='mode',
            options=[
                {'label': 'Tracking', 'value': 'track'},
                {'label': 'Analyze', 'value': 'analyze'},
                {'label': 'Correlation', 'value': 'corr'}
            ],
            value='track'
        )
    ], style={'width': '200px'}),
    html.Div([
        html.Label("Data folder"),
        dcc.Input(id='data-folder', value='data', type='text')
    ]),
    html.Div([
        html.Label("Frames"),
        dcc.Input(id='frames', value='', type='number')
    ]),
    html.Button('Run', id='run-btn', n_clicks=0),
    html.Div(id='output')
])


def run_tracking(source, max_frames):
    tracker = Tracking(source=source)
    tracker.run_analysis(max_frames=max_frames)

def run_analyze(data_folder, max_frames):
    analysis = MovementAnalysis(data_folder=data_folder, max_frames=max_frames)
    analysis.run_analysis()

def run_corr(data_folder, max_frames):
    # Use functions directly to obtain matplotlib figures
    datasets = corr.load_datasets(data_folder)
    if not datasets:
        return None
    if max_frames is not None:
        datasets = [(n, d.iloc[:max_frames]) for n, d in datasets]
    datasets = corr.trim_datasets_to_smallest(datasets)
    fig_corr = corr.plot_correlation_matrices(datasets)
    avg = corr.compute_average_keypoint_correlations(datasets)
    fig_heat = corr.plot_skeleton_heatmap(avg)
    buf1 = io.BytesIO()
    fig_corr.savefig(buf1, format='png')
    buf1.seek(0)
    buf2 = io.BytesIO()
    fig_heat.savefig(buf2, format='png')
    buf2.seek(0)
    img1 = base64.b64encode(buf1.read()).decode('utf-8')
    img2 = base64.b64encode(buf2.read()).decode('utf-8')
    return img1, img2


@app.callback(Output('output', 'children'),
              Input('run-btn', 'n_clicks'),
              State('mode', 'value'),
              State('data-folder', 'value'),
              State('frames', 'value'))
def run_action(n_clicks, mode, data_folder, frames):
    if n_clicks == 0:
        return ''
    max_frames = int(frames) if frames not in (None, '', 'None') else None
    if mode == 'track':
        threading.Thread(target=run_tracking, args=('0', max_frames)).start()
        return html.Div('Running tracking... Check the opened window to stop.')
    elif mode == 'analyze':
        threading.Thread(target=run_analyze, args=(data_folder, max_frames)).start()
        return html.Div('Running analysis... check console for results.')
    else:
        result = run_corr(data_folder, max_frames)
        if result is None:
            return html.Div('No datasets found.')
        img1, img2 = result
        return html.Div([
            html.Img(src='data:image/png;base64,{}'.format(img1)),
            html.Br(),
            html.Img(src='data:image/png;base64,{}'.format(img2))
        ])


if __name__ == '__main__':
    app.run_server(debug=False)
