# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import base64
import os
import typing as t

import dash
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from webapp_photo_luminescence.tabs import (
    common,
    upload
)


img_download = dcc.Download("img-download")
img_download_button = dbc.Button(
    [
        "Download Image",
        img_download
    ],
    id="img-download-button",
    color="primary",
    className="mt-2",
    disabled=True
)
graph = common.create_graph(id="streak-image-graph")
options = common.create_options_layout(
    options_components=None,
    download_components=[
        img_download_button
    ]
)
layout = common.create_layout(graph, options)


@dash.callback(
    dash.Output(graph, "figure"),
    dash.Input(upload.files_dropdown, "value"),
    dash.State(upload.upload_dir_store, "data"),
    dash.Input(upload.filter_radio_items, "value"),
    prevent_initial_call=True
)
def update_streak_image(
    selected_items: list[str] | None,
    upload_dir: str | None,
    filter_type: str | None
) -> go.Figure:
    assert upload_dir is not None
    assert upload_dir.startswith(upload.UPLOAD_BASEDIR)
    if not selected_items:
        return go.Figure()
    filepaths = [os.path.join(upload_dir, item) for item in selected_items]
    item_to_data = {
        os.path.split(filepath)[-1]: upload.load_pldata(filepath, filter_type)
        for filepath in filepaths if os.path.exists(filepath)
    }
    fig = go.Figure(
        [
            go.Surface(
                z=data.streak_image,
                x=data.wavelength,
                y=data.time,
                name=key
            ) for key, data in item_to_data.items()
        ]
    )
    fig.update_traces(
        hovertemplate=""
        "Wavelength: %{x:.2f} nm<br>"
        "Time: %{y:.2f} ns<br>"
        "Intensity: %{z:d}<br>"
        "<extra></extra>"
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Time (ns)",
            zaxis_title="Intensity (arb.units)",
            aspectmode="auto"
        ),
        margin=dict(l=40, r=40, b=20, t=10),
    )
    fig.update_traces(
        contours_x=dict(
            show=True,
            usecolormap=True,
            highlightcolor="cyan"
        ),
        contours_y=dict(
            show=True,
            usecolormap=True,
            highlightcolor="cyan"
        ),
        contours_z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="limegreen",
            project_z=True
        )
    )
    return fig


@dash.callback(
    dash.Output(img_download_button, "disabled"),
    dash.Input(upload.files_dropdown, "value")
)
def update_download_button_ability(
    selected_items: list[str] | None
) -> bool:
    if not selected_items:
        return True
    return bool(len(selected_items) != 1)


@dash.callback(
    dash.Output(img_download, "data"),
    dash.Input(img_download_button, "n_clicks"),
    dash.State(upload.files_dropdown, "value"),
    dash.State(upload.upload_dir_store, "data"),
    dash.State(upload.filter_radio_items, "value"),
    prevent_initial_call=True
)
def update_download_content(
    n_clicks: int | None,
    selected_items: list[str] | None,
    upload_dir: str | None,
    filter_type: str | None
) -> dict[str, t.Any]:
    assert upload_dir is not None
    assert upload_dir.startswith(upload.UPLOAD_BASEDIR)
    if not selected_items:
        raise dash.exceptions.PreventUpdate
    item = selected_items[0]
    filepath = os.path.join(upload_dir, item)
    if not os.path.exists(filepath):
        raise dash.exceptions.PreventUpdate
    data = upload.load_pldata(filepath, filter_type)
    return dict(
        filename=(filter_type+"-" if filter_type else "") + item,
        content=base64.b64encode(data.to_raw_binary()).decode(),
        type="application/octet-stream",
        base64=True
    )
