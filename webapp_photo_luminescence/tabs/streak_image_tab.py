# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import base64
import typing as t

import dash
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from webapp_photo_luminescence import (
    uploadbar,
    sidebar
)
from webapp_photo_luminescence.tabs import common


graph = common.create_graph(id="streak-image-graph")
img_download = dcc.Download("img-download")
img_download_button = dbc.Button(
    [
        "Download Image",
        img_download
    ],
    color="primary",
    className="mt-2",
    id="img-download-button"
)
options = common.create_options_layout(
    options_components=None,
    download_components=[
        img_download_button
    ]
)
layout = common.create_layout(graph, options, None)


@dash.callback(
    dash.Output(graph, "figure"),
    dash.Input(uploadbar.uploaded_files_dropdown, "value"),
    dash.State(uploadbar.uploaded_files_store, "data"),
    dash.Input(sidebar.filter_radio_items, "value"),
    prevent_initial_call=True
)
def update_streak_image(
    selected_items: list[str] | None,
    uploaded_files: dict[str, str] | None,
    filter_type: str | None,
) -> go.Figure:
    if not uploaded_files or not selected_items:
        raise dash.exceptions.PreventUpdate
    item_to_data = {
        item: uploadbar.load_pldata(uploaded_files[item], filter_type)
        for item in filter(uploaded_files.__contains__, selected_items)
    }
    fig = go.Figure(
        [
            go.Surface(
                z=data.streak_image,
                x=data.wavelength,
                y=data.time,
                name=key,
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
            highlightcolor="cyan",
        ),
        contours_y=dict(
            show=True,
            usecolormap=True,
            highlightcolor="cyan",
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
    dash.Input(uploadbar.uploaded_files_dropdown, "value")
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
    dash.State(uploadbar.uploaded_files_dropdown, "value"),
    dash.State(uploadbar.uploaded_files_store, "data"),
    dash.State(sidebar.filter_radio_items, "value"),
    prevent_initial_call=True
)
def update_download_content(
    n_clicks: int | None,
    selected_items: list[str] | None,
    uploaded_files: dict[str, str] | None,
    filter_type: str | None,
) -> dict[str, t.Any]:
    if not selected_items or not uploaded_files:
        raise dash.exceptions.PreventUpdate
    item = selected_items[0]
    if item not in uploaded_files:
        raise dash.exceptions.PreventUpdate
    data = uploadbar.load_pldata(
        uploaded_files[item],
        filter_type
    )
    return dict(
        filename=(filter_type+"-" if filter_type else "") + item,
        content=base64.b64encode(data.to_raw_binary()).decode(),
        type="application/octet-stream",
        base64=True,
    )
