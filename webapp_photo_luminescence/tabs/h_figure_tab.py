# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import functools
import os
import typing as t

import dash
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from tlab_analysis import photo_luminescence as pl

from webapp_photo_luminescence.tabs import (
    common,
    upload
)


graph = common.create_graph(id="h-figure-graph")
peak_vline_switch = dbc.Switch(
    id="h-peak-vline-switch",
    label="Vertical Line at Peak",
    value=False,
)
FWHM_range_switch = dbc.Switch(
    id="h-FWHM-range-switch",
    label="FWHM Range",
    value=False,
    className="",
)
download = dcc.Download(
    id="h-csv-download"
)
download_button = dbc.Button(
    [
        "Download CSV",
        download
    ],
    id="h-csv-download-button"
)
options = common.create_options_layout(
    options_components=[
        peak_vline_switch,
        FWHM_range_switch
    ],
    download_components=[
        download_button
    ]
)
table = common.create_table(id="h-table")
layout = common.create_layout(graph, options, table)


@functools.lru_cache(maxsize=8)
def load_time_resolved(
    filepath: str,
    filter_type: str | None,
) -> pl.TimeResolved[pl.Data]:
    data = upload.load_pldata(filepath, filter_type)
    tr = data.time_resolved()
    tr.df["name"] = os.path.split(filepath)[-1]
    return tr


@dash.callback(
    dash.Output(graph, "figure"),
    dash.Input(upload.files_dropdown, "value"),
    dash.State(upload.upload_dir_store, "data"),
    dash.Input(upload.filter_radio_items, "value"),
    dash.Input(peak_vline_switch, "value"),
    dash.Input(FWHM_range_switch, "value"),
    prevent_initial_call=True
)
def update_graph(
    selected_items: list[str] | None,
    upload_dir: str | None,
    filter_type: str | None,
    show_peak_vline: bool,
    show_FWHM_range: bool,
) -> go.Figure:
    assert upload_dir is not None
    assert upload_dir.startswith(upload.UPLOAD_BASEDIR)
    if not selected_items:
        return go.Figure()
    filepaths = [os.path.join(upload_dir, item) for item in selected_items]
    trs = [
        load_time_resolved(
            filepath,
            filter_type
        ) for filepath in filepaths if os.path.exists(filepath)
    ]
    df = pd.concat([tr.df for tr in trs])
    fig = px.line(
        df,
        x="wavelength",
        y="intensity",
        color="name",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    if show_peak_vline:
        for tr in trs:
            fig.add_vline(
                tr.peak_wavelength,
                annotation=dict(
                    text=f"Wavelength: {tr.peak_wavelength:.2f} nm",
                    hovertext=f"Intensity: {tr.peak_intensity:.4g}"
                )
            )
    if show_FWHM_range:
        for i, tr in enumerate(trs):
            fig.add_vrect(
                *tr.half_range,
                fillcolor=px.colors.qualitative.Set1[i],
                opacity=0.10,
                annotation=dict(
                    text=f"FWHM: {tr.FWHM:.3g} nm",
                    hovertext=""
                    f"Left: {tr.half_range[0]:.2f} nm<br>"
                    f"Right: {tr.half_range[1]:.2f} nm"
                )
            )
    fig.update_traces(
        hovertemplate=""
        "Wavelength: %{x:.2f} nm<br>"
        "Intensity: %{y:.4g}<br>"
        "<extra></extra>"
    )
    fig.update_layout(
        legend=dict(
            font=dict(size=14),
            yanchor="top",
            y=-0.1,
            xanchor="left",
            x=0
        ),
    )
    fig.update_xaxes(
        title_text="<b>Wavelength (nm)</b>"
    )
    fig.update_yaxes(
        title_text="<b>Intensity (arb. units)</b>",
        range=(0, df["intensity"].max() * 1.05)
    )
    return fig


@dash.callback(
    dash.Output(table, "data"),
    dash.Input(upload.files_dropdown, "value"),
    dash.State(upload.upload_dir_store, "data"),
    dash.Input(upload.filter_radio_items, "value"),
    prevent_initial_call=True
)
def update_table(
    selected_items: list[str] | None,
    upload_dir: str | None,
    filter_type: str | None,
) -> dict[str, t.Any] | None:
    assert upload_dir is not None
    assert upload_dir.startswith(upload.UPLOAD_BASEDIR)
    if not selected_items:
        return None
    filepaths = [os.path.join(upload_dir, item) for item in selected_items]
    trs = [
        load_time_resolved(
            filepath,
            filter_type
        ) for filepath in filepaths if os.path.exists(filepath)
    ]
    df = pd.concat([tr.df for tr in trs])
    return dict(df.to_dict("records"))


@dash.callback(
    dash.Output(download_button, "disabled"),
    dash.Input(upload.files_dropdown, "value")
)
def update_download_button_ability(selected_items: list[str] | None) -> bool:
    if selected_items is None:
        return True
    return bool(len(selected_items) != 1)


@dash.callback(
    dash.Output(download, "data"),
    dash.Input(download_button, "n_clicks"),
    dash.State(upload.files_dropdown, "value"),
    dash.State(upload.upload_dir_store, "data"),
    dash.State(upload.filter_radio_items, "value"),
    prevent_initial_call=True
)
def download_csv(
    n_clicks: int,
    selected_items: list[str] | None,
    upload_dir: str | None,
    filter_type: str | None,
) -> dict[str, t.Any]:
    assert upload_dir is not None
    assert upload_dir.startswith(upload.UPLOAD_BASEDIR)
    if not selected_items:
        raise dash.exceptions.PreventUpdate
    item = selected_items[0]
    filepath = os.path.join(upload_dir, item)
    if not os.path.exists(filepath):
        raise dash.exceptions.PreventUpdate
    tr = load_time_resolved(
        filepath,
        filter_type,
    )
    filename = "h-" \
        + (filter_type+"-" if filter_type else "") \
        + item \
        + ".csv"
    return dict(
        filename=filename,
        content=tr.df.to_csv(index=False),
        type="text/csv",
        base64=False,
    )
