# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import typing as t

import dash
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from tlab_analysis import photo_luminescence as pl

from webapp_photo_luminescence import (
    uploadbar,
    sidebar,
)
from webapp_photo_luminescence.tabs import common


graph = common.create_graph(id="h-figure-graph")
peak_vline_switch = dbc.Switch(
    id="h-peak-vline-switch",
    label="Vertical Line at Peak",
    value=True,
    className="mt-2",
)
FWHM_range_switch = dbc.Switch(
    id="h-FWHM-range-switch",
    label="FWHM Range",
    value=True,
    className="",
)
csv_download = dcc.Download(
    id="h-csv-download"
)
csv_download_button = dbc.Button(
    [
        "Download CSV",
        csv_download
    ],
    id="h-csv-download-button"
)
options = common.create_options_layout(
    options_components=[
        peak_vline_switch,
        FWHM_range_switch
    ],
    download_components=[
        csv_download_button
    ]
)
table = common.create_table(id="h-table")
layout = common.create_layout(graph, options, table)


@dash.callback(
    dash.Output(graph, "figure"),
    dash.Output(table, "data"),
    dash.Input(uploadbar.uploaded_files_dropdown, "value"),
    dash.State(uploadbar.uploaded_files_store, "data"),
    dash.Input(sidebar.filter_radio_items, "value"),
    dash.Input(peak_vline_switch, "value"),
    dash.Input(FWHM_range_switch, "value"),
    prevent_initial_call=True
)
def update_graph_and_table(
    selected_items: list[str] | None,
    uploaded_files: dict[str, str] | None,
    filter_type: str | None,
    show_peak_vline: bool,
    show_FWHM_range: bool,
) -> tuple[go.Figure, dict[str, t.Any]]:
    if not uploaded_files or not selected_items:
        return go.Figure(), dict()
    trs: list[pl.TimeResolved] = []
    for item in filter(uploaded_files.__contains__, selected_items):
        data = uploadbar.load_pldata(uploaded_files[item], filter_type)
        tr = data.time_resolved()
        tr.df["time"] = f"{tr.range[0]:.2f}"
        tr.df["FWHM"] = tr.FWHM
        tr.df["name"] = item
        trs.append(tr)
    df = pd.concat([tr.df for tr in trs])
    fig = px.line(
        df,
        x="wavelength",
        y="intensity",
        color="name",
        custom_data=["FWHM"],
        animation_frame="time"
    )
    if show_peak_vline:
        for tr in trs:
            fig.add_vline(tr.peak_wavelength)
    if show_FWHM_range:
        for i, tr in enumerate(trs):
            fig.add_vrect(
                *tr.half_range,
                fillcolor=px.colors.qualitative.Plotly[i],
                opacity=0.10
            )
    fig.update_traces(
        hovertemplate=""
        "Wavelength: %{x:.2f} nm<br>"
        "Intensity: %{y:d}<br>"
        "FWHM: %{customdata[0]:.2f} nm"
        "<extra></extra>"
    )
    fig.update_layout(
        legend=dict(
            font=dict(size=12),
            yanchor="top",
            y=-0.1,
            xanchor="left",
            x=0
        ),
    )
    fig.update_xaxes(
        title_text="<br>Wavelength (nm)</br>"
    )
    fig.update_yaxes(
        title_text="<br>Intensity (arb. units)</br>",
        range=(0, df["intensity"].max() * 1.05)
    )
    return fig, df.to_dict("records")


@dash.callback(
    dash.Output(csv_download_button, "disabled"),
    dash.Input(uploadbar.uploaded_files_dropdown, "value")
)
def update_download_button_ability(selected_items: list[str] | None) -> bool:
    return not bool(selected_items)


@dash.callback(
    dash.Output(csv_download, "data"),
    dash.Input(csv_download_button, "n_clicks"),
    dash.State(table, "data")
)
def download_csv(
    n_clicks: int,
    data: dict[str, t.Any] | None
) -> dict[str, t.Any]:
    if not data:
        raise dash.exceptions.PreventUpdate
    return dict(
        filename="h-figure.csv",
        type="text/csv",
        base64=False,
        content=pd.DataFrame(data).to_csv(index=False)
    )
