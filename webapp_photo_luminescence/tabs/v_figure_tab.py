# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import typing as t

import dash
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import numpy.typing as npt
from tlab_analysis import photo_luminescence as pl

from webapp_photo_luminescence import (
    uploadbar,
    sidebar
)
from webapp_photo_luminescence.tabs import common


graph = common.create_graph(id="v-figure-graph")
wavelength_slider = dcc.RangeSlider(
    id="v-wavelength-slider",
    min=0,
    max=800,
    value=(0, 800),
    step=1,
    marks=None,
    tooltip={"placement": "bottom", "always_visible": True},
    className="",
)
fitting_curve_switch = dbc.Switch(
    id="fitting-curve-switch",
    label="Fitting",
    value=True,
    className="mt-2",
)
csv_download = dcc.Download(
    id="v-csv-download"
)
csv_download_button = dbc.Button(
    [
        "Download CSV",
        csv_download
    ],
    id="v-csv-download-button"
)
options = common.create_options_layout(
    options_components=[
        dbc.Label("Wavelength Range"),
        wavelength_slider,
        fitting_curve_switch,
    ],
    download_components=[
        csv_download_button
    ]
)
table = common.create_table(id="v-table")
layout = common.create_layout(graph, options, table)


@dash.callback(
    dash.Output(graph, "figure"),
    dash.Output(table, "data"),
    dash.Input(uploadbar.uploaded_files_dropdown, "value"),
    dash.State(uploadbar.uploaded_files_store, "data"),
    dash.Input(sidebar.filter_radio_items, "value"),
    dash.Input(wavelength_slider, "value"),
    dash.Input(fitting_curve_switch, "value"),
    prevent_initial_call=True
)
def update_graph_and_table(
    selected_items: list[str] | None,
    uploaded_files: dict[str, str] | None,
    filter_type: str | None,
    wavelength_range: list[int],
    fitting: bool,
) -> tuple[go.Figure, dict[str, t.Any]]:
    if not uploaded_files or not selected_items:
        return go.Figure(), dict()
    wrs: list[pl.WavelengthResolved[pl.Data]] = []
    traces: list[go.Trace] = []
    for item in filter(uploaded_files.__contains__, selected_items):
        data = uploadbar.load_pldata(uploaded_files[item], filter_type)
        wr = data.wavelength_resolved(tuple(wavelength_range[:2]))
        wr.df["fit"] = np.nan
        wr.df["name"] = item
        wrs.append(wr)
    if fitting:
        for wr in wrs:
            params, cov = wr.fit(double_exponential, bounds=(0.0, np.inf))
            fast, slow = sorted((params[:2], params[2:]), key=lambda x: x[1])  # type: ignore
            a = int(fast[0] / (fast[0] + slow[0]) * 100)
            traces.append(
                go.Scatter(
                    x=wr.df["time"],
                    y=wr.df["fit"],
                    line=dict(color="black"),
                    name=f"Double Exponential Approximation "
                    f"a : b = {a}:{100-a}, "
                    f"τ₁ = {fast[1]:.2f} ns, "
                    f"τ₂ = {slow[1]:.2f} ns"
                )
            )
    df = pd.concat([wr.df for wr in wrs])
    fig = px.line(
        df,
        x="time",
        y="intensity",
        color="name",
    )
    fig.add_traces(traces)
    fig.update_traces(
        hovertemplate=""
        "Time: %{x:.2f} ns<br>"
        "Intensity: %{y:d}<br>"
        "<extra></extra>"
    )
    fig.update_layout(
        legend=dict(
            font=dict(size=12),
            yanchor="top",
            y=-0.15,
            xanchor="left",
            x=0
        )
    )
    fig.update_xaxes(
        title_text="<b>Time (ns)</b>",
    )
    fig.update_yaxes(
        title_text="<b>Intensity (arb. units)</b>",
        range=np.log10(
            [
                df["intensity"][np.isclose(df["time"], 0.0)].min(),
                df["intensity"].max()
            ]
        ) * [1.0, 1.05],
        type="log"
    )
    return fig, df.to_dict("records")


@dash.callback(
    dash.Output(wavelength_slider, "min"),
    dash.Output(wavelength_slider, "max"),
    dash.Input(uploadbar.uploaded_files_dropdown, "value"),
    dash.State(uploadbar.uploaded_files_store, "data"),
    prevent_initial_call=True,
)
def update_wavelength_slider_range(
    selected_items: list[str] | None,
    uploaded_files: dict[str, str] | None,
) -> tuple[float, float]:
    if not selected_items or not uploaded_files:
        raise dash.exceptions.PreventUpdate
    wavelength = np.concatenate(
        [
            uploadbar.load_pldata(uploaded_files[item]).wavelength
            for item in filter(uploaded_files.__contains__, selected_items)
        ]
    )
    return int(wavelength.min()), int(wavelength.max())


def double_exponential(
    time: npt.NDArray[np.float32],
    a: float,
    tau1: float,
    b: float,
    tau2: float
) -> npt.NDArray[np.float32]:
    return a * np.exp(-time/tau1) + b * np.exp(-time/tau2)


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
        filename="v-figure.csv",
        type="text/csv",
        base64=False,
        content=pd.DataFrame(data).to_csv(index=False)
    )
