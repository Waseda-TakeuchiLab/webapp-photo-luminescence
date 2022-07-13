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
import numpy as np
import numpy.typing as npt
from tlab_analysis import photo_luminescence as pl

from webapp_photo_luminescence.tabs import (
    common,
    upload
)


wavelength_slider = dcc.RangeSlider(
    id="v-wavelength-slider",
    min=0,
    max=800,
    value=(0, 800),
    step=1,
    marks=None,
    tooltip={"placement": "bottom", "always_visible": True}
)
fitting_curve_switch = dbc.Switch(
    id="v-fitting-curve-switch",
    label="Fitting",
    value=True,
    className="mt-2"
)
log_intensity_switch = dbc.Switch(
    id="v-log-intensity-switch",
    label="Log Intensity",
    value=True
)
normalize_intensity_switch = dbc.Switch(
    id="v-normalize-intensity-switch",
    label="Normalize Intensity",
    value=False
)
download = dcc.Download(
    id="v-csv-download"
)
download_button = dbc.Button(
    [
        "Download CSV",
        download
    ],
    id="v-csv-download-button"
)
graph = common.create_graph(id="v-figure-graph")
options = common.create_options_layout(
    options_components=[
        dbc.Label("Wavelength Range"),
        wavelength_slider,
        fitting_curve_switch,
        log_intensity_switch,
        normalize_intensity_switch
    ],
    download_components=[
        download_button
    ]
)
table = common.create_table(id="v-table")
layout = common.create_layout(graph, options, table)


@functools.lru_cache(maxsize=16)
def load_wavelength_resolved(
    filepath: str,
    filter_type: str | None = None,
    wavelength_range: tuple[float, float] | None = None,
    fitting: bool = False,
    normalize_intensity: bool = False,
) -> pl.WavelengthResolved[pl.Data]:
    data = upload.load_pldata(filepath, filter_type)
    wr = data.wavelength_resolved(wavelength_range)
    if normalize_intensity:
        wr.df["intensity"] /= wr.df["intensity"].max()
    wr.df["fit"] = np.nan
    wr.df["name"] = os.path.split(filepath)[-1]
    if fitting:
        params, cov = wr.fit(double_exponential, bounds=(0.0, np.inf))
        fast, slow = sorted((params[:2], params[2:]), key=lambda x: x[1])  # type: ignore
        a = int(fast[0] / (fast[0] + slow[0]) * 100)
        wr.df.attrs["fit"] = {
            "a": a,
            "tau1": fast[1],
            "b": 100 - a,
            "tau2": slow[1],
            "params": params,
            "cov": cov
        }
    return wr


@dash.callback(
    dash.Output(graph, "figure"),
    dash.Input(upload.files_dropdown, "value"),
    dash.State(upload.upload_dir_store, "data"),
    dash.Input(upload.filter_radio_items, "value"),
    dash.Input(wavelength_slider, "value"),
    dash.Input(fitting_curve_switch, "value"),
    dash.Input(log_intensity_switch, "value"),
    dash.Input(normalize_intensity_switch, "value"),
    prevent_initial_call=True
)
def update_graph(
    selected_items: list[str] | None,
    upload_dir: str | None,
    filter_type: str | None,
    wavelength_range: list[int],
    fitting: bool,
    log_y: bool,
    normalize_intensity: bool
) -> go.Figure:
    upload_dir = upload.validate_upload_dir(upload_dir)
    if not selected_items:
        return go.Figure()
    filepaths = [os.path.join(upload_dir, item) for item in selected_items]
    wrs = [
        load_wavelength_resolved(
            filepath,
            filter_type,
            tuple(wavelength_range[:2]),
            fitting,
            normalize_intensity
        ) for filepath in filepaths if os.path.exists(filepath)
    ]
    df = pd.concat([wr.df for wr in wrs])
    fig = px.line(
        df,
        x="time",
        y="intensity",
        color="name",
        color_discrete_sequence=px.colors.qualitative.Set1,
        log_y=log_y
    )
    fig.add_traces(
        [
            go.Scatter(
                x=wr.df["time"],
                y=wr.df["fit"],
                line=dict(color="black"),
                name=f"Double Exponential Approximation "
                f"a : b = {wr.df.attrs['fit']['a']}:{wr.df.attrs['fit']['b']}, "
                f"τ₁ = {wr.df.attrs['fit']['tau1']:.3g} ns, "
                f"τ₂ = {wr.df.attrs['fit']['tau2']:.3g} ns"
            ) for wr in wrs if "fit" in wr.df.attrs
        ]
    )
    fig.update_traces(
        hovertemplate=""
        "Time: %{x:.3g} ns<br>"
        "Intensity: %{y:.4g}<br>"
        "<extra></extra>"
    )
    fig.update_layout(
        legend=dict(
            font=dict(size=14),
            yanchor="top",
            y=-0.15,
            xanchor="left",
            x=0
        )
    )
    fig.update_xaxes(
        title_text="<b>Time (ns)</b>"
    )
    range_y = np.array(
        [
            df["intensity"][np.isclose(df["time"], 0.0)].min(),
            df["intensity"].max()
        ]
    )
    fig.update_yaxes(
        title_text="<b>Intensity (arb. units)</b>",
        range=(np.log10(range_y) if log_y else range_y) * [1.0, 1.05]
    )
    return fig


@dash.callback(
    dash.Output(table, "data"),
    dash.Input(upload.files_dropdown, "value"),
    dash.State(upload.upload_dir_store, "data"),
    dash.Input(upload.filter_radio_items, "value"),
    dash.Input(wavelength_slider, "value"),
    dash.Input(fitting_curve_switch, "value"),
    dash.Input(normalize_intensity_switch, "value"),
    prevent_initial_call=True
)
def update_table(
    selected_items: list[str] | None,
    upload_dir: str | None,
    filter_type: str | None,
    wavelength_range: list[int],
    fitting: bool,
    normalize_intensity: bool
) -> list[dict[str, t.Any]] | None:
    upload_dir = upload.validate_upload_dir(upload_dir)
    if not selected_items:
        return None
    filepaths = [os.path.join(upload_dir, item) for item in selected_items]
    wrs = [
        load_wavelength_resolved(
            filepath,
            filter_type,
            tuple(wavelength_range[:2]),
            fitting,
            normalize_intensity
        ) for filepath in filepaths if os.path.exists(filepath)
    ]
    df = pd.concat([wr.df for wr in wrs])
    return df.to_dict("records")  # type: ignore


@dash.callback(
    dash.Output(wavelength_slider, "min"),
    dash.Output(wavelength_slider, "max"),
    dash.Input(upload.files_dropdown, "value"),
    dash.State(upload.upload_dir_store, "data"),
    prevent_initial_call=True
)
def update_wavelength_slider_range(
    selected_items: list[str] | None,
    upload_dir: str | None
) -> tuple[float, float]:
    upload_dir = upload.validate_upload_dir(upload_dir)
    if not selected_items:
        raise dash.exceptions.PreventUpdate
    filepaths = [os.path.join(upload_dir, item) for item in selected_items]
    wavelength = np.concatenate(
        [
            upload.load_pldata(filepath).wavelength
            for filepath in filepaths if os.path.exists(filepath)
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
    dash.State(wavelength_slider, "value"),
    dash.State(fitting_curve_switch, "value"),
    dash.State(normalize_intensity_switch, "value"),
    prevent_initial_call=True
)
def download_csv(
    n_clicks: int,
    selected_items: list[str] | None,
    upload_dir: str | None,
    filter_type: str | None,
    wavelength_range: list[int],
    fitting: bool,
    normalize_intensity: bool
) -> dict[str, t.Any]:
    upload_dir = upload.validate_upload_dir(upload_dir)
    if not selected_items:
        raise dash.exceptions.PreventUpdate
    item = selected_items[0]
    filepath = os.path.join(upload_dir, item)
    if not os.path.exists(filepath):
        raise dash.exceptions.PreventUpdate
    wr = load_wavelength_resolved(
        filepath,
        filter_type,
        tuple(wavelength_range[:2]),
        fitting,
        normalize_intensity
    )
    filename = f"v({wavelength_range[0]}-{wavelength_range[1]})-" \
        + (filter_type+"-" if filter_type else "") \
        + item \
        + ".csv"
    return dict(
        filename=filename,
        content=wr.df.to_csv(index=False),
        type="text/csv",
        base64=False
    )
