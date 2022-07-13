# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import base64
import datetime
import io
import os
import typing as t

import dash
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import tlab_pptx

from webapp_photo_luminescence.tabs import (
    h_figure_tab,
    v_figure_tab,
    upload
)


download = dcc.Download("pptx-download")
download_button = dbc.Button(
    [
        "Download PowerPoint",
        download
    ],
    id="pptx-download-button",
    disabled=True,
    color="primary",
    className="mt-2"
)
datepicker = dcc.DatePickerSingle(
    id="experiment-date-picker-single",
    with_portal=True,
    persistence_type="session"
)


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
    dash.State(v_figure_tab.wavelength_slider, "value"),
    dash.State(h_figure_tab.graph, "figure"),
    dash.State(v_figure_tab.graph, "figure"),
    dash.State(datepicker, "date"),
    prevent_initial_call=True
)
def download_powerpoint(
    n_clicks: int,
    selected_items: list[str] | None,
    upload_dir: str | None,
    filter_type: str | None,
    wavelength_range: list[int],
    h_fig: dict[str, t.Any] | None,
    v_fig: dict[str, t.Any] | None,
    experiment_date: str | None
) -> dict[str, t.Any]:
    upload_dir = upload.validate_upload_dir(upload_dir)
    if not selected_items:
        raise dash.exceptions.PreventUpdate
    item = selected_items[0]
    filepath = os.path.join(upload_dir, item)
    if not os.path.exists(filepath):
        raise dash.exceptions.PreventUpdate
    tr = h_figure_tab.load_time_resolved(
        filepath,
        filter_type
    )
    wr = v_figure_tab.load_wavelength_resolved(
        filepath,
        filter_type,
        tuple(wavelength_range[:2]),
        fitting=True
    )
    prs = tlab_pptx.PhotoLuminescencePresentation(
        title="title",
        excitation_wavelength=405,  # TODO: Retrieve from `item`
        excitation_power=5,         # TODO: Retrieve from `item`
        time_range=10,              # TODO: Retrieve from `item`
        center_wavelength=int(tr.peak_wavelength),
        FWHM=tr.FWHM,
        frame=10000,                # TODO: Retrieve from `item`
        date=datetime.date.fromisoformat(experiment_date) if experiment_date else datetime.date.today(),
        h_fig=go.Figure(h_fig),
        v_fig=go.Figure(v_fig),
        a=int(wr.df.attrs["fit"]["a"]),
        b=int(wr.df.attrs["fit"]["b"]),
        tau1=float(wr.df.attrs['fit']['tau1']),
        tau2=float(wr.df.attrs['fit']['tau2'])
    )
    with io.BytesIO() as f:
        prs.save(f)
        f.seek(0)
        raw = f.read()
    filename = (filter_type+"-" if filter_type else "") + item + ".pptx"
    return dict(
        filename=filename,
        content=base64.b64encode(raw).decode(),
        type="application/octet-stream",
        base64=True
    )
