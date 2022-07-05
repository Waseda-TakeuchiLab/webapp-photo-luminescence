# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import base64
import pathlib
import functools
import dataclasses
import tempfile

import dash
from dash import dcc
import dash_bootstrap_components as dbc
import numpy as np
import cv2
from tlab_analysis import photo_luminescence as pl


UPLOAD_DIR = pathlib.Path("/tmp/upload")
UPLOAD_DIR.mkdir(exist_ok=True)


upload_button = dbc.Button(
    "Upload Files",
    id="upload-button",
    color="primary",
    className="my-2"
)
uploaded_files_dropdown = dcc.Dropdown(
    options=[],
    value=None,
    multi=True,
    clearable=False,
    id="uploaded-files-dropdown",
    className="mt-3 w-100",
)
file_uploader = dcc.Upload(
    [
        upload_button,
    ],
    multiple=True,
    className=""
)
uploaded_files_store = dcc.Store(
    "uploaded-files-store",
    storage_type="memory",
    data=dict()
)
layout = dbc.Row(
    [
        dbc.Col(uploaded_files_dropdown),
        dbc.Col(file_uploader, width="auto"),
        uploaded_files_store
    ],
    className="frex-nowrap ms-auto",
    style={"width": "100%"}
)


@dash.callback(
    dash.Output(uploaded_files_store, "data"),
    dash.Input(file_uploader, "contents"),
    dash.State(file_uploader, "filename"),
    dash.State(uploaded_files_store, "data"),
    prevent_initial_call=True
)
def update_uploaded_files(
    contents: list[str] | None,
    filenames: list[str] | None,
    uploaded_files: dict[str, str] | None
) -> dict[str, str]:
    if contents is None or filenames is None:
        raise dash.exceptions.PreventUpdate
    if not isinstance(uploaded_files, dict):
        uploaded_files = dict()
    for filename, content in zip(filenames, contents):
        with tempfile.NamedTemporaryFile("wb", dir=UPLOAD_DIR, delete=False) as f:
            content_type, content_string = content.split(",")
            f.write(base64.b64decode(content_string))
            uploaded_files[filename] = f.name
    return uploaded_files


@dash.callback(
    dash.Output(uploaded_files_dropdown, "options"),
    dash.Input(uploaded_files_store, "data"),
    prevent_initial_call=True,
)
def update_dropdown_options(
    uploaded_files: dict[str, str] | None
) -> list[str]:
    if not uploaded_files:
        raise dash.exceptions.PreventUpdate
    return list(reversed(uploaded_files.keys()))


@dash.callback(
    dash.Output(uploaded_files_dropdown, "value"),
    dash.Input(uploaded_files_dropdown, "options"),
    dash.State(file_uploader, "filename"),
    dash.Input(uploaded_files_dropdown, "value"),
    prevent_initial_call=True,
)
def update_dropdown_value(
    options: list[str] | None,
    filename: list[str] | None,
    current_value: list[str] | None
) -> list[str]:
    if current_value is None:
        current_value = list()
    if not filename:
        return current_value
    else:
        return current_value + filename


@functools.lru_cache(maxsize=8)
def load_pldata(
    filename: str,
    filter_type: str | None = None
) -> pl.Data:
    data = pl.Data.from_raw_file(UPLOAD_DIR / filename)
    data = filter2d(data, filter_type)
    return data


def filter2d(data: pl.Data, type: str | None) -> pl.Data:
    if type == "mean":
        data_dict = dataclasses.asdict(data)
        kernel = np.ones((5, 5), np.float32) / 25
        data_dict["intensity"] = cv2.filter2D(
            data.streak_image,
            -1,
            kernel
        ).flatten()
        return pl.Data(**data_dict)
    elif type == "gaussian":
        data_dict = dataclasses.asdict(data)
        data_dict["intensity"] = cv2.GaussianBlur(
            data.streak_image,
            (5, 5),
            1
        ).flatten()
        return pl.Data(**data_dict)
    else:
        return data
