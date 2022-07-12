# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import base64
import dataclasses
import functools
import os
import pathlib
import tempfile

import dash
from dash import dcc
import dash_bootstrap_components as dbc
import numpy as np
import cv2
from tlab_analysis import photo_luminescence as pl


UPLOAD_BASEDIR = "/tmp/app"

filter_radio_items = dbc.RadioItems(
    id="filter-radio-items",
    options=[
        {"label": "None", "value": None},
        {"label": "Mean", "value": "mean"},
        {"label": "Gaussian", "value": "gaussian"}
    ],
    value=None
)
upload_button = dbc.Button(
    "Upload Files",
    id="upload-button",
    color="primary",
    className="my-2"
)
files_dropdown = dcc.Dropdown(
    options=[],
    value=None,
    multi=True,
    clearable=False,
    id="uploaded-files-dropdown",
    className="mt-3 w-100"
)
file_uploader = dcc.Upload(
    [
        upload_button
    ],
    multiple=True
)
upload_dir_store = dcc.Store(
    "upload-dir-store",
    storage_type="session",
    data=""
)
last_uploaded_store = dcc.Store(
    "last-upload-store",
    storage_type="memory",
    data=list()
)


@dash.callback(
    dash.Output(upload_dir_store, "data"),
    dash.Input(upload_dir_store, "data")
)
def update_upload_dir(_: dict[str, str] | None) -> str:
    if upload_dir_store:
        raise dash.exceptions.PreventUpdate
    if not os.path.exists(UPLOAD_BASEDIR):
        os.mkdir(UPLOAD_BASEDIR)
    tempdir = tempfile.mkdtemp(dir=UPLOAD_BASEDIR)
    return tempdir


@dash.callback(
    dash.Output(last_uploaded_store, "data"),
    dash.Input(file_uploader, "contents"),
    dash.State(file_uploader, "filename"),
    dash.State(upload_dir_store, "data"),
    prevent_initial_call=True
)
def on_upload_files(
    contents: list[str] | None,
    filenames: list[str] | None,
    upload_dir: str | None
) -> list[str]:
    assert upload_dir is not None
    assert upload_dir.startswith(UPLOAD_BASEDIR)
    if contents is None or filenames is None:
        raise dash.exceptions.PreventUpdate
    for filename, content in zip(filenames, contents):
        with open(os.path.join(upload_dir, filename), "wb") as f:
            content_type, content_string = content.split(",")
            f.write(base64.b64decode(content_string))
    return filenames


@dash.callback(
    dash.Output(files_dropdown, "options"),
    dash.Input(last_uploaded_store, "data"),
    dash.State(upload_dir_store, "data"),
    prevent_initial_call=True,
)
def update_dropdown_options(
    last_uploaded_files: list[str] | None,
    upload_dir: str | None
) -> list[str]:
    assert upload_dir is not None
    assert upload_dir.startswith(UPLOAD_BASEDIR)
    return [path.name for path in pathlib.Path(upload_dir).iterdir() if path.is_file()]


@dash.callback(
    dash.Output(files_dropdown, "value"),
    dash.Input(last_uploaded_store, "data"),
    dash.State(files_dropdown, "value"),
    prevent_initial_call=True
)
def update_dropdown_value(
    filenames: list[str] | None,
    current_value: list[str] | None
) -> list[str]:
    if current_value is None:
        current_value = list()
    if not filenames:
        return current_value
    else:
        return current_value + filenames


@functools.lru_cache(maxsize=8)
def load_pldata(
    filepath: str,
    filter_type: str | None = None
) -> pl.Data:
    data = pl.Data.from_raw_file(filepath)
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
