# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import dash_bootstrap_components as dbc

from . import (
    upload,
    streak_image_tab,
    h_figure_tab,
    v_figure_tab
)


uploadbar = dbc.Row(
    [
        dbc.Col(upload.files_dropdown),
        dbc.Col(upload.file_uploader, width="auto"),
        upload.upload_dir_store,
        upload.last_uploaded_store
    ],
    className="frex-nowrap ms-auto",
    style={"width": "100%"}
)
tabs = dbc.Tabs(
    [
        dbc.Tab(
            streak_image_tab.layout,
            id="streak-image-tab",
            label="Streak Image",
        ),
        dbc.Tab(
            h_figure_tab.layout,
            id="h-figure-tab",
            label="H-Figure",
        ),
        dbc.Tab(
            v_figure_tab.layout,
            id="v-figure-tab",
            label="V-Figure",
        ),
    ],
    id="figure-tabs",
    className="nav-fill"
)
layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(uploadbar)
            ]
        ),
        dbc.Row(
            [
                dbc.Col(tabs)
            ]
        ),
    ],
    fluid=True
)
