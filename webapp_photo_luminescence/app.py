# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import os
import pathlib

import dash
import dash_bootstrap_components as dbc

from webapp_photo_luminescence import (
    navbar,
    sidebar,
    uploadbar,
    tabs
)


main_container = dbc.Container(
    [
        dbc.Row(uploadbar.layout),
        dbc.Row(tabs.layout),
    ]
)
layout = dbc.Container(
    [
        dbc.Row(navbar.layout),
        dbc.Row(
            [
                dbc.Col(sidebar.layout, width=2, class_name="bg-light"),
                dbc.Col(main_container, width=10)
            ],
            style={"height": "100vh"}
        )
    ],
    fluid=True
)


URL_BASE_PATH = os.environ.get("URL_BASE_PATH", "/")
app = dash.Dash(
    __name__,
    title="PL Analysis",
    url_base_pathname=URL_BASE_PATH,
    external_stylesheets=[dbc.themes.MATERIA],
    extra_hot_reload_paths=list(pathlib.Path(__file__).parent.glob("**/*.py")),
    suppress_callback_exceptions=True
)
app.layout = layout
