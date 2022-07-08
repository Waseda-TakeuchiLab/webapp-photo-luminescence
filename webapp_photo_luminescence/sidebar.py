# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
from dash import html
import dash_bootstrap_components as dbc

from webapp_photo_luminescence.tabs import upload


layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("Filter", className="mt-2"),
                        html.Div(upload.filter_radio_items, className="mx-2")
                    ]
                ),
            ],
            style={"height": "70vh"}
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                    ]
                ),
            ],
            style={"height": "30vh"}
        )
    ],
)
