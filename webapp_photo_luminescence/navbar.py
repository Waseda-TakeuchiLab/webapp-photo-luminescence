# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import dash
from dash import html
import dash_bootstrap_components as dbc

import webapp_photo_luminescence

toggler = dbc.NavbarToggler(n_clicks=0)
collapse = dbc.Collapse(
    is_open=False,
    navbar=True
)
layout = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.NavbarBrand(
                                [
                                    "Photo Luminescence ",
                                    html.Small("v"+webapp_photo_luminescence.__version__)
                                ]
                            ),
                        ),
                    ],
                    align="center",
                ),
                href="#",
                style={"textDecoration": "none"}
            ),
            toggler,
            collapse,
        ],
        fluid=True
    ),
    color="dark",
    dark=True,
    style={"height": "5vh"}
)


@dash.callback(
    dash.Output(collapse, "is_open"),
    dash.Input(toggler, "n_clicks"),
    dash.State(collapse, "is_open")
)
def toggle_navbar_collapse(
    n: int,
    is_open: bool
) -> bool:
    if n:
        return not is_open
    return is_open
