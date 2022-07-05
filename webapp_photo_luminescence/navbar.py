# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
from dash import html
import dash_bootstrap_components as dbc

import webapp_photo_luminescence


layout = dbc.Navbar(
    [
        dbc.NavbarBrand(
            [
                "Photo Luminescence ",
                html.Small("v"+webapp_photo_luminescence.__version__)
            ],
            href="/"
        )
    ],
    color="dark",
    dark=True,
    style={"height": "5vh"}
)
