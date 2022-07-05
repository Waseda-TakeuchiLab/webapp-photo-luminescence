# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
from dash import html
import dash_bootstrap_components as dbc


filter_radio_items = dbc.RadioItems(
    id="filter-radio-items",
    options=[
        {"label": "None", "value": None},
        {"label": "Mean", "value": "mean"},
        {"label": "Gaussian", "value": "gaussian"}
    ],
    value=None
)
filter_options_section = dbc.Row(
    [
        html.H5("Filter", className="fw-bold fst-italic mt-2"),
        html.Div(filter_radio_items, className="ms-2")
    ],
    className="ms-1"
)
layout = html.Div(
    [
        filter_options_section,
    ],
    style={"height": "100vh"},
)
