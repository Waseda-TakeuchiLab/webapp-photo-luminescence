# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import typing as t

from dash import dcc, dash_table
from dash.development import base_component
import dash_bootstrap_components as dbc


Child = t.Union[base_component.Component, str, int]
Children = t.Union[list[Child], Child, None]


def create_graph(**kwargs: t.Any) -> dcc.Graph:
    _kwargs = dict(
        config=dict(doubleClick="reset"),
        style={"height": "70vh"}
    )
    _kwargs.update(kwargs)
    return dcc.Graph(**_kwargs)


def create_table(**kwargs: t.Any) -> dash_table.DataTable:
    _kwargs = dict(
        page_size=100,
        style_table={"height": "80vh", "overflowY": "auto"},
    )
    _kwargs.update(kwargs)
    return dash_table.DataTable(**_kwargs)


def create_options_layout(
    options_components: Children,
    download_components: Children
) -> dbc.Container:
    container = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        options_components,
                        width=6,
                        lg=12,
                        style={"height": "60vh"}
                    ),
                    dbc.Col(
                        download_components,
                        width=6,
                        lg=12,
                        style={"height": "auto"}
                    ),
                ],
            ),
        ],
        className="mt-2"
    )
    return container


def create_layout(
    graph: dcc.Graph | None = None,
    options: dbc.Container | None = None,
    table: dash_table.DataTable | None = None
) -> dbc.Container:
    container = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(graph, type="circle"),
                        width=12,
                        lg=9
                    ),
                    dbc.Col(
                        options,
                        width=12,
                        lg=3
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(table)
                ]
            )
        ]
    )
    return container
