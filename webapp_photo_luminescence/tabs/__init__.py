# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import dash_bootstrap_components as dbc

from webapp_photo_luminescence.tabs import (
    streak_image_tab,
    h_figure_tab,
    v_figure_tab
)

layout = dbc.Tabs(
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
