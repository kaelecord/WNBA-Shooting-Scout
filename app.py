# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:14:53 2024

@author: kaele
"""

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, State, Input, Output, callback
import copy
import io
import base64
from datetime import date
from wnba_functions import *

shots_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQq2hScV4DjC1ixL3thUGt_tomrMEuyBmIF9ltYbx-BTkPuQsx0ZpDiGISCmMNEfDfMmZz7HidA0Lz8/pub?output=csv'
wnba_shots = pd.read_csv(shots_url)
#wnba_shots['game_date'] = pd.to_datetime(wnba_shots['game_date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')

zone_averages_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQqOSPgCgqQ-fqQdAOf5Z7I5NY_sMKDNHzsLuSYzxuK-kmU4nuJ-6TSbnEsPd7nLNFrmVsOepNtVUG6/pub?output=csv'
wnba_zone_league_averages = pd.read_csv(zone_averages_url)

wnba_team_map = {3: 'Dallas Wings',
                 5: 'Indiana Fever',
                 6: 'Los Angeles Sparks',
                 8: 'Minnesota Lynx',
                 9: 'New York Liberty',
                 11: 'Phoenix Mercury',
                 14: 'Seattle Storm',
                 16: 'Washington Mystics',
                 17: 'Las Vegas Aces',
                 18: 'Connecticut Sun',
                 19: 'Chicago Sky',
                 20: 'Atlanta Dream'}

wnba_team_id_map = {value: key for key, value in wnba_team_map.items()}

wnba_team_logo_map = {3: 'https://a.espncdn.com/i/teamlogos/wnba/500/dal.png',
                 5: 'https://a.espncdn.com/i/teamlogos/wnba/500/ind.png',
                 6: 'https://a.espncdn.com/i/teamlogos/wnba/500/la.png',
                 8: 'https://a.espncdn.com/i/teamlogos/wnba/500/min.png',
                 9: 'https://a.espncdn.com/i/teamlogos/wnba/500/ny.png',
                 11: 'https://a.espncdn.com/i/teamlogos/wnba/500/phx.png',
                 14: 'https://a.espncdn.com/i/teamlogos/wnba/500/sea.png',
                 16: 'https://a.espncdn.com/i/teamlogos/wnba/500/wsh.png',
                 17: 'https://a.espncdn.com/i/teamlogos/wnba/500/lv.png',
                 18: 'https://a.espncdn.com/i/teamlogos/wnba/500/conn.png',
                 19: 'https://a.espncdn.com/i/teamlogos/wnba/500/chi.png',
                 20: 'https://a.espncdn.com/i/teamlogos/wnba/500/atl.png'}
wnba_team_primary_color_map = {3: '#002b5c',
                 5: '#002d62',
                 6: '#552583',
                 8: '#266092',
                 9: '#86cebc',
                 11: '#3c286e',
                 14: '#2c5235',
                 16: '#e03a3e',
                 17: '#a7a8aa',
                 18: '#f05023',
                 19: '#5091cd',
                 20: '#e31837'}
wnba_team_secondary_color_map = {3: '#c4d600',
                 5: '#e03a3e',
                 6: '#fdb927',
                 8: '#79bc43',
                 9: '#000000',
                 11: '#e56020',
                 14: '#fee11a',
                 16: '#002b5c',
                 17: '#000000',
                 18: '#0a2240',
                 19: '#ffd520',
                 20: '#5091cc'}

# add visualization code here
app = Dash("WNBA Shooting Dash")

app.layout = html.Div(children=[
    html.H1(children='WNBA Shooting Dashboard',
            style={"display": "flex",
                   "justify-content": "center",
                   "align-items": "center"}),
    # Filter div
    html.Div([
        #left half
        html.Div([html.Label("Team", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='team-dropdown',
                options=sorted(wnba_shots['team_display_name_shooter'].unique()),
                placeholder="Select a Team",
                value='Indiana Fever',
                searchable=True,
                clearable=True, style={'width':'75%'})],
                 style={"display": "flex",
                        "justify-content": "center",
                        "align-items": "center",
                        "width": "50%"}),
        #right half
        html.Div([html.Label("Player", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='player-dropdown',
                options=sorted(wnba_shots['athlete_display_name_shooter'].unique()),
                placeholder="Select a Player",
                value='Caitlin Clark',
                searchable=True,
                clearable=True, style={'width':'75%'})],
                style={"display": "flex",
                        "justify-content": "center",
                        "align-items": "center",
                        "width": "50%"})
    ], style={"display": "flex", "width": "100%"}),
    # row 1 div. Card on left, hexbin on right
    html.Div([
        html.Div([
        #player/team name div background color is teams main color white text with first and last name on different lines
        html.Div(id='card-header-div',style={"height": "10%"}),
        # player/team stats div
        html.Div(id='card-stats-div', style={"display": "flex", "height": "60%", "width":"100%", "justify-content": "center", "align-items": "center"}),
        # player/team scoring by game div
        html.Div([dcc.Graph(id='card-scoring-bar', style={"width": "100%", "height": "100%", "overflow": "hidden"})],
                 style={"display": "flex", "height": "30%", "width":"100%", "justify-content": "center", "align-items": "center"})
    ],id='card-div',
             style={"display": "flex",
                    "flex-direction": "column",
                    "width": "50%",
                    "border": "1px solid black",
                    "border-radius": "5px",
                    "padding": "15px",
                    "box-sizing": "border-box"}
             ),
              html.Div([
            # Here is your new div content (hexbin, or any other new div)
            html.Div(id='hexbin-div')  # Update with your specific content
        ], style={
            "display": "flex",
            "flex-direction": "column",
            "width": "50%",  # 50% of the parent container
            "padding": "15px",
            "box-sizing": "border-box"
        })], style={'display': 'flex', 'width': '100%', 'height':'50%'}),

    # row 2 div
    html.Div([
        #filters div
        html.Div([
            html.Div([html.Label('Chart Filters', style={'font-weight': 'bold'})],
                     style={'display': 'flex',  # Enable flexbox
                            'justify-content': 'center',  # Center horizontally
                            'align-items': 'center',
                            'margin-bottom': '10px'}),
            html.Div([dcc.RadioItems(
                options=[
                    {'label':'Player Chart', 'value':'player'},
                    {'label':'Team Chart', 'value':'team'}
                ],
                value='player', inline=True,
                id='chart-team-player-radio'
            )],
                     style={'display':'flex',
                            'justify-content':'center',
                            'align-items':'center',
                            'padding':'10px'}),
            html.Div([html.Label("Date Range", style={'font-weight': 'bold'}),
                      dcc.DatePickerRange(
                        id='chart-date-range',
                        min_date_allowed=date(2024, 5, 14),
                        max_date_allowed=date(2024, 9, 17),
                        initial_visible_month=date(2024, 5, 14),
                        start_date=date(2024, 5, 14),
                        end_date=date(2024, 9, 17)
                      )], style={'padding':'5px'}),
            html.Div([html.Label("Game", style={'font-weight': 'bold'}),
                      dcc.Dropdown(
                          id='game-dropdown',
                          options=sorted(list(wnba_shots['game_info'].unique())),
                          value= [],
                          searchable=True,
                          multi=True
                      )], style={'padding':'5px'}),
            html.Div([html.Label("Opponent", style={'font-weight': 'bold'}),
                      dcc.Dropdown(
                        id='opponent-dropdown',
                        options=list(wnba_team_id_map.keys()),
                        placeholder="Opponent:",
                        value=[],
                        searchable=True,
                        multi=True,
                      )], style={'padding':'5px'}),
            html.Div([html.Label("Location", style={'font-weight': 'bold'}),
                      dcc.Checklist(
                          id='location-checklist',
                          options=[
                              {'label': 'Home', 'value': 'home'},
                              {'label': 'Away', 'value': 'away'}
                              ],
                          value=['home', 'away'],
                          inline=True
                      )], style={'padding':'5px'}), #location
            html.Div([html.Label("Half", style={'font-weight': 'bold'}),
                      dcc.Checklist(
                        id='half-checklist',
                        options=[
                            {'label': '1st Half', 'value': 1},
                            {'label': '2nd Half', 'value': 2}
                        ],
                        value=[1,2],
                        inline=True
                      )], style={'padding':'5px'}),
            html.Div([html.Label("Quarter", style={'font-weight': 'bold'}),
                      dcc.Checklist(
                        id='qtr-checklist',
                        options=[
                            {'label': '1st Quarter', 'value': 1},
                            {'label': '2nd Quarter', 'value': 2},
                            {'label': '3rd Quarter', 'value': 3},
                            {'label': '4th Quarter', 'value': 4},
                            {'label': 'Overtime', 'value': 5}
                        ],
                        value=[1,2,3,4,5],
                        inline=True
                      )], style={'padding':'5px'}),
            html.Div([html.Label("Assisted?", style={'font-weight': 'bold'}),
                      dcc.Checklist(
                        id='assisted-checklist',
                        options=[
                            {'label': 'Yes', 'value': True},
                            {'label': 'No', 'value': False}
                        ],
                        value=[True, False],
                        inline=True
                      ),
                      html.Label("Blocked?", style={'font-weight': 'bold'}),
                      dcc.Checklist(
                        id='blocked-checklist',
                        options=[
                            {'label': 'Yes', 'value': True},
                            {'label': 'No', 'value': False}
                        ],
                        value=[True, False],
                        inline=True
                      )], style={'padding':'5px'}),
            # Apply Button
            html.Div([html.Button('Apply Filter(s)', id='apply-button', n_clicks=0)],
                     style={'display': 'flex',  # Enable flexbox
                            'justify-content': 'center',  # Center horizontally
                            'align-items': 'center',
                            'margin': '10px'})
            ], style={
            'width': '33%',  # Filters div takes 33%
            'float': 'left',  # Float left for side-by-side layout
            'padding': '10px',
            'box-sizing': 'border-box'
        }),
        #chart div
        html.Div([
            #chart selection div
            html.Div([dcc.RadioItems(
                options=[
                    {'label':'Make/Miss', 'value':'standard'},
                    {'label':'Shot Zone', 'value':'zone'},
                    {'label':'Heat', 'value':'heat'},
                    {'label':'Stream', 'value':'stream'}
                ],
                value='standard', inline=True,
                id='chart-selection-radio'
            )],id='chart-selection-div',
                     style={
                         'text-align': 'center',  # Center the radio buttons
                         'margin-bottom': '10px'
                     }),
            # chart display div
            html.Div([dcc.Graph(id='chart-display')], id='chart-display-div',
                     style={
                                'display': 'flex',  # Enable flexbox
                                'justify-content': 'center',  # Center horizontally
                                'align-items': 'center',  # Center vertically
                                'width': '100%',
                                'height': '100%',  # Ensure it takes full height
                                'min-height': '400px',  # Set a minimum height
                                'border': '1px solid #ddd'
                            })
        ], style={
            'width': '67%',  # Chart div takes 67%
            'float': 'right',  # Float right for side-by-side layout
            'box-sizing': 'border-box'
            })
    ],
             style={
                 'display': 'flex',  # Flexbox for responsive layout
                 'flex-direction': 'row',  # Horizontal layout
                 'width': '100%',
                 'box-sizing': 'border-box',
                 'height': '50%'
             })
], style={'height': '100%'})
@callback(
    Output('team-dropdown','options'),
    Input('player-dropdown', 'value')
)
def chained_callback_team(player):
  dff = copy.deepcopy(wnba_shots)

  if player is not None:
    dff = dff[dff['athlete_display_name_shooter'] == player]

  return sorted(dff['team_display_name_shooter'].unique())

@callback(
    Output('player-dropdown', 'options'),
    Input('team-dropdown', 'value')
)
def chained_callback_player(team):
  dff = copy.deepcopy(wnba_shots)

  if team is not None:
    dff = dff[dff['team_display_name_shooter'] == team]

  return sorted(dff['athlete_display_name_shooter'].unique())

@callback(
    Output('game-dropdown', 'options'),
    [
        Input('team-dropdown', 'value'),
        Input('player-dropdown', 'value'),
        Input('opponent-dropdown', 'value')]
)
def chained_callback_game_info(team, player, opponents):
  dff = copy.deepcopy(wnba_shots)

  if player is not None:
    if len(opponents) > 0:
      dff = dff[(dff['opponent_display_name'].isin(opponents)) & (dff['athlete_display_name_shooter'] == player)]
    else:
      dff = dff[dff['athlete_display_name_shooter'] == player]
  elif team is not None:
    if len(opponents) > 0:
      dff = dff[(dff['opponent_display_name'].isin(opponents)) & (dff['team_display_name_shooter'] == team)]
    else:
      dff = dff[dff['team_display_name_shooter'] == team]
  game_info = sorted(dff['game_info'].unique())
  try:
    game_info.remove('2024-07-20 vs Team USA Team USA')
    game_info.remove('2024-07-20 @ Team WNBA Team WNBA')
  except:
    pass
  return game_info

@callback(
    Output('opponent-dropdown', 'options'),
    [Input('team-dropdown', 'value'),
    Input('player-dropdown', 'value'),
    Input('game-dropdown', 'value')]
)
def chained_callback_opponent(team, player, game_info):
  dff = copy.deepcopy(wnba_shots)

  if player is not None:
    if len(game_info) > 0:
      dff = dff[(dff['game_info'].isin(game_info)) & (dff['athlete_display_name_shooter'] == player)]
    else:
      dff = dff[dff['athlete_display_name_shooter'] == player]
  elif team is not None:
    if len(game_info) > 0:
      dff = dff[(dff['game_info'].isin(game_info)) & (dff['team_display_name_shooter'] == team)]
    else:
      dff = dff[dff['team_display_name_shooter'] == team]
  opponents = sorted(dff['opponent_display_name'].unique())
  try:
    opponents.remove('Team USA Team USA')
    opponents.remove('Team WNBA Team WNBA')
  except:
    pass
  return opponents

@callback(
    Output('card-scoring-bar', 'figure'),
    Input('team-dropdown', 'value'),
    Input('player-dropdown', 'value')
)
def update_card_scoring_chart(team, player):
  if player is not None:
    fig = create_card_player_game_score_chart(player)
  elif team is not None:
    fig = create_card_team_game_score_chart(team)

  fig.update_layout(
        autosize=False,
        margin=dict(l=0, r=0, t=0, b=0),  # Set margins to zero
        height=None,  # Allow it to adapt to the container
        width=None,
    )
  return fig
@callback(
    Output('card-stats-div', 'children'),
    Input('team-dropdown', 'value'),
    Input('player-dropdown', 'value')
)
def update_card_stats(team, player):
  if player is not None:
    ppg, fg_percent, three_point_percent, ft_percent = get_card_player_percentages(player)
  elif team is not None:
    ppg, fg_percent, three_point_percent, ft_percent = get_card_team_percentages(team)
  else:
    ppg, fg_percent, three_point_percent, ft_percent = 0, 0, 0, 0

  return [
        # Container div with flex-direction column for vertical alignment
        html.Div([
            # Points Per Game (PPG) in bold and large font
            html.Div([
                html.Div(f"{ppg:.2f}", style={"font-size": "60px", "font-weight": "bold"}),  # Large PPG number
                html.Div("points per game", style={"font-size": "20px", "margin":"5px"})  # Smaller label
            ], style={"margin": "5px"}),

            # Container for the stats (FG%, 3FG%, FT%) horizontally aligned
            html.Div([
                # FG% stats
                html.Div([
                    html.Div(f"{fg_percent}", style={"font-size": "30px", "font-weight": "bold"}),  # Larger number for FG %
                    html.Div("fg", style={"font-size": "20px", "margin":"5px"})  # Smaller text for label
                ], style={"margin": "10px", "display": "flex", "align-items": "center"}),

                # 3FG% stats
                html.Div([
                    html.Div(f"{three_point_percent}", style={"font-size": "30px", "font-weight": "bold"}),  # Larger number for 3FG %
                    html.Div("3fg", style={"font-size": "20px", "margin":"5px"})  # Smaller text for label
                ], style={"margin": "10px", "display": "flex", "align-items": "center"}),

                # FT% stats
                html.Div([
                    html.Div(f"{ft_percent}", style={"font-size": "30px", "font-weight": "bold"}),  # Larger number for FT %
                    html.Div("ft", style={"font-size": "20px"})  # Smaller text for label
                ], style={"margin": "10px", "display": "flex", "align-items": "center"})
            ], style={"display": "flex", "align-items": "left", "justify-content": "left", "width": "100%"})  # Align items horizontally
        ], style={"display": "flex", "flex-direction": "column", "align-items": "left", "width":"100%"})  # Stack PPG on top, and stats on same line
    ]
@callback(
    Output('card-header-div', 'children'),
    Output('card-header-div', 'style'),
    Input('team-dropdown', 'value'),
    Input('player-dropdown', 'value')
)
def update_card_header(team, player):
  if player is not None:
    name = player
    color, image_url = get_team_color_headshot_for_player(player)
  elif team is not None:
    name = team
    color, image_url = get_team_color_logo_for_team(team)
  children = [
      html.Div(
          html.Img(src=image_url, style={"height": "250px", "width":"auto"}),
          style={"flex": "0 0 auto", "margin-right": "15px"}
      ),
      html.Div(
          name,
          style={"flex": "1", "font-size": "40px", "color": "white", "font-weight": "bold", "text-align": "right", "padding":"5px"}
      )
  ]

    # Header background style
  style = {
      "display": "flex",
      "align-items": "center",
      "border-radius": "5px",
      "width" : "100%",
      "background-color": color
  }

  return children, style

@callback(
    Output('hexbin-div', 'children'),
    Input('team-dropdown', 'value'),
    Input('player-dropdown', 'value')
)
def update_hexbin_shot_chart(team, player):
    # Clear the div content initially by returning an empty list
    children = []
    fig =plt.close()
    if player is not None:
        # Generate a new figure or data
        fig = create_player_hexbin(player)
    elif team is not None:
        fig = create_team_hexbin(team)

    if fig:
        # Convert the figure to a Base64 image string and update the div
        img = io.BytesIO()
        fig.savefig(img, format='png')  # Save the figure to the BytesIO object
        img.seek(0)  # Rewind the BytesIO object to the beginning
        img_str = base64.b64encode(img.getvalue()).decode('utf8')  # Encode the image as Base64

        # Add the new content (the image) to the div
        children = [html.Img(src=f"data:image/png;base64,{img_str}", style={"width": "80%", "height": "auto"})]

    return children

@callback(
    Output("chart-display", "figure"),
    # Define filters as state variables, so the callback isn't triggered on change
    [Input('apply-button', 'n_clicks'),
     Input("chart-selection-radio", "value")],
    [State('chart-team-player-radio','value'),
     State('team-dropdown','value'),
     State('player-dropdown','value'),
     State('chart-date-range','start_date'),
     State('chart-date-range','end_date'),
     State('opponent-dropdown','value'),
     State('location-checklist','value'),
     State('half-checklist','value'),
     State('qtr-checklist','value'),
     State('game-dropdown','value'),
     State('assisted-checklist', 'value'),
     State('blocked-checklist', 'value')]
)
def update_plot(n_clicks, shot_chart_selection, chart_type, team, player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked):
    # Only update the plot if the Apply button has been clicked
    if n_clicks > 0:
      if chart_type == 'player':
         if shot_chart_selection == "standard":
            return create_player_shot_chart_filters(player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)
         elif shot_chart_selection == "zone":
            return create_player_zone_chart_filters(player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)
         elif shot_chart_selection == "heat":
            return create_player_heatmap_filters(player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)
         elif shot_chart_selection == "stream":
            return create_player_stream_plot(player)
      elif chart_type == 'team':
          if shot_chart_selection == "standard":
            return create_team_shot_chart_filters(team, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)
          elif shot_chart_selection == "zone":
              return create_team_zone_chart_filters(team, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)
          elif shot_chart_selection == "heat":
              return create_team_heatmap_filters(team, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)
          elif shot_chart_selection == "stream":
              return create_team_stream_plot(team)
    elif chart_type == 'player':
        if shot_chart_selection == "standard":
            return create_player_shot_chart(player)
        elif shot_chart_selection == "zone":
            return create_player_zone_chart(player)
        elif shot_chart_selection == "heat":
            return create_player_heatmap(player)
        elif shot_chart_selection == "stream":
            return create_player_stream_plot(player)
    elif chart_type == 'team':
        if shot_chart_selection == "standard":
            return create_team_shot_chart(team)
        elif shot_chart_selection == "zone":
            return create_team_zone_chart(team)
        elif shot_chart_selection == "heat":
            return create_team_heatmap(team)
        elif shot_chart_selection == "stream":
            return create_team_stream_plot(team)

# DO NOT EDIT BELOW THIS LINE (except to change `jupyter_height`)
if __name__ == '__main__':
    app.run(debug=True)