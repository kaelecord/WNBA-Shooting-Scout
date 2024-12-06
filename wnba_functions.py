# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:12:07 2024

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

def hex_to_rgba(hex_color, opacity=0.2):
    # Remove the "#" from the hex color string if present
    hex_color = hex_color.lstrip('#')
    # Convert hex to RGB
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    # Return the RGBA format with the given opacity
    return f'rgba({r}, {g}, {b}, {opacity})'

def check_three_zones(x, y):
  if y <= -118/3:
    if x > 0:
      return "LC3" # left corner 3
    else:
      return "RC3" # right corner 3
  elif ((28/9)*(x-15))>=y:
    return "LW3" # left wing 3
  elif((-28/9)*(x+15))>=y:
    return "RW3" # right wing 3
  else:
    return "C3" # center 3 (aka top of key)

def check_two_zones(x, y):
  # 1st check if the shot falls within the inner most circle
  if ((x**2) + (y+42)**2) <= 81:
    return "Rim2"
  # Next check if the shots falls in the second ring of zones
  elif ((x**2) + (y+42)**2) <= 289:
    if ((28/9)*(x-15))>=y:
      return "LMR2" # Left mid range 2
    elif ((-28/9)*(x+15))>=y:
      return "RMR2" # right mid range 2
    else:
      return "CMR2" # center mid range 2
  # if shot is not in either one of those "rings" then it falls in the outer most band on zones
  else:
    if ((12/9)*(x-25))-20 >=y:
      return "LCL2" # left corner long 2
    elif ((28/9)*(x-15))>=y:
      return "LWL2" # left wing long 2
    elif ((-12/9)*(x+25))-20 >=y:
      return "RCL2" # right corner long 2
    elif ((-28/9)*(x+15))>=y:
      return "RWL2" # right wing long 2
    else:
      return "CL2" # center long 2
  
def get_shot_zone(row):
    if row['shot_type'] == '3 Pointer':
        return check_three_zones(row['coordinate_x'], row['coordinate_y'])
    elif row['shot_type'] == 'Free Throw':
        return 'FT'
    else:  # For 2 Pointers or other shot types
        return check_two_zones(row['coordinate_x'], row['coordinate_y'])

def get_arc_coordinates(center, radius, start_angle, end_angle, num_points=100):
    """
    Generate a list of coordinates along an arc of a circle.

    Args:
    - center (tuple): The center of the circle (x_c, y_c).
    - radius (float): The radius of the circle.
    - start_angle (float): The starting angle of the arc in radians.
    - end_angle (float): The ending angle of the arc in radians.
    - num_points (int): The number of points to generate along the arc.

    Returns:
    - list of tuples: A list of (x, y) coordinates along the arc.
    """
    # Generate angles between start and end


    # Generate angles between start and end in a smooth way
    angles = np.linspace(start_angle, end_angle, num_points)

    # Calculate coordinates based on the parametric equations of the circle
    x_coords = center[0] + radius * np.cos(angles)
    y_coords = center[1] + radius * np.sin(angles)

    # Return as a list of (x, y) tuples
    return list(x_coords), list(y_coords)

def get_rim2_coords():
  x, y = get_arc_coordinates((0,-42), 9,((-np.pi/6)-0.06), (((7*np.pi)/6) + 0.06), 1000)
  x.insert(0,0)
  y.insert(0,-47)
  x.insert(1,7.48331) # from desmos
  y.insert(1,-47)
  x.append(-7.48331)
  y.append(-47)
  x.append(0)
  y.append(-47)
  return x, y

def get_rmr2_coords():
  x, y = get_arc_coordinates((0,-42), 9, ((np.pi/2)+0.4705), (((7*np.pi)/6)+0.06552)) #inner arc
  x.reverse()
  y.reverse()
  outer_x, outer_y = get_arc_coordinates((0,-42), 17, ((np.pi/2)+0.395), (((7*np.pi)/6)-0.225)) #outer arc
  x.insert(0,-12)
  y.insert(0,-47)
  x.insert(1,-7.48331)
  y.insert(1,-47)
  x.append(-6.54331)
  y.append(-26.30971)
  x.extend(outer_x)
  y.extend(outer_y)
  x.append(-12)
  y.append(-47)

  return x, y

def get_lmr2_coords():
  x, y = get_arc_coordinates((0,-42), 9, ((np.pi/2)-0.4705), (((-1*np.pi)/6)-0.06552)) #inner arc
  x.reverse()
  y.reverse()
  outer_x, outer_y = get_arc_coordinates((0,-42), 17, ((np.pi/2)-0.395), (((-1*np.pi)/6)+0.225)) #outer arc
  x.insert(0,12)
  y.insert(0,-47)
  x.insert(1,7.48331)
  y.insert(1,-47)
  x.append(6.54331)
  y.append(-26.30971)
  x.extend(outer_x)
  y.extend(outer_y)
  x.append(12)
  y.append(-47)

  return x, y

def get_cmr2_coords():
  x, y = get_arc_coordinates((0,-42), 9, ((np.pi/2)+0.4705), ((np.pi/2)-0.4705)) #inner arc
  outer_x, outer_y = get_arc_coordinates((0,-42), 17, ((np.pi/2)-0.395), ((np.pi/2)+0.395)) #outer arc
  x.append(6)
  y.append(-28)
  x.extend(outer_x)
  y.extend(outer_y)
  x.append(-6)
  y.append(-28)
  x.append(-4.07873)
  y.append(-33.97728)

  return x,y

def get_rc3_coords():
  x = [-25, -22, -22, -25, -25]
  y = [-47, -47, (-118/3), (-118/3), -47]

  return x, y

def get_lc3_coords():
  x = [25, 22, 22, 25, 25]
  y = [-47, -47, (-118/3), (-118/3), -47]

  return x, y

def get_rw3_coords():
  x = [-22, -25, -25, -15, -7.83761]
  y = [(-118/3), (-118/3),0, 0, -22.28299]
  three_arc_x, three_arc_y = get_arc_coordinates((0,-43), 22.15, ((np.pi/2)+0.361676), (np.pi-0.16627))
  x.extend(three_arc_x)
  y.extend(three_arc_y)

  return x, y

def get_c3_coords():
  x = [-7.83761, -15, 15, 7.83761]
  y = [-22.28299, 0, 0, -22.28299]

  three_arc_x, three_arc_y = get_arc_coordinates((0,-43), 22.15, ((np.pi/2)-0.361676), ((np.pi/2)+0.361676))

  x.extend(three_arc_x)
  y.extend(three_arc_y)

  return x, y

def get_lw3_coords():
  x = [22, 25, 25, 15, 7.83761]
  y = [(-118/3), (-118/3),0, 0, -22.28299]

  three_arc_x, three_arc_y = get_arc_coordinates((0,-43), 22.15, ((np.pi/2)-0.361676), (0.16627))
  x.extend(three_arc_x)
  y.extend(three_arc_y)

  return x, y

def get_rcl2_coords():
  x = [-16.24808, -22, -22]
  y = [-47, -47, (-118/3)]
  outer_x, outer_y = get_arc_coordinates((0,-43), 22.15, (np.pi-0.16627), (((3*np.pi)/4)+0.141803)) #outer arc
  inner_x, inner_y = get_arc_coordinates((0,-42), 17, (((3*np.pi)/4)+0.27101), (((7*np.pi)/6)-0.225)) #inner arc
  x.extend(outer_x)
  y.extend(outer_y)
  x.extend(inner_x)
  y.extend(inner_y)

  return x, y

def get_rwl2_coords():
  x, y = get_arc_coordinates((0,-42), 17, ((np.pi/2)+0.395), (((3*np.pi)/4)+0.27101)) # inner arc
  outer_x, outer_y = get_arc_coordinates((0,-43), 22.15, (((3*np.pi)/4)+0.1418), ((np.pi/2)+0.361676))
  x.extend(outer_x)
  y.extend(outer_y)
  x.append(x[0])
  y.append(y[0])

  return x, y

def get_cl2_coords():
  x,y = get_arc_coordinates((0,-42), 17, ((np.pi/2)-0.395), ((np.pi/2)+0.395)) #inner arc
  outer_x, outer_y = get_arc_coordinates((0,-43), 22.15, ((np.pi/2)+0.361676), ((np.pi/2)-0.361676))
  x.extend(outer_x)
  y.extend(outer_y)
  x.append(x[0])
  y.append(y[0])

  return x, y

def get_lwl2_coords():
  x, y = get_arc_coordinates((0,-43), 22.15, ((np.pi/4)-0.141803), ((np.pi/2)-0.361676)) #outer arc
  inner_x, inner_y = get_arc_coordinates((0,-42), 17, ((np.pi/2)-0.395), ((np.pi/4)-0.27101)) #inner arc
  x.extend(inner_x)
  y.extend(inner_y)
  x.append(x[0])
  y.append(y[0])

  return x, y

def get_lcl2_coords():
  x = [16.24808, 22, 22]
  y = [-47, -47, (-118/3)]

  outer_x, outer_y = get_arc_coordinates((0,-43), 22.15, (0.16627), ((np.pi/4)-0.141803)) #outer arc
  inner_x, inner_y = get_arc_coordinates((0,-42), 17, ((np.pi/4)-0.27101), (((-1*np.pi)/6)+0.225)) #inner arc
  x.extend(outer_x)
  y.extend(outer_y)
  x.extend(inner_x)
  y.extend(inner_y)

  return x, y

rim2_x, rim2_y = get_rim2_coords()
rmr2_x, rmr2_y = get_rmr2_coords()
lmr2_x, lmr2_y = get_lmr2_coords()
cmr2_x, cmr2_y = get_cmr2_coords()
rc3_x, rc3_y = get_rc3_coords()
lc3_x, lc3_y = get_lc3_coords()
rw3_x, rw3_y = get_rw3_coords()
c3_x, c3_y = get_c3_coords()
lw3_x, lw3_y = get_lw3_coords()
rcl2_x, rcl2_y = get_rcl2_coords()
rwl2_x, rwl2_y = get_rwl2_coords()
cl2_x, cl2_y = get_cl2_coords()
lwl2_x, lwl2_y = get_lwl2_coords()
lcl2_x, lcl2_y = get_lcl2_coords()

zone_coordinates = {
    "Rim2": (rim2_x, rim2_y),
    "RMR2": (rmr2_x, rmr2_y),
    "LMR2": (lmr2_x, lmr2_y),
    "CMR2": (cmr2_x, cmr2_y),
    "RC3": (rc3_x, rc3_y),
    "LC3": (lc3_x, lc3_y),
    "RW3": (rw3_x, rw3_y),
    "C3": (c3_x, c3_y),
    "LW3": (lw3_x, lw3_y),
    "RCL2": (rcl2_x, rcl2_y),
    "RWL2": (rwl2_x, rwl2_y),
    "CL2": (cl2_x, cl2_y),
    "LWL2": (lwl2_x, lwl2_y),
    "LCL2": (lcl2_x, lcl2_y)
}

zone_annotation_coordinates = {
    "Rim2": (0, -38),
    "RMR2": (-12, -37),
    "LMR2": (12, -37),
    "CMR2": (0, 35),
    "RC3": (-23.5, -43),
    "LC3": (23.5, -43),
    "RW3": (-19, -19),
    "C3": (0, -17),
    "LW3": (19, -19),
    "RCL2": (-19, -38),
    "RWL2": (-12, -27),
    "CL2": (0, -23),
    "LWL2": (12, -27),
    "LCL2": (19, -38)
}
def get_team_shots_data(team):
  temp = wnba_shots[(wnba_shots['team_display_name_shooter'] == team) & (wnba_shots['coordinate_y'] <= 0) & (wnba_shots['shot_zone'] != 'FT')].reset_index(drop=True)
 
  return temp

def get_player_shots_data(player):
  temp = wnba_shots[(wnba_shots['athlete_display_name_shooter'] == player) & (wnba_shots['coordinate_y'] <= 0) & (wnba_shots['shot_zone'] != 'FT')].reset_index(drop=True)
  
  return temp

def get_team_shots_data_filters(team, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked):
  temp = copy.deepcopy(wnba_shots)

  if start_date is not None:
    start_date_string = start_date
  else:
    start_date_string = "2024-05-14"
  if end_date is not None:
    end_date_string = end_date
  else:
    end_date_string = '2024-09-17'
  if len(opponent) > 0:
      opponent_ids = [wnba_team_id_map[opponent_name] for opponent_name in opponent]
  else:
      opponent_ids = list(wnba_team_id_map.values())
  if len(location) > 0:
    chosen_location = location
  else:
    chosen_location = ['home', 'away']
  if len(half) > 0:
    chosen_half = half
  else:
    chosen_half = []
  if len(qtr) > 0:
    chosen_qtr = qtr
  else:
    chosen_qtr = []
  if len(game) > 0:
    chosen_game = game
  else:
    chosen_game = list(wnba_shots['game_info'].unique())
  if len(assisted) > 0:
    chosen_assisted = assisted
  else:
    chosen_assisted = [True,False]
  if len(blocked) > 0:
    chosen_blocked = blocked
  else:
    chosen_blocked = [True, False]
  temp = wnba_shots[(wnba_shots['team_display_name_shooter'] == team) & (wnba_shots['coordinate_y'] <= 0) & (wnba_shots['shot_zone'] != 'FT') & (wnba_shots['game_date'] >= start_date_string) & (wnba_shots['game_date'] <= end_date_string) & (wnba_shots['opponent_id'].isin(opponent_ids)) & (wnba_shots['location'].isin(chosen_location)) & (wnba_shots['half'].isin(chosen_half)) & (wnba_shots['qtr'].isin(chosen_qtr)) & (wnba_shots['game_info'].isin(chosen_game)) & (wnba_shots['assisted'].isin(chosen_assisted)) & (wnba_shots['blocked'].isin(chosen_blocked))
].reset_index(drop=True)
  return temp

def get_player_shots_data_filters(player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked):
    temp = copy.deepcopy(wnba_shots)

    if start_date is not None:
      #start_date_object = date.fromisoformat(start_date)
      #start_date_string = start_date_object.strftime('%YYYY-%MM-%DD')
      start_date_string = start_date
    else:
      start_date_string = "2024-05-14"
    if end_date is not None:
      #end_date_object = date.fromisoformat(end_date)
      #end_date_string = end_date_object.strftime('%YYYY-%MM-%DD')
      end_date_string = end_date
    else:
      end_date_string = '2024-09-17'
    if len(opponent) > 0:
      opponent_ids = [wnba_team_id_map[opponent_name] for opponent_name in opponent]
    else:
      opponent_ids = list(wnba_team_id_map.values())
    if len(location) > 0:
      chosen_location = location
    else:
      chosen_location = ['home', 'away']
    if len(half) > 0:
      chosen_half = half
    else:
      chosen_half = []
    if len(qtr) > 0:
      chosen_qtr = qtr
    else:
      chosen_qtr = []
    if len(game) > 0:
      chosen_game = game
    else:
      chosen_game = list(wnba_shots['game_info'].unique())
    if len(assisted) > 0:
      chosen_assisted = assisted
    else:
      chosen_assisted = [True,False]
    if len(blocked) > 0:
      chosen_blocked = blocked
    else:
      chosen_blocked = [True, False]

    temp = wnba_shots[(wnba_shots['athlete_display_name_shooter'] == player) & (wnba_shots['coordinate_y'] <= 0) & (wnba_shots['shot_zone'] != 'FT') & (wnba_shots['game_date'] >= start_date_string) & (wnba_shots['game_date'] <= end_date_string) & (wnba_shots['opponent_id'].isin(opponent_ids)) & (wnba_shots['location'].isin(chosen_location)) & (wnba_shots['half'].isin(chosen_half)) & (wnba_shots['qtr'].isin(chosen_qtr)) & (wnba_shots['game_info'].isin(chosen_game)) & (wnba_shots['assisted'].isin(chosen_assisted)) & (wnba_shots['blocked'].isin(chosen_blocked))
].reset_index(drop=True)
    return temp

def get_team_totals_data(team):
  team_id = wnba_team_id_map[team]
  wnba_team_data = wnba_shots[wnba_shots['offensive_team_id'] == team_id].reset_index(drop=True)
  wnba_team_totals = wnba_team_data.groupby(['game_id', 'offensive_team_id']).agg({'game_date':'max','score_value': 'sum'}).reset_index()
  wnba_team_totals = wnba_team_totals[wnba_team_totals['offensive_team_id'].isin(wnba_team_map.keys())]
  wnba_team_totals = pd.merge(wnba_team_totals, wnba_shots[['game_date', 'home_team_id', 'offensive_team_id','away_team_id']].drop_duplicates(), on=['game_date', 'offensive_team_id'], how='left')
  wnba_team_totals['is_home_team'] = wnba_team_totals['offensive_team_id'] == wnba_team_totals['home_team_id']
  wnba_team_totals['opponent_team_id'] = wnba_team_totals['away_team_id'].where(wnba_team_totals['is_home_team'] == True, wnba_team_totals['home_team_id'])
  wnba_team_totals['team_name'] = wnba_team_totals['offensive_team_id'].apply(lambda x: wnba_team_map[x])
  wnba_team_totals['team_logo'] = wnba_team_totals['offensive_team_id'].apply(lambda x: wnba_team_logo_map[x])
  wnba_team_totals['opponent_name'] = wnba_team_totals['opponent_team_id'].apply(lambda x: wnba_team_map[x])
  wnba_team_totals['opponent_logo'] = wnba_team_totals['opponent_team_id'].apply(lambda x: wnba_team_logo_map[x])
  wnba_team_totals['team_color'] = wnba_team_totals['offensive_team_id'].apply(lambda x: wnba_team_primary_color_map[x])
  wnba_team_totals['team_alternate_color'] = wnba_team_totals['offensive_team_id'].apply(lambda x: wnba_team_secondary_color_map[x])

  wnba_team_totals.sort_values(by='game_date', ascending=True).reset_index(drop=True)
  return wnba_team_totals

def get_player_totals_data(player):
  wnba_player_data = wnba_shots[wnba_shots['athlete_display_name_shooter'] == player].reset_index(drop=True)

  wnba_player_totals = wnba_player_data.groupby(['game_id', 'shooter_id', 'athlete_display_name_shooter', 'offensive_team_id', 'athlete_headshot_href_shooter']).agg({'game_date':'max','score_value': 'sum'}).reset_index()
  wnba_player_totals = wnba_player_totals[wnba_player_totals['offensive_team_id'].isin(wnba_team_map.keys())]
  wnba_player_totals = pd.merge(wnba_player_totals, wnba_shots[['game_date', 'home_team_id', 'offensive_team_id','away_team_id']].drop_duplicates(), on=['game_date', 'offensive_team_id'], how='left')
  wnba_player_totals['is_home_team'] = wnba_player_totals['offensive_team_id'] == wnba_player_totals['home_team_id']
  wnba_player_totals['opponent_team_id'] = wnba_player_totals['away_team_id'].where(wnba_player_totals['is_home_team'] == True, wnba_player_totals['home_team_id'])
  wnba_player_totals['team_name'] = wnba_player_totals['offensive_team_id'].apply(lambda x: wnba_team_map[x])
  wnba_player_totals['team_logo'] = wnba_player_totals['offensive_team_id'].apply(lambda x: wnba_team_logo_map[x])
  wnba_player_totals['opponent_name'] = wnba_player_totals['opponent_team_id'].apply(lambda x: wnba_team_map[x])
  wnba_player_totals['opponent_logo'] = wnba_player_totals['opponent_team_id'].apply(lambda x: wnba_team_logo_map[x])
  wnba_player_totals['team_color'] = wnba_player_totals['offensive_team_id'].apply(lambda x: wnba_team_primary_color_map[x])
  wnba_player_totals['team_alternate_color'] = wnba_player_totals['offensive_team_id'].apply(lambda x: wnba_team_secondary_color_map[x])

  wnba_player_totals.sort_values(by='game_date', ascending=True).reset_index(drop=True)

  return wnba_player_totals

def get_team_stream_data(team):
  team_id = wnba_team_id_map[team]
  wnba_team_data = wnba_shots[wnba_shots['offensive_team_id'] == team_id].reset_index(drop=True)

  wnba_team_data['shot_distance_rounded'] = wnba_team_data['shot_distance'].round(0)
  wnba_team_data['made_three'] = wnba_team_data['score_value'].apply(lambda x: 1 if x == 3 else 0)

  wnba_team_stream = wnba_team_data[wnba_team_data['shot_type']!='Free Throw'].groupby(['offensive_team_id', 'shot_distance_rounded']).agg({'shooter_id': 'count',
                                                                                                                                    'scoring_play': 'sum',
                                                                                                                                    'made_three': 'sum'}).reset_index()
  wnba_team_stream = wnba_team_stream[wnba_team_stream['offensive_team_id'].isin(wnba_team_map.keys())]
  wnba_team_stream['team_name'] = wnba_team_stream['offensive_team_id'].apply(lambda x: wnba_team_map[x])
  wnba_team_stream['team_logo'] = wnba_team_stream['offensive_team_id'].apply(lambda x: wnba_team_logo_map[x])
  wnba_team_stream['team_color'] = wnba_team_stream['offensive_team_id'].apply(lambda x: wnba_team_primary_color_map[x])
  wnba_team_stream['team_alternate_color'] = wnba_team_stream['offensive_team_id'].apply(lambda x: wnba_team_secondary_color_map[x])
  wnba_team_stream['fg_pct'] = wnba_team_stream['scoring_play'] / wnba_team_stream['shooter_id']
  wnba_team_stream['efg_pct'] = (wnba_team_stream['scoring_play'] + (wnba_team_stream['made_three']/2)) / wnba_team_stream['shooter_id']

  wnba_team_stream.rename(columns={'shot_distance_rounded': 'shot_distance',
                                  'offensive_team_id': 'team_id',
                                  'shooter_id': 'shot_count'}, inplace=True)

  team_data = wnba_team_stream[wnba_team_stream['shot_count']>=10].reset_index(drop=True)

  return team_data

def get_player_stream_data(player):
  wnba_player_data = wnba_shots[wnba_shots['athlete_display_name_shooter'] == player].reset_index(drop=True)

  wnba_player_data['shot_distance_rounded'] = wnba_player_data['shot_distance'].round(0)
  wnba_player_data['made_three'] = wnba_player_data['score_value'].apply(lambda x: 1 if x == 3 else 0)

  wnba_player_stream = wnba_player_data[wnba_player_data['shot_type']!='Free Throw'].groupby(['shot_distance_rounded', 'shooter_id', 'athlete_display_name_shooter', 'offensive_team_id', 'athlete_headshot_href_shooter']).agg({'shot_type': 'count',
                                                                                                                                    'scoring_play': 'sum',
                                                                                                                                    'made_three': 'sum'}).reset_index()
  wnba_player_stream = wnba_player_stream[wnba_player_stream['offensive_team_id'].isin(wnba_team_map.keys())]
  wnba_player_stream['team_name'] = wnba_player_stream['offensive_team_id'].apply(lambda x: wnba_team_map[x])
  wnba_player_stream['team_logo'] = wnba_player_stream['offensive_team_id'].apply(lambda x: wnba_team_logo_map[x])
  wnba_player_stream['team_color'] = wnba_player_stream['offensive_team_id'].apply(lambda x: wnba_team_primary_color_map[x])
  wnba_player_stream['team_alternate_color'] = wnba_player_stream['offensive_team_id'].apply(lambda x: wnba_team_secondary_color_map[x])
  wnba_player_stream['fg_pct'] = wnba_player_stream['scoring_play'] / wnba_player_stream['shot_type']
  wnba_player_stream['efg_pct'] = (wnba_player_stream['scoring_play'] + (wnba_player_stream['made_three']/2)) / wnba_player_stream['shot_type']

  wnba_player_stream.rename(columns={'athlete_display_name_shooter':'player_name',
                                    'athlete_headshot_href_shooter':'player_headshot',
                                    'shot_distance_rounded': 'shot_distance',
                                  'offensive_team_id': 'team_id',
                                  'shot_type': 'shot_count'}, inplace=True)
  #wnba_player_stream
  player_data = wnba_player_stream[wnba_player_stream['shot_count']>=5].reset_index(drop=True)

  return player_data

def get_team_zone_data(team):  
  wnba_team_data = get_team_shots_data(team)
  team_zone_averages = wnba_team_data.groupby(['shot_zone']).agg({'shooter_id':'count','scoring_play': 'sum'}).reset_index()
  team_zone_averages.rename(columns={'shooter_id':'zone_attempts', 'scoring_play':'zone_makes', 'shot_zone':'zone'}, inplace=True)
  team_zone_averages['zone_shooting_pct'] = team_zone_averages['zone_makes']/team_zone_averages['zone_attempts']

  team_zone_averages = pd.merge(team_zone_averages, wnba_zone_league_averages, on='zone', suffixes=('_team', '_league'))
  team_zone_averages['zone_shooting_pct_diff'] = team_zone_averages['zone_shooting_pct_team'] - team_zone_averages['zone_shooting_pct_league']

  color_bins = [-float('inf'), -0.05, -0.025, 0, 0.025, 0.05, float('inf')]  # Define the ranges for color categories
  color_labels = ['#03045e', '#00b4d8', '#caf0f8', '#df8080', '#cb0b0a', '#8e0413']  # Color labels for each range

  # Create new column 'color' based on `zone_shooting_pct_diff`
  team_zone_averages['color'] = pd.cut(team_zone_averages['zone_shooting_pct_diff'], bins=color_bins, labels=color_labels, right=False)

  return team_zone_averages

def get_player_zone_data(player):
  wnba_player_data = get_player_shots_data(player)
  player_zone_averages = wnba_player_data.groupby(['shot_zone']).agg({'shooter_id':'count','scoring_play': 'sum'}).reset_index()
  player_zone_averages.rename(columns={'shooter_id':'zone_attempts', 'scoring_play':'zone_makes', 'shot_zone':'zone'}, inplace=True)
  player_zone_averages['zone_shooting_pct'] = player_zone_averages['zone_makes']/player_zone_averages['zone_attempts']

  player_zone_averages = pd.merge(player_zone_averages, wnba_zone_league_averages, on='zone', suffixes=('_team', '_league'))
  player_zone_averages['zone_shooting_pct_diff'] = player_zone_averages['zone_shooting_pct_team'] - player_zone_averages['zone_shooting_pct_league']

  color_bins = [-float('inf'), -0.05, -0.025, 0, 0.025, 0.05, float('inf')]  # Define the ranges for color categories
  color_labels = ['#03045e', '#00b4d8', '#caf0f8', '#df8080', '#cb0b0a', '#8e0413']  # Color labels for each range

  # Create new column 'color' based on `zone_shooting_pct_diff`
  player_zone_averages['color'] = pd.cut(player_zone_averages['zone_shooting_pct_diff'], bins=color_bins, labels=color_labels, right=False)

  return player_zone_averages

def get_team_zone_data_filters(team, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked):
  wnba_team_data = get_team_shots_data_filters(team, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)

  team_zone_averages = wnba_team_data.groupby(['shot_zone']).agg({'shooter_id':'count','scoring_play': 'sum'}).reset_index()
  team_zone_averages.rename(columns={'shooter_id':'zone_attempts', 'scoring_play':'zone_makes', 'shot_zone':'zone'}, inplace=True)
  team_zone_averages['zone_shooting_pct'] = team_zone_averages['zone_makes']/team_zone_averages['zone_attempts']

  team_zone_averages = pd.merge(team_zone_averages, wnba_zone_league_averages, on='zone', suffixes=('_team', '_league'))
  team_zone_averages['zone_shooting_pct_diff'] = team_zone_averages['zone_shooting_pct_team'] - team_zone_averages['zone_shooting_pct_league']

  color_bins = [-float('inf'), -0.05, -0.025, 0, 0.025, 0.05, float('inf')]  # Define the ranges for color categories
  color_labels = ['#03045e', '#00b4d8', '#caf0f8', '#df8080', '#cb0b0a', '#8e0413']  # Color labels for each range

  # Create new column 'color' based on `zone_shooting_pct_diff`
  team_zone_averages['color'] = pd.cut(team_zone_averages['zone_shooting_pct_diff'], bins=color_bins, labels=color_labels, right=False)

  return team_zone_averages

def get_player_zone_data_filters(player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked):
  wnba_player_data = get_player_shots_data_filters(player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)

  player_zone_averages = wnba_player_data.groupby(['shot_zone']).agg({'shooter_id':'count','scoring_play': 'sum'}).reset_index()
  player_zone_averages.rename(columns={'shooter_id':'zone_attempts', 'scoring_play':'zone_makes', 'shot_zone':'zone'}, inplace=True)
  player_zone_averages['zone_shooting_pct'] = player_zone_averages['zone_makes']/player_zone_averages['zone_attempts']

  player_zone_averages = pd.merge(player_zone_averages, wnba_zone_league_averages, on='zone', suffixes=('_team', '_league'))
  player_zone_averages['zone_shooting_pct_diff'] = player_zone_averages['zone_shooting_pct_team'] - player_zone_averages['zone_shooting_pct_league']

  color_bins = [-float('inf'), -0.05, -0.025, 0, 0.025, 0.05, float('inf')]  # Define the ranges for color categories
  color_labels = ['#03045e', '#00b4d8', '#caf0f8', '#df8080', '#cb0b0a', '#8e0413']  # Color labels for each range

  # Create new column 'color' based on `zone_shooting_pct_diff`
  player_zone_averages['color'] = pd.cut(player_zone_averages['zone_shooting_pct_diff'], bins=color_bins, labels=color_labels, right=False)

  return player_zone_averages

def get_team_color_headshot_for_player(player):
  player_data = wnba_shots[wnba_shots['athlete_display_name_shooter'] == player].reset_index(drop=True)
  color = '#'+player_data['team_color_shooter'].iloc[0]
  headshot = player_data['athlete_headshot_href_shooter'].iloc[0]
  return color, headshot

def get_team_color_logo_for_team(team):
  team_id = wnba_team_id_map[team]
  color = wnba_team_primary_color_map[team_id]
  logo = wnba_team_logo_map[team_id]
  return color, logo

def get_team_logo_headshot_for_player(player):
  player_data = wnba_shots[wnba_shots['athlete_display_name_shooter'] == player].reset_index(drop=True)
  team_logo = player_data['team_logo_shooter'].iloc[0]
  headshot = player_data['athlete_headshot_href_shooter'].iloc[0]
  return team_logo, headshot

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, -43), radius=.75, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-2.5, -43.75), 5, -.1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    #outer_box = Rectangle((-80, -70), 160, 190, linewidth=lw, color=color,
                          #fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-8, -47), 16, 19, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, -28), 12, 12, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, -28), 12, 12, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, -43), 4, 4, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-22, -47), 0, 7.67, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((22, -47), 0, 7.67, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the
    # threes
    three_arc = Arc((0, -43), 44.3, 50, theta1=9, theta2=171, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 0), 12, 12, theta1=180, theta2=0,
                           linewidth=lw, color=color)


    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, inner_box, top_free_throw, bottom_free_throw, restricted, corner_three_a, corner_three_b, three_arc, center_outer_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-25, -47), 50, 47, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

def create_player_hexbin(player):
  player_data = wnba_shots[(wnba_shots['athlete_display_name_shooter'] == player) & (wnba_shots['shot_zone'] != "FT")].reset_index(drop=True)
  player_zone = get_player_zone_data(player)
  # Create a single figure
  fig, ax = plt.subplots(figsize=(7.5, 7.05))
  ax.set_xlim(-25, 25)
  ax.set_ylim(-47, 0)

  total_shots = len(player_data)
  combined_hex = ax.hexbin(
        player_data.coordinate_x, player_data.coordinate_y,
        extent=(-25, 25, -47, 0), gridsize=40, mincnt=1, visible=False
    )
  total_shots_by_hex = combined_hex.get_array()
  total_freq_by_hex = total_shots_by_hex / total_shots  # Frequency across all zones
  max_size = max(total_freq_by_hex)  # Normalize sizes by the largest hex

  # Extract hexbin offsets for the combined data
  combined_x, combined_y = combined_hex.get_offsets().T
  plt.close()

  # Loop through each unique shot zone
  for zone in player_data['shot_zone'].unique():
      # Filter data for the current zone
      player_one_zone = player_data[player_data['shot_zone'] == zone]

      # Create the hexbin plot
      zone_hex = ax.hexbin(
            player_one_zone.coordinate_x, player_one_zone.coordinate_y,
            extent=(-25, 25, -47, 0), gridsize=40, mincnt=1, visible=False
        )

      # Get hexbin data for the current zone
      zone_shots_by_hex = zone_hex.get_array()

      # Match current zone hexbin offsets to combined hexbin offsets
      zone_indices = [
          (i, combined_x[i], combined_y[i])
          for i in range(len(combined_x))
          if (combined_x[i], combined_y[i]) in zip(zone_hex.get_offsets()[:, 0], zone_hex.get_offsets()[:, 1])
      ]

      # Sizes are based on the total shot data's frequency
      sizes = [total_freq_by_hex[i] / max_size * 150 for i, _, _ in zone_indices]

      # Scatter plot for the zone
      color = player_zone[player_zone['zone'] == zone]['color'].values[0]
      for idx, size in zip(zone_indices, sizes):
          ax.scatter(
              combined_x[idx[0]], combined_y[idx[0]],
              s=size, color=color, marker='h', label=zone
          )

  # Add shot chart title

  # Draw the court
  draw_court(ax=ax, outer_lines=True,)
  #cur_axes = plt.gca()
  #cur_axes.axes.get_xaxis().set_visible(False)
  #cur_axes.axes.get_yaxis().set_visible(False)
  #for spine in cur_axes.spines.values():
    #spine.set_visible(False)
  ax.axis('off')
  # Show the final combined plot
  return fig

def create_team_hexbin(team):
  team_id = wnba_team_id_map[team]
  team_data = wnba_shots[(wnba_shots['offensive_team_id'] == team_id) & (wnba_shots['shot_zone'] != "FT")].reset_index(drop=True)
  team_zone = get_team_zone_data(team)
  # Create a single figure
  fig, ax = plt.subplots(figsize=(5, 4.7))
  ax.set_xlim(-25, 25)
  ax.set_ylim(-47, 0)

  total_shots = len(team_data)
  combined_hex = ax.hexbin(
        team_data.coordinate_x, team_data.coordinate_y,
        extent=(-25, 25, -47, 0), gridsize=40, mincnt=1, visible=False
    )
  total_shots_by_hex = combined_hex.get_array()
  total_freq_by_hex = total_shots_by_hex / total_shots  # Frequency across all zones
  max_size = max(total_freq_by_hex)  # Normalize sizes by the largest hex

  # Extract hexbin offsets for the combined data
  combined_x, combined_y = combined_hex.get_offsets().T
  plt.close()

  # Loop through each unique shot zone
  for zone in team_data['shot_zone'].unique():
      # Filter data for the current zone
      team_one_zone = team_data[team_data['shot_zone'] == zone]

      # Create the hexbin plot
      zone_hex = ax.hexbin(
            team_one_zone.coordinate_x, team_one_zone.coordinate_y,
            extent=(-25, 25, -47, 0), gridsize=40, mincnt=1, visible=False
        )

      # Get hexbin data for the current zone
      zone_shots_by_hex = zone_hex.get_array()

      # Match current zone hexbin offsets to combined hexbin offsets
      zone_indices = [
          (i, combined_x[i], combined_y[i])
          for i in range(len(combined_x))
          if (combined_x[i], combined_y[i]) in zip(zone_hex.get_offsets()[:, 0], zone_hex.get_offsets()[:, 1])
      ]

      # Sizes are based on the total shot data's frequency
      sizes = [total_freq_by_hex[i] / max_size * 150 for i, _, _ in zone_indices]

      # Scatter plot for the zone
      color = team_zone[team_zone['zone'] == zone]['color'].values[0]
      for idx, size in zip(zone_indices, sizes):
          ax.scatter(
              combined_x[idx[0]], combined_y[idx[0]],
              s=size, color=color, marker='h', label=zone
          )

  # Add shot chart title

  # Draw the court
  draw_court(ax=ax, outer_lines=True,)
  #cur_axes = plt.gca()
  #cur_axes.axes.get_xaxis().set_visible(False)
  #cur_axes.axes.get_yaxis().set_visible(False)
  #for spine in cur_axes.spines.values():
    #spine.set_visible(False)
  ax.axis('off')

  # Show the final combined plot
  return fig

def draw_plotly_court(fig, fig_width=600, margins=1):
    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200, closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    fig_height = fig_width * (47 + 2 * margins) / (50 + 2 * margins)
    fig.update_layout(width=fig_width, height=fig_height)

    # Set axes ranges
    fig.update_xaxes(range=[-25 - margins, 25 + margins])
    fig.update_yaxes(range=[-47 - margins, 0 + margins])

    threept_break_y = -47 + 7.694444444#89.47765084
    three_line_col = "#777777"
    main_line_col = "#777777"

    fig.update_layout(
        # Line Horizontal
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        shapes=[
            dict(
                type="rect", x0=-25, y0=-47, x1=25, y1=0,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="rect", x0=-8, y0=-47, x1=8, y1=-28,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="rect", x0=-6, y0=-47, x1=6, y1=-28,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="circle", x0=-6, y0=-34, x1=6, y1=-22, xref="x", yref="y",
                line=dict(color=main_line_col, width=1),
                # fillcolor='#dddddd',
                layer='below'
            ),
            #dict(
            #    type="line", x0=-60, y0=137.5, x1=60, y1=137.5,
            #    line=dict(color='red', width=1),
            #    layer='below'
            #),
            #dict(
            #   type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,
            #    line=dict(color="#ec7607", width=1),
            #    fillcolor='#ec7607',
            #),
            dict(
                type="circle", x0=-0.75, y0=-43.75, x1=0.75, y1=-42.25, xref="x", yref="y",
                line=dict(color="#ec7607", width=1),
            ),
            dict(
                type="line", x0=-2.5, y0=-43.75, x1=2.5, y1=-43.75,
                line=dict(color="#ec7607", width=1),
            ),
            dict(type="path",
                 path=ellipse_arc(y_center=-43, a=4, b=4, start_angle=0, end_angle=np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="path",
                 path=ellipse_arc(x_center=0, y_center=-43, a=22.27, b=22.15, start_angle=0.15, end_angle=np.pi - 0.15),
                 line=dict(color=three_line_col, width=1), layer='below'),
            dict(
                type="line", x0=-22, y0=-47, x1=-22, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=22, y0=-47, x1=22, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-25, y0=-19, x1=-22, y1=-19,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=25, y0=-19, x1=22, y1=-19,
                line=dict(color=main_line_col, width=1), layer='below'
            ),

            dict(
                type="line", x0=-9, y0=-40, x1=-8, y1=-40,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-9, y0=-39, x1=-8, y1=-39,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-9, y0=-35, x1=-8, y1=-35,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-9, y0=-32, x1=-8, y1=-32,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=9, y0=-40, x1=8, y1=-40,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=9, y0=-39, x1=8, y1=-39,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=9, y0=-35, x1=8, y1=-35,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=9, y0=-32, x1=8, y1=-32,
                line=dict(color=main_line_col, width=1), layer='below'
            ),

            dict(type="path",
                 path=ellipse_arc(y_center=0, a=6, b=6, start_angle=-0, end_angle=-np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),

        ]
    )
    return True

def create_team_shot_chart(team):
  team_shots = get_team_shots_data(team)

  make_color = f"#{team_shots['team_color_shooter'][0]}"
  miss_color = f"#{team_shots['team_alternate_color_shooter'][0]}"

  scored_shots = team_shots[team_shots['scoring_play'] == 1]
  missed_shots = team_shots[team_shots['scoring_play'] == 0]
  if miss_color == "#0":
    miss_color = '#000000'

  # Create hover text for scored shots
  scored_text = [
      f"Shooter: {row['athlete_display_name_shooter']}<br>Assisted: {'Yes' if row['assisted'] else 'No'}"
      for _, row in scored_shots.iterrows()
  ]

  # Create hover text for missed shots
  missed_text = [
      f"Shooter: {row['athlete_display_name_shooter']}<br>Blocked: {'Yes' if row['blocked'] else 'No'}"
      for _, row in missed_shots.iterrows()
  ]

  fig = go.Figure()
  draw_plotly_court(fig)
  fig.add_trace(go.Scatter(
      x=scored_shots['coordinate_x'],
      y=scored_shots['coordinate_y'],
      mode='markers',
      marker=dict(color=make_color, symbol='circle', size=8, line=dict(width=1)),
      text=scored_text,
      hoverinfo="text",
      name='Made Shot'
  ))

  # Scatter plot for missed shots (red open circles)
  fig.add_trace(go.Scatter(
      x=missed_shots['coordinate_x'],
      y=missed_shots['coordinate_y'],
      mode='markers',
      marker=dict(color=miss_color, symbol='circle-open', size=8, line=dict(width=1), opacity=0.3),
      text=missed_text,
      hoverinfo="text",
      name='Missed Shot'
  ))
  fig.add_layout_image(
    go.layout.Image(
        source=team_shots['team_logo_shooter'][0],
        xref="x", yref="y", x=-24, y=-1, sizex=7.5, sizey=7.5,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))
  fig.update_layout(showlegend=False,
                    autosize=True,  # Enable autosizing to fill the container
                    margin=dict(l=0, r=0, t=0, b=0),  # Remove any margins
                    height=None,  # Avoid fixed height
                    width=None)   # Avoid fixed width

  return fig

def create_team_shot_chart_filters(team, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked):
  team_shots = get_team_shots_data_filters(team, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)

  make_color = f"#{team_shots['team_color_shooter'][0]}"
  miss_color = f"#{team_shots['team_alternate_color_shooter'][0]}"
  if miss_color == "#0":
    miss_color = '#000000'

  scored_shots = team_shots[team_shots['scoring_play'] == 1]
  missed_shots = team_shots[team_shots['scoring_play'] == 0]
  

  # Create hover text for scored shots
  scored_text = [
      f"Shooter: {row['athlete_display_name_shooter']}<br>Assisted: {'Yes' if row['assisted'] else 'No'}"
      for _, row in scored_shots.iterrows()
  ]

  # Create hover text for missed shots
  missed_text = [
      f"Shooter: {row['athlete_display_name_shooter']}<br>Blocked: {'Yes' if row['blocked'] else 'No'}"
      for _, row in missed_shots.iterrows()
  ]

  fig = go.Figure()
  draw_plotly_court(fig)
  fig.add_trace(go.Scatter(
      x=scored_shots['coordinate_x'],
      y=scored_shots['coordinate_y'],
      mode='markers',
      marker=dict(color=make_color, symbol='circle', size=8, line=dict(width=1)),
      text=scored_text,
      hoverinfo="text",
      name='Made Shot'
  ))

  # Scatter plot for missed shots (red open circles)
  fig.add_trace(go.Scatter(
      x=missed_shots['coordinate_x'],
      y=missed_shots['coordinate_y'],
      mode='markers',
      marker=dict(color=miss_color, symbol='circle-open', size=8, line=dict(width=1), opacity=0.3),
      text=missed_text,
      hoverinfo="text",
      name='Missed Shot'
  ))
  fig.add_layout_image(
    go.layout.Image(
        source=team_shots['team_logo_shooter'][0],
        xref="x", yref="y", x=-24, y=-1, sizex=7.5, sizey=7.5,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))
  fig.update_layout(showlegend=False,
                    autosize=True,  # Enable autosizing to fill the container
                    margin=dict(l=0, r=0, t=0, b=0),  # Remove any margins
                    height=None,  # Avoid fixed height
                    width=None)   # Avoid fixed width

  return fig

def create_player_shot_chart(player):
  player_shots = get_player_shots_data(player)

  make_color = f"#{player_shots['team_color_shooter'][0]}"
  miss_color = f"#{player_shots['team_alternate_color_shooter'][0]}"
  if miss_color == "#0":
    miss_color = '#000000'
    
  scored_shots = player_shots[player_shots['scoring_play'] == 1]
  missed_shots = player_shots[player_shots['scoring_play'] == 0]

  # Create hover text for scored shots
  scored_text = [
      f"Shooter: {row['athlete_display_name_shooter']}<br>Assisted: {'Yes' if row['assisted'] else 'No'}"
      for _, row in scored_shots.iterrows()
  ]

  # Create hover text for missed shots
  missed_text = [
      f"Shooter: {row['athlete_display_name_shooter']}<br>Blocked: {'Yes' if row['blocked'] else 'No'}"
      for _, row in missed_shots.iterrows()
  ]

  fig = go.Figure()
  draw_plotly_court(fig)
  fig.add_trace(go.Scatter(
      x=scored_shots['coordinate_x'],
      y=scored_shots['coordinate_y'],
      mode='markers',
      marker=dict(color=make_color, symbol='circle', size=8, line=dict(width=1)),
      text=scored_text,
      hoverinfo="text",
      name='Made Shot'
  ))

  # Scatter plot for missed shots (red open circles)
  fig.add_trace(go.Scatter(
      x=missed_shots['coordinate_x'],
      y=missed_shots['coordinate_y'],
      mode='markers',
      marker=dict(color=miss_color, symbol='circle-open', size=8, line=dict(width=1), opacity=0.3),
      text=missed_text,
      hoverinfo="text",
      name='Missed Shot'
  ))
  fig.add_layout_image(
    go.layout.Image(
        source=player_shots['athlete_headshot_href_shooter'][0],
        xref="x", yref="y", x=-25, y=0, sizex=11, sizey=7.5,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))
  fig.add_layout_image(
    go.layout.Image(
        source=player_shots['team_logo_shooter'][0],
        xref="x", yref="y", x=-25, y=0, sizex=3, sizey=3,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))
  fig.update_layout(showlegend=False,
                    autosize=True,  # Enable autosizing to fill the container
                    margin=dict(l=0, r=0, t=0, b=0),  # Remove any margins
                    height=None,  # Avoid fixed height
                    width=None)   # Avoid fixed width

  return fig

def create_player_shot_chart_filters(player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked):
  player_shots = get_player_shots_data_filters(player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)

  make_color = f"#{player_shots['team_color_shooter'][0]}"
  miss_color = f"#{player_shots['team_alternate_color_shooter'][0]}"
  if miss_color == "#0":
    miss_color = '#000000'
    
  scored_shots = player_shots[player_shots['scoring_play'] == 1]
  missed_shots = player_shots[player_shots['scoring_play'] == 0]

  # Create hover text for scored shots
  scored_text = [
      f"Shooter: {row['athlete_display_name_shooter']}<br>Assisted: {'Yes' if row['assisted'] else 'No'}"
      for _, row in scored_shots.iterrows()
  ]

  # Create hover text for missed shots
  missed_text = [
      f"Shooter: {row['athlete_display_name_shooter']}<br>Blocked: {'Yes' if row['blocked'] else 'No'}"
      for _, row in missed_shots.iterrows()
  ]

  fig = go.Figure()
  draw_plotly_court(fig)
  fig.add_trace(go.Scatter(
      x=scored_shots['coordinate_x'],
      y=scored_shots['coordinate_y'],
      mode='markers',
      marker=dict(color=make_color, symbol='circle', size=8, line=dict(width=1)),
      text=scored_text,
      hoverinfo="text",
      name='Made Shot'
  ))

  # Scatter plot for missed shots (red open circles)
  fig.add_trace(go.Scatter(
      x=missed_shots['coordinate_x'],
      y=missed_shots['coordinate_y'],
      mode='markers',
      marker=dict(color=miss_color, symbol='circle-open', size=8, line=dict(width=1), opacity=0.3),
      text=missed_text,
      hoverinfo="text",
      name='Missed Shot'
  ))
  fig.add_layout_image(
    go.layout.Image(
        source=player_shots['athlete_headshot_href_shooter'][0],
        xref="x", yref="y", x=-25, y=0, sizex=11, sizey=7.5,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))
  fig.add_layout_image(
    go.layout.Image(
        source=player_shots['team_logo_shooter'][0],
        xref="x", yref="y", x=-25, y=0, sizex=3, sizey=3,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))
  fig.update_layout(showlegend=False,
                    autosize=True,  # Enable autosizing to fill the container
                    margin=dict(l=0, r=0, t=0, b=0),  # Remove any margins
                    height=None,  # Avoid fixed height
                    width=None)   # Avoid fixed width

  return fig

def create_team_heatmap(team):
  team_shots = get_team_shots_data(team)
  custom_colorscale = [
        [0, 'rgba(0, 0, 0, 0)'],  # Fully transparent at the lowest density
        [0.01, 'rgb(206, 206, 206)'], #gray
        [0.1, 'rgb(90, 4, 124)'],
        [0.2, 'rgb(133, 0, 123)'],
        [0.4, 'rgb(170, 0, 117)'],
        [0.5, 'rgb(200, 26, 108)'],
        [0.6, 'rgb(225, 59, 98)'],
        [0.7, 'rgb(243, 92, 88)'],
        [0.8, 'rgb(255, 126, 78)'],
        [0.9, 'rgb(255, 159, 72)'],
        [1, 'rgb(255, 226, 84)']
    ]
  shot_text = [
      f"Player: {row['athlete_display_name_shooter']}<br>Result: {'Make' if row['scoring_play']==1 else 'Miss'}"
      for _, row in team_shots.iterrows()
  ]
  fig = go.Figure()
  #draw_plotly_court(fig)
  fig.add_trace(go.Histogram2dContour(
        x=team_shots['coordinate_x'],
        y=team_shots['coordinate_y'],
        colorscale=custom_colorscale,  # Choose a colorscale for the density
        contours=dict(
            start=0,  # Start contour level
            end=40,  # End contour level (this controls how fine the contours are)
            size=4  # The size of each contour level (smaller values make the contours more sensitive)
        ),
        showscale=False,  # Show color scale
        opacity=0.8,  # Set the opacity to make it blend well with the court background
        hoverinfo="skip"
    ))

  fig.add_trace(go.Scatter(
      x=team_shots['coordinate_x'],
      y=team_shots['coordinate_y'],
      mode='markers',
      marker=dict(color='white', symbol='circle', size=2),
      text=shot_text,
      hoverinfo="text"
  ))
  draw_plotly_court(fig)
  fig.add_layout_image(
    go.layout.Image(
        source=team_shots['team_logo_shooter'][0],
        xref="x", yref="y", x=-24, y=-1, sizex=7.5, sizey=7.5,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))

  return fig

def create_team_heatmap_filters(team, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked):
  team_shots = get_team_shots_data_filters(team, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)
  if len(team_shots) >= 1000:
    contour_end = 40
    contour_size= 4
  else:
    contour_end = 10
    contour_size = 1
  custom_colorscale = [
        [0, 'rgba(0, 0, 0, 0)'],  # Fully transparent at the lowest density
        [0.01, 'rgb(206, 206, 206)'], #gray
        [0.1, 'rgb(90, 4, 124)'],
        [0.2, 'rgb(133, 0, 123)'],
        [0.4, 'rgb(170, 0, 117)'],
        [0.5, 'rgb(200, 26, 108)'],
        [0.6, 'rgb(225, 59, 98)'],
        [0.7, 'rgb(243, 92, 88)'],
        [0.8, 'rgb(255, 126, 78)'],
        [0.9, 'rgb(255, 159, 72)'],
        [1, 'rgb(255, 226, 84)']
    ]
  shot_text = [
      f"Player: {row['athlete_display_name_shooter']}<br>Result: {'Make' if row['scoring_play']==1 else 'Miss'}"
      for _, row in team_shots.iterrows()
  ]
  fig = go.Figure()
  #draw_plotly_court(fig)
  fig.add_trace(go.Histogram2dContour(
        x=team_shots['coordinate_x'],
        y=team_shots['coordinate_y'],
        colorscale=custom_colorscale,  # Choose a colorscale for the density
        contours=dict(
            start=0,  # Start contour level
            end=contour_end,  # End contour level (this controls how fine the contours are)
            size=contour_size  # The size of each contour level (smaller values make the contours more sensitive)
        ),
        showscale=False,  # Show color scale
        opacity=0.8,  # Set the opacity to make it blend well with the court background
        hoverinfo="skip"
    ))

  fig.add_trace(go.Scatter(
      x=team_shots['coordinate_x'],
      y=team_shots['coordinate_y'],
      mode='markers',
      marker=dict(color='white', symbol='circle', size=2),
      text=shot_text,
      hoverinfo="text"
  ))
  draw_plotly_court(fig)
  fig.add_layout_image(
    go.layout.Image(
        source=team_shots['team_logo_shooter'][0],
        xref="x", yref="y", x=-24, y=-1, sizex=7.5, sizey=7.5,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))

  return fig

def create_player_heatmap(player):
  player_shots = get_player_shots_data(player)

  custom_colorscale = [
        [0, 'rgba(0, 0, 0, 0)'],  # Fully transparent at the lowest density
        [0.01, 'rgb(206, 206, 206)'], #gray
        [0.1, 'rgb(90, 4, 124)'],
        [0.2, 'rgb(133, 0, 123)'],
        [0.4, 'rgb(170, 0, 117)'],
        [0.5, 'rgb(200, 26, 108)'],
        [0.6, 'rgb(225, 59, 98)'],
        [0.7, 'rgb(243, 92, 88)'],
        [0.8, 'rgb(255, 126, 78)'],
        [0.9, 'rgb(255, 159, 72)'],
        [1, 'rgb(255, 226, 84)']
    ]
  shot_text = [
      f"Result: {'Make' if row['scoring_play']==1 else 'Miss'}"
      for _, row in player_shots.iterrows()
  ]
  fig = go.Figure()
  #draw_plotly_court(fig)
  fig.add_trace(go.Histogram2dContour(
        x=player_shots['coordinate_x'],
        y=player_shots['coordinate_y'],
        colorscale=custom_colorscale,  # Choose a colorscale for the density
        contours=dict(
            start=0,  # Start contour level
            end=20,  # End contour level (this controls how fine the contours are)
            size=4  # The size of each contour level (smaller values make the contours more sensitive)
        ),
        showscale=False,  # Show color scale
        opacity=0.8,  # Set the opacity to make it blend well with the court background
        hoverinfo="skip"
    ))

  fig.add_trace(go.Scatter(
      x=player_shots['coordinate_x'],
      y=player_shots['coordinate_y'],
      mode='markers',
      marker=dict(color='white', symbol='circle', size=2),
      text=shot_text,
      hoverinfo="text"
  ))
  draw_plotly_court(fig)
  fig.add_layout_image(
    go.layout.Image(
        source=player_shots['athlete_headshot_href_shooter'][0],
        xref="x", yref="y", x=-25, y=0, sizex=11, sizey=7.5,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))
  fig.add_layout_image(
    go.layout.Image(
        source=player_shots['team_logo_shooter'][0],
        xref="x", yref="y", x=-25, y=0, sizex=3, sizey=3,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))

  return fig

def create_player_heatmap_filters(player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked):
  player_shots = get_player_shots_data_filters(player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)
  if len(player_shots) >= 100:
    contour_end = 20
    contour_size= 4
  else:
    contour_end = 5
    contour_size = 1
  custom_colorscale = [
        [0, 'rgba(0, 0, 0, 0)'],  # Fully transparent at the lowest density
        [0.01, 'rgb(206, 206, 206)'], #gray
        [0.1, 'rgb(90, 4, 124)'],
        [0.2, 'rgb(133, 0, 123)'],
        [0.4, 'rgb(170, 0, 117)'],
        [0.5, 'rgb(200, 26, 108)'],
        [0.6, 'rgb(225, 59, 98)'],
        [0.7, 'rgb(243, 92, 88)'],
        [0.8, 'rgb(255, 126, 78)'],
        [0.9, 'rgb(255, 159, 72)'],
        [1, 'rgb(255, 226, 84)']
    ]
  shot_text = [
      f"Result: {'Make' if row['scoring_play']==1 else 'Miss'}"
      for _, row in player_shots.iterrows()
  ]
  fig = go.Figure()
  #draw_plotly_court(fig)
  fig.add_trace(go.Histogram2dContour(
        x=player_shots['coordinate_x'],
        y=player_shots['coordinate_y'],
        colorscale=custom_colorscale,  # Choose a colorscale for the density
        contours=dict(
            start=0,  # Start contour level
            end=contour_end,  # End contour level (this controls how fine the contours are)
            size=contour_size  # The size of each contour level (smaller values make the contours more sensitive)
        ),
        showscale=False,  # Show color scale
        opacity=0.8,  # Set the opacity to make it blend well with the court background
        hoverinfo="skip"
    ))

  fig.add_trace(go.Scatter(
      x=player_shots['coordinate_x'],
      y=player_shots['coordinate_y'],
      mode='markers',
      marker=dict(color='white', symbol='circle', size=2),
      text=shot_text,
      hoverinfo="text"
  ))
  draw_plotly_court(fig)
  fig.add_layout_image(
    go.layout.Image(
        source=player_shots['athlete_headshot_href_shooter'][0],
        xref="x", yref="y", x=-25, y=0, sizex=11, sizey=7.5,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))
  fig.add_layout_image(
    go.layout.Image(
        source=player_shots['team_logo_shooter'][0],
        xref="x", yref="y", x=-25, y=0, sizex=3, sizey=3,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))

  return fig

def create_team_zone_chart(team):
  data = get_team_zone_data(team)
  fig = go.Figure()
  draw_plotly_court(fig)
  for zone, (x_coords, y_coords) in zone_coordinates.items():
      # Extract the color value for the zone (ensure it's a string)
      try:
          color = str(data[data['zone'] == zone]['color'].values[0])  # Extracting the color value
          hover_text = f"Zone: {zone}<br>Shooting % Diff: {round(data[data['zone'] == zone]['zone_shooting_pct_diff'].values[0]*100,2)}%"
          annotation_text = f"{data[data['zone'] == zone]['zone_makes'].values[0]}/{data[data['zone'] == zone]['zone_attempts'].values[0]}<br>({round(data[data['zone'] == zone]['zone_shooting_pct_team'].values[0]*100,2)}%)"
      except:
          color = 'white'
          hover_text = "No Shots Taken in Zone"
          annotation_text = "0/0"

      # Add the trace to the figure
      fig.add_trace(go.Scatter(x=x_coords, y=y_coords,
                              mode='lines', fill='toself',
                              line=dict(color=color, width=1),
                              marker=dict(size=0),
                              name = hover_text
                              ))
      fig.add_annotation(x=zone_annotation_coordinates[zone][0], y=zone_annotation_coordinates[zone][1],
                        text = annotation_text,
                        showarrow=False,
                        font=dict(size=12, color='black'),
                        align = 'center')

  fig.add_layout_image(
    go.layout.Image(
        source=get_team_color_logo_for_team(team)[1],
        xref="x", yref="y", x=24, y=0, sizex=5, sizey=5,
        xanchor="right", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))
  fig.update_layout(showlegend=False)

  return fig

def create_team_zone_chart_filters(team, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked):
  data = get_team_zone_data_filters(team, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)
  fig = go.Figure()
  draw_plotly_court(fig)
  for zone, (x_coords, y_coords) in zone_coordinates.items():
      # Extract the color value for the zone (ensure it's a string)
      try:
          color = str(data[data['zone'] == zone]['color'].values[0])  # Extracting the color value
          hover_text = f"Zone: {zone}<br>Shooting % Diff: {round(data[data['zone'] == zone]['zone_shooting_pct_diff'].values[0]*100,2)}%"
          annotation_text = f"{data[data['zone'] == zone]['zone_makes'].values[0]}/{data[data['zone'] == zone]['zone_attempts'].values[0]}<br>({round(data[data['zone'] == zone]['zone_shooting_pct_team'].values[0]*100,2)}%)"
      except:
          color = 'white'
          hover_text = "No Shots Taken in Zone"
          annotation_text = "0/0"

      # Add the trace to the figure
      fig.add_trace(go.Scatter(x=x_coords, y=y_coords,
                              mode='lines', fill='toself',
                              line=dict(color=color, width=1),
                              marker=dict(size=0),
                              name = hover_text
                              ))
      fig.add_annotation(x=zone_annotation_coordinates[zone][0], y=zone_annotation_coordinates[zone][1],
                        text = annotation_text,
                        showarrow=False,
                        font=dict(size=12, color='black'),
                        align = 'center')
  fig.add_layout_image(
    go.layout.Image(
        source=get_team_color_logo_for_team(team)[1],
        xref="x", yref="y", x=24, y=0, sizex=5, sizey=5,
        xanchor="right", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))
  fig.update_layout(showlegend=False)

  return fig

def create_player_zone_chart(player):
  data = get_player_zone_data(player)
  fig = go.Figure()
  draw_plotly_court(fig)
  for zone, (x_coords, y_coords) in zone_coordinates.items():
      # Extract the color value for the zone (ensure it's a string)
      try:
          color = str(data[data['zone'] == zone]['color'].values[0])  # Extracting the color value
          hover_text = f"Zone: {zone}<br>Shooting % Diff: {round(data[data['zone'] == zone]['zone_shooting_pct_diff'].values[0]*100,2)}%"
          annotation_text = f"{data[data['zone'] == zone]['zone_makes'].values[0]}/{data[data['zone'] == zone]['zone_attempts'].values[0]}<br>({round(data[data['zone'] == zone]['zone_shooting_pct_team'].values[0]*100,2)}%)"
      except:
          color = 'white'
          hover_text = "No Shots Taken in Zone"
          annotation_text = "0/0"

      # Add the trace to the figure
      fig.add_trace(go.Scatter(x=x_coords, y=y_coords,
                              mode='lines', fill='toself',
                              line=dict(color=color, width=1),
                              marker=dict(size=0),
                              name = hover_text
                              ))
      fig.add_annotation(x=zone_annotation_coordinates[zone][0], y=zone_annotation_coordinates[zone][1],
                        text = annotation_text,
                        showarrow=False,
                        font=dict(size=12, color='black'),
                        align = 'center')
  team_logo, player_headshot = get_team_logo_headshot_for_player(player)
  fig.add_layout_image(
      go.layout.Image(
          source=team_logo,
          xref="x", yref="y", x=25, y=0, sizex=3, sizey=3,
          xanchor="right", yanchor="top",
          sizing="stretch", opacity=1, layer="above"))
  fig.add_layout_image(
      go.layout.Image(
          source=player_headshot,
          xref="x", yref="y", x=24, y=-1, sizex=11, sizey=7,
          xanchor="right", yanchor="top",
          sizing="stretch", opacity=1, layer="above"))
  fig.update_layout(showlegend=False)

  return fig

def create_player_zone_chart_filters(player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked):
  data = get_player_zone_data_filters(player, start_date, end_date, opponent, location, half, qtr, game, assisted, blocked)
  fig = go.Figure()
  draw_plotly_court(fig)
  for zone, (x_coords, y_coords) in zone_coordinates.items():
      # Extract the color value for the zone (ensure it's a string)
      try:
          color = str(data[data['zone'] == zone]['color'].values[0])  # Extracting the color value
          hover_text = f"Zone: {zone}<br>Shooting % Diff: {round(data[data['zone'] == zone]['zone_shooting_pct_diff'].values[0]*100,2)}%"
          annotation_text = f"{data[data['zone'] == zone]['zone_makes'].values[0]}/{data[data['zone'] == zone]['zone_attempts'].values[0]}<br>({round(data[data['zone'] == zone]['zone_shooting_pct_team'].values[0]*100,2)}%)"
      except:
          color = 'white'
          hover_text = "No Shots Taken in Zone"
          annotation_text = "0/0"

      # Add the trace to the figure
      fig.add_trace(go.Scatter(x=x_coords, y=y_coords,
                              mode='lines', fill='toself',
                              line=dict(color=color, width=1),
                              marker=dict(size=0),
                              name = hover_text
                              ))
      fig.add_annotation(x=zone_annotation_coordinates[zone][0], y=zone_annotation_coordinates[zone][1],
                        text = annotation_text,
                        showarrow=False,
                        font=dict(size=12, color='black'),
                        align = 'center')
  team_logo, player_headshot = get_team_logo_headshot_for_player(player)
  fig.add_layout_image(
      go.layout.Image(
          source=team_logo,
          xref="x", yref="y", x=25, y=0, sizex=3, sizey=3,
          xanchor="right", yanchor="top",
          sizing="stretch", opacity=1, layer="above"))
  fig.add_layout_image(
      go.layout.Image(
          source=player_headshot,
          xref="x", yref="y", x=24, y=-1, sizex=11, sizey=7,
          xanchor="right", yanchor="top",
          sizing="stretch", opacity=1, layer="above"))
  fig.update_layout(showlegend=False)

  return fig

def create_card_team_game_score_chart(team):
  data = get_team_totals_data(team)
  data['game_number'] = data.index + 1

  fig = px.bar(data, x='game_number', y='score_value', text='score_value',
              color_discrete_sequence=[data['team_color'][0]])
  fig.update_traces(textposition="outside")


  fig.update_traces(
    hovertemplate=(
        "Game Number: %{customdata[0]}<br>" +
        "Game Date: %{customdata[1]}<br>" +
        "Home/Away: %{customdata[2]}<br>" +
        "Points Scored: %{y}"
    ),
    customdata=list(zip(data['game_number'], data['game_date'], ['Home' if home else 'Away' for home in data['is_home_team']])),
    textposition="outside"
  )

  fig.update_layout(
    template = 'plotly_white',
    xaxis_title=None, yaxis_title=None)

  fig.update_xaxes(showgrid=False, showticklabels=False)  # Remove x-axis grid and labels
  fig.update_yaxes(showgrid=False, showticklabels=False)
  return fig

def create_card_player_game_score_chart(player):
  data = get_player_totals_data(player)
  data['game_number'] = data.index + 1
  data['away_flag'] = ['' if home else '*' for home in data['is_home_team']]

  fig = px.bar(data, x='game_number', y='score_value', text='score_value',
              color_discrete_sequence=data['team_color'])
  fig.update_traces(marker_color=data['team_color'], textposition="outside")

  fig.update_traces(
    hovertemplate=(
        "Game Number: %{customdata[0]}<br>" +
        "Game Date: %{customdata[1]}<br>" +
        "Home/Away: %{customdata[2]}<br>" +
        "Points Scored: %{y}"
    ),
    customdata=list(zip(data['game_number'],data['game_date'], ['Home' if home else 'Away' for home in data['is_home_team']]))
  )

  fig.update_layout(
    template = 'plotly_white',
    xaxis_title=None, yaxis_title=None)

  fig.update_xaxes(showgrid=False, showticklabels=False)  # Remove x-axis grid and labels
  fig.update_yaxes(showgrid=False, showticklabels=False)
  return fig

def create_team_stream_plot(team):
  data = get_team_stream_data(team)

  made_threes = data['made_three'].sum()
  made_twos = data['scoring_play'].sum() - made_threes
  fg_pct = (made_twos + (made_threes)) / data['shot_count'].sum()
  efg_pct = (made_twos + (1.5*made_threes)) / data['shot_count'].sum()

  max_shot_count = data['shot_count'].max()
  min_shot_count = data['shot_count'].min()

  # Calculate the shot count factor for each data point
  data['shot_count_factor'] = (data['shot_count'] / max_shot_count) * 0.2  # Factor between 0 and 0.1

  # Create y_upper and y_lower based on the shot_count_factor
  y_upper = data['fg_pct'] + data['shot_count_factor']
  y_lower = data['fg_pct'] - data['shot_count_factor']

  # Smooth the bands using CubicSpline interpolation
  cs_upper = CubicSpline(data['shot_distance'], y_upper, bc_type='clamped')
  cs_lower = CubicSpline(data['shot_distance'], y_lower, bc_type='clamped')

  # Generate smooth values for the upper and lower bounds
  x_smooth = np.linspace(data['shot_distance'].min(), data['shot_distance'].max(), 500)  # More points for smoothness
  y_upper_smooth = cs_upper(x_smooth)
  y_lower_smooth = cs_lower(x_smooth)

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=data['shot_distance'], y=data['fg_pct'],
                          line_shape='spline', mode='lines', name = '', # so name doesn't pop up next to hover next when its not needed
                          line=dict(color=hex_to_rgba(data['team_color'][0], 1)),
                          showlegend=False))
  fig.add_trace(go.Scatter(
      x=np.concatenate([x_smooth, x_smooth[::-1]]),  # smooth x, then reversed smooth x
      y=np.concatenate([y_upper_smooth, y_lower_smooth[::-1]]),  # smooth upper, then reversed smooth lower
      fill='toself',
      fillcolor=hex_to_rgba(data['team_alternate_color'][0], 0.2),
      line=dict(color=hex_to_rgba(data['team_alternate_color'][0], 0.2)),
      hoverinfo="skip",
      showlegend=False
      )
  )
  fig.update_layout(
      xaxis = dict(
          tickmode = 'linear',
          tick0 = 0,
          dtick = 1
          ),
      yaxis=dict(
        tickformat=".0%",  # Format y-axis as percentage
        ticks="inside",
        tick0 = 0,
        dtick = 0.1,
        showgrid=True  # Display gridlines for the y-axis if needed
        ),
      xaxis_title='Shot Distance (ft.)',
      yaxis_title='Field Goal %')

  fig.update_traces(
    hovertemplate=(
        "Shot Distance: %{customdata[0]} ft.<br>" +
        "FG%: %{y}"
    ),
    customdata=list(zip(data['shot_distance']))
  )

  fig.add_hline(y=fg_pct, line_dash="dot", line_color=hex_to_rgba(data['team_color'][0], 1),
                annotation_text=f'FG% - {round(fg_pct*100,2)}%',
                annotation_font_size=10,
                annotation_font_color=hex_to_rgba(data['team_color'][0], 1),
                annotation_position="bottom right")
  fig.add_hline(y=efg_pct, line_dash="dot", line_color=hex_to_rgba(data['team_color'][0], 1),
                annotation_text=f'eFG% - {round(efg_pct*100,2)}%',
                annotation_font_size=10,
                annotation_font_color=hex_to_rgba(data['team_color'][0], 1),
                annotation_position="top right")
  fig.add_layout_image(
    go.layout.Image(
        source=data['team_logo'][0],
        xref="x", yref="y", x=data['shot_distance'].max()-2, y=0.8, sizex=2, sizey=.3,
        xanchor="right", yanchor="top",
        sizing="stretch", opacity=1, layer="above"))

  fig.update_layout(template = 'plotly_white',
                    margin=dict(l=0, r=0, t=0, b=0),  # Remove any margins
                    height=600,  # Avoid fixed height
                    width=1200)   # Avoid fixed width
  return fig

def create_player_stream_plot(player):
  data = get_player_stream_data(player)

  made_threes = data['made_three'].sum()
  made_twos = data['scoring_play'].sum() - made_threes
  fg_pct = (made_twos + (made_threes)) / data['shot_count'].sum()
  efg_pct = (made_twos + (1.5*made_threes)) / data['shot_count'].sum()

  max_shot_count = data['shot_count'].max()
  min_shot_count = data['shot_count'].min()

  # Calculate the shot count factor for each data point
  data['shot_count_factor'] = (data['shot_count'] / max_shot_count) * 0.2  # Factor between 0 and 0.1

  # Create y_upper and y_lower based on the shot_count_factor
  y_upper = data['fg_pct'] + data['shot_count_factor']
  y_lower = data['fg_pct'] - data['shot_count_factor']

  # Smooth the bands using CubicSpline interpolation
  cs_upper = CubicSpline(data['shot_distance'], y_upper, bc_type='clamped')
  cs_lower = CubicSpline(data['shot_distance'], y_lower, bc_type='clamped')

  # Generate smooth values for the upper and lower bounds
  x_smooth = np.linspace(data['shot_distance'].min(), data['shot_distance'].max(), 500)  # More points for smoothness
  y_upper_smooth = cs_upper(x_smooth)
  y_lower_smooth = cs_lower(x_smooth)

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=data['shot_distance'], y=data['fg_pct'],
                          line_shape='spline', mode='lines', name="Field Goal %",
                          line=dict(color=hex_to_rgba(data['team_color'][0], 1)),
                          showlegend=False))
  fig.add_trace(go.Scatter(
      x=np.concatenate([x_smooth, x_smooth[::-1]]),  # smooth x, then reversed smooth x
      y=np.concatenate([y_upper_smooth, y_lower_smooth[::-1]]),  # smooth upper, then reversed smooth lower
      fill='toself',
      fillcolor=hex_to_rgba(data['team_alternate_color'][0], 0.2),
      line=dict(color=hex_to_rgba(data['team_alternate_color'][0], 0.2)),
      hoverinfo="skip",
      showlegend=False
      )
  )
  fig.update_layout(
      xaxis = dict(
          tickmode = 'linear',
          tick0 = 1,
          dtick = 1
          ),
      yaxis=dict(
        tickformat=".0%",  # Format y-axis as percentage
        ticks="inside",
        showgrid=True  # Display gridlines for the y-axis if needed
        ),
      xaxis_title='Shot Distance (ft.)', yaxis_title='Field Goal %')
  fig.update_layout(yaxis = dict(
      tickmode = 'linear',
      tick0 = 0,
      dtick = 0.1
  ))
  fig.add_hline(y=fg_pct, line_dash="dot", line_color=hex_to_rgba(data['team_color'][0], 1),
                annotation_text=f'FG% - {round(fg_pct*100,2)}%',
                annotation_font_size=10,
                annotation_font_color=hex_to_rgba(data['team_color'][0], 1),
                annotation_position="bottom right")
  fig.add_hline(y=efg_pct, line_dash="dot", line_color=hex_to_rgba(data['team_color'][0], 1),
                annotation_text=f'eFG% - {round(efg_pct*100,2)}%',
                annotation_font_size=10,
                annotation_font_color=hex_to_rgba(data['team_color'][0], 1),
                annotation_position="top right")
  if len(data) >= 10:
    fig.add_layout_image(
      go.layout.Image(
          source=data['player_headshot'][0],
          xref="paper", yref="paper", x=.99, y=.95, sizex=0.1, sizey=0.2,
          xanchor="right", yanchor="top",
          sizing="stretch", opacity=1, layer="above"))
    fig.add_layout_image(
      go.layout.Image(
          source=data['team_logo'][0],
          xref="paper", yref="paper", x=1, y=1, sizex=0.04, sizey=0.08,
          xanchor="right", yanchor="top",
          sizing="stretch", opacity=1, layer="above"))

  fig.update_layout(template = 'plotly_white',
                    margin=dict(l=0, r=0, t=0, b=0),  # Remove any margins
                    height=600,  # Avoid fixed height
                    width=1200)   # Avoid fixed width

  return fig

def get_card_player_percentages(player):
  wnba_shots.loc[wnba_shots['shot_type'] == 'Layup', 'shot_type'] = '2 Pointer'
  player_data = wnba_shots[wnba_shots['athlete_display_name_shooter'] == player].reset_index(drop=True)
  player_scoring = player_data.groupby(['game_id']).agg({'score_value': 'sum'}).reset_index()
  ppg = round(player_scoring['score_value'].mean(),2)

  player_shooting = player_data.groupby(['shot_type']).agg({'offensive_team_id': 'count',
                                              'scoring_play': 'sum'}).reset_index()

  two_point_attempts = player_shooting[player_shooting['shot_type'] == '2 Pointer']['offensive_team_id'].sum()
  two_point_makes = player_shooting[player_shooting['shot_type'] == '2 Pointer']['scoring_play'].sum()
  three_point_attempts = player_shooting[player_shooting['shot_type'] == '3 Pointer']['offensive_team_id'].sum()
  three_point_makes = player_shooting[player_shooting['shot_type'] == '3 Pointer']['scoring_play'].sum()

  field_goal_attempts = two_point_attempts+three_point_attempts
  field_goal_makes = two_point_makes+three_point_makes
  field_goal_percent = str(round((field_goal_makes / field_goal_attempts)*100,2))+"%"
  three_point_percent = str(round((three_point_makes / three_point_attempts)*100,2))+"%"
  ft_percent = str(round((player_shooting[player_shooting['shot_type'] == 'Free Throw']['scoring_play'].sum() / player_shooting[player_shooting['shot_type'] == 'Free Throw']['offensive_team_id'].sum())*100,2))+"%"

  return ppg, field_goal_percent, three_point_percent, ft_percent

def get_card_team_percentages(team):
  wnba_shots.loc[wnba_shots['shot_type'] == 'Layup', 'shot_type'] = '2 Pointer'
  team_id = wnba_team_id_map[team]

  team_data = wnba_shots[wnba_shots['offensive_team_id'] == team_id].reset_index(drop=True)
  team_scoring = team_data.groupby(['game_id']).agg({'score_value': 'sum'}).reset_index()
  ppg = round(team_scoring['score_value'].mean(),2)

  team_shooting = team_data.groupby(['shot_type']).agg({'offensive_team_id': 'count',
                                              'scoring_play': 'sum'}).reset_index()

  two_point_attempts = team_shooting[team_shooting['shot_type'] == '2 Pointer']['offensive_team_id'].sum()
  two_point_makes = team_shooting[team_shooting['shot_type'] == '2 Pointer']['scoring_play'].sum()
  three_point_attempts = team_shooting[team_shooting['shot_type'] == '3 Pointer']['offensive_team_id'].sum()
  three_point_makes = team_shooting[team_shooting['shot_type'] == '3 Pointer']['scoring_play'].sum()

  field_goal_attempts = two_point_attempts+three_point_attempts
  field_goal_makes = two_point_makes+three_point_makes
  field_goal_percent = str(round((field_goal_makes / field_goal_attempts)*100,2))+"%"
  three_point_percent = str(round((three_point_makes / three_point_attempts)*100,2))+"%"
  ft_percent = str(round((team_shooting[team_shooting['shot_type'] == 'Free Throw']['scoring_play'].sum() / team_shooting[team_shooting['shot_type'] == 'Free Throw']['offensive_team_id'].sum())*100,2))+"%"

  return ppg, field_goal_percent, three_point_percent, ft_percent