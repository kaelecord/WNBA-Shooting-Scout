# WNBA Shooting Scout
Web App hosting WNBA shooting visualizations.

Web App Link: [WNBA Shooting Scout](https://wnba-shooting-scout.onrender.com/)  
[Video Demo Link](https://indiana-my.sharepoint.com/:v:/r/personal/kecord_iu_edu/Documents/Data%20Viz%20Final%20Project%20Demo.webm?csf=1&web=1&e=N1iVNj&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D)

## 3 Key Components
1. Player/Team Card
    * Player name and headshot/Team name and logo
    * Season shooting stats
    * Game-by-game scoring bar chart
2. Player/Team Season Shot Chart (hex)
    * Static hexbin shot chart
    * Size denotes shot count
    * Color denotes shooting % compared to WNBA League Average
        * Dark Red (+5+%), Red (+2.5-5%), Light Red (+0-2.5%), Light Blue (-0-2.5%), Blue (-2.5-5%), and Dark Blue (-5+%)
3. 4 Filterable Shot Charts
    * Available shot charts:
        * Make/Miss
        * Shot Zone
        * Heat Map
        * Stream Chart
            * FG% by shot distance
                * Line represents FG% at shot distance
                * Band represents shot count at shot distance
    * Available filters:
        * Game date range
        * Specific game
        * Opponent
        * Game location (Home/Away)
        * Half
        * Quarter
        * Assisted
        * Blocked
