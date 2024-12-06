# WNBA Shooting Scout
The goal of this web app is to provide access to WNBA shooting charts that can be hard to find at times. Similar plots can be found on [WNBA.com](https://stats.wnba.com/), however I find it hard to navigate to get the exact shot chart you are looking for. Inside this web app you'll be able to get a player or team's shot chart, shot zone chart, heat map, and stream plot for a wide range of specific scenarios. Do you want to see how your favorite team shot against their bitter rival? Simply select your team, find their rival in the opponent filter, and have fun exploring. Maybe you want to see how Caitlin Clark performed after the Olympic break. Well then you're in the right spot! Select the Fever, Caitlin Clark, and filter her shot chart by the date range you are looking for and you're off to the races. Now, hand up,🙋‍♂️, I'm very aware this is not the most visually appealing web app you have come across. Getting the formatting as it sits right now was already more than enough CSS than I bargained for, however in the future I hope to update this as I learn more. Along those same lines, another flaw that became very apparent was how well (or the lack thereof) the current design transfers to mobile. Once again, not in my CSS bag right now. 

Web App Link: [WNBA Shooting Scout](https://wnba-shooting-scout.onrender.com/)  
[Video Demo Link](https://indiana-my.sharepoint.com/:v:/r/personal/kecord_iu_edu/Documents/Data%20Viz%20Final%20Project%20Demo.webm?csf=1&web=1&e=N1iVNj&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D)

## 3 Key Components
1. Player/Team Card
    * Player name and headshot/Team name and logo
    * Season shooting stats
    * Game-by-game scoring bar chart
    * <img src="https://github.com/kaelecord/WNBA-Shooting-Scout/blob/main/examples%20images/player%20card.png" alt="Caitlin Clark Player Card" width="250" height="auto"><img src="https://github.com/kaelecord/WNBA-Shooting-Scout/blob/main/examples%20images/team%20card.png" alt="Indiana Fever Team Card" width="250" height="auto">
2. Player/Team Season Shot Chart (hex)
    * Static hexbin shot chart
    * Size denotes shot count
    * Color denotes shooting % compared to WNBA League Average
        * Dark Red (+5+%), Red (+2.5-5%), Light Red (+0-2.5%), Light Blue (-0-2.5%), Blue (-2.5-5%), and Dark Blue (-5+%)
    * Caitlin Clark:<img src="https://github.com/kaelecord/WNBA-Shooting-Scout/blob/main/examples%20images/player%20hexbin.png" alt="Caitlin Clark Season Hex Shot Chart" width="250" height="auto">
    * Indiana Fever: <img src="https://github.com/kaelecord/WNBA-Shooting-Scout/blob/main/examples%20images/team%20card.png" alt="Indiana Fever Season Hex Shot Chart" width="250" height="auto">
3. 4 Filterable Shot Charts
    * Available shot charts:
        * Make/Miss
           * <img src="https://github.com/kaelecord/WNBA-Shooting-Scout/blob/main/examples%20images/player%20shot%20chart.png" alt="Caitlin Clark Player Card" width="250" height="auto"><img src="https://github.com/kaelecord/WNBA-Shooting-Scout/blob/main/examples%20images/team%20shot%20chart.png" alt="Indiana Fever Team Card" width="250" height="auto">
        * Shot Zone
           * <img src="https://github.com/kaelecord/WNBA-Shooting-Scout/blob/main/examples%20images/player%20shot%20zone.png" alt="Caitlin Clark Player Card" width="250" height="auto"><img src="https://github.com/kaelecord/WNBA-Shooting-Scout/blob/main/examples%20images/team%20shot%20zone.png" alt="Indiana Fever Team Card" width="250" height="auto">
        * Heat Map
           * <img src="https://github.com/kaelecord/WNBA-Shooting-Scout/blob/main/examples%20images/player%20heat%20map.png" alt="Caitlin Clark Player Card" width="250" height="auto"><img src="https://github.com/kaelecord/WNBA-Shooting-Scout/blob/main/examples%20images/team%20heat%20map.png" alt="Indiana Fever Team Card" width="250" height="auto">
        * Stream Chart
            * FG% by shot distance
                * Line represents FG% at shot distance
                * Band represents shot count at shot distance
        * <img src="https://github.com/kaelecord/WNBA-Shooting-Scout/blob/main/examples%20images/player%20stream.png" alt="Caitlin Clark Player Card" width="250" height="auto"><img src="https://github.com/kaelecord/WNBA-Shooting-Scout/blob/main/examples%20images/team%20steam.png" alt="Indiana Fever Team Card" width="250" height="auto">
    * Available filters:
        * Game date range
        * Specific game
        * Opponent
        * Game location (Home/Away)
        * Half
        * Quarter
        * Assisted
        * Blocked
