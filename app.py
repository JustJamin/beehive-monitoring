import dash
import dash_leaflet as dl
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import datetime, timezone
import time
import os


# -------------------------
# Configuration
# -------------------------
# InfluxDB Cloud credentials
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN")
INFLUX_URL = os.environ.get("INFLUX_URL")
INFLUX_ORG = os.environ.get("INFLUX_ORG")
INFLUX_BUCKET = "ls300-tracking-demo"

MEASUREMENT = "tracker_data"
START_TIME = "2025-06-04T06:17:00Z"  # Adjust as needed

# -------------------------
# Connect to InfluxDB
# -------------------------
client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

# -------------------------
# Load all data
# -------------------------

def load_all_data():
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {START_TIME})
      |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "device", "latitude", "longitude", "temperature", "humidity", "speed", "altitude", "pressure", "batteryVoltage", "counter", "heading", "hoursUptime", "satId", "userButton", "hall"])
    '''
    df = query_api.query_data_frame(query)

    # Handle case when multiple DataFrames are returned
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True)

    # Sometimes empty queries return metadata-only frames
    if df.empty or '_time' not in df.columns:
        print("Warning: No data returned from InfluxDB or missing '_time' column.")
        return pd.DataFrame()  # return empty DataFrame to avoid crash later

    df = df.rename(columns={"_time": "time"})
    df["time"] = pd.to_datetime(df["time"])
    return df

# Global data store
data_df = load_all_data()

# Assign a unique color per device
device_colors = {}
color_palette = ["red", "blue", "green", "purple", "yellow", "orange", "pink", "magenta", "cyan", "lime"]

# -------------------------
# Create Dash App
# -------------------------
app = dash.Dash(__name__)
app.title = "LS300 Tracker Demo"
server = app.server

# CSS page formatting
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                margin: 0;
                background: #3c1361;
                height: 100%;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Map centre (I can make this dynamic later)
if not data_df.empty:
    map_center = [data_df["latitude"].mean(), data_df["longitude"].mean()]
else:
    map_center = [51, -1]  # Default fallback

app.layout = html.Div(
    style={
        "backgroundColor": "#3c1361",  # A rich purple hue
        "color": "white",              # Ensures good contrast for text
        "height": "100vh",             # Full vertical height
        "padding": "10px",
        "fontFamily": "Ubuntu"
    },
    children = [
        html.Img(
            src="/assets/logo.png",
            style={
                "position": "absolute",
                "top": "10px",
                "right": "10px",
                "height": "120px",
                "zIndex": "999"
            }
        ),
        html.H2("Lacuna Space Tracker Demo",
            style={
                "fontSize": "3em",  # Increase size as desired
                "marginBottom": "20px"
            }
        ),
        dl.Map(center=map_center, zoom=12, id="live-map", style={'width': '100%', 'height': 'calc(100vh - 120px)'}, children=[
            dl.TileLayer(),
            dl.LayerGroup(id="marker-layer")
        ]),
        dcc.Interval(id="interval", interval=30*1000, n_intervals=0)
    ]
)

# -------------------------
# Helper to format hover popup
# -------------------------
def format_popup(row):
    return html.Div([
        html.B(f"Device: {row['device']}"),
        html.Br(),
        f"Time: {row['time'].strftime('%Y-%m-%d %H:%M:%S')}",
        html.Br(),
        f"Lat: {row.get('latitude', 'N/A')}°",
        html.Br(),
        f"Lon: {row.get('longitude', 'N/A')}°",
        html.Br(),
        f"Temp: {row.get('temperature', 'N/A')}°C",
        html.Br(),
        f"Humidity: {row.get('humidity', 'N/A')}%",
        html.Br(),
        f"Speed: {row.get('speed', 'N/A')} m/s",
        html.Br(),
        f"Pressure: {row.get('pressure', 'N/A')} hPa",
        html.Br(),
        f"Battery: {row.get('batteryVoltage', 'N/A')} V",
        html.Br(),
        f"Uptime: {row.get('hoursUptime', 'N/A')} h"
    ])

# -------------------------
# Callback to update map
# -------------------------
@app.callback(
    Output("marker-layer", "children"),
    Input("interval", "n_intervals")
)
def update_map(n):
    global data_df
    # Query new data
    if data_df.empty or "time" not in data_df.columns:
        return []

    try:
        last_time = data_df["time"].max().isoformat()
    except Exception as e:
        print(f"Error extracting last_time: {e}")
        return []
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: time(v: "{last_time}"))
      |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "device", "latitude", "longitude", "temperature", "humidity", "speed", "altitude", "pressure", "batteryVoltage", "counter", "heading", "hoursUptime", "satId", "userButton", "hall"])
    '''
    new_df = query_api.query_data_frame(query)
    if not new_df.empty:
        new_df = new_df.rename(columns={"_time": "time"})
        new_df["time"] = pd.to_datetime(new_df["time"])
        data_df = pd.concat([data_df, new_df], ignore_index=True)
        data_df.drop_duplicates(subset=["device", "time"], inplace=True)
        data_df.sort_values(by=["device", "time"], inplace=True)

    # Filter to only include devices starting with "satellite"
    # To disable this filter, comment out the following line
    data_df = data_df[data_df["device"].str.startswith("satellite")]

    global device_colors  # ensure we’re modifying the global mapping

    # Assign colors only to new devices
    for device in data_df["device"].dropna().unique():
        if device not in device_colors:
            device_colors[device] = color_palette[len(device_colors) % len(color_palette)]

    # Build markers
    markers = []
    for _, row in data_df.iterrows():
        device = row.get("device", "unknown")
        if device not in device_colors:
            # Assign next available colour (loop if more devices than colours)
            device_colors[device] = color_palette[len(device_colors) % len(color_palette)]

        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            markers.append(
                dl.CircleMarker(
                    center=(row["latitude"], row["longitude"]),
                    radius=8,
                    color=device_colors[device],
                    fill=True,
                    fillOpacity=0.2,
                    children=dl.Tooltip(children=format_popup(row))
                )
            )
    
    # Build lines per device
    lines = []
    for device, group in data_df.groupby("device"):
        group = group.sort_values("time")  # ensure ordered path
        coords = list(zip(group["latitude"], group["longitude"]))
        color = device_colors.get(device, "black")  # fallback

        if len(coords) >= 2:
            lines.append(
                dl.Polyline(
                    positions=coords,
                    color=color,
                    weight=4,
                    dashArray="5, 5",  # "on, off" pattern for dashes
                    opacity=0.6
                )
            )
    return markers + lines

# -------------------------
# Run the app
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)