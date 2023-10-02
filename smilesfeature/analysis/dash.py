import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import numpy as np
from matplotlib import pyplot as plt
import base64
from io import BytesIO

def array_to_base64(array):
    """
    Convert a NumPy array to a base64-encoded PNG image.

    Parameters:
        array (numpy.ndarray): The input NumPy array representing an image.

    Returns:
        str: A base64-encoded PNG image.
    """
    fig, ax = plt.subplots()
    ax.imshow(array, cmap='gray')
    ax.plot([-10, 10], [-10, 10], 'r--')  # Add y=x line
    ax.axis('equal')  # Set aspect ratio to be equal
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def create_inline_dash_dashboard(df, true_col, predicted_col):
    """
    Create an inline Dash dashboard to visualize scatter plots and hover over points to view images.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing data to visualize.
        true_col (str): The name of the true values column.
        predicted_col (str): The name of the predicted values column.

    Example:
        # Assuming df contains columns 'true_column_name', 'predicted_column_name', and 'image_array'.
        >>> create_inline_dash_dashboard(df, 'true_column_name', 'predicted_column_name')
    """
    encoded_images = [array_to_base64(array) for array in df['image_array']]

    app = dash.Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(
            id='scatter-plot',
            figure={
                'data': [
                    {
                        'x': df[true_col],
                        'y': df[predicted_col],
                        'mode': 'markers',
                        'type': 'scatter',
                        'name': 'Data',
                    },
                    {
                        'x': [-10, 10],
                        'y': [-10, 10],
                        'mode': 'lines',
                        'name': 'y=x',
                    },
                ],
                'layout': {
                    'xaxis': {'title': 'True Values', 'range': [min(df[true_col]), max(df[true_col])]},
                    'yaxis': {'title': 'Predicted Values', 'range': [min(df[predicted_col]), max(df[predicted_col])]},
                    'aspectratio': {'x': 1, 'y': 2},  # Set 1:1 aspect ratio
                },
            }
        ),
        html.Div([
            html.Img(id='hover-image', src='', style={'height': '200px'})
        ])
    ])

    @app.callback(
        Output('hover-image', 'src'),
        Input('scatter-plot', 'hoverData')
    )
    def show_hover_image(hoverData):
        if hoverData:
            index = hoverData['points'][0]['pointIndex']
            return f"data:image/png;base64,{encoded_images[index]}"
        return ""

    # Run the Dash app in inline mode within the Jupyter Notebook
    app.run_server(mode='inline')

