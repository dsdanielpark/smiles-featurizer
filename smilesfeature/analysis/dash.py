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
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def create_dash_dashboard(df, true_col, predicted_col):
    """
    Create a Dash dashboard to visualize scatter plots and hover over points to view images.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing data to visualize.
        true_col (str): The name of the true values column.
        predicted_col (str): The name of the predicted values column.

    Example:
        # Assuming df contains columns 'true_column_name', 'predicted_column_name', and 'image_array'.
        >>> create_dash_dashboard(df, 'true_column_name', 'predicted_column_name')
    """
    encoded_images = [array_to_base64(array) for array in df['image_array']]

    app = dash.Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(
            id='scatter-plot',
            figure=px.scatter(x=df[true_col], y=df[predicted_col], labels={'x':'True Values', 'y':'Predicted Values'})
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

    if __name__ == '__main__':
        app.run_server(debug=True)
