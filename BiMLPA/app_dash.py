import os
import random
import tempfile
import base64
import io
import sys
from collections import Counter, defaultdict

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np

from bimlpa import BiMLPA_SqrtDeg, _labels_to_partition
from modularity import suzuki_modularity

# Initialize the Dash app with a modern theme
app = dash.Dash(__name__, 
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    'https://use.fontawesome.com/releases/v5.15.4/css/all.css',
                    'https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap'
                ],
                suppress_callback_exceptions=True,
                title="BiMLPA - Community Detection")

# Custom CSS for modern styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Poppins', sans-serif;
                background-color: #f8f9fa;
            }
            .card {
                border-radius: 10px;
                border: none;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .card:hover {
                box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
            }
            .card-header {
                background-color: #ffffff;
                border-bottom: 1px solid rgba(0, 0, 0, 0.05);
                border-top-left-radius: 10px !important;
                border-top-right-radius: 10px !important;
                padding: 0.8rem 1.25rem;
            }
            .dash-graph {
                border-radius: 10px;
                overflow: hidden;
            }
            .dash-table-container {
                border-radius: 10px;
                overflow: hidden;
            }
            .upload-box {
                border-radius: 10px !important;
                border: 2px dashed #dee2e6 !important;
                transition: border-color 0.3s ease;
            }
            .upload-box:hover {
                border-color: #007bff !important;
            }
            .btn-primary {
                background-color: #4361ee;
                border-color: #4361ee;
                border-radius: 8px;
                padding: 0.5rem 1.5rem;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            .btn-primary:hover {
                background-color: #3a56d4;
                border-color: #3a56d4;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(67, 97, 238, 0.3);
            }
            .btn-primary:active {
                transform: translateY(0);
            }
            .nav-pills .nav-link.active {
                background-color: #4361ee;
            }
            .alert {
                border-radius: 8px;
            }
            h1, h2, h3, h4, h5, h6 {
                font-weight: 600;
                color: #333;
            }
            .card-body {
                padding: 1.5rem;
            }
            .form-label {
                font-weight: 500;
                margin-bottom: 0.5rem;
                color: #555;
            }
            .dash-dropdown .Select-control {
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }
            .dash-dropdown .Select-control:hover {
                border-color: #4361ee;
            }
            .rc-slider-track {
                background-color: #4361ee;
            }
            .rc-slider-handle {
                border-color: #4361ee;
            }
            .rc-slider-handle:hover {
                border-color: #3a56d4;
            }
            .rc-slider-handle:active {
                border-color: #3a56d4;
                box-shadow: 0 0 5px #4361ee;
            }
            .badge {
                font-weight: 500;
                padding: 0.5em 0.8em;
                border-radius: 6px;
            }
            .table-container {
                border-radius: 10px;
                overflow: hidden;
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

# Global storage for graph data (in a real app, use dcc.Store)
app_data = {
    'graph': None,
    'communities_sorted': None,
    'node_labels': None,
    'full_table_data': None  # Add this to store non-truncated data for download
}

def generate_bipartite_from_t2dm(file_path: str):
    """
    Generate bipartite graph from T2DM drug-target dataset.
    Handles duplicate drugs by creating unique drug nodes.
    Supports CSV, TSV, and TXT formats with automatic delimiter detection.
    Raises appropriate exceptions for validation errors.
    """
    try:
        # Detect delimiter based on file extension
        file_extension = file_path.lower().split('.')[-1]
        
        if file_extension == 'tsv':
            delimiter = '\t'
        elif file_extension == 'txt':
            # Try to detect delimiter from first line
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if '\t' in first_line:
                    delimiter = '\t'
                elif ',' in first_line:
                    delimiter = ','
                else:
                    # Default to tab for .txt files
                    delimiter = '\t'
        else:  # csv or other
            delimiter = ','
        
        # Read file with detected delimiter and error handling
        df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8')
    except pd.errors.EmptyDataError:
        raise ValueError("File kosong. Harap upload file yang berisi data.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Format file tidak valid: {str(e)}")
    except UnicodeDecodeError:
        # Try alternative encoding
        try:
            df = pd.read_csv(file_path, sep=delimiter, encoding='latin-1')
        except Exception as e:
            raise ValueError(f"Gagal membaca file (encoding error): {str(e)}")
    except Exception as e:
        raise ValueError(f"Gagal membaca file: {str(e)}")
    
    # Validate required columns
    if 'Drug' not in df.columns:
        available_cols = ', '.join(df.columns.tolist())
        raise ValueError(f"Kolom 'Drug' tidak ditemukan. Kolom yang tersedia: {available_cols}")
    
    if 'Gene' not in df.columns:
        available_cols = ', '.join(df.columns.tolist())
        raise ValueError(f"Kolom 'Gene' tidak ditemukan. Kolom yang tersedia: {available_cols}")
    
    # Check if dataframe has any data
    if len(df) == 0:
        raise ValueError("File CSV tidak memiliki data (hanya header).")
    
    # Remove rows with missing values
    original_len = len(df)
    df = df.dropna(subset=['Drug', 'Gene'])
    
    if len(df) == 0:
        raise ValueError("Tidak ada baris dengan data Drug dan Gene yang valid (semua baris memiliki nilai kosong).")
    
    if len(df) < original_len:
        print(f"Warning: {original_len - len(df)} baris diabaikan karena memiliki nilai kosong.")
    
    # Create bipartite graph
    G = nx.Graph()
    
    # Get unique drugs and genes
    unique_drugs = df['Drug'].unique()
    unique_genes = df['Gene'].unique()
    
    # Validate that we have both drugs and genes
    if len(unique_drugs) == 0:
        raise ValueError("Tidak ada drug yang ditemukan dalam file.")
    
    if len(unique_genes) == 0:
        raise ValueError("Tidak ada gene yang ditemukan dalam file.")
    
    # Add drug nodes (bipartite=0)
    for drug in unique_drugs:
        G.add_node(drug, bipartite=0, display=drug, side="drug")
    
    # Add gene nodes (bipartite=1)
    for gene in unique_genes:
        G.add_node(gene, bipartite=1, display=gene, side="gene")
    
    # Add edges from the dataframe
    edge_count = 0
    for _, row in df.iterrows():
        drug = row['Drug']
        gene = row['Gene']
        G.add_edge(drug, gene)
        edge_count += 1
    
    # Validate graph structure
    if G.number_of_nodes() == 0:
        raise ValueError("Graf tidak memiliki node. Periksa format data.")
    
    if G.number_of_edges() == 0:
        raise ValueError("Graf tidak memiliki edge. Periksa format data.")
    
    # Check if graph is actually bipartite
    if not nx.is_bipartite(G):
        raise ValueError("Graf yang dihasilkan bukan bipartit. Periksa data untuk memastikan tidak ada koneksi Drug-Drug atau Gene-Gene.")
    
    return G

def create_bipartite_graph_plotly(G, node_labels, communities_sorted, max_nodes_per_side=50, show_labels=True, node_size=8, selected_community=None, limit_nodes=True):
    """Convert networkx graph to plotly visualization"""
    left = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    right = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]
    
    warnings = []
    
    if not left or not right:
        warnings.append("Atribut bipartite pada node tidak lengkap. Visualisasi mungkin tidak rapi.")

    # If individual community mode, filter nodes to only show selected community
    if selected_community is not None and selected_community <= len(communities_sorted):
        selected_comm = communities_sorted[selected_community - 1]
        left = [n for n in left if n in selected_comm]
        right = [n for n in right if n in selected_comm]
        communities_sorted = [selected_comm]  # Only show this community
        warnings.append(f"Menampilkan hanya Komunitas {selected_community}")

    # Filter nodes if too many (only if limit_nodes is True)
    if limit_nodes:
        if len(left) > max_nodes_per_side:
            # Sort by degree and take top nodes
            left = sorted(left, key=lambda n: G.degree(n), reverse=True)[:max_nodes_per_side]
            warnings.append(f"Menampilkan {len(left)} drug nodes teratas (dari {len([n for n, d in G.nodes(data=True) if d.get('bipartite') == 0])} total)")
        
        if len(right) > max_nodes_per_side:
            # Sort by degree and take top nodes
            right = sorted(right, key=lambda n: G.degree(n), reverse=True)[:max_nodes_per_side]
            warnings.append(f"Menampilkan {len(right)} gene nodes teratas (dari {len([n for n, d in G.nodes(data=True) if d.get('bipartite') == 1])} total)")
    else:
        # Show all nodes without limit
        warnings.append(f"Menampilkan semua {len(left)} drug nodes dan {len(right)} gene nodes")

    # Create subgraph with filtered nodes
    subgraph_nodes = set(left + right)
    G_sub = G.subgraph(subgraph_nodes)
    
    # Buat mapping komunitas berdasarkan urutan yang ditampilkan
    community_order = {}
    for idx, comm in enumerate(communities_sorted, start=1):
        for node in comm:
            community_order[node] = idx if selected_community is None else selected_community

    # Calculate positions
    def side_positions(nodes, x_const):
        groups = defaultdict(list)
        for n in nodes:
            if n in community_order:
                groups[community_order[n]].append(n)
        ordered_groups = sorted(groups.items(), key=lambda kv: kv[0])
        y = 0.0
        pos = {}
        for _, members in ordered_groups:
            members_sorted = sorted(members, key=lambda n: G_sub.degree(n), reverse=True)
            for n in members_sorted:
                pos[n] = (x_const, y)
                y -= 0.25
            y -= 0.8
        return pos

    pos_left = side_positions(left, -1.0)
    pos_right = side_positions(right, 1.0)
    pos = {**pos_left, **pos_right}

    # Prepare data for plotly
    edge_x = []
    edge_y = []
    for edge in G_sub.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#888888'),  # Moderately thicker than original, not too thick
        hoverinfo='none',
        mode='lines',
        opacity=0.3,  # Medium opacity for better visibility without being too bold
    )

    # Prepare node traces by community
    node_traces = []
    
    # More diverse color palette with reds, yellows, blues, browns, greens, etc.
    community_colors = [
        '#E63946',  # Red
        '#FFB703',  # Yellow/Gold
        '#2A9D8F',  # Teal green
        '#8B4513',  # Brown
        '#4361EE',  # Royal blue
        '#FF6B6B',  # Light red
        '#1E88E5',  # Blue
        '#43AA8B',  # Green
        '#F9844A',  # Orange
        '#9C27B0',  # Purple
        '#795548',  # Dark brown
        '#009688',  # Turquoise
        '#F94144',  # Bright red
        '#6A0572',  # Deep purple
        '#F8961E',  # Deep orange
        '#577590',  # Steel blue
        '#90BE6D',  # Leaf green
        '#FF9E00',  # Amber
        '#8338EC',  # Violet
        '#264653',  # Dark teal
        '#00B4D8',  # Bright blue
        '#E76F51',  # Terra cotta
        '#6D6875',  # Slate gray
        '#06D6A0'   # Mint
    ]
    
    for comm_idx in sorted(set(community_order.values())):
        # Get nodes in this community
        comm_nodes_left = [n for n in left if n in community_order and community_order[n] == comm_idx]
        comm_nodes_right = [n for n in right if n in community_order and community_order[n] == comm_idx]
        
        color = community_colors[(comm_idx-1) % len(community_colors)]
        
        # Left nodes (squares)
        if comm_nodes_left:
            node_trace_left = go.Scatter(
                x=[pos[n][0] for n in comm_nodes_left],
                y=[pos[n][1] for n in comm_nodes_left],
                mode='markers+text' if show_labels else 'markers',
                marker=dict(
                    symbol='square',
                    size=node_size,
                    color=color,
                    line=dict(width=1, color='#ffffff'),
                    opacity=0.9
                ),
                text=[str(G_sub.nodes[n].get("display", G_sub.nodes[n].get("label", n))) for n in comm_nodes_left] if show_labels else "",
                textposition="middle right",
                textfont=dict(size=7, color='#333333'),
                hovertext=[f"<b>{G_sub.nodes[n].get('display', n)}</b><br>Type: Drug<br>Community: {comm_idx}<br>Connections: {G_sub.degree(n)}" for n in comm_nodes_left],
                hoverinfo='text',
                name=f'Community {comm_idx} (Left)'
            )
            node_traces.append(node_trace_left)
        
        # Right nodes (circles)
        if comm_nodes_right:
            node_trace_right = go.Scatter(
                x=[pos[n][0] for n in comm_nodes_right],
                y=[pos[n][1] for n in comm_nodes_right],
                mode='markers+text' if show_labels else 'markers',
                marker=dict(
                    symbol='circle',
                    size=node_size,
                    color=color,
                    line=dict(width=1, color='#ffffff'),
                    opacity=0.9
                ),
                text=[str(G_sub.nodes[n].get("display", G_sub.nodes[n].get("label", n))) for n in comm_nodes_right] if show_labels else "",
                textposition="middle left",
                textfont=dict(size=7, color='#333333'),
                hovertext=[f"<b>{G_sub.nodes[n].get('display', n)}</b><br>Type: Gene<br>Community: {comm_idx}<br>Connections: {G_sub.degree(n)}" for n in comm_nodes_right],
                hoverinfo='text',
                name=f'Community {comm_idx} (Right)'
            )
            node_traces.append(node_trace_right)

    # Create figure with modern styling
    title = f'BiMLPA Community Detection Results - Komunitas {selected_community}' if selected_community else 'BiMLPA Community Detection Results'
    fig = go.Figure(data=[edge_trace] + node_traces,
                    layout=go.Layout(
                        title=dict(
                            text=title,
                            font=dict(family="Poppins, sans-serif", size=18, color='#333333'),
                            x=0.5,
                            xanchor='center'
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white',  # Changed to pure white for better contrast
                        paper_bgcolor='white',  # Changed to pure white for better contrast
                    ))
    
    return fig, warnings

# Define layout
def get_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("BiMLPA Community Detection", className="display-5 fw-bold text-primary mb-0"),
                    html.P("Bipartite Multilabel Propagation Algorithm", className="lead text-muted")
                ], className="text-center py-4")
            ], width=12)
        ]),
        dbc.Row([
            # Sidebar (left)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-file-upload me-2 text-primary"),
                        "Input Data"
                    ], className="fw-bold"),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt fa-2x mb-2 text-primary"),
                                html.Div([
                                    'Drag & Drop atau ',
                                    html.A('Pilih File CSV/TSV/TXT', className="text-primary fw-bold")
                                ], className="text-center")
                            ], className="d-flex flex-column align-items-center justify-content-center h-100"),
                            style={
                                'width': '100%',
                                'height': '120px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center'
                            },
                            className="upload-box mb-3"
                        ),
                        html.Div(id='upload-status')
                    ])
                ], className="mb-4 shadow-sm"),
                
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-sliders-h me-2 text-primary"),
                        "Parameter"
                    ], className="fw-bold"),
                    dbc.CardBody([
                        html.Label("Threshold (BiMLPA)", className="form-label"),
                        dcc.Slider(
                            id='threshold-slider',
                            min=0.0, max=1.0, step=0.05, value=0.5,
                            marks={0: '0', 0.5: '0.5', 1: '1'},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-4"
                        ),
                        
                        html.Label("Max propagating labels", className="form-label"),
                        dbc.Input(id='max-prop-input', type="number", value=3, min=1, step=1, className="mb-3"),
                        
                        html.Label("Max MM iter", className="form-label"),
                        dbc.Input(id='max-mm-input', type="number", value=100, min=1, step=1, className="mb-3"),
                        
                        html.Label("Max MS iter", className="form-label"),
                        dbc.Input(id='max-ms-input', type="number", value=100, min=1, step=1, className="mb-3")
                    ])
                ], className="mb-4 shadow-sm"),
                
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-network me-2 text-primary"), 
                        "Visualisasi"
                    ], className="fw-bold"),
                    dbc.CardBody([
                        html.Label("Mode Visualisasi", className="form-label"),
                        dbc.RadioItems(
                            id='viz-mode',
                            options=[
                                {'label': ' Tampilkan semua komunitas', 'value': 'all'},
                                {'label': ' Tampilkan per komunitas', 'value': 'individual'}
                            ],
                            value='all',
                            className="mb-3"
                        ),
                        
                        html.Div([
                            html.Label("Pilih Komunitas", className="form-label"),
                            dcc.Dropdown(
                                id='community-dropdown',
                                placeholder="Pilih komunitas...",
                                className="mb-3 dash-dropdown"
                            )
                        ], id='community-selector', style={'display': 'none'}),
                        
                        dbc.Checklist(
                            id='limit-nodes',
                            options=[{'label': ' Batasi jumlah node per sisi', 'value': 'limit'}],
                            value=['limit'],
                            className="mb-2"
                        ),
                        
                        html.Div([
                            html.Label("Maks node per sisi", className="form-label mt-2"),
                            dcc.Slider(
                                id='max-nodes-slider',
                                min=10, max=200, step=10, value=50,
                                marks={10: '10', 50: '50', 100: '100', 200: '200'},
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mb-4"
                            )
                        ], id='node-limit-controls'),
                        
                        dbc.Checklist(
                            id='show-labels',
                            options=[{'label': ' Tampilkan label node', 'value': 'show'}],
                            value=['show'],
                            className="mb-3"
                        ),
                        
                        html.Label("Ukuran node", className="form-label"),
                        dcc.Slider(
                            id='node-size-slider',
                            min=5, max=20, step=1, value=8,
                            marks={5: '5', 10: '10', 15: '15', 20: '20'},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-4"
                        )
                    ])
                ], className="mb-4 shadow-sm"),
                
                dbc.Button([
                    html.I(className="fas fa-play me-2"),
                    "Jalankan BiMLPA"
                ], id="run-button", color="primary", className="w-100 mb-4 py-3")
                
            ], width=3, className="pe-4"),
            
            # Main content (right)
            dbc.Col([
                html.Div(id="graph-summary-container"),
                
                # Results container
                html.Div([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-project-diagram me-2 text-primary"),
                            "Hasil Komunitas"
                        ], className="fw-bold"),
                        dbc.CardBody(
                            html.Div(id="communities-output")  # Remove spinner
                        )
                    ], className="mb-4 shadow-sm"),
                    
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-network-wired me-2 text-primary"),
                            "Visualisasi"
                        ], className="fw-bold"),
                        dbc.CardBody([
                            html.Div(id="visualization-warnings", className="mb-3"),
                            dcc.Graph(id="graph-visualization", style={"height": "600px"})  # Remove spinner
                        ])
                    ], className="mb-4 shadow-sm"),
                    
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-chart-line me-2 text-primary"),
                            "Modularity"
                        ], className="fw-bold"),
                        dbc.CardBody(
                            html.Div(id="modularity-output")  # Remove spinner
                        )
                    ], className="mb-4 shadow-sm"),
                    
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-table me-2 text-primary"),
                            "Tabel Node → Komunitas",
                            dbc.Button([
                                html.I(className="fas fa-download me-2"),
                                "Unduh CSV"
                            ], id="download-button", color="primary", size="sm", className="float-end")
                        ], className="fw-bold"),
                        dbc.CardBody([
                            dcc.Download(id="download-data"),
                            dash_table.DataTable(  # Remove spinner
                                id='node-community-table',
                                style_table={
                                    'overflowX': 'auto',
                                    'borderRadius': '10px'
                                },
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '12px 15px',
                                    'font-family': 'Poppins, system-ui',
                                    'fontSize': '14px',
                                    'maxWidth': '300px',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                    'whiteSpace': 'nowrap'
                                },
                                style_cell_conditional=[
                                    {
                                        'if': {'column_id': 'community'},
                                        'width': '15%',
                                        'maxWidth': '120px',
                                        'minWidth': '100px'
                                    },
                                    {
                                        'if': {'column_id': 'total_nodes'},
                                        'width': '15%',
                                        'maxWidth': '120px',
                                        'minWidth': '100px'
                                    },
                                    {
                                        'if': {'column_id': 'drugs'},
                                        'width': '35%',
                                        'maxWidth': '400px',
                                        'minWidth': '200px'
                                    },
                                    {
                                        'if': {'column_id': 'proteins'},
                                        'width': '35%',
                                        'maxWidth': '400px',
                                        'minWidth': '200px'
                                    }
                                ],
                                style_header={
                                    'backgroundColor': '#f1f3f9',
                                    'fontWeight': 'bold',
                                    'borderBottom': '2px solid #dee2e6',
                                    'color': '#333'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': '#f8f9fa'
                                    },
                                    {
                                        'if': {'column_id': 'community'},
                                        'fontWeight': 'bold',
                                        'color': '#4361ee'
                                    }
                                ],
                                page_size=10,
                                page_action='native',
                                sort_action='native',
                                filter_action='native'
                            )
                        ])
                    ], className="mb-4 shadow-sm")
                ], id='results-container', style={'display': 'none'}),
                
                # Simplified welcome container - align with top of sidebar
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-network-wired fa-3x mb-3 text-primary"),
                                html.H4("BiMLPA Community Detection", className="mb-3"),
                                html.P([
                                    "Deteksi komunitas pada graf bipartit menggunakan algoritma BiMLPA.",
                                    html.Br(),
                                    "Pilih file CSV, TSV, atau TXT dengan kolom 'Drug' dan 'Gene' untuk memulai."
                                ], className="mb-0 text-muted")
                            ], className="text-center py-4")
                        ])
                    ], className="shadow-sm")  # Removed my-5 class to align with top
                ], id="welcome-container", className="mt-0")  # Added mt-0 to ensure no top margin
            ], width=9)
        ])
    ], fluid=True, className="py-4 px-4", style={'maxWidth': '1800px'})

app.layout = get_layout()

# Global variable to store the uploaded file path
uploaded_file_path = None

# Import Plotly Express for color scales
import plotly.express as px
from dash.exceptions import PreventUpdate

# Define callbacks
@app.callback(
    Output('seed-container', 'style'),
    Input('deterministic-mode', 'value')
)
def toggle_seed_input(deterministic_mode):
    # Always hide seed container since we're always deterministic
    return {'display': 'none'}

@app.callback(
    Output('node-limit-controls', 'style'),
    Input('limit-nodes', 'value')
)
def toggle_node_limit_controls(limit_nodes):
    if 'limit' in limit_nodes:
        return {'display': 'block'}
    return {'display': 'none'}

@app.callback(
    Output('community-selector', 'style'),
    Input('viz-mode', 'value')
)
def toggle_community_selector(viz_mode):
    if viz_mode == 'individual':
        return {'display': 'block'}
    return {'display': 'none'}

@app.callback(
    Output('edge-list-options', 'style'),
    Input('data-mode', 'value')
)
def toggle_edge_list_options(data_mode):
    # Remove this callback - no longer needed
    return {'display': 'none'}

@app.callback(
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_upload_status(contents, filename):
    if contents is not None:
        return html.Div([
            html.I(className="fas fa-check-circle text-success me-2"),
            f"File dipilih: {filename}"
        ])
    return html.Div("Belum ada file yang dipilih", className="text-muted fst-italic")

# Add this function at the top level of your code, outside any callbacks:
def create_toggle_callback(collapse_id):
    @app.callback(
        Output(collapse_id, "is_open"),
        Input(f"toggle-{collapse_id}", "n_clicks"),
        State(collapse_id, "is_open"),
        prevent_initial_call=True
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open
    return toggle_collapse

@app.callback(
    Output({'type': 'community-collapse', 'index': dash.dependencies.MATCH}, 'is_open'),
    Input({'type': 'community-button', 'index': dash.dependencies.MATCH}, 'n_clicks'),
    State({'type': 'community-collapse', 'index': dash.dependencies.MATCH}, 'is_open'),
    prevent_initial_call=True
)
def toggle_community_collapse(n_clicks, is_open):
    if n_clicks is None:
        raise PreventUpdate
    return not is_open

@app.callback(
    [Output('graph-visualization', 'figure', allow_duplicate=True),
     Output('visualization-warnings', 'children', allow_duplicate=True)],
    [Input('viz-mode', 'value'),
     Input('community-dropdown', 'value'),
     Input('limit-nodes', 'value')],
    [State('max-nodes-slider', 'value'),
     State('show-labels', 'value'),
     State('node-size-slider', 'value')],
    prevent_initial_call=True
)
def update_visualization_mode(viz_mode, selected_community, limit_nodes, max_nodes_per_side, show_labels, node_size):
    # Check if we have stored graph data
    if not app_data['graph'] or not app_data['communities_sorted']:
        raise PreventUpdate
    
    G = app_data['graph']
    communities_sorted = app_data['communities_sorted']
    node_labels = app_data['node_labels']
    
    show_node_labels = 'show' in show_labels
    limit_nodes_enabled = 'limit' in limit_nodes
    
    if viz_mode == 'all':
        # Show all communities
        fig, vis_warnings = create_bipartite_graph_plotly(
            G, node_labels, communities_sorted, 
            max_nodes_per_side, show_node_labels, node_size,
            limit_nodes=limit_nodes_enabled
        )
    elif viz_mode == 'individual' and selected_community:
        # Show only selected community
        fig, vis_warnings = create_bipartite_graph_plotly(
            G, node_labels, communities_sorted, 
            max_nodes_per_side, show_node_labels, node_size, 
            selected_community=selected_community,
            limit_nodes=limit_nodes_enabled
        )
    else:
        # Default case - show all
        fig, vis_warnings = create_bipartite_graph_plotly(
            G, node_labels, communities_sorted, 
            max_nodes_per_side, show_node_labels, node_size,
            limit_nodes=limit_nodes_enabled
        )
    
    # Display warnings if any
    warnings_display = []
    if vis_warnings:
        for warning in vis_warnings:
            warnings_display.append(dbc.Alert(warning, color="info", dismissable=True))
    
    return fig, warnings_display

@app.callback(
    [Output('results-container', 'style'),
     Output('graph-summary-container', 'children'),
     Output('communities-output', 'children'),
     Output('graph-visualization', 'figure'),
     Output('visualization-warnings', 'children'),
     Output('modularity-output', 'children'),
     Output('node-community-table', 'data'),
     Output('node-community-table', 'columns'),
     Output('community-dropdown', 'options', allow_duplicate=True)],
    [Input('run-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename'),
     State('threshold-slider', 'value'),
     State('max-prop-input', 'value'),
     State('max-mm-input', 'value'),
     State('max-ms-input', 'value'),
     State('max-nodes-slider', 'value'),
     State('show-labels', 'value'),
     State('node-size-slider', 'value'),
     State('limit-nodes', 'value')],
    prevent_initial_call=True,
    running=[
        # Show loading indicator on button while callback is running
        (Output('run-button', 'disabled'), True, False),
        # Show loading overlay on results while callback is running
        (Output('results-container', 'className'), 'loading', '')
    ]
)
def run_bimlpa(n_clicks, contents, filename, threshold, 
               max_prop, max_mm, max_ms, max_nodes_per_side, show_labels, node_size, limit_nodes):
    if n_clicks is None or contents is None:
        empty_fig = go.Figure()
        return {'display': 'none'}, None, None, empty_fig, None, None, [], [], []
    
    # Validate file extension - now accepts CSV, TSV, and TXT
    valid_extensions = ['.csv', '.tsv', '.txt']
    if not any(filename.lower().endswith(ext) for ext in valid_extensions):
        empty_fig = go.Figure()
        error_message = dbc.Alert([
            html.H5("Format File Tidak Valid", className="alert-heading"),
            html.P(f"File yang diupload: {filename}"),
            html.Hr(),
            html.P("Harap upload file dengan format CSV (.csv), TSV (.tsv), atau TXT (.txt)", className="mb-0")
        ], color="danger")
        return {'display': 'none'}, error_message, None, empty_fig, None, None, [], [], []
    
    # Decode and save uploaded content
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
    except Exception as e:
        empty_fig = go.Figure()
        error_message = dbc.Alert([
            html.H5("Gagal Membaca File", className="alert-heading"),
            html.P(f"Error: {str(e)}"),
            html.Hr(),
            html.P("File mungkin corrupt atau tidak dalam format yang benar.", className="mb-0")
        ], color="danger")
        return {'display': 'none'}, error_message, None, empty_fig, None, None, [], [], []
    
    # Create temp file to store the data - preserve original extension
    path = None
    try:
        # Get file extension
        file_ext = os.path.splitext(filename)[1] if '.' in filename else '.csv'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, mode='wb') as tmp:
            tmp.write(decoded)
            path = tmp.name
    except Exception as e:
        empty_fig = go.Figure()
        error_message = dbc.Alert([
            html.H5("Gagal Menyimpan File Sementara", className="alert-heading"),
            html.P(f"Error: {str(e)}"),
        ], color="danger")
        return {'display': 'none'}, error_message, None, empty_fig, None, None, [], [], []
    
    try:
        # Load graph from T2DM file (CSV/TSV/TXT)
        try:
            G = generate_bipartite_from_t2dm(path)
        except ValueError as e:
            # Specific validation errors from generate_bipartite_from_t2dm
            empty_fig = go.Figure()
            error_message = dbc.Alert([
                html.H5("Error Validasi Data", className="alert-heading"),
                html.P(str(e)),
                html.Hr(),
                html.P([
                    "Pastikan file memiliki:",
                    html.Ul([
                        html.Li("Format CSV, TSV, atau TXT yang valid"),
                        html.Li("Kolom 'Drug' dan 'Gene'"),
                        html.Li("Minimal satu baris data (selain header)"),
                        html.Li("Tidak ada nilai kosong pada kolom Drug dan Gene"),
                        html.Li("Untuk TSV/TXT: gunakan tab sebagai separator")
                    ])
                ], className="mb-0")
            ], color="danger")
            if path and os.path.exists(path):
                os.unlink(path)
            return {'display': 'none'}, error_message, None, empty_fig, None, None, [], [], []
        except Exception as e:
            # Unexpected errors during graph generation
            empty_fig = go.Figure()
            error_message = dbc.Alert([
                html.H5("Error Tidak Terduga", className="alert-heading"),
                html.P(f"Gagal membuat graf: {str(e)}"),
                html.Hr(),
                html.P("Silakan periksa format file dan coba lagi.", className="mb-0")
            ], color="danger")
            if path and os.path.exists(path):
                os.unlink(path)
            return {'display': 'none'}, error_message, None, empty_fig, None, None, [], [], []
            
        # Clean up temp file
        if path and os.path.exists(path):
            os.unlink(path)
        
        # Validate graph has sufficient data
        if G.number_of_nodes() < 2:
            empty_fig = go.Figure()
            error_message = dbc.Alert([
                html.H5("Graf Terlalu Kecil", className="alert-heading"),
                html.P(f"Graf hanya memiliki {G.number_of_nodes()} node."),
                html.Hr(),
                html.P("Diperlukan minimal 2 node untuk analisis komunitas.", className="mb-0")
            ], color="warning")
            return {'display': 'none'}, error_message, None, empty_fig, None, None, [], [], []
        
        if G.number_of_edges() < 1:
            empty_fig = go.Figure()
            error_message = dbc.Alert([
                html.H5("Graf Tidak Memiliki Koneksi", className="alert-heading"),
                html.P("Graf tidak memiliki edge (koneksi antar node)."),
                html.Hr(),
                html.P("Diperlukan minimal 1 edge untuk analisis komunitas.", className="mb-0")
            ], color="warning")
            return {'display': 'none'}, error_message, None, empty_fig, None, None, [], [], []
        
        # Backup 'label' to 'display' if needed
        for n, data in G.nodes(data=True):
            if 'label' in data and not isinstance(data['label'], dict):
                data['display'] = data['label']
                del data['label']
        
        left = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
        right = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]
        
        # Graph summary
        graph_summary = dbc.Card([
            dbc.CardHeader("Ringkasan Graf", className="fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H4(G.number_of_nodes(), className="text-center"),
                        html.P("Nodes", className="text-center text-muted")
                    ], width=4),
                    dbc.Col([
                        html.H4(G.number_of_edges(), className="text-center"),
                        html.P("Edges", className="text-center text-muted")
                    ], width=4),
                    dbc.Col([
                        html.H4(f"{len(left)} | {len(right)}", className="text-center"),
                        html.P("Left | Right", className="text-center text-muted")
                    ], width=4)
                ])
            ])
        ], className="mb-4 shadow-sm")
        
        # Run BiMLPA - hapus semua yang terkait seed
        try:
            # Always use sqrtdeg variant tanpa seed
            algo = BiMLPA_SqrtDeg(G, threshold, int(max_prop), int(max_mm), int(max_ms))
            algo.start()
        except Exception as e:
            empty_fig = go.Figure()
            error_message = dbc.Alert([
                html.H5("Error Saat Menjalankan BiMLPA", className="alert-heading"),
                html.P(f"Error: {str(e)}"),
                html.Hr(),
                html.P("Coba sesuaikan parameter atau periksa data input.", className="mb-0")
            ], color="danger")
            return {'display': 'none'}, graph_summary, error_message, empty_fig, None, None, [], [], []
        
        # Get communities
        try:
            node_labels, communities = _labels_to_partition(G)
            
            if len(communities) == 0:
                empty_fig = go.Figure()
                warning_message = dbc.Alert([
                    html.H5("Tidak Ada Komunitas Terdeteksi", className="alert-heading"),
                    html.P("BiMLPA tidak menemukan komunitas dalam graf."),
                    html.Hr(),
                    html.P("Coba ubah parameter threshold atau periksa struktur data.", className="mb-0")
                ], color="warning")
                return {'display': 'none'}, graph_summary, warning_message, empty_fig, None, None, [], [], []
            
        except Exception as e:
            empty_fig = go.Figure()
            error_message = dbc.Alert([
                html.H5("Error Saat Mengekstrak Komunitas", className="alert-heading"),
                html.P(f"Error: {str(e)}"),
            ], color="danger")
            return {'display': 'none'}, graph_summary, error_message, empty_fig, None, None, [], [], []
        
        # SIMPLIFIED COMMUNITY CARDS - Only use one approach
        communities_sorted = sorted(communities, key=len, reverse=True)
        
        # Create a header for total communities
        communities_header = html.Div([
            html.H5(f"Total Komunitas Terdeteksi: {len(communities_sorted)}", 
                    className="mb-3 text-primary")
        ])
        
        # Create expandable community cards with properly working pattern-matching callbacks
        community_cards = []
        
        for idx, comm in enumerate(communities_sorted, start=1):
            # Separate nodes by bipartite type
            left_nodes = [n for n in comm if G.nodes[n].get("bipartite") == 0]
            right_nodes = [n for n in comm if G.nodes[n].get("bipartite") == 1]
            
            # Sort nodes by name
            left_nodes_display = sorted([G.nodes[n].get("display", G.nodes[n].get("label", n)) for n in left_nodes], key=str)
            right_nodes_display = sorted([G.nodes[n].get("display", G.nodes[n].get("label", n)) for n in right_nodes], key=str)
            
            # Create card with pattern-matching IDs and CLEARLY CLICKABLE BUTTON
            community_cards.append(
                dbc.Card([
                    # Card header with visibly clickable button
                    dbc.CardHeader([
                        # Using a clearly visible button style
                        html.Button(
                            [
                                html.Span(f"Komunitas {idx} ({len(comm)} nodes)", className="me-2"),
                                html.Span(f"Drug: {len(left_nodes)}", className="badge bg-primary me-1"),
                                html.Span(f"Gene: {len(right_nodes)}", className="badge bg-success"),
                                html.I(className="fas fa-chevron-down float-end mt-1")
                            ],
                            id={'type': 'community-button', 'index': idx},
                            className="btn btn-outline-primary w-100 text-start",
                            style={"cursor": "pointer"}
                        )
                    ], className="p-0"),
                    
                    # Collapsible content
                    dbc.Collapse(
                        dbc.CardBody([
                            # Two column layout for displaying nodes
                            dbc.Row([
                                # Left nodes column
                                dbc.Col([
                                    html.Div("Drug/Left Nodes:", className="fw-bold text-primary border-bottom pb-1 mb-2"),
                                    html.Div(
                                        html.Ul([html.Li(str(m)) for m in left_nodes_display], className="mb-0 ps-3"),
                                        style={"max-height": "200px", "overflow-y": "auto"}
                                    ) if left_nodes_display else html.P("None", className="text-muted fst-italic")
                                ], width=6),
                                
                                # Right nodes column
                                dbc.Col([
                                    html.Div("Gene/Right Nodes:", className="fw-bold text-success border-bottom pb-1 mb-2"),
                                    html.Div(
                                        html.Ul([html.Li(str(m)) for m in right_nodes_display], className="mb-0 ps-3"),
                                        style={"max-height": "200px", "overflow-y": "auto"}
                                    ) if right_nodes_display else html.P("None", className="text-muted fst-italic")
                                ], width=6)
                            ])
                        ]),
                        id={'type': 'community-collapse', 'index': idx},
                        is_open=idx==1  # Only first community open by default
                    )
                ], className="mb-4")
            )
        
        # Final communities output
        communities_output = html.Div([
            communities_header,
            html.Div(community_cards)
        ])
        
        # Store graph data for individual community visualization
        app_data['graph'] = G
        app_data['communities_sorted'] = communities_sorted
        app_data['node_labels'] = node_labels
        
        # Visualization
        try:
            show_node_labels = 'show' in show_labels
            limit_nodes_enabled = 'limit' in limit_nodes
            fig, vis_warnings = create_bipartite_graph_plotly(G, node_labels, communities_sorted, max_nodes_per_side, show_node_labels, node_size, limit_nodes=limit_nodes_enabled)
        except Exception as e:
            fig = go.Figure()
            vis_warnings = [f"Error saat membuat visualisasi: {str(e)}"]
        
        # Display warnings if any
        warnings_display = []
        if vis_warnings:
            for warning in vis_warnings:
                warnings_display.append(dbc.Alert(warning, color="warning", dismissable=True))
        
        # Modularity
        try:
            # Use the new calculation functions with proper partition
            community_mapping = {}
            for idx, comm in enumerate(communities_sorted, start=1):
                for node in comm:
                    community_mapping[node] = idx
            
            qm = suzuki_modularity(G)
            
            modularity_output = dbc.Row([
                dbc.Col([
                    html.H5(f"{qm:.4f}", className="text-center"),
                    html.P("Suzuki Modularity", className="text-center text-muted")
                ], width=12)
            ])
        except Exception as e:
            modularity_output = dbc.Alert([
                html.P(f"Gagal menghitung modularity: {str(e)}", className="mb-0")
            ], color="warning")
        
        # Create community mapping (consistent with display)
        community_mapping = {}
        for idx, comm in enumerate(communities_sorted, start=1):
            for node in comm:
                community_mapping[node] = idx
        
        # Node community table
        try:
            # Kelompokkan data per komunitas, tampilkan dalam satu baris per komunitas
            rows = []
            full_rows = []  # Create a separate list for full data
            for idx, comm in enumerate(communities_sorted, start=1):
                # Drug nodes
                drug_nodes = [n for n in comm if G.nodes[n].get("bipartite") == 0]
                drug_names = sorted([str(G.nodes[n].get("display", G.nodes[n].get("label", n))) for n in drug_nodes])
                
                # Protein nodes (bipartite=1, bisa disebut Gene/Protein)
                protein_nodes = [n for n in comm if G.nodes[n].get("bipartite") == 1]
                protein_names = sorted([str(G.nodes[n].get("display", G.nodes[n].get("label", n))) for n in protein_nodes])
                
                # Create complete text for CSV download
                drugs_text = ", ".join(drug_names) if drug_names else "None"
                proteins_text = ", ".join(protein_names) if protein_names else "None"
                
                # Save full data for download
                full_rows.append({
                    "community": idx,
                    "drugs": drugs_text,  # Full text tanpa truncation
                    "proteins": proteins_text,  # Full text tanpa truncation
                    "total_nodes": len(comm)
                })
                
                # Create truncated text for display
                rows.append({
                    "community": idx,
                    "drugs": truncate_text(drugs_text, 150),
                    "proteins": truncate_text(proteins_text, 150),
                    "total_nodes": len(comm)
                })

            # Store the complete data for download
            app_data['full_table_data'] = full_rows
            
            table_data = rows
            table_columns = [
                {"name": "Community", "id": "community"},
                {"name": "Drugs", "id": "drugs"},
                {"name": "Proteins", "id": "proteins"},
                {"name": "Total Nodes", "id": "total_nodes"}
            ]
        except Exception as e:
            table_data = []
            table_columns = []
            warnings_display.append(dbc.Alert(f"Error saat membuat tabel: {str(e)}", color="warning", dismissable=True))
        
        # Create dropdown options for community selector
        community_dropdown_options = [{'label': f'Komunitas {i}', 'value': i} for i in range(1, len(communities_sorted) + 1)]
        
        return {'display': 'block'}, graph_summary, communities_output, fig, warnings_display, modularity_output, table_data, table_columns, community_dropdown_options
    
    except Exception as e:
        # Handle unexpected errors
        empty_fig = go.Figure()
        error_message = dbc.Alert([
            html.H5("Error Tidak Terduga", className="alert-heading"),
            html.P(f"Terjadi kesalahan yang tidak terduga: {str(e)}"),
            html.Hr(),
            html.P([
                "Silakan coba:",
                html.Ul([
                    html.Li("Periksa format file (CSV/TSV/TXT)"),
                    html.Li("Pastikan file memiliki kolom 'Drug' dan 'Gene'"),
                    html.Li("Untuk TSV/TXT: pastikan menggunakan tab sebagai separator"),
                    html.Li("Sesuaikan parameter BiMLPA"),
                    html.Li("Refresh halaman dan coba lagi")
                ])
            ], className="mb-0")
        ], color="danger")
        
        # Clean up temp file if exists
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except:
                pass
        
        return {'display': 'none'}, error_message, None, empty_fig, None, None, [], [], []

@app.callback(
    Output('download-data', 'data'),
    Input('download-button', 'n_clicks'),
    prevent_initial_call=True
)
def download_csv(n_clicks):
    if n_clicks is None or app_data['full_table_data'] is None:
        return None
    
    # Use the full data without truncation for CSV download
    df = pd.DataFrame(app_data['full_table_data'])
    return dcc.send_data_frame(df.to_csv, "communities.csv", index=False)

def truncate_text(text, max_length=100):
    """Truncate text with ellipsis if too long"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

# Import signal for better handling of keyboard interrupts
import signal

# Add signal handler for graceful shutdown
def signal_handler(sig, frame):
    print('\nShutting down Dash server gracefully...')
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Run the app
if __name__ == '__main__':
    try:
        print("Starting Dash server... Press Ctrl+C to stop.")
        # Windows-specific fix for socket issues
        if sys.platform.startswith('win'):
            # Use a different port or disable reloader for Windows
            try:
                app.run(debug=True, use_reloader=False, host='127.0.0.1', port=8050)
            except OSError as e:
                if "socket" in str(e).lower():
                    print("Socket error detected. Trying alternative configuration...")
                    # Try without debug mode
                    app.run(debug=False, host='127.0.0.1', port=8050)
                else:
                    raise e
        else:
            app.run(debug=True)
    except KeyboardInterrupt:
        print('\nShutting down Dash server gracefully...')
    except Exception as e:
        print(f'An error occurred: {str(e)}')
    finally:
        print('Server shutdown complete.')
