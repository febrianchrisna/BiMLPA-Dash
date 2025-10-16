import networkx as nx
import argparse
import pandas as pd


def generate_network(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    u_list = []
    v_list = []
    for l in lines:
        l = l.strip()
        if not l or l.startswith('#'):
            continue
        parts = l.split()
        if len(parts) < 2:
            continue
        try:
            u, v = int(parts[0]), int(parts[1])
        except ValueError:
            # Abaikan baris yang tidak bisa di-cast ke int
            continue
        u_list.append(u)
        v_list.append(v)
    if not u_list or not v_list:
        return nx.Graph()
    u_max = max(u_list)
    v_min = min(v_list)
    if u_max < v_min:
        padding = 0
    elif v_min == 0:
        padding = u_max + 1
    else:
        padding = u_max

    G = nx.Graph()
    u_set = list(set(u_list))
    v_set = list(map(lambda x: x+padding, set(v_list)))

    G.add_nodes_from(u_set, bipartite=0)
    G.add_nodes_from(v_set, bipartite=1)

    for i in range(len(u_list)):
        G.add_edge(u_list[i], v_list[i]+padding)
    return G


def generate_network_with_name(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    u_list = []
    v_list = []
    for l in lines:
        l = l.strip()
        if not l or l.startswith('#'):
            continue
        parts = l.split('\t')
        if len(parts) < 2:
            continue
        u, v = parts[0], parts[1]
        u_list.append(u)
        v_list.append(v)

    G = nx.Graph()
    u_set = list(set(u_list))
    v_set = list(set(v_list))

    G.add_nodes_from(u_set, bipartite=0)
    G.add_nodes_from(v_set, bipartite=1)

    for i in range(len(u_list)):
        G.add_edge(u_list[i], v_list[i])
    return G


def _print_summary(G: nx.Graph):
    left = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    right = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    print(f"Left (bipartite=0): {len(left)} | Right (bipartite=1): {len(right)}")


def _save_graph(G: nx.Graph, out_path: str):
    out_lower = out_path.lower()
    if out_lower.endswith('.gpickle'):
        nx.write_gpickle(G, out_path)
    elif out_lower.endswith('.graphml'):
        nx.write_graphml(G, out_path)
    elif out_lower.endswith('.edgelist'):
        nx.write_edgelist(G, out_path, data=False)
    else:
        raise ValueError("Ekstensi output tidak dikenal. Gunakan .gpickle, .graphml, atau .edgelist")


def generate_network_from_csv(path):
    """
    Generate bipartite graph from CSV file with Drug,Gene columns.
    Creates bipartite graph with drugs as left nodes (bipartite=0)
    and genes as right nodes (bipartite=1).
    Compatible with T2DM drug-target dataset format.
    """
    # Read CSV file
    df = pd.read_csv(path)
    
    # For T2DM format, expect 'Drug' and 'Gene' columns
    if 'Drug' not in df.columns or 'Gene' not in df.columns:
        raise ValueError("CSV harus memiliki kolom 'Drug' dan 'Gene'")
    
    # Extract unique values
    drugs = df['Drug'].dropna().unique()
    genes = df['Gene'].dropna().unique()
    
    # Create bipartite graph
    G = nx.Graph()
    
    # Add drug nodes (bipartite=0, side="drug")
    for drug in drugs:
        G.add_node(drug, bipartite=0, side="drug", display=drug)
    
    # Add gene nodes (bipartite=1, side="gene") 
    for gene in genes:
        G.add_node(gene, bipartite=1, side="gene", display=gene)
    
    # Add edges from the dataframe
    for _, row in df.iterrows():
        drug = row['Drug']
        gene = row['Gene']
        if pd.notna(drug) and pd.notna(gene):
            G.add_edge(drug, gene)
    
    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bipartite NetworkX graph from an edge list or CSV.")
    parser.add_argument("path", nargs='?', help="Path ke file input")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ints", action="store_true", help="Edge list 2 kolom (integer IDs)")
    mode.add_argument("--names", action="store_true", help="Edge list 2 kolom (nama, dipisah TAB)")
    mode.add_argument("--csv", action="store_true", help="CSV file dengan kolom Drug,Gene")
    parser.add_argument("--out", help="Simpan graf ke file (.gpickle | .graphml | .edgelist)")
    args = parser.parse_args()

    if not args.path:
        parser.print_help()
        exit()

    if args.csv:
        G = generate_network_from_csv(args.path)
    else:
        if args.names:
            G = generate_network_with_name(args.path)
        else:
            G = generate_network(args.path)

    _print_summary(G)
    
    if args.out:
        _save_graph(G, args.out)
        print(f"Tersimpan: {args.out}")
