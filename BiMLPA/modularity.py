from collections import defaultdict


def calculate_murata_modularity(G, partition):
    """
    Murata Modularity (2010)
    
    Q_B = Q_B^{X→Y} + Q_B^{Y→X}
    
    Equations:
    - e_lm = jumlah_edges / M  (Eq. 9)
    - a_l = Σ_m e_lm           (Eq. 10)
    - Q_B^l = e_l,ψ(l) - a_l * a_ψ(l)  (Eq. 11)
    - Q_B = Σ Q_B^l            (Eq. 14)
    
    Args:
        G: bipartite NetworkX graph  
        partition: dict mapping node -> community
    
    Returns:
        float: modularity score
    """
    # Separate nodes by bipartite attribute
    X = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    Y = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]
    
    # Total edges M
    M = G.number_of_edges()
    if M == 0:
        return 0.0
    
    # Count edges between communities (raw count)
    e_count = defaultdict(lambda: defaultdict(int))
    for u, v in G.edges():
        if u in partition and v in partition:
            if G.nodes[u].get("bipartite") == 0:  # u in X
                e_count[partition[u]][partition[v]] += 1
            else:  # v in X
                e_count[partition[v]][partition[u]] += 1
    
    # Normalize by M to get e_lm (Equation 9)
    e_lm = defaultdict(lambda: defaultdict(float))
    for l in e_count:
        for m in e_count[l]:
            e_lm[l][m] = e_count[l][m] / M
    
    # Calculate a_l for X communities (Equation 10)
    a_X = {}
    for l in e_lm:
        a_X[l] = sum(e_lm[l].values())
    
    # Calculate a_m for Y communities (transpose)
    all_Y_comms = set(m for cols in e_lm.values() for m in cols.keys())
    a_Y = {}
    for m in all_Y_comms:
        a_Y[m] = sum(e_lm[l].get(m, 0) for l in e_lm.keys())
    
    # Q_X→Y (Equation 11 & 12)
    Q_XY = 0.0
    for l in e_lm.keys():
        if e_lm[l]:
            # Find corresponding community (argmax)
            psi_l = max(e_lm[l].items(), key=lambda x: x[1])[0]
            # Q_B^l = e_l,ψ(l) - a_l * a_ψ(l)
            Q_XY += (e_lm[l][psi_l] - a_X[l] * a_Y[psi_l])
    
    # Build transpose for Y→X
    e_T = defaultdict(lambda: defaultdict(float))
    for l in e_lm:
        for m in e_lm[l]:
            e_T[m][l] = e_lm[l][m]
    
    # Q_Y→X (Equation 11 & 13)
    Q_YX = 0.0
    for m in e_T.keys():
        if e_T[m]:
            # Find corresponding community (argmax)
            phi_m = max(e_T[m].items(), key=lambda x: x[1])[0]
            # Q_B^m = e_m,φ(m) - a_m * a_φ(m)
            Q_YX += (e_T[m][phi_m] - a_Y[m] * a_X[phi_m])
    
    # Q_B = Q_XY + Q_YX (Equation 14)
    return Q_XY + Q_YX