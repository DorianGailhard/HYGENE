import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull

def plot_hypergraphs(pred_hypergraphs, 
                     node_size=200, edge_width=2, 
                     node_color='skyblue', edge_color='salmon', 
                     alpha=0.7, figsize=(50, 50)):
    
    num_graphs = len(pred_hypergraphs)
    n_cols = int(min(5, np.ceil(np.sqrt(num_graphs))))
    n_rows = int(np.ceil(num_graphs / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = np.atleast_2d(axs)

    for idx, H in enumerate(pred_hypergraphs):
        row, col = divmod(idx, n_cols)
        ax = axs[row, col]
        
        # Clique expansion since hnx doesn't have spring layout
        G = nx.Graph()
        for edge in H.edges:
            nodes = list(H.edges[edge])
            for u in range(len(nodes)):
                for v in range(u + 1, len(nodes)):
                    G.add_edge(nodes[u], nodes[v])

        # Use the clique expansion to compute layout
        pos = nx.spring_layout(G, k=1, scale=1, iterations=100)
        
        # Draw hyperedges as convex hulls or straight lines
        for edge in H.edges:
            nodes = list(H.edges[edge])
            if len(nodes) > 2:
                points = np.array([pos[node] for node in nodes])
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                
                hull_color = np.random.random(3) * 0.5 + 0.5
                ax.fill(hull_points[:, 0], hull_points[:, 1], color=hull_color, alpha=0.3)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 
                            color=edge_color, linewidth=edge_width, alpha=alpha)
            elif len(nodes) == 2:
                node1, node2 = nodes
                ax.plot([pos[node1][0], pos[node2][0]], [pos[node1][1], pos[node2][1]],
                        color=edge_color, linewidth=edge_width, alpha=alpha)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, 
                               node_color=node_color, alpha=1, 
                               linewidths=1, edgecolors='black')
        
        ax.axis('off')
        ax.set_title(f"N = {len(H.nodes)}", fontsize=40)

    # Hide any unused subplots
    for i in range(num_graphs, n_rows * n_cols):
        axs.flat[i].axis('off')

    plt.tight_layout()
    return fig