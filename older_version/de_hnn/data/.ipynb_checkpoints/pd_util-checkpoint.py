import networkx as nx
import gudhi as gd
import numpy as np

def ego_graph(G, n, radius=1, center=True, undirected=False, distance=None):
    if undirected:
        if distance is not None:
            sp, _ = nx.single_source_dijkstra(
                G.to_undirected(), n, cutoff=radius, weight=distance
            )
        else:
            sp = dict(
                nx.single_source_shortest_path_length(
                    G.to_undirected(), n, cutoff=radius
                )
            )
    else:
        if distance is not None:
            sp, _ = nx.single_source_dijkstra(G, n, cutoff=radius, weight=distance)
        else:
            sp = dict(nx.single_source_shortest_path_length(G, n, cutoff=radius))

    H = list(G.subgraph(sp).copy().edges)
    return H, sp



def gen_local_pd(G, s_n, radius, processed=True):
    edgs, loc_delay_dict = ego_graph(G, s_n, radius)
    #print(sub_g.number_of_nodes())
    if len(edgs) <= 2:
        return np.array([[0., 0.]]), np.array([[0., 0.]]), [0, 0, 0, 0]
    d_lst = [[] for idx in range(radius)]

    for k, v in loc_delay_dict.items():
        d_lst[v-1].append(k)
    
    to_level = radius
    st = gd.SimplexTree()

    for edg in edgs:
        st.insert(edg)

    for delay in range(to_level):
        nodes = d_lst[delay]
        for node in nodes:
            st.assign_filtration([node], delay)

    st.extend_filtration()
    dgms = st.extended_persistence()
    
    if processed:
        zero_dgm = np.array([list([tp[1][0], tp[1][1]]) for tp in dgms[0]] + [list([tp[1][0], tp[1][1]]) for tp in dgms[2]])
        one_dgm = np.array([list([tp[1][1], tp[1][0]]) for tp in dgms[-1]])
        neigh_lst = [len(val) for val in d_lst]
        
        if zero_dgm.shape[-1] != 2:
            zero_dgm = np.array([[0., 0.]])
                            
        if one_dgm.shape[-1] != 2:
            one_dgm = np.array([[0., 0.]])

        if len(neigh_lst) < 4:
            neigh_lst = neigh_lst + [0 for idx in range(4 - len(neigh_lst))]
        
    return zero_dgm, one_dgm, neigh_lst
