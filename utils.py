import numpy as np
import pandas as pd
import pylab as plt
from netgraph import Graph
import networkx as nx
from ABM import *



def describe_network(model):
    """Return the average clustering coefficient and characteristic
    path length."""
    
    cc = nx.average_clustering(model.G)
    try:
        l = nx.average_shortest_path_length(model.G)
    except:
        l = "disconnected"

    print('Average clustering coefficient:', cc)
    print('Characteristic path length:', l)
    

def simulate_transimission(n_simulations=50, total_steps=100,
                           m=4, no_agents=50, ptrans=0.10,
                           infection_period= 30, infection_sd=10,
                           target_hubs=True):
    """Return a list of dataFrames containing the result of simulation."""
    df_list = []
    for i in range(n_simulations):
        model = NetworkInfectionModel(N=no_agents, ptrans=ptrans,
                                      infection_period= infection_period, 
                                      infection_sd=infection_sd, seed=i,
                                      m=m, target_hubs=target_hubs)
        for i in range(total_steps):
            model.step()

        agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
        # note that recovered nodes are still believer of disinformation
        agent_df.loc[agent_df['State'] == 2, 'State'] = 1
        df = agent_df.groupby('Step')['State'].sum().reset_index()
        df.rename(columns={'State': 'No. of Disinformed'}, inplace=True)
        df_list.append(df)
    return df_list
    

def simulate_community_transmission(n_simulations=50, total_steps=100,
                                    no_agents=50, ptrans=0.10,
                                    infection_period= 30, infection_sd=10):

    """Return a list of dataFrames containing the result of simulation for
    a network with 2 communities."""

    df_list = []
    modularity = []
    for i in range(n_simulations):
        model = CommunityInfectionModel(N=no_agents, ptrans=ptrans,
                                infection_period=infection_period,
                                infection_sd=infection_sd,
                                partition = 0.70, seed=i)
    
        for i in range(total_steps):
            model.step()
            
        modularity.append(model.modularity)
        agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
        # note that recovered nodes are still believer of disinformation
        agent_df.loc[agent_df['State'] == 2, 'State'] = 1
        df = agent_df.groupby('Step')['State'].sum().reset_index()
        df.rename(columns={'State': 'No. of Disinformed'}, inplace=True)
        df_list.append(df)
    return df_list, modularity
    

def plot_communities(model,fig,layout='spring',title=''):
    """Plot the community network."""
    
    cmap =["teal", "violet", "violet"]  # color teal--> susceptible,
                                        # violet--> infected & recovered
    graph = model.G
    plt.clf()
    ax=fig.add_subplot()
    states = [int(i.state) for i in model.grid.get_all_cell_contents()]

    colors = [cmap[i] for i in states]
    
    node_color = {node: color for node, color in zip(model.G.nodes, colors)}

    Graph(graph,
          node_color=node_color, node_edge_width=0, edge_alpha=0.1,
          node_layout='community',
          node_layout_kwargs=dict(node_to_community=model.node_to_community),
          edge_layout='bundled', edge_layout_kwargs=dict(k=2000), seed=0)


def plot_network(model,fig,layout='spring',title=''):
    """Plot the network."""
    
    cmap =["lightblue", "lightcoral", "lightgreen"]
    graph = model.G
    
    if layout == 'kamada-kawai':      
        pos = nx.kamada_kawai_layout(graph)  
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    else:
        pos = nx.spring_layout(graph, seed=8)  
        
    
    plt.clf()
    ax=fig.add_subplot()
    states = [int(i.state) for i in model.grid.get_all_cell_contents()]
    colors = [cmap[i] for i in states]

    nx.draw(graph, pos, node_size=[v * 100 for v in dict(graph.degree).values()], edge_color='gray', node_color=colors, with_labels=True,
            alpha=0.9,font_size=14,ax=ax)
    ax.set_title(title)
    

def plot_SIR(model, no_agents, filepath=None):
    """Plot the SIR within the given time steps."""
    
    spread_df = model.datacollector.get_model_vars_dataframe()
    plt.figure(figsize=(10,5))
    plt.grid()
    plt.plot(spread_df.index, (spread_df['Susceptible']/no_agents)*100, '+-',
             label='Susceptible', color='tab:blue')
    plt.plot(spread_df.index, (spread_df['Infected']/no_agents)*100, '+-',
             label='Infected', color='tab:red')
    plt.plot(spread_df.index, (spread_df['Recovered']/no_agents)*100, '+-',
             label='Recovered', color='tab:green')
    plt.xlabel('Time Step')
    plt.ylabel('% of Total Agents')
    plt.legend(bbox_to_anchor=(1,1), fontsize=12)
    plt.savefig(filepath+".png", dpi=300)
    plt.show()