import numpy as np
import pandas as pd
import pylab as plt
import time, enum, math
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import networkx.algorithms.community as nx_comm
import scipy.stats as ss
import operator
import random



class State(enum.IntEnum):
    SUSCEPTIBLE = 0    #neutral
    INFECTED = 1       #believed the rumor, can influence their neighbors
    RECOVERED = 2      #will not influence their neighbors

class MyAgent(Agent):
    """An agent in an epidemic model."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = State.SUSCEPTIBLE
        self.infection_time = 0
           
    def move(self):
        """Move the agent"""
        # check all possible neighbors in the network
        possible_steps = [
            node
            for node in self.model.grid.get_neighbors(self.pos, include_center=False)
            if self.model.grid.is_cell_empty(node)
                        ]
        if len(possible_steps) > 0:
            new_position = self.random.choice(possible_steps)                 
            self.model.grid.move_agent(self, new_position)

    def status(self):
        """Check infection status"""
            
        if self.state == State.INFECTED:     
            t = self.model.schedule.time-self.infection_time
            if t > self.recovery_time:          
                self.state = State.RECOVERED
        

    def contact(self):
        """Find close contacts and infect neighbors if self is infected, or 
        if self is susceptible and any of your neighbors are infected,
        self is influenced by the misinformation.
        """
        
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)

        if self.state == State.INFECTED:
            susceptible_neighbors = [
                                     agent
                                     for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)
                                     if agent.state is State.SUSCEPTIBLE
                                     ]
            
            if len(susceptible_neighbors) > 0.5*len(neighbors_nodes):
                
                for a in susceptible_neighbors:
                    if self.random.random() < self.model.ptrans:
                        a.state = State.INFECTED
                        a.recovery_time = a.model.get_recovery_time()

    def step(self):
        self.status()
        self.move()
        self.contact()

    def toJSON(self):        
        d = self.unique_id
        return json.dumps(d, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)

class NetworkInfectionModel(Model):
    """A model for disinformation spread."""

    def __init__(self, N=10, ptrans=0.1, infection_period=10, m=4,
                 infection_sd=2, p_infect = 0.05, seed=143,
                 target_hubs=True):
    
        self.num_nodes = N    # nodes = agents
        self.infection_period = infection_period
        self.infection_sd = infection_sd
        self.ptrans = ptrans
        self.p_infect = p_infect
        self.target_hubs = target_hubs
        self.m = m
        self.seed = seed

        self.G = nx.barabasi_albert_graph(n=self.num_nodes, m=self.m, seed=self.seed)
        top = list(nx.degree_centrality(self.G).items())
        top.sort(key=operator.itemgetter(1), reverse=True)
        top_nodes = [nd for nd, c in top]
        central_nodes = top_nodes[:(int(self.num_nodes*self.p_infect))]
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.running = True

        # Create agents
        for i, node in enumerate(self.G.nodes()):
            a = MyAgent(i+1, self)
            self.schedule.add(a)
            #add agent
            self.grid.place_agent(a, node)
            # init disbelief

            if self.target_hubs:  #hubs are spreader
                if node in central_nodes:
                    a.recovery_time = self.get_recovery_time()
                    a.state = State.INFECTED

                else:
                    a.state = State.SUSCEPTIBLE  
                    
            else: #random spreaders
                #make some agents infected at start
                infected = np.random.choice([0,1], p=[(1-p_infect), p_infect])
                if infected == 1:
                    a.recovery_time = self.get_recovery_time()
                    a.state = State.INFECTED
                    
        self.datacollector = DataCollector({
            'Susceptible': 'susceptible',
            'Infected': 'infected',
            'Recovered': 'recovered'},
             agent_reporters={"State": "state"})

    def get_recovery_time(self):
        return int(self.random.normalvariate(self.infection_period,
                                             self.infection_sd))
    
    @property
    def susceptible(self):
        
        sus = [agent for agent in self.schedule.agents 
               if agent.state is State.SUSCEPTIBLE]
        return len(sus)

    @property
    def infected(self):
        inf = [agent for agent in self.schedule.agents 
               if agent.state is State.INFECTED]
        return len(inf)
    
    @property
    def recovered(self):
        rec = [agent for agent in self.schedule.agents 
               if agent.state is State.RECOVERED]
        return len(rec)
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        
        
class CommunityInfectionModel(Model):
    """A model for disinformation spread."""

    def __init__(self, N=10, ptrans=0.01, infection_period=10,
                 infection_sd=1, partition = 0.70, seed=143):
    
        self.num_nodes = N    # nodes = agents
        self.infection_period = infection_period
        self.infection_sd = infection_sd
        self.partition = partition
        self.node_to_community = None
        self.believer = None
        self.disbeliever = None
        self.modularity = None
        self.ptrans = ptrans
        self.seed = seed
        self.list_nodes = None
        
        # create a modular graph
        partition_sizes = [int(self.num_nodes*self.partition),
                           int(self.num_nodes*(1-self.partition))]
        self.G = nx.random_partition_graph(partition_sizes, 0.6, 0.4, seed=self.seed)

        # since we created the graph, we know the best partition:        
        self.node_to_community = dict()
        node = 0
        for community_id, size in enumerate(partition_sizes):
            for _ in range(size):
                self.node_to_community[node] = community_id
                node += 1
                
        #compute modularity
        
        communities = set([group for group in self.node_to_community.values()])
        nodes_in_community = {}
        for i in communities:
            nodes_in_community[i] = {node for node, community_id in 
                                     self.node_to_community.items() 
                                     if community_id==i}
            
        self.list_nodes = list(nodes_in_community.values())
        
        self.modularity = nx_comm.modularity(self.G, self.list_nodes)
        
        self.disbeliever = [node for node, community_id in self.node_to_community.items() if community_id==0]
        self.believer = [node for node, community_id in self.node_to_community.items() if community_id==1]

        self.grid = NetworkGrid(self.G)
        
        self.schedule = RandomActivation(self)
        self.running = True

        # Create agents
        for i, node in enumerate(self.G.nodes()):
            a = MyAgent(i+1, self)
            self.schedule.add(a)
            #add agent
            self.grid.place_agent(a, node)

            if node in self.believer:
                a.recovery_time = self.get_recovery_time()
                a.state = State.INFECTED

            else:
                a.state = State.SUSCEPTIBLE  

     
        self.datacollector = DataCollector({
            'Susceptible': 'susceptible',
            'Infected': 'infected',
            'Recovered': 'recovered'},
             agent_reporters={"State": "state"})

    def get_recovery_time(self):
        return int(self.random.normalvariate(self.infection_period,
                                             self.infection_sd))

    
    @property
    def susceptible(self):
        
        sus = [agent for agent in self.schedule.agents 
               if agent.state is State.SUSCEPTIBLE]
        return len(sus)

    @property
    def infected(self):
        inf = [agent for agent in self.schedule.agents 
               if agent.state is State.INFECTED]
        return len(inf)
    
    @property
    def recovered(self):
        rec = [agent for agent in self.schedule.agents 
               if agent.state is State.RECOVERED]
        return len(rec)
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()