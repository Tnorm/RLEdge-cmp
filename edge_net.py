# FNSS NETWORK EXAMPLE
import fnss
import random
import networkx as nx


# drones = 50, edge_nodes = 5, cloud=1
topology = fnss.two_tier_topology(n_core=1, n_edge=5, n_hosts=5)


# assign capacities
# let's set links connecting servers to edge switches to 5 Mbps.
# and links connecting core and edge switches to 5 Mbps.

# get list of core_edge links and edge_leaf links
link_types = nx.get_edge_attributes(topology, 'type')
core_edge_links = [link for link in link_types
                   if link_types[link] == 'core_edge']
edge_leaf_links = [link for link in link_types
                   if link_types[link] == 'edge_leaf']

# assign capacities
fnss.set_capacities_constant(topology, 5, 'Mbps', edge_leaf_links)
fnss.set_capacities_constant(topology, 5, 'Mbps', core_edge_links)

# add traffic
traffic_matrix = fnss.static_traffic_matrix(topology, mean=2, stddev=0.2, max_u=0.5)


# assign weight 1 to all links
fnss.set_weights_constant(topology, 1)

# assign delay of 10 ms to each link
fnss.set_delays_constant(topology, 10, 'ms', edge_leaf_links)
fnss.set_delays_constant(topology, 10, 'ms', core_edge_links)

def action_failure(source_nodes, receiver_nodes):
    source = random.choice(source_nodes)
    receiver = random.choice(receiver_nodes)
    return {'source': source, 'receiver': receiver}


def rand_failure(links):
    link = random.choice(links)
    return {'link': link, 'action': 'down'}


def rand_mobility(nodes):
    node = random.choice(nodes)
    return {'node': node}

def rand_request(source_nodes, receiver_nodes):
    source = random.choice(source_nodes)
    receiver = random.choice(receiver_nodes)
    return {'source': source, 'receiver': receiver}

event_schedule = fnss.poisson_process_event_schedule(
                        avg_interval=5,
                        t_start=0,
                        duration=5000,
                        t_unit='ms',
                        event_generator=rand_failure,  # event gen function
                        links=list(topology.edges()),  # 'links' argument
                        )

node_types = nx.get_node_attributes(topology, 'type')

drones = [nodes for nodes in node_types
              if node_types[nodes] == 'host']
cloud = [0]
edges = [nodes for nodes in node_types
              if node_types[nodes] == 'switch' and nodes != 0]

event_schedule_mob = fnss.poisson_process_event_schedule(
                        avg_interval=5,
                        t_start=0,
                        duration=5000,
                        t_unit='ms',
                        event_generator=rand_mobility,  # event gen function
                        nodes=drones,  # 'nodes' argument
                        )

event_schedule2 = fnss.poisson_process_event_schedule(
                        avg_interval=10,
                        t_start=0,
                        duration=5000,
                        t_unit='ms',
                        event_generator=rand_request,  # event gen function
                        source_nodes=cloud,  # rand_request argument
                        receiver_nodes=drones  # rand_request argument
                        )

# save topology to a file
fnss.write_topology(topology, 'topology.xml')
fnss.write_event_schedule(event_schedule, 'event_schedule.xml')
fnss.write_event_schedule(event_schedule2, 'event_schedule2.xml')
fnss.write_event_schedule(event_schedule_mob, 'event_schedule_mob.xml')
fnss.write_traffic_matrix(traffic_matrix, 'traffic_matrix.xml')