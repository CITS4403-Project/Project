# CITS 4403 Project Concepts

Github Repo: https://github.com/CITS4403-Project/Project

## Concept 1: Emergent Wealth Inequality in the Yard-Sale Model

### Overview

This project investigates how wealth inequality can emerge from simple local interactions between agents, even when all agents begin with equal wealth and the total wealth of the system remains constant.

At each time step, two connected agents participate in a transaction. A random process determines the winner, while the amount exchanged depends on a configurable rule or range. Repeated interactions may produce nonlinear and highly unequal wealth distributions at the system level.

### Baseline model

- All agents begin with the same amount of wealth.
- The total wealth of the system is conserved.
- Two agents are selected to trade at each time step.
- The transaction amount is determined by a configurable range or by a proportion of the poorer agent’s wealth.
- A coin toss, with winning probability \(p\), determines the direction of wealth transfer.

### Model extensions

- **Transaction size:** Vary the amount or proportion of wealth exchanged.
- **Biased interactions:** Adjust \(p\) to represent unequal economic advantage.
- **Taxation and redistribution:** Tax transaction gains and redistribute the collected wealth.
- **Static trading network:** Restrict transactions to agents connected through a fixed network.
- **Dynamic trading network:** Allow links to disconnect or reconnect with probability \(r\).
- **Initial wealth distribution:** Compare equal, random, and highly unequal initial conditions.
- **Network topology:** Compare random, small-world, scale-free, and lattice networks.

### Complex-systems phenomena

The model can be used to study:

- Emergent wealth inequality
- Positive feedback and wealth condensation
- Sensitivity to initial conditions
- Self-organisation
- Path dependence
- The influence of network topology
- Possible phase transitions between relatively equal and wealth-condensed states

------

## Concept 2: Robustness and Cascading Failure in the Transperth Transport Network

### Overview

This project models the Transperth rail system as a complex network and investigates its robustness under random failures and targeted attacks.

The railway system is represented as a graph:

\[ G=(V,E), \]

where \(V\) represents stations and \(E\) represents direct railway connections. The project examines how local failures can propagate through the network and cause large-scale disruption.

### Failure scenarios

- **Random failure:** Randomly remove stations or railway connections.
- **Targeted attack:** Remove stations with high degree, betweenness centrality, passenger flow, or other measures of importance.
- **Major-station failure:** Trigger a failure at an important interchange and examine how disruption spreads.
- **Cascading failure:** Redistribute passengers from failed stations onto neighbouring stations, potentially causing them to exceed their capacity and fail.

### Phase transition and critical tolerance

The proportion of removed stations or edges can be gradually increased to identify a critical threshold at which the network’s largest connected component collapses.

Relevant measurements include:

- Size of the largest connected component
- Number of isolated stations
- Average shortest-path length
- Network efficiency
- Passenger demand that can still be served
- Critical failure threshold
- Scale and duration of cascading failures

### Multilayer transport network

Bus routes and major roads can be introduced as a second network layer:

- Rail stations form the rail layer.
- Bus routes or road connections form the backup layer.
- Replacement buses create temporary edges between disconnected regions.
- Different bus-deployment strategies can be compared.

This produces a multilayer or interdependent transport network in which failures and recovery processes occur across multiple transport modes.

### Complex-systems phenomena

The project can investigate:

- Network robustness
- Cascading failure
- Percolation and phase transitions
- Critical nodes
- Nonlinear system collapse
- Multilayer networks
- Adaptive recovery and resilience
