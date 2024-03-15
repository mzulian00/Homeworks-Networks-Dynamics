import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import random
from Sparse_Graph import SparseGraph, random_graph
from Epidemics import Simulator, EpidemicSimulation


vaccination_plan = [0, 5, 15, 25, 35, 45, 55, 60, 60, 60, 60, 60, 60, 60, 60]
I0 = [1, 1, 3, 5, 9, 17, 32, 32, 17, 5, 2, 1, 0, 0, 0, 0]
vaccination_plan_H1N1 = [5, 9, 16, 24, 32, 40, 47, 54, 59, 60, 60, 60, 60, 60, 60, 60]

sim_k_regular = Simulator(graph_type = 'K_REGULAR')
# sim_k_regular.run(k=4)
sim_k_regular.plot_results()


sim_no_vax = Simulator()
# sim_no_vax.run()
sim_no_vax.plot_results()

sim_vax = Simulator(vaccination_plan=vaccination_plan)
# sim_vax.run()
sim_vax.plot_results()



sim_vax = Simulator(num_nodes=934,num_weeks=16,vaccination_plan=vaccination_plan_H1N1)
k, b, r = sim_vax.H1N1(num_simulations=3)
print(k,b,r)

sim_vax.plot_results_H1N1()




