import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from Sparse_Graph import SparseGraph, random_graph, k_regular_graph

DEBUG_EPIDEMICS = False
DEBUG_RUN = False

class EpidemicSimulation:
    """
    This class simulate an epidemic given a network and the epidemic parameters
    """
    def __init__(self, G, beta=0.3, rho=0.6):
        self.G = SparseGraph(G)
        self.beta = beta
        self.rho = rho
        self.num_initial_Vaccinated_nodes = 0
        
    def set_initial_infected_nodes(self, num_initial_Infected_nodes):
        nodes = np.random.choice(self.G.nodes_list, size=num_initial_Infected_nodes, replace=False)
        for node in nodes: 
            self.G.set_node_state(node, 'I')

    def set_initial_vaccinated_nodes(self, num_initial_Vaccinated_nodes):
        nodes_with_state, _ = self.G.get_dictionaries()
        non_vaccinated_nodes = [node for node in nodes_with_state['NOVAX'] if node not in nodes_with_state['I']]
        self.vaccinate(non_vaccinated_nodes, num_initial_Vaccinated_nodes)
        self.num_initial_Vaccinated_nodes = num_initial_Vaccinated_nodes

    def infect(self, susceptible_nodes, infected_nodes):
        """
        infected_nodes infect some of the susceptible_nodes with probability
        P = 1-(1-beta)^m and return how many has been infected this week
        """
        newly_I = 0
        for susceptible_node in susceptible_nodes:
            m = sum([1 for neighbor in self.G.neighbors(susceptible_node) if neighbor in infected_nodes])
            infection_probability = [1-(1-self.beta)**m , (1-self.beta)**m]
            is_infected = np.random.choice([True, False], p=infection_probability)
            if is_infected:
                self.G.set_node_state(susceptible_node, 'I')
                newly_I += 1
        return newly_I
    
    def recover(self, infected_nodes):
        """
        recover some of the infected_nodes with probability P = rho
        """
        recover_probability = [self.rho, 1-self.rho]
        for infected_node in infected_nodes:
            has_recovered = np.random.choice([True, False], p=recover_probability)
            if has_recovered:
                self.G.set_node_state(infected_node, 'R')
     
    def vaccinate(self, non_vaccinated_nodes, num_nodes_to_be_vaccinated):
        """
        vaccinate a number of num_nodes_to_be_vaccinated random unvaccinated nodes
        """
        vaccinated_nodes = np.random.choice(non_vaccinated_nodes, size=num_nodes_to_be_vaccinated, replace=False)
        for node in vaccinated_nodes:
            self.G.set_node_state(node, 'V')

    def weekly_vaccination(self, vaccination_plan, week):
        """
        compute how many nodes vaccinate this week using the percentage given by Vacc(t)
        """
        if week == 0:
            newly_V = int(vaccination_plan[0] * self.G.num_nodes / 100) - self.num_initial_Vaccinated_nodes
        else:
            newly_V = int((vaccination_plan[week] - vaccination_plan[week-1]) * self.G.num_nodes / 100)
        return newly_V

    def SIR(self, num_weeks=15):
        newly_infected_per_week, total_susceptible_per_week = [], []
        total_infected_per_week, total_recovered_per_week   = [], []
        
        for week in range(num_weeks):
            nodes_with_state, num_nodes_with_state = self.G.get_dictionaries()

            newly_I = self.infect(nodes_with_state['S'], nodes_with_state['I'])

            self.recover(nodes_with_state['I'])

            nodes_with_state, num_nodes_with_state = self.G.get_dictionaries()

            newly_infected_per_week.append(newly_I)
            total_susceptible_per_week.append(num_nodes_with_state['S'])
            total_infected_per_week.append(num_nodes_with_state['I'])
            total_recovered_per_week.append(num_nodes_with_state['R'])
                
        return newly_infected_per_week, total_susceptible_per_week, total_infected_per_week, total_recovered_per_week
    
    def SIRV(self, num_weeks=15, vaccination_plan=[0,5,15,25,35,45,55,60,60,60,60,60,60,60,60]):
        if num_weeks != len(vaccination_plan):
            print('exit')
            exit()
        
        newly_infected_per_week, newly_vaccinated_per_week  = [], []
        total_susceptible_per_week, total_infected_per_week = [], []
        total_recovered_per_week, total_vaccinated_per_week = [], []

        if DEBUG_EPIDEMICS:
            self.G.plot_Graph()
        
        for week in range(num_weeks):
            nodes_with_state, _ = self.G.get_dictionaries()

            newly_V = self.weekly_vaccination(vaccination_plan,week)
            self.vaccinate(nodes_with_state['NOVAX'], newly_V)

            nodes_with_state, num_nodes_with_state = self.G.get_dictionaries()

            newly_I = self.infect(nodes_with_state['S'], nodes_with_state['I'])

            self.recover(nodes_with_state['I'])

            nodes_with_state, num_nodes_with_state = self.G.get_dictionaries()

            newly_infected_per_week.append(newly_I)
            newly_vaccinated_per_week.append(newly_V)
            total_susceptible_per_week.append(num_nodes_with_state['S'])
            total_infected_per_week.append(num_nodes_with_state['I'])
            total_recovered_per_week.append(num_nodes_with_state['R'])
            total_vaccinated_per_week.append(num_nodes_with_state['V'])

            if DEBUG_EPIDEMICS and week == 5:
                self.G.plot_Graph()
                
        if DEBUG_EPIDEMICS:
            self.G.plot_Graph()
 
        return newly_infected_per_week, total_susceptible_per_week, total_infected_per_week, total_recovered_per_week, newly_vaccinated_per_week, total_vaccinated_per_week


class Simulator():
    """
    This is the main class of the program, it's in charge of creating Graphs, 
    give them to the epidemics and simulate n times, keeping track of the average results.

    """
    def __init__(self,num_simulations=100, num_weeks=15, num_nodes=500, num_initial_I_nodes=10, perc_initial_V_nodes=0, vaccination_plan=[], graph_type = 'RANDOM'):
        if not (graph_type == 'RANDOM' or graph_type == 'K_REGULAR'):
            print('exit')
            exit()
        self.num_simulations = num_simulations
        self.num_weeks = num_weeks
        self.num_nodes = num_nodes
        self.num_initial_I_nodes = num_initial_I_nodes 
        self.num_initial_V_nodes = int(perc_initial_V_nodes * num_nodes / 100)
        if len(vaccination_plan) != num_weeks:
             vaccination_plan = []
        self.vaccination_plan = vaccination_plan
        self.VAX = len(vaccination_plan) > 0
        self.graph_type = graph_type
        self.result = Simulator.WrapperResult(self)
    class WrapperResult():
        """
        this is a subclass used to initialize and update averages of Simulator
        """
        def __init__(self, simulation):
            self.empty = True
            self.num_simulations = simulation.num_simulations
            self.avg_newly_I = np.zeros(simulation.num_weeks)
            self.avg_total_S = np.zeros(simulation.num_weeks)
            self.avg_total_I = np.zeros(simulation.num_weeks)
            self.avg_total_R = np.zeros(simulation.num_weeks)
            if simulation.VAX:
                self.avg_newly_V = np.zeros(simulation.num_weeks)
                self.avg_total_V = np.zeros(simulation.num_weeks)
            else:
                self.avg_newly_V, self.avg_total_V = [],[]

        def update_avg(self,avg, new_val):
            if self.empty == True:
                self.empty = False
            avg += np.array(new_val) / self.num_simulations
            

    def run(self, k=6, beta=0.3, rho=0.7):
        self.k = k
        self.beta = beta
        self.rho = rho

        self.result = Simulator.WrapperResult(self)

        for simulation in range(self.num_simulations):

            # choose the graph to use
            if self.graph_type == 'RANDOM':
                graph = random_graph(self.num_nodes, k)
            elif self.graph_type == 'K_REGULAR':
                graph = k_regular_graph(self.num_nodes, k)
          
            epidemic = EpidemicSimulation(graph, beta, rho)

            # Initialize epidemic with random initial infected nodes
            epidemic.set_initial_infected_nodes(self.num_initial_I_nodes)

            # Simulate the epidemic
            if self.VAX:
                epidemic.set_initial_vaccinated_nodes(self.num_initial_V_nodes)
                newly_I, total_S, total_I, total_R, newly_V, total_V = epidemic.SIRV(self.num_weeks, self.vaccination_plan)
            else:
                newly_I, total_S, total_I, total_R = epidemic.SIR(self.num_weeks)
            
            # Update averages
            self.result.update_avg(self.result.avg_newly_I,   newly_I)
            self.result.update_avg(self.result.avg_total_S,total_S)
            self.result.update_avg(self.result.avg_total_I,   total_I)
            self.result.update_avg(self.result.avg_total_R,  total_R)
            if self.VAX:
                self.result.update_avg(self.result.avg_newly_V, newly_V)
                self.result.update_avg(self.result.avg_total_V, total_V)        

    def H1N1(self, k0=10, beta0=0.3, rho0=0.6, num_simulations=10, I0 = [1, 3, 5, 9, 17, 32, 32, 17, 5, 2, 1, 0, 0, 0, 0]):
        if self.VAX == False:
            return

        self.num_simulations = num_simulations
        self.newly_infected_reference = I0
        optimal_k, optimal_beta, optimal_rho = k0, beta0, rho0
        k, beta, rho = k0, beta0, rho0
        k_tmp, beta_tmp, rho_tmp = k0, beta0, rho0
        I_tmp = [0 for i in range(16)]
        
        # initial value for RMSE
        self.run(k,beta,rho)
        RMSE_min = self.RMSE()
        rmse_min_tmp = RMSE_min
        self.I = self.result.avg_newly_I

        print(f'k0={k0}, beta0={beta0:.2f}, rho0={rho0:.2f}, RMSE0={RMSE_min}')

        # creation of the search space as a combination of all the possible directions (-delta, +0, +delta), excluding (+0, +0, +0)
        movements = [{'k':k,'beta':b,'rho':r} for k in [-1,0,1] for b in [-1,0,1] for r in [-1,0,1] if [k,b,r]!=[0,0,0]]

        num_steps = 20
        for step in range(num_steps):
            for index in range(len(movements)):

                k =    optimal_k + movements[index]['k'] * 1 # deltak = 1
                beta = optimal_beta + movements[index]['beta'] * 0.1 # deltabeta = 0.1
                rho =  optimal_rho + movements[index]['rho'] * 0.1 # deltarho = 0.1

                # check boundaries: k is non-negative, beta and rho are probabilities (0,1]
                if k > 100 or k < 1 or beta > 1 or beta < 0.1 or rho > 1 or rho < 0.1:
                    print(f'step {step} index {index}')
                    continue 

                self.run(k,beta,rho)
                RMSE = self.RMSE()

                # if there is an improvement update values k,b,r
                if RMSE < rmse_min_tmp:
                    rmse_min_tmp = RMSE
                    I_tmp = self.result.avg_newly_I
                    k_tmp, beta_tmp, rho_tmp = k, beta, rho
                    print(f'step {step:2} index {index}, k={k}, beta={beta:.2f}, rho={rho:.2f}, RMSE={RMSE:.3f}')
                else:
                    print(f'step {step:2} index {index}, k={k}, beta={beta:.2f}, rho={rho:.2f}')

            # if no improvement end algorithm
            if k_tmp == optimal_k and beta_tmp == optimal_beta and rho_tmp == optimal_rho:
                break
            else:
                RMSE_min = rmse_min_tmp
                I = I_tmp
                optimal_k, optimal_beta, optimal_rho = k_tmp, beta_tmp, rho_tmp


        self.RMSE_min, self.I = RMSE_min, I
        self.optimal_k, self.optimal_beta, self.optimal_rho = optimal_k, optimal_beta, optimal_rho
        return optimal_k, optimal_beta, optimal_rho 
    

    def H1N1_new(self, k0=10, beta0=0.3, rho0=0.6, num_simulations=10, I0 = [1, 3, 5, 9, 17, 32, 32, 17, 5, 2, 1, 0, 0, 0, 0]):
        if self.VAX == False:
            return
        
        self.num_simulations = num_simulations
        self.newly_infected_reference = I0
        optimal_k, optimal_beta, optimal_rho = k0, beta0, rho0
        k, beta, rho = k0, beta0, rho0
        k_tmp, beta_tmp, rho_tmp = k0, beta0, rho0
        I_tmp = [0 for i in range(16)]
        
        self.run(k,beta,rho)
        RMSE_min = self.RMSE()
        rmse_min_tmp = RMSE_min
        self.I = self.result.avg_newly_I

        print(f'k0={k0}, beta0={beta0:.2f}, rho0={rho0:.2f}, RMSE0={RMSE_min}')

        movements = [{'k':k,'beta':b,'rho':r} for k in [-1,0,1] for b in [-1,0,1] for r in [-1,0,1] if [k,b,r]!=[0,0,0]]

        # define the set of deltas to use
        delta = {
            'k' :   [1, 1, 1],
            'beta': [0.1, 0.05, 0.01],
            'rho' : [0.1, 0.05, 0.01]
        }

        num_steps = 10
        num_delta_types = len(delta['k'])
        delta_type = 0
        # exactly as befor but now when it stops, change deltas and become more precise
        while delta_type < num_delta_types:
            for step in range(num_steps):
                for index in range(len(movements)):
                    k =    optimal_k + movements[index]['k'] * delta['k'][delta_type]
                    beta = optimal_beta + movements[index]['beta'] * delta['beta'][delta_type]
                    rho =  optimal_rho + movements[index]['rho'] * delta['rho'][delta_type]

                    if k > 100 or k < 1 or beta > 1 or beta < 0.1 or rho > 1 or rho < 0.1:
                        print(f'delta_type {delta_type:2}, step {step:2}')
                        continue 

                    self.run(k,beta,rho)
                    RMSE = self.RMSE()

                    if RMSE < rmse_min_tmp:
                        rmse_min_tmp = RMSE
                        I_tmp = self.result.avg_newly_I
                        k_tmp, beta_tmp, rho_tmp = k, beta, rho
                        print(f'delta_type {delta_type:2}, step {step:2}, index {index:2}, k={k}, beta={beta:.2f}, rho={rho:.2f}, RMSE={RMSE:.3f}')
                    else:
                        print(f'delta_type {delta_type:2}, step {step:2}, index {index:2}, k={k}, beta={beta:.2f}, rho={rho:.2f}')


                if k_tmp == optimal_k and beta_tmp == optimal_beta and rho_tmp == optimal_rho:
                    break
                else:
                    RMSE_min= rmse_min_tmp
                    I = I_tmp
                    optimal_k, optimal_beta, optimal_rho = k_tmp, beta_tmp, rho_tmp
            delta_type += 1

        self.RMSE_min, self.I = RMSE_min, I
        self.optimal_k, self.optimal_beta, self.optimal_rho = optimal_k, optimal_beta, optimal_rho
        return optimal_k, optimal_beta, optimal_rho 

    def RMSE(self):
        n = len(self.result.avg_newly_I)
        I = self.result.avg_newly_I
        I0 = self.newly_infected_reference
        return np.sqrt( 1/n * sum([ (I[t]-I0[t])**2 for t in range(n) ]) )

    def plot_results(self):
        if self.result.empty:
            return
        
        print('---------------------------------------------------------------------------')
        print(f"k = {self.k}, " + r"$\beta$ = " + f"{self.beta:.2f}, " + r"$\rho$ = " + f"{self.rho:.2f}")
        print(f'avg newly infected = {self.result.avg_newly_I.astype(int)}')
        print(f'avg total susceptible = {self.result.avg_total_S.astype(int)}')
        print(f'avg total infected = {self.result.avg_total_I.astype(int)}')
        print(f'avg total recovered = {self.result.avg_total_R.astype(int)}')
        if self.VAX:
            print(f'avg total vaccinated = {self.result.avg_total_V.astype(int)}')
            print(f'avg newly vaccinated = {self.result.avg_newly_V.astype(int)}')
        print('---------------------------------------------------------------------------')

        # Plot results
        weeks = np.arange(1,self.num_weeks+1)

        # plt.figure(figsize=(10, 6))
        plt.plot(weeks, self.result.avg_newly_I, label='Average Newly Infected')
        plt.xlabel('Week')
        plt.xticks(weeks)
        plt.ylabel('Number of Individuals')
        plt.legend()
        plt.grid(True)
        plt.show()

        if self.VAX:
            plt.plot(weeks, self.result.avg_newly_V, label='Average Newly Vaccinated')
            plt.xlabel('Week')
            plt.xticks(weeks)
            plt.ylabel('Number of Individuals')
            plt.legend()
            plt.grid(True)
            plt.show()

        plt.plot(weeks, self.result.avg_total_S, label='Avg S')
        plt.plot(weeks, self.result.avg_total_I, label='Avg I')
        plt.plot(weeks, self.result.avg_total_R, label='Avg R')
        if self.VAX:
            plt.plot(weeks, self.result.avg_total_V, label='Avg V')
        plt.xlabel('Week')
        plt.xticks(weeks)
        plt.ylabel('Number of Individuals')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_results_H1N1(self):

        print('---------------------------------------------------------------------------')
        print(f"k={self.optimal_k}, beta={self.optimal_beta:.2f}, rho={self.optimal_rho:.2f}, RMSE={self.RMSE_min:.2f}")
        print(f'I  = {self.I.astype(int)}')
        print(f'I0 = {np.array(self.newly_infected_reference)}')
        print('---------------------------------------------------------------------------')
        weeks = np.arange(1,len(self.newly_infected_reference)+1)
        
        plt.figure(figsize=(10, 6))
        plt.title(f"k = {self.optimal_k}, " + r"$\beta$ = " + f"{self.optimal_beta:.2f}, " + r"$\rho$ = " + f"{self.optimal_rho:.2f}, RMSE={self.RMSE_min:.2f}")
        plt.plot(weeks, self.I, label='I')
        plt.plot(weeks, self.newly_infected_reference, label='I0')
        plt.xlabel('weeks')
        plt.xticks(weeks)
        plt.ylabel('Infected people')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
