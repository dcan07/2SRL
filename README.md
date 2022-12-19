# 2SRL
Two-stage reinforcement learning framework for solving two-stage stochastic programs

- baseline_solvers.py are the heuristic and exact solution functions that are utilized. Considered approaches are Gurobi, SDDiP, Adaptive Fixing and Primal Effective Capacity Heuristic
- dataset.py contains the dataset object that contains the problem parameters and characteristics 
- generate_config.py is modified to generate model configs, change agent and instance parameters
- pointer_network.py is the pointer network model that actor and critic networks are based on
- train_agent2.py is the agent 2 training code
- train_agent1.py is the agent 1 training code and it requires a trained agent 2
- solve_test_set.py solves the test instances to be used during testing
- test_agent.py generates predictions for test instances and compares with other approaches
