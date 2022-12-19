# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:39:04 2022

@author: dy234
"""

import sys


from msppy.msp import MSIP
from msppy.solver import Extensive,SDDiP
from msppy.evaluation import EvaluationTrue
import gurobipy
from gurobipy import GRB
import numpy as np
from docplex.mp.model import Model
import time
from torch.utils.data import DataLoader
import itertools
from sklearn.metrics import accuracy_score
import platform



if platform.system()=='Windows':
    TwoStageRLpath = 'C:/Users/dy234/Documents/paper3/TwoStageRL'
    msppypath = 'C:/Users/dy234/Documents/paper3/msppy-master'
    n_cores= 8

else: 
    TwoStageRLpath = '/home/d/dy234/paper3/TwoStageRL'
    msppypath = '/home/d/dy234/paper3/msppy-master'
    n_cores= 16

    
if TwoStageRLpath not in sys.path:
    sys.path.append(TwoStageRLpath)
if msppypath not in sys.path:
    sys.path.append(msppypath)




def solve2stageknapsackcplex(c, A, b, q, T, W, h, MIPGap=0, use_set_time = False, set_time = -1, **kwargs):
    '''
    Solve the 2 stage knapsack problem given in formulation:
        max cx + E[Q(x,$)]
        st Ax<=b
        where Q(x,$) is
        min qy
        st Tx+Wy <= h
    Only q is a random vector
    
    c = size=(n_firststage_variables)
    A = size=(n_firststage_constraints,n_firststage_variables)
    b = size=(n_firststage_constraints)
    q = size=(n_secondstage_variable, *)
    T = size=(n_secondstage_constraints,n_firststage_variables,*) 
    W = size=(n_secondstage_constraints,n_secondstage_variables,*)
    h = size=(n_secondstage_constraints,*)
    * is 1 if known n_scenarios otherwise

    '''
    n_firststage_variables = len(c)
    n_firststage_constraints = len(A)
    n_secondstage_variables = len(W[0])
    n_secondstage_constraints = len(W)
    n_scenarios = max(len(q[0]),len(T[0,0]),len(W[0,0]),len(h[0])) 
    
    # Decision variables
    mdl = Model("mknapsack")
    #mdl.context.cplex_parameters.threads = 1
    x = mdl.binary_var_list(n_firststage_variables)
    y = mdl.binary_var_matrix(n_scenarios,n_secondstage_variables)

        
    # Objective function
    mdl.first_stage = mdl.sum( (c[i]*x[i]) for i in range(n_firststage_variables))
    # Known q
    if q.shape[1]==1:
        mdl.second_stage  = mdl.sum( (q[j,0]*y[s,j]) for j in range(n_secondstage_variables) for s in range(n_scenarios))
    # Random q
    else:
        mdl.second_stage  = mdl.sum( (q[j,s]*y[s,j]) for j in range(n_secondstage_variables) for s in range(n_scenarios))
    mdl.maximize( mdl.first_stage  + (mdl.second_stage/n_scenarios))
    
    # Constraints
    for m in range(n_firststage_constraints):
        mdl.add_constraint(mdl.sum(x[i]*A[m,i] for i in range(n_firststage_variables)) <= b[m])
    
        #'''
    # There is some random elements in constraints
    # The indexes are used to determine if the given matrix is known or random
    # If known, shape would be 1 and we should only consider the scenario at index 0,
    # If it truly does have more than 1 scenario, consider all
    h_index = [0 if h.shape[-1]==1 else ct for ct in range(n_scenarios)]
    T_index = [0 if T.shape[-1]==1 else ct for ct in range(n_scenarios)]
    W_index = [0 if W.shape[-1]==1 else ct for ct in range(n_scenarios)]
    for s in range(n_scenarios):
        for n in range(n_secondstage_constraints):
            mdl.add_constraint(mdl.sum(x[i]*T[n,i,T_index[s]] for i in range(n_firststage_variables))+
                               mdl.sum(y[s,j]*W[n,j,W_index[s]] for j in range(n_secondstage_variables))<=h[n,h_index[s]])
    #'''
    mdl.context.cplex_parameters.threads = n_cores
    if use_set_time:
        mdl.parameters.timelimit = set_time
    if MIPGap>0:
        mdl.parameters.mip.tolerances.mipgap.set(MIPGap)
    solution = mdl.solve(clean_before_solve=True, log_output = False, **kwargs)

    #print('cplex :', mdl.objective_value)
    #cplex_firststage_var=np.array([x[i].solution_value for i in range(n_firststage_variables)] )
    #cplex_secondstage_var=np.array([[y[s, j].solution_value for j in range(n_secondstage_variables)] for s in range(n_scenarios)])
    cplex_firststage_var=[x[i].solution_value for i in range(n_firststage_variables)] 
    cplex_secondstage_var=[[y[s, j].solution_value for j in range(n_secondstage_variables)] for s in range(n_scenarios)]
    if solution is not None:
        return mdl.objective_value, mdl.solve_details.time, cplex_firststage_var, cplex_secondstage_var, mdl.solve_details.mip_relative_gap, mdl.solve_details.best_bound
    else:
        return -1, -1, -1, -1, -1, -1

def solve2stageknapsackgurobi(c, A, b, q, T, W, h, MIPGap=0, use_set_time = False, set_time = -1, **kwargs):
    '''
    Solve the 2 stage knapsack problem given in formulation:
        max cx + E[Q(x,$)]
        st Ax<=b
        where Q(x,$) is
        min qy
        st Tx+Wy <= h
    Only q is a random vector
    
    c = size=(n_firststage_variables)
    A = size=(n_firststage_constraints,n_firststage_variables)
    b = size=(n_firststage_constraints)
    q = size=(n_secondstage_variable, *)
    T = size=(n_secondstage_constraints,n_firststage_variables,*) 
    W = size=(n_secondstage_constraints,n_secondstage_variables,*)
    h = size=(n_secondstage_constraints,*)
    * is 1 if known n_scenarios otherwise

    '''
    n_firststage_variables = len(c)
    n_firststage_constraints = len(A)
    n_secondstage_variables = len(W[0])
    n_secondstage_constraints = len(W)
    n_scenarios = max(len(q[0]),len(T[0,0]),len(W[0,0]),len(h[0])) 
    
    # Decision variables
    mdl = gurobipy.Model("mknapsack")
    x = mdl.addVars(n_firststage_variables, vtype=GRB.BINARY)
    y = mdl.addVars(n_scenarios,n_secondstage_variables, vtype=GRB.BINARY)

        
    # Objective function
    first_stage = gurobipy.quicksum((c[i]*x[i]) for i in range(n_firststage_variables))
    # Known q
    if q.shape[1]==1:
        second_stage  = gurobipy.quicksum((q[j,0]*y[s,j]) for j in range(n_secondstage_variables) for s in range(n_scenarios))
    # Random q
    else:
        second_stage  = gurobipy.quicksum((q[j,s]*y[s,j]) for j in range(n_secondstage_variables) for s in range(n_scenarios))
    mdl.setObjective( first_stage  + (second_stage/n_scenarios), GRB.MAXIMIZE)
   
    # Constraints
    for m in range(n_firststage_constraints):
        mdl.addConstr(gurobipy.quicksum(x[i]*A[m,i] for i in range(n_firststage_variables)) <= b[m])
    
        #'''
    # There is some random elements in constraints
    # The indexes are used to determine if the given matrix is known or random
    # If known, shape would be 1 and we should only consider the scenario at index 0,
    # If it truly does have more than 1 scenario, consider all
    h_index = [0 if h.shape[-1]==1 else ct for ct in range(n_scenarios)]
    T_index = [0 if T.shape[-1]==1 else ct for ct in range(n_scenarios)]
    W_index = [0 if W.shape[-1]==1 else ct for ct in range(n_scenarios)]
    for s in range(n_scenarios):
        for n in range(n_secondstage_constraints):
            mdl.addConstr(gurobipy.quicksum(x[i]*T[n,i,T_index[s]] for i in range(n_firststage_variables))+
                               gurobipy.quicksum(y[s,j]*W[n,j,W_index[s]] for j in range(n_secondstage_variables))<=h[n,h_index[s]])
    mdl.setParam('OutputFlag', 0)
    if use_set_time:
        mdl.Params.TimeLimit = set_time
    if MIPGap>0:
        mdl.Params.MIPGap = MIPGap
    mdl.params.threads = n_cores 
    mdl.optimize(**kwargs)

    firststage_var=[x[i].x for i in range(n_firststage_variables)] 
    secondstage_var=[[y[s, j].x for j in range(n_secondstage_variables)] for s in range(n_scenarios)]
    if mdl.Status != 3:
        return mdl.ObjVal, mdl.Runtime, firststage_var, secondstage_var, mdl.MIPGap, mdl.ObjBound
    else:
        return -1, -1, -1, -1, -1, -1

def solve2stageknapsackgurobiLP(c, A, b, q, T, W, h, x_zero, x_one, y_zero, y_one):
    '''
    Solve the 2 stage knapsack problem given in formulation:
        max cx + E[Q(x,$)]
        st Ax<=b
        where Q(x,$) is
        min qy
        st Tx+Wy <= h
    Only q is a random vector
    
    c = size=(n_firststage_variables)
    A = size=(n_firststage_constraints,n_firststage_variables)
    b = size=(n_firststage_constraints)
    q = size=(n_secondstage_variable, *)
    T = size=(n_secondstage_constraints,n_firststage_variables,*) 
    W = size=(n_secondstage_constraints,n_secondstage_variables,*)
    h = size=(n_secondstage_constraints,*)
    * is 1 if known n_scenarios otherwise

    '''
    n_firststage_variables = len(c)
    n_firststage_constraints = len(A)
    n_secondstage_variables = len(W[0])
    n_secondstage_constraints = len(W)
    n_scenarios = max(len(q[0]),len(T[0,0]),len(W[0,0]),len(h[0])) 
    
    # Decision variables
    mdl = gurobipy.Model("mknapsack")
    #mdl.context.cplex_parameters.threads = 1
    x = mdl.addVars(n_firststage_variables, vtype=GRB.CONTINUOUS, ub =1)
    y = mdl.addVars(n_scenarios,n_secondstage_variables, vtype=GRB.CONTINUOUS, ub =1)

        
    # Objective function
    first_stage = gurobipy.quicksum((c[i]*x[i]) for i in range(n_firststage_variables))
    # Known q
    if q.shape[1]==1:
        second_stage  = gurobipy.quicksum((q[j,0]*y[s,j]) for j in range(n_secondstage_variables) for s in range(n_scenarios))
    # Random q
    else:
        second_stage  = gurobipy.quicksum((q[j,s]*y[s,j]) for j in range(n_secondstage_variables) for s in range(n_scenarios))
    mdl.setObjective( first_stage  + (second_stage/n_scenarios), GRB.MAXIMIZE)
   
    # Constraints
    for m in range(n_firststage_constraints):
        mdl.addConstr(gurobipy.quicksum(x[i]*A[m,i] for i in range(n_firststage_variables)) <= b[m])
    
    
    
    # There is some random elements in constraints
    # The indexes are used to determine if the given matrix is known or random
    # If known, shape would be 1 and we should only consider the scenario at index 0,
    # If it truly does have more than 1 scenario, consider all
    h_index = [0 if h.shape[-1]==1 else ct for ct in range(n_scenarios)]
    T_index = [0 if T.shape[-1]==1 else ct for ct in range(n_scenarios)]
    W_index = [0 if W.shape[-1]==1 else ct for ct in range(n_scenarios)]
    for s in range(n_scenarios):
        for n in range(n_secondstage_constraints):
            mdl.addConstr(gurobipy.quicksum(x[i]*T[n,i,T_index[s]] for i in range(n_firststage_variables))+
                               gurobipy.quicksum(y[s,j]*W[n,j,W_index[s]] for j in range(n_secondstage_variables))<=h[n,h_index[s]])
    
    # Add zero variables as constraints
    for i in range(n_firststage_variables):
        if x_zero[i]==1:
            mdl.addConstr(x[i] == 0)
    for s in range(n_scenarios):
        for j in range(n_secondstage_variables):
            if y_zero[s,j]==1:
                mdl.addConstr(y[s,j] == 0)
    
    # Add one variables as constraints
    for i in range(n_firststage_variables):
        if x_one[i]==1:
            mdl.addConstr(x[i] == 1)
    for s in range(n_scenarios):
        for j in range(n_secondstage_variables):
            if y_one[s,j]==1:
                mdl.addConstr(y[s,j] == 1)
            
    mdl.setParam('OutputFlag', 0)
    mdl.params.threads = n_cores       
    # Add variables 
    mdl.optimize()

    firststage_var=[x[i].x for i in range(n_firststage_variables)] 
    secondstage_var=[[y[s, j].x for j in range(n_secondstage_variables)] for s in range(n_scenarios)]
    if mdl.Status != 3:
        return mdl.ObjVal, mdl.Runtime, np.array(firststage_var), np.array(secondstage_var)
    else:
        return -1, -1, -1, -1 

def solve2stageknapsackheuristic(c, A, b, q, T, W, h, gamma = 0.25):
    # AF heuristic. This arrays are intialized as zeros. If their indexes are 1, it menas that they are in the set
    n_firststage_variables = len(c)
    n_secondstage_variables = len(W[0])
    n_scenarios = max(len(q[0]),len(T[0,0]),len(W[0,0]),len(h[0])) 
    
    x_zero = np.zeros(shape =(n_firststage_variables) )
    x_one = np.zeros(shape =(n_firststage_variables) )
    y_zero = np.zeros(shape = (n_scenarios,n_secondstage_variables))
    y_one = np.zeros(shape = (n_scenarios,n_secondstage_variables))
    
    # Step 2
    # Initially sove the lp problem 
    heuristic_start_time = time.time()
    _, _, heur_dec_var1, heur_dec_var2, = solve2stageknapsackgurobiLP(c, A, b, q, T, W, h, x_zero, x_one, y_zero, y_one)
    
    

    # Step 3
    # Fix variables form x_zero, x_one, y_zero, y_one  
    for i in range(n_firststage_variables):
        if heur_dec_var1[i] <= gamma:
            x_zero[i] = 1
        elif heur_dec_var1[i] == 1:
            x_one[i] = 1
    for s in range(n_scenarios):
        for j in range(n_secondstage_variables):
            if heur_dec_var2[s][j] <= gamma:
                y_zero[s,j] = 1
            elif heur_dec_var2[s][j] == 1:
                y_one[s,j] = 1

                

    
    # Step 4
    # While there are fractional solutions
    while not ( np.equal(np.mod(heur_dec_var1, 1), 0).all() and np.equal(np.mod(heur_dec_var2, 1), 0).all()):
        
        # Solve LP
        heur_obj, _, heur_dec_var1, heur_dec_var2, = solve2stageknapsackgurobiLP(c, A, b, q, T, W, h, x_zero, x_one, y_zero, y_one)

    
        # Fix variables form x_zero and x_one     
        for i in range(n_firststage_variables):
            if heur_dec_var1[i] == 0:
                x_zero[i] = 1
            elif heur_dec_var1[i] == 1:
                x_one[i] = 1
        for s in range(n_scenarios):
            for j in range(n_secondstage_variables):
                if heur_dec_var2[s,j] == 0:
                    y_zero[s,j] = 1
                elif heur_dec_var2[s,j] == 1:
                    y_one[s,j] = 1
                    
        
        # Get the non zero min and make it zero
        #first stage min index
        i0 = np.where( heur_dec_var1==np.min(heur_dec_var1[np.nonzero(heur_dec_var1)]))
        i0 = i0[0][0]

        #second stage min index
        i1,i2 = np.where( heur_dec_var2==np.min(heur_dec_var2[np.nonzero(heur_dec_var2)]))
        i1, i2 = i1[0], i2[0]
        
        # Get the min of both stages
        if heur_dec_var1[i0] < heur_dec_var2[i1,i2] :
            x_zero[i0] = 1
        else:
            y_zero[i1, i2] = 1
        
    # Now finalize
    heuristic_end_time = time.time()

    heur_time = heuristic_end_time-heuristic_start_time
    
    
    # Objective function
    first_stage = np.sum([c[i]*heur_dec_var1[i] for i in range(n_firststage_variables)])
    # Known q
    if q.shape[1]==1:
        second_stage  = np.sum([[q[j,0]*heur_dec_var2[s,j] for j in range(n_secondstage_variables)] for s in range(n_scenarios)])
    # Random q
    else:
        second_stage  = np.sum([[q[j,s]*heur_dec_var2[s,j] for j in range(n_secondstage_variables)] for s in range(n_scenarios)])
    heur_obj = first_stage  + (second_stage/n_scenarios)




    return heur_obj, heur_time, heur_dec_var1, heur_dec_var2


def smallestN_indices(a, N):
    idx = a.ravel().argsort()[:N]
    return np.stack(np.unravel_index(idx, a.shape)).T


def solve2stageknapsackheuristic2(c, A, b, q, T, W, h, gamma = 0.2):
    # AF heuristic. This arrays are intialized as zeros. If their indexes are 1, it menas that they are in the set
    n_firststage_variables = len(c)
    n_secondstage_variables = len(W[0])
    n_scenarios = max(len(q[0]),len(T[0,0]),len(W[0,0]),len(h[0])) 
    
    x_zero = np.zeros(shape =(n_firststage_variables) )
    x_one = np.zeros(shape =(n_firststage_variables) )
    y_zero = np.zeros(shape = (n_scenarios,n_secondstage_variables))
    y_one = np.zeros(shape = (n_scenarios,n_secondstage_variables))
    x_flag = np.zeros(shape =(n_firststage_variables) )
    y_flag = np.zeros(shape = (n_scenarios,n_secondstage_variables))
    
    # Step 2
    # Initially sove the lp problem 
    heuristic_start_time = time.time()
    
    ######
    n_firststage_constraints = len(A)
    n_secondstage_constraints = len(W)
    
    # Decision variables
    mdl = gurobipy.Model("mknapsack")
    #mdl.context.cplex_parameters.threads = 1
    x = mdl.addVars(n_firststage_variables, vtype=GRB.CONTINUOUS, ub =1)
    y = mdl.addVars(n_scenarios,n_secondstage_variables, vtype=GRB.CONTINUOUS, ub =1)

        
    # Objective function
    first_stage = gurobipy.quicksum((c[i]*x[i]) for i in range(n_firststage_variables))
    # Known q
    if q.shape[1]==1:
        second_stage  = gurobipy.quicksum((q[j,0]*y[s,j]) for j in range(n_secondstage_variables) for s in range(n_scenarios))
    # Random q
    else:
        second_stage  = gurobipy.quicksum((q[j,s]*y[s,j]) for j in range(n_secondstage_variables) for s in range(n_scenarios))
    mdl.setObjective( first_stage  + (second_stage/n_scenarios), GRB.MAXIMIZE)
   
    # Constraints
    for m in range(n_firststage_constraints):
        mdl.addConstr(gurobipy.quicksum(x[i]*A[m,i] for i in range(n_firststage_variables)) <= b[m])
    
    
    
    # There is some random elements in constraints
    # The indexes are used to determine if the given matrix is known or random
    # If known, shape would be 1 and we should only consider the scenario at index 0,
    # If it truly does have more than 1 scenario, consider all
    h_index = [0 if h.shape[-1]==1 else ct for ct in range(n_scenarios)]
    T_index = [0 if T.shape[-1]==1 else ct for ct in range(n_scenarios)]
    W_index = [0 if W.shape[-1]==1 else ct for ct in range(n_scenarios)]
    for s in range(n_scenarios):
        for n in range(n_secondstage_constraints):
            mdl.addConstr(gurobipy.quicksum(x[i]*T[n,i,T_index[s]] for i in range(n_firststage_variables))+
                               gurobipy.quicksum(y[s,j]*W[n,j,W_index[s]] for j in range(n_secondstage_variables))<=h[n,h_index[s]])
    
    
            
    mdl.setParam('OutputFlag', 0)
    mdl.params.threads = n_cores       
    # Add variables 
    mdl.optimize()

    heur_dec_var1=np.array([x[i].x for i in range(n_firststage_variables)] )
    heur_dec_var2=np.array([[y[s, j].x for j in range(n_secondstage_variables)] for s in range(n_scenarios)])
    heur_dec_var1_previous=np.copy(heur_dec_var1)
    heur_dec_var2_previous=np.copy(heur_dec_var2)
    
    # Step 3
    # Fix variables form x_zero, x_one, y_zero, y_one  
    for i in range(n_firststage_variables):
        if heur_dec_var1[i] <= gamma:
            x_zero[i] = 1
        elif heur_dec_var1[i] == 1:
            x_one[i] = 1
    for s in range(n_scenarios):
        for j in range(n_secondstage_variables):
            if heur_dec_var2[s,j] <= gamma:
                y_zero[s,j] = 1
            elif heur_dec_var2[s,j] == 1:
                y_one[s,j] = 1

                

    temp_ct = 0 
    # Step 4
    # While there are fractional solutions
    #while not ( np.equal(np.mod(heur_dec_var1, 1), 0).all() and np.equal(np.mod(heur_dec_var2, 1), 0).all()):
    while (np.mean(x_flag)<1 or np.mean(y_flag)<1):
        #print('iter: ' ,temp_ct)
        #print(np.equal(np.mod(heur_dec_var1, 1), 0).all())
        #print(np.equal(np.mod(heur_dec_var2, 1), 0).all())
        #print(heur_dec_var1)
        #print(heur_dec_var2)
        #if np.mean(y_flag)>=1:
        #if temp_ct==1200:
        #    for s in range(n_scenarios):
        #        for j in range(n_secondstage_variables):
        #            if heur_dec_var2[s,j] != 0 and heur_dec_var2[s,j] != 1:
        #                print('here ', heur_dec_var2[s,j])
        #                print(s,j,y_flag[s,j])
        #print(np.mean(x_flag))
        #print(np.mean(y_flag))
        temp_ct+=1
        # Solve LP
        # Add zero variables as constraints
        for i in range(n_firststage_variables):
            if x_zero[i]==1 and x_flag[i]==0:
                mdl.addConstr(x[i] == 0)
                x_flag[i]=1
        for s in range(n_scenarios):
            for j in range(n_secondstage_variables):
                if y_zero[s,j]==1 and y_flag[s,j]==0:
                    mdl.addConstr(y[s,j] == 0)
                    y_flag[s,j]=1
        
        # Add one variables as constraints
        for i in range(n_firststage_variables):
            if x_one[i]==1 and x_flag[i]==0:
                mdl.addConstr(x[i] == 1)
                x_flag[i]=1
        for s in range(n_scenarios):
            for j in range(n_secondstage_variables):
                if y_one[s,j]==1 and y_flag[s,j]==0:
                    mdl.addConstr(y[s,j] == 1)
                    y_flag[s,j]=1
        mdl.optimize()
        
        heur_dec_var1=np.array([x[i].x for i in range(n_firststage_variables)] )
        heur_dec_var2=np.array([[y[s, j].x for j in range(n_secondstage_variables)] for s in range(n_scenarios)])

    
        # Fix variables form x_zero and x_one     
        for i in range(n_firststage_variables):
            if heur_dec_var1[i] == 0:
                x_zero[i] = 1
            elif heur_dec_var1[i] == 1:
                x_one[i] = 1
        for s in range(n_scenarios):
            for j in range(n_secondstage_variables):
                if heur_dec_var2[s,j] == 0:
                    y_zero[s,j] = 1
                elif heur_dec_var2[s,j] == 1:
                    y_one[s,j] = 1
                    
        
        # Get the non zero min and make it zero
        #first stage min index
        
        heur_dec_var1_copy=np.copy(heur_dec_var1)
        heur_dec_var2_copy=np.copy(heur_dec_var2)
        
        heur_dec_var1_copy[heur_dec_var1<=0]=1
        heur_dec_var2_copy[heur_dec_var2<=0]=1
        ind1 = smallestN_indices(heur_dec_var1_copy, max(int(x_flag.size/1000),1))
        ind2 = smallestN_indices(heur_dec_var2_copy, max(int(y_flag.size/1000),1))
        #ind = smallestN_indices(heur_dec_var_copy, 1)
        # If not fixing a single variable
        #print(len(ind2))
        if len(ind2)>1:
            for ix in range(len(ind1)):
                i0 = ind1[ix,0]
                x_zero[i0] = 1
                #print(heur_dec_var1_copy[i0])
            for ix in range(len(ind2)):
                i1, i2= ind2[ix,0], ind2[ix,1]
                y_zero[i1,i2] = 1
                #if temp_ct==1201:
                #    print(i1,i2,heur_dec_var2_copy[i1,i2])
        else:
            # Get the min of both stages
            i0 = ind1[0,0]
            i1, i2= ind2[0,0], ind2[0,1]
            if heur_dec_var1_copy[i0] < heur_dec_var2_copy[i1,i2] :
                x_zero[i0] = 1
                #print(i1,i2,heur_dec_var1_copy[i1,i2])
            else:
                y_zero[i1, i2] = 1
                #print(i1,i2,heur_dec_var2_copy[i1,i2])
        
        # Stop iteration if not changed
        if np.all(heur_dec_var1_previous==heur_dec_var1) and np.all(heur_dec_var2_previous==heur_dec_var2):
            for s in range(n_scenarios):
                for j in range(n_secondstage_variables):
                    if heur_dec_var2[s,j] != 0 and heur_dec_var2[s,j] != 1:
                        heur_dec_var2[s,j]=0
            for i in range(n_firststage_variables):
                if heur_dec_var1[i] != 0 and heur_dec_var1[i] != 1:
                    heur_dec_var1[i]=0
            break
        # Otherwise continue
        else:
            heur_dec_var1_previous=np.copy(heur_dec_var1)
            heur_dec_var2_previous=np.copy(heur_dec_var2)
        
        
        

        
    # Now finalize
    heuristic_end_time = time.time()
    
    # Objective function
    first_stage = np.sum([c[i]*heur_dec_var1[i] for i in range(n_firststage_variables)])
    # Known q
    if q.shape[1]==1:
        second_stage  = np.sum([[q[j,0]*heur_dec_var2[s,j] for j in range(n_secondstage_variables)] for s in range(n_scenarios)])
    # Random q
    else:
        second_stage  = np.sum([[q[j,s]*heur_dec_var2[s,j] for j in range(n_secondstage_variables)] for s in range(n_scenarios)])
    heur_obj = first_stage  + (second_stage/n_scenarios)


    heur_time = heuristic_end_time-heuristic_start_time

    return heur_obj, heur_time, heur_dec_var1, heur_dec_var2

def solve2stageknapsackgurobiregularized(c, A, b, q, T, W, h, w_kminus1 = None, xbar_kminus1 = None, rho = 0, fixed_variables=None, MIPGap=0, use_set_time = False, set_time = -1, **kwargs):
    '''
    Solve the 2 stage knapsack problem given in formulation:
        max cx + E[Q(x,$)]
        st Ax<=b
        where Q(x,$) is
        min qy
        st Tx+Wy <= h
    Only q is a random vector
    
    c = size=(n_firststage_variables)
    A = size=(n_firststage_constraints,n_firststage_variables)
    b = size=(n_firststage_constraints)
    q = size=(n_secondstage_variable, *)
    T = size=(n_secondstage_constraints,n_firststage_variables,*) 
    W = size=(n_secondstage_constraints,n_secondstage_variables,*)
    h = size=(n_secondstage_constraints,*)
    * is 1 if known n_scenarios otherwise

    '''
    n_firststage_variables = len(c)
    n_firststage_constraints = len(A)
    n_secondstage_variables = len(W[0])
    n_secondstage_constraints = len(W)
    n_scenarios = max(len(q[0]),len(T[0,0]),len(W[0,0]),len(h[0])) 
    
    # Decision variables
    mdl = gurobipy.Model("mknapsack")
    x = mdl.addVars(n_firststage_variables, vtype=GRB.BINARY)
    y = mdl.addVars(n_scenarios,n_secondstage_variables, vtype=GRB.BINARY)

        
    # Objective function
    first_stage = gurobipy.quicksum((c[i]*x[i]) for i in range(n_firststage_variables))
    # Known q
    if q.shape[1]==1:
        second_stage  = gurobipy.quicksum((q[j,0]*y[s,j]) for j in range(n_secondstage_variables) for s in range(n_scenarios))
    # Random q
    else:
        second_stage  = gurobipy.quicksum((q[j,s]*y[s,j]) for j in range(n_secondstage_variables) for s in range(n_scenarios))
    
    if w_kminus1 is not None:
        reg1 = gurobipy.quicksum((w_kminus1[i]*x[i]) for i in range(n_firststage_variables))
        reg2 = gurobipy.quicksum(rho[i] * (x[i]-xbar_kminus1[i]) * (x[i]-xbar_kminus1[i]) for i in range(n_firststage_variables))
        mdl.setObjective( first_stage  + (second_stage/n_scenarios) - reg1 - (reg2/2), GRB.MAXIMIZE)
    
    else:
        mdl.setObjective( first_stage  + (second_stage/n_scenarios), GRB.MAXIMIZE)
   
    # Constraints
    for m in range(n_firststage_constraints):
        mdl.addConstr(gurobipy.quicksum(x[i]*A[m,i] for i in range(n_firststage_variables)) <= b[m])
    
        #'''
    # There is some random elements in constraints
    # The indexes are used to determine if the given matrix is known or random
    # If known, shape would be 1 and we should only consider the scenario at index 0,
    # If it truly does have more than 1 scenario, consider all
    h_index = [0 if h.shape[-1]==1 else ct for ct in range(n_scenarios)]
    T_index = [0 if T.shape[-1]==1 else ct for ct in range(n_scenarios)]
    W_index = [0 if W.shape[-1]==1 else ct for ct in range(n_scenarios)]
    for s in range(n_scenarios):
        for n in range(n_secondstage_constraints):
            mdl.addConstr(gurobipy.quicksum(x[i]*T[n,i,T_index[s]] for i in range(n_firststage_variables))+
                               gurobipy.quicksum(y[s,j]*W[n,j,W_index[s]] for j in range(n_secondstage_variables))<=h[n,h_index[s]])
    
    # Fixed variables
    for i in range(n_firststage_variables):
        if fixed_variables is not None:
            if fixed_variables[i] != -1:
                mdl.addConstr(x[i]==fixed_variables[i])

    mdl.setParam('OutputFlag', 0)
    if use_set_time:
        mdl.Params.TimeLimit = set_time
    if MIPGap>0:
        mdl.Params.MIPGap = MIPGap
    mdl.params.threads = n_cores 
    mdl.optimize(**kwargs)

    firststage_var=[x[i].x for i in range(n_firststage_variables)] 
    secondstage_var=[[y[s, j].x for j in range(n_secondstage_variables)] for s in range(n_scenarios)]
    if mdl.Status != 3:
        return mdl.ObjVal, mdl.Runtime, firststage_var, secondstage_var, mdl.MIPGap, mdl.ObjBound
    else:
        return -1, -1, -1, -1, -1, -1



def solve2stageknapsackprogressivehedging(c, A, b, q, T, W, h, epsilon = 0.01, mu=2, timelimit=3600, iterlimit = 10000):
    # PH heuristic. 
    n_firststage_variables = len(c)
    n_secondstage_variables = len(W[0])
    n_scenarios = max(len(q[0]),len(T[0,0]),len(W[0,0]),len(h[0])) 
    
    
    # Time it 
    ph_start_time = time.time()
    
    # STEP 1
    k=0
    
    # STEP 2
    x_k = np.empty(shape =(n_scenarios,n_firststage_variables) )
    y_k = np.empty(shape =(n_scenarios,n_secondstage_variables) )
    for s in range(n_scenarios):
        _, _, x_k[s], var2, _, _ = solve2stageknapsackgurobiregularized(c, A, b, 
                                                                         np.expand_dims(q[:,0 if q.shape[-1]==1 else s], axis =-1), 
                                                                         np.expand_dims(T[:,:,0 if T.shape[-1]==1 else s], axis =-1), 
                                                                         np.expand_dims(W[:,:,0 if W.shape[-1]==1 else s], axis =-1), 
                                                                         np.expand_dims(h[:,0 if h.shape[-1]==1 else s], axis =-1))
        # Set second stage variable
        y_k[s] = var2[0]
    
    # STEP 3
    xbar_k = np.mean(x_k, axis=0)
    
    # Calculate rho
    rho = c / (np.max(x_k) - np.min(x_k) + 1)
    
    # STEP 4
    w_k = np.empty(shape =(n_scenarios,n_firststage_variables) )
    for s in range(n_scenarios):
        w_k[s] = rho * (x_k[s] - xbar_k)
        
    
    
    # Start loop
    g_k = 5000000
    # Variables are fixed if the index is not -1 to that value
    fixed_variables = np.full_like(xbar_k, -1.)
    # previous x_k is hold in order to check fixing
    #x_previous = np.full(shape = (mu*n_scenarios,n_scenarios,n_firststage_variables),fill_value = -1)    
    x_previous = np.expand_dims(x_k, axis = 0)
    
    
    
    
    
    while g_k>epsilon and (time.time()-ph_start_time)<timelimit and k < iterlimit:
        
        # STEP 5
        k += 1
        xbar_kminus1 = xbar_k.copy() 
        w_kminus1 = w_k.copy()
        
        # STEP 6
        x_k = np.empty(shape =(n_scenarios,n_firststage_variables) )
        y_k = np.empty(shape =(n_scenarios,n_secondstage_variables))
        for s in range(n_scenarios):
            _, _, x_k[s], var2, _, _ = solve2stageknapsackgurobiregularized(c, A, b, 
                                                                             np.expand_dims(q[:,0 if q.shape[-1]==1 else s], axis =-1), 
                                                                             np.expand_dims(T[:,:,0 if T.shape[-1]==1 else s], axis =-1), 
                                                                             np.expand_dims(W[:,:,0 if W.shape[-1]==1 else s], axis =-1), 
                                                                             np.expand_dims(h[:,0 if h.shape[-1]==1 else s], axis =-1),
                                                                             w_kminus1[s],
                                                                             xbar_kminus1,
                                                                             rho,
                                                                             fixed_variables)
            # Set second stage variable
            y_k[s] = var2[0]
        
        # STEP 7
        xbar_k = np.mean(x_k, axis=0)
        
        # STEP 8
        w_k = np.empty(shape =(n_scenarios,n_firststage_variables) )
        for s in range(n_scenarios):
            w_k[s] = w_kminus1[s] + (rho * (x_k[s] - xbar_k))
            
        # STEP 9
        g_k = 0
        for s in range(n_scenarios):
            g_k += np.linalg.norm(x_k[s] - xbar_k)
            
        # Set previous
        x_previous = np.concatenate((x_previous,np.expand_dims(x_k,axis=0)), axis = 0 )
        # make sure you get last mu*n_scenarios elements
        x_previous = x_previous[ -mu*n_scenarios:]
        # if we wait for enough iterations
        if k>mu*n_scenarios:
            # Check if values are stabilized
            stabilized = np.mean(x_previous, axis = (0,1)) == x_k[0]
            for ix,stable in enumerate(stabilized):
                # If stabilized, fix the variable
                if stable:
                    fixed_variables[ix] = x_k[0,ix]

    
   
        
    # Now finalize
    ph_end_time = time.time()

    ph_time = ph_end_time-ph_start_time
    
    # Variables
    ph_dec_var1 = xbar_k
    heur_dec_var2 = y_k
    
    # objective
    ph_obj = np.sum(c * ph_dec_var1) + (np.sum(np.transpose(q) * heur_dec_var2) / n_scenarios)

    return ph_obj, ph_time, ph_dec_var1, heur_dec_var2


def solve2stageknapsack(c, A, b, q, T, W, h, MIPGap=0, only_record_progress = True, use_set_obj = False, set_obj = -1, use_set_time = False, set_time = -1, **kwargs):
    '''
    Solve the 2 stage knapsack problem given in formulation:
        max cx + E[Q(x,$)]
        st Ax<=b
        where Q(x,$) is
        min qy
        st Tx+Wy <= h
    Only q is a random vector
    
    c = size=(n_firststage_variables)
    A = size=(n_firststage_constraints,n_firststage_variables)
    b = size=(n_firststage_constraints)
    q = size=(n_secondstage_variable, *)
    T = size=(n_secondstage_constraints,n_firststage_variables,*) 
    W = size=(n_secondstage_constraints,n_secondstage_variables,*)
    h = size=(n_secondstage_constraints,*)
    * is 1 if known n_scenarios otherwise

    '''
    n_firststage_variables = len(c)
    n_firststage_constraints = len(A)
    n_secondstage_variables = len(W[0])
    n_secondstage_constraints = len(W)
    n_scenarios = max(len(q[0]),len(T[0,0]),len(W[0,0]),len(h[0]))
    
    #q = np.squeeze(q,axis=-1)
    #W = np.expand_dims(W,axis=-1)
  
    two_stage = MSIP(T=2, sense =-1)
    for t in range(2):
        model = two_stage.models[t]
        model.params.threads = n_cores  
        if t == 0:
            
            x_now, x_past = model.addStateVars(n_firststage_variables, vtype='B', name='x'+str(t), obj = c)
            for m in range(n_firststage_constraints):
                model.addConstr(gurobipy.quicksum(x_now[i]*A[m,i] for i in range(n_firststage_variables)) <= b[m])
            
        else:
            # Subproblems do not have an objective coefficient for first stage variables
            x_now, x_past = model.addStateVars(n_firststage_variables, vtype='B', name='x'+str(t), obj = 0)
            # Known q
            if q.shape[1]==1:
                y = model.addVars(n_secondstage_variables, obj = q, vtype='B', name='y'+str(t))
            # Random q
            else:
                y = model.addVars(n_secondstage_variables, uncertainty = q.transpose(), vtype='B', name='y'+str(t))
            # Constraints
            for n in range(n_secondstage_constraints):
                # Add uncertanities if their shape is not 1, i,e, there are scenarios
                if h.shape[-1]>1:
                    unc_h = {'rhs': h[n]}
                else: 
                    unc_h = {}
                if T.shape[-1]>1:
                    unc_T = {x_past[i]:T[n,i] for i in range(n_firststage_variables)}
                else: 
                    unc_T = {}
                if W.shape[-1]>1:
                    unc_W = {y[j]:W[n,j] for j in range(n_secondstage_variables)}
                else: 
                    unc_W = {}
                unc = {**unc_h, **unc_T, **unc_W} 
                model.addConstr(gurobipy.quicksum(x_past[i]*T[n,i,0] for i in range(n_firststage_variables))+
                                gurobipy.quicksum(y[j]*W[n,j,0] for j in range(n_secondstage_variables))<=h[n,0],uncertainty=unc)
            
    extensive_formulation = Extensive(two_stage)
    
    obj, soltime, times, gaps, objs, bounds = extensive_formulation.solve(outputFlag=0,MIPGap=MIPGap, only_record_progress = only_record_progress, use_set_obj = use_set_obj, set_obj = set_obj, use_set_time = use_set_time, set_time = set_time, **kwargs)
    
    first_stage_vars = [extensive_formulation.getVarByName(name+'(0,)').X for name in ['x0['+str(i)+']' for i in range(n_firststage_variables)]]
    second_stage_vars = []
    for j in range(n_scenarios):
        second_stage_vars.append([extensive_formulation.getVarByName(name+'(0,'+str(j)+')').X for name in ['y1['+str(i)+']' for i in range(n_secondstage_variables)]])
    
    #variables = extensive_formulation.getVars()
    if extensive_formulation.Status == 3:
        return -1, -1, -1, -1, -1, -1, -1, -1
    else: 
        #print('gurobi: ',obj)
        return obj, soltime, first_stage_vars, second_stage_vars, times, gaps, objs, bounds

def solve2stageknapsackSDDIP(c, A, b, q, T, W, h, cuts=['B','SB','LG'], **kwargs):
    '''
    Solve the 2 stage knapsack problem given in formulation:
        max cx + E[Q(x,$)]
        st Ax<=b
        where Q(x,$) is
        min qy
        st Tx+Wy <= h
    Only q is a random vector
    
    c = size=(n_firststage_variables)
    A = size=(n_firststage_constraints,n_firststage_variables)
    b = size=(n_firststage_constraints)
    q = size=(n_secondstage_variable, *)
    T = size=(n_secondstage_constraints,n_firststage_variables,*) 
    W = size=(n_secondstage_constraints,n_secondstage_variables,*)
    h = size=(n_secondstage_constraints,*)
    * is 1 if known n_scenarios otherwise

    '''
    n_firststage_variables = len(c)
    n_firststage_constraints = len(A)
    n_secondstage_variables = len(W[0])
    n_secondstage_constraints = len(W)
    #n_scenarios = max(len(q[0]),len(T[0,0]),len(W[0,0]),len(h[0]))
    
    #q = np.squeeze(q,axis=-1)
    #W = np.expand_dims(W,axis=-1)
  
    two_stage = MSIP(T=2, sense =-1)
    for t in range(2):
        model = two_stage.models[t]
        model.params.threads = n_cores
        if t == 0:
            
            x_now, x_past = model.addStateVars(n_firststage_variables, vtype='B', name='x'+str(t), obj = c)
            for m in range(n_firststage_constraints):
                model.addConstr(gurobipy.quicksum(x_now[i]*A[m,i] for i in range(n_firststage_variables)) <= b[m])
            
        else:
            # Subproblems do not have an objective coefficient for first stage variables
            x_now, x_past = model.addStateVars(n_firststage_variables, vtype='B', name='x'+str(t), obj = 0)
            # Known q
            if q.shape[1]==1:
                y = model.addVars(n_secondstage_variables, obj = q, vtype='B', name='y'+str(t))
            # Random q
            else:
                y = model.addVars(n_secondstage_variables, uncertainty = q.transpose(), vtype='B', name='y'+str(t))
            # Constraints
            for n in range(n_secondstage_constraints):
                # Add uncertanities if their shape is not 1, i,e, there are scenarios
                if h.shape[-1]>1:
                    unc_h = {'rhs': h[n]}
                else: 
                    unc_h = {}
                if T.shape[-1]>1:
                    unc_T = {x_past[i]:T[n,i] for i in range(n_firststage_variables)}
                else: 
                    unc_T = {}
                if W.shape[-1]>1:
                    unc_W = {y[j]:W[n,j] for j in range(n_secondstage_variables)}
                else: 
                    unc_W = {}
                unc = {**unc_h, **unc_T, **unc_W} 
                model.addConstr(gurobipy.quicksum(x_past[i]*T[n,i,0] for i in range(n_firststage_variables))+
                                gurobipy.quicksum(y[j]*W[n,j,0] for j in range(n_secondstage_variables))<=h[n,0],uncertainty=unc)
    
    SDDIP_formulation = SDDiP(two_stage)
    SDDIP_formulation.solve(cuts, **kwargs)
    obj, soltime, other_variables = SDDIP_formulation.MSP.db, SDDIP_formulation.total_time, SDDIP_formulation.other_variables[0]
    first_stage_vars = [SDDIP_formulation.MSP.models[0].getVarByName(name).X for name in ['x0['+str(i)+']' for i in range(n_firststage_variables)]]
    second_stage_vars = other_variables
    
    if SDDIP_formulation.MSP.models[1].Status == 3:
        return -1, -1, -1, -1
    else: 
        return obj, soltime, first_stage_vars, second_stage_vars
    



