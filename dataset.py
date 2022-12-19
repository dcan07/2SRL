import platform
import sys
if platform.system()=='Windows':
    TwoStageRLpath = 'C:/Users/dy234/Documents/paper3/TwoStageRL'
    msppypath = 'C:/Users/dy234/Documents/paper3/msppy-master'

else: 
    TwoStageRLpath = '/home/d/dy234/paper3/TwoStageRL'
    msppypath = '/home/d/dy234/paper3/msppy-master'
   
if TwoStageRLpath not in sys.path:
    sys.path.append(TwoStageRLpath)
if msppypath not in sys.path:
    sys.path.append(msppypath)
    

import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from baseline_solvers import solve2stageknapsack
from baseline_solvers import solvemultistageknapsackgurobi
import os

class TwoStage_MKP_Dataset(Dataset):
    def __init__(self, n_problems, n_firststage_variables, n_firststage_constraints, n_secondstage_variables, n_secondstage_constraints, n_scenarios, seed, first_stage_scaler, second_stage_scaler, random_q, random_T, random_W, random_h, mean_first_stage_decision ,remaining_h_ratio):
        '''
        Create dataset for the 2 stage knapsack problem given in formulation:
            max cx + E[Q(x,$)]
            st Ax<=b
            where Q(x,$) is
            min qy
            st Tx+Wy <= h
        q, T, h can be a random vector
        '''
        self.n_problems = n_problems
        self.n_firststage_variables = n_firststage_variables
        self.n_firststage_constraints = n_firststage_constraints
        self.n_secondstage_variables = n_secondstage_variables
        self.n_secondstage_constraints = n_secondstage_constraints
        self.n_scenarios = n_scenarios
        self.seed = seed
        self.first_stage_scaler = first_stage_scaler
        self.second_stage_scaler = second_stage_scaler
        self.random_q = random_q
        self.random_T = random_T
        self.random_W = random_W
        self.random_h = random_h
        self.mean_first_stage_decision = mean_first_stage_decision
        self.remaining_h_ratio = remaining_h_ratio

        c_size = (n_problems, n_firststage_variables)
        A_size = (n_problems, n_firststage_constraints,n_firststage_variables)
        b_size = (n_problems, n_firststage_constraints)

        if random_q:
            q_size = (n_problems, n_secondstage_variables, n_scenarios)
        else:
            q_size = (n_problems, n_secondstage_variables, 1)
            
        if random_T:
            T_size = (n_problems, n_secondstage_constraints,n_firststage_variables, n_scenarios) 
        else:
            T_size = (n_problems, n_secondstage_constraints,n_firststage_variables, 1) 
        
        if random_W:
            W_size = (n_problems, n_secondstage_constraints,n_secondstage_variables, n_scenarios)
        else:
            W_size = (n_problems, n_secondstage_constraints,n_secondstage_variables, 1)
                
        if random_h:
            h_size = (n_problems, n_secondstage_constraints, n_scenarios)
        else:
            h_size = (n_problems, n_secondstage_constraints, 1)
        

        # Set seed before each random generation
    
        # c
        np.random.seed(seed)
        self.c = np.random.randint(1,21,size=c_size).astype(np.float32)
        # A
        np.random.seed(seed+1)
        self.A = np.random.randint(1,21,size=A_size).astype(np.float32)
        # b
        np.random.seed(seed+2)
        #self.b = np.random.randint(201,601,size=b_size).astype(np.float32)
        #b = np.random.randint(0.4*np.mean(A)*n_firststage_variables,0.6*np.mean(A)*n_firststage_variables,size=b_size).astype(np.float32)
        self.b = np.random.randint(0.4*np.mean(self.A)*self.n_firststage_variables,0.6*np.mean(self.A)*self.n_firststage_variables,size=b_size).astype(np.float32)
        # q
        np.random.seed(seed+3)
        self.q = np.random.randint(1,21,size=q_size).astype(np.float32)
        # T
        np.random.seed(seed+4)
        self.T = np.random.randint(1,21,size=T_size).astype(np.float32)
        # W
        np.random.seed(seed+5)
        self.W = np.random.randint(1,21,size=W_size).astype(np.float32)
        # h
        np.random.seed(seed+6)
        #self.h = np.random.randint(1301,1601,size=h_size).astype(np.float32)
        #h = np.random.randint(0.4*((np.mean(T)*n_firststage_variables)+(np.mean(W)*n_secondstage_variables)),0.6*((np.mean(T)*n_firststage_variables)+(np.mean(W)*n_secondstage_variables)),size=h_size).astype(np.float32)
        self.h = np.random.randint(0.4*((np.mean(self.T)*self.n_firststage_variables)+(np.mean(self.W)*self.n_secondstage_variables)),0.6*((np.mean(self.T)*self.n_firststage_variables)+(np.mean(self.W)*self.n_secondstage_variables)),size=h_size).astype(np.float32)
        self.seed += 7
        
   
        # Scaling first stage is easy since there is no preceeding decisions
        # A_norm is the weights normalized by capacity
        self.A_norm = self.A/np.expand_dims(self.b, axis=-1)
        # x is weights and objective function data
        self.first_stage_data = np.concatenate((np.expand_dims(self.c, axis=1), self.A_norm), axis=1)
        # MinMax Scaler
        # If there is no fitted scaler, i.e., training data
        # We fit the scaler using training data, and use it for test data
        if first_stage_scaler is None:
            self.first_stage_scaler = MinMaxScaler(feature_range=(-1, 1))
            self.first_stage_scaler.fit(np.transpose(self.first_stage_data,axes=(0,2,1)).reshape(self.first_stage_data.shape[0]*self.first_stage_data.shape[2],self.first_stage_data.shape[1])) 
        else:
            self.first_stage_scaler = first_stage_scaler
        # Scale seperately for each item
        for i in range(self.first_stage_data.shape[2]):
            self.first_stage_data[:, :, i] = self.first_stage_scaler.transform(self.first_stage_data[:, :, i])

        # Second stage data   
        # Scaling second stage data is tricky since the remaing h actually changes based on first stage decisions 
        # First do scaling with a predetermined average binary variables from first stage
        # This would be ok to train second stage model
        # Calculate the mean first stage decision if not predetermined by solving first 10 instance from the dataset
        if self.mean_first_stage_decision is None:
            first_stage_vars_list = []
            second_stage_vars_list = []
            for i in range(50):
                _, _, first_stage_vars, second_stage_vars, _, _, _, _ = solve2stageknapsack(self.c[i], self.A[i], self.b[i], self.q[i], self.T[i], self.W[i], self.h[i], MIPGap=0, only_record_progress = True, use_set_obj = False, set_obj = -1, use_set_time = False, set_time = -1)
                #_, _, first_stage_vars, second_stage_vars, _, _, _, _ = solve2stageknapsack(c[i], A[i], b[i], q[i], T[i], W[i], h[i], MIPGap=0, only_record_progress = True, use_set_obj = False, set_obj = -1, use_set_time = False, set_time = -1)
                first_stage_vars = np.array(first_stage_vars).astype(np.float32)
                first_stage_vars_list.append(first_stage_vars)
                #first_stage_vars_list.append(first_stage_vars)
                second_stage_vars = np.array(second_stage_vars).astype(np.float32)
                second_stage_vars_list.append(second_stage_vars)
            #self.mean_first_stage_decision = np.mean(first_stage_vars_list)
            first_stage_vars_list = np.array(first_stage_vars_list)
            first_stage_vars_list = np.expand_dims(first_stage_vars_list, axis=-1)
            second_stage_vars_list = np.array(second_stage_vars_list)
            second_stage_vars_list = np.expand_dims(second_stage_vars_list, axis=-1)
            #second_stage_vars_list = second_stage_vars_list.transpose((0,2,1))
            
            lhs = self.T[:(i+1)]*np.expand_dims(first_stage_vars_list, axis=1)
            #lhs = lhs.astype(np.float32)
            lhs = np.sum(lhs, axis=2)
            self.remaining_h_ratio = lhs / self.h[:(i+1)]

            self.mean_first_stage_decision = np.mean(self.remaining_h_ratio)
        
            print('\n\n Mean first Stage: ',np.mean(first_stage_vars_list))
            print('\n\n Mean second Stage: ',np.mean(second_stage_vars_list))
        # Then for the first stage training, it sould be scaled with the actual decision variables determined by the model
        self.update_second_stage_data()





    def __getitem__(self, index):
        first_stage_data = self.first_stage_data[index]
        second_stage_data = self.second_stage_data[index]
        c = self.c[index]
        A = self.A[index]
        b = self.b[index]
        q = self.q[index]
        T = self.T[index]
        W = self.W[index]
        h = self.h[index]
        remaining_h = self.remaining_h[index]
        
        

        return first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h

    def __len__(self):
        return len(self.h)
    
    def first_stage_scaler(self):
        return self.first_stage_scaler
    def second_stage_scaler(self):
        return self.second_stage_scaler

    def generate_random_first_stage_decisions(self):
        # Generate random first stage variables based on proportion
        # This is helpful to fit the scaler and train second stage model

        np.random.seed(self.seed)
        self.first_stage_decision = np.random.binomial(1, self.mean_first_stage_decision, size=(self.n_problems, self.n_firststage_variables, 1))
        self.first_stage_decision = np.zeros(shape=(self.n_problems, self.n_firststage_variables, 1))

        self.seed += 1

    def check_first_stage_feasibility(self):
        lhs = np.sum(self.A*self.first_stage_decision.transpose((0,2,1)), axis=-1)
        lhs = lhs.astype(np.float32)
        return np.all((self.b-lhs)>=0)
        
    def update_remaining_h(self):
        lhs = self.T*np.expand_dims(self.first_stage_decision, axis=1)
        lhs = lhs.astype(np.float32)
        lhs = np.sum(lhs, axis=2)
        self.remaining_h = self.h - lhs
    def update_remaining_h_constant(self):

        #lhs = np.sum(self.T, axis=2)*self.mean_first_stage_decision
        self.remaining_h = self.h * (1-self.mean_first_stage_decision)
    def update_remaining_h_ratio_dist(self):
        mean_first_stage_random = np.random.choice(self.remaining_h_ratio.flatten(), size = self.h.shape)
        self.remaining_h = self.h * (1-mean_first_stage_random)

    
    def scale_second_stage_data(self):
        # W_norm is the weights normalized by capacity
        self.W_norm = self.W/np.expand_dims(self.remaining_h, axis=-2)
        # If only obj is random
        if self.W_norm.shape[-1]<self.q.shape[-1]:
            self.tiled_arr=np.tile(self.W_norm, (1,1,1,self.n_scenarios))
            self.second_stage_data = np.concatenate((np.expand_dims(self.q, axis=1), self.tiled_arr), axis=1)
            #tiled_arr.shape
        # If only T,W, and/or h are random
        elif self.W_norm.shape[-1]>self.q.shape[-1]:
            self.tiled_arr=np.tile(self.q, (1,1,self.n_scenarios))
            self.second_stage_data = np.concatenate((np.expand_dims(self.tiled_arr, axis=1), self.W_norm), axis=1)

        # All are random
        else:
            self.second_stage_data = np.concatenate((np.expand_dims(self.q, axis=1), self.W_norm), axis=1)
            
        if self.second_stage_scaler is None:
            self.second_stage_scaler = MinMaxScaler(feature_range=(-1, 1))
            self.second_stage_scaler.fit(np.transpose(self.second_stage_data,axes=(0,3,2,1)).reshape(-1,self.second_stage_data.shape[1])) 

        # Scale seperately for each item
        for i in range(self.second_stage_data.shape[2]):
            for j in range(self.second_stage_data.shape[3]):
                self.second_stage_data[:, :, i, j] = self.second_stage_scaler.transform(self.second_stage_data[:, :, i, j])
    
    def update_second_stage_data(self, first_stage_decision = None):
        
        if first_stage_decision is not None:
            self.first_stage_decision = first_stage_decision
        else:
            self.generate_random_first_stage_decisions()
        # Ensure that we have a feasbile first stage solution
        
        #while not self.check_first_stage_feasibility():
        #    self.generate_random_first_stage_decisions()
        #    print('inf')
        self.update_remaining_h_ratio_dist()
        self.scale_second_stage_data()

class TwoStage_MKP_TestDataset(Dataset):
    def __init__(self, c, A, b, q, T, W, h, n_problems, n_firststage_variables, n_firststage_constraints, n_secondstage_variables, n_secondstage_constraints, n_scenarios, seed, first_stage_scaler, second_stage_scaler, random_q, random_T, random_W, random_h, mean_first_stage_decision ,remaining_h_ratio):
        '''
        Create dataset for the 2 stage knapsack problem given in formulation:
            max cx + E[Q(x,$)]
            st Ax<=b
            where Q(x,$) is
            min qy
            st Tx+Wy <= h
        q, T, h can be a random vector
        '''
        self.n_problems = n_problems
        self.n_firststage_variables = n_firststage_variables
        self.n_firststage_constraints = n_firststage_constraints
        self.n_secondstage_variables = n_secondstage_variables
        self.n_secondstage_constraints = n_secondstage_constraints
        self.n_scenarios = n_scenarios
        self.seed = seed
        self.first_stage_scaler = first_stage_scaler
        self.second_stage_scaler = second_stage_scaler
        self.random_q = random_q
        self.random_T = random_T
        self.random_W = random_W
        self.random_h = random_h
        self.mean_first_stage_decision = mean_first_stage_decision
        self.remaining_h_ratio = remaining_h_ratio

        

    
        # c
        self.c = c
        # A
        self.A = A
        # b
        self.b = b
        # q
        self.q = q
        # T
        self.T = T
        # W
        self.W = W
        # h
        self.h = h

   
        # Scaling first stage is easy since there is no preceeding decisions
        # A_norm is the weights normalized by capacity
        self.A_norm = self.A/np.expand_dims(self.b, axis=-1)
        # x is weights and objective function data
        self.first_stage_data = np.concatenate((np.expand_dims(self.c, axis=1), self.A_norm), axis=1)
        # MinMax Scaler
        # If there is no fitted scaler, i.e., training data
        # We fit the scaler using training data, and use it for test data
        if first_stage_scaler is None:
            self.first_stage_scaler = MinMaxScaler(feature_range=(-1, 1))
            self.first_stage_scaler.fit(np.transpose(self.first_stage_data,axes=(0,2,1)).reshape(self.first_stage_data.shape[0]*self.first_stage_data.shape[2],self.first_stage_data.shape[1])) 
        else:
            self.first_stage_scaler = first_stage_scaler
        # Scale seperately for each item
        for i in range(self.first_stage_data.shape[2]):
            self.first_stage_data[:, :, i] = self.first_stage_scaler.transform(self.first_stage_data[:, :, i])

        # Second stage data   
        # Scaling second stage data is tricky since the remaing h actually changes based on first stage decisions 
        # First do scaling with a predetermined average binary variables from first stage
        # This would be ok to train second stage model
        # Calculate the mean first stage decision if not predetermined by solving first 10 instance from the dataset
        if self.mean_first_stage_decision is None:
            first_stage_vars_list = []
            second_stage_vars_list = []
            for i in range(20):
                _, _, first_stage_vars, second_stage_vars, _, _, _, _ = solve2stageknapsack(self.c[i], self.A[i], self.b[i], self.q[i], self.T[i], self.W[i], self.h[i], MIPGap=0, only_record_progress = True, use_set_obj = False, set_obj = -1, use_set_time = False, set_time = -1)
                #_, _, first_stage_vars, second_stage_vars, _, _, _, _ = solve2stageknapsack(c[i], A[i], b[i], q[i], T[i], W[i], h[i], MIPGap=0, only_record_progress = True, use_set_obj = False, set_obj = -1, use_set_time = False, set_time = -1)
                first_stage_vars = np.array(first_stage_vars).astype(np.float32)
                first_stage_vars_list.append(first_stage_vars)
                #first_stage_vars_list.append(first_stage_vars)
                second_stage_vars = np.array(second_stage_vars).astype(np.float32)
                second_stage_vars_list.append(second_stage_vars)
            #self.mean_first_stage_decision = np.mean(first_stage_vars_list)
            first_stage_vars_list = np.array(first_stage_vars_list)
            first_stage_vars_list = np.expand_dims(first_stage_vars_list, axis=-1)
            second_stage_vars_list = np.array(second_stage_vars_list)
            second_stage_vars_list = np.expand_dims(second_stage_vars_list, axis=-1)
            #second_stage_vars_list = second_stage_vars_list.transpose((0,2,1))
            
            lhs = self.T[:(i+1)]*np.expand_dims(first_stage_vars_list, axis=1)
            #lhs = lhs.astype(np.float32)
            lhs = np.sum(lhs, axis=2)
            self.remaining_h_ratio = lhs / self.h[:(i+1)]

            self.mean_first_stage_decision = np.mean(self.remaining_h_ratio)

        # Then for the first stage training, it sould be scaled with the actual decision variables determined by the model
        self.update_second_stage_data()





    def __getitem__(self, index):
        first_stage_data = self.first_stage_data[index]
        second_stage_data = self.second_stage_data[index]
        c = self.c[index]
        A = self.A[index]
        b = self.b[index]
        q = self.q[index]
        T = self.T[index]
        W = self.W[index]
        h = self.h[index]
        remaining_h = self.remaining_h[index]
        
        

        return first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h

    def __len__(self):
        return len(self.h)
    
    def first_stage_scaler(self):
        return self.first_stage_scaler
    def second_stage_scaler(self):
        return self.second_stage_scaler

    def generate_random_first_stage_decisions(self):
        # Generate random first stage variables based on proportion
        # This is helpful to fit the scaler and train second stage model

        np.random.seed(self.seed)
        self.first_stage_decision = np.random.binomial(1, self.mean_first_stage_decision, size=(self.n_problems, self.n_firststage_variables, 1))
        self.first_stage_decision = np.zeros(shape=(self.n_problems, self.n_firststage_variables, 1))

        self.seed += 1
 
    def check_first_stage_feasibility(self):
        lhs = np.sum(self.A*self.first_stage_decision.transpose((0,2,1)), axis=-1)
        lhs = lhs.astype(np.float32)
        return np.all((self.b-lhs)>=0)
        
    def update_remaining_h(self):
        lhs = self.T*np.expand_dims(self.first_stage_decision, axis=1)
        lhs = lhs.astype(np.float32)
        lhs = np.sum(lhs, axis=2)
        self.remaining_h = self.h - lhs
    def update_remaining_h_constant(self):

        #lhs = np.sum(self.T, axis=2)*self.mean_first_stage_decision
        self.remaining_h = self.h * (1-self.mean_first_stage_decision)
    def update_remaining_h_ratio_dist(self):
        mean_first_stage_random = np.random.choice(self.remaining_h_ratio.flatten(), size = self.h.shape)
        self.remaining_h = self.h * (1-mean_first_stage_random)

    
    def scale_second_stage_data(self):
        # W_norm is the weights normalized by capacity
        self.W_norm = self.W/np.expand_dims(self.remaining_h, axis=-2)
        # If only obj is random
        if self.W_norm.shape[-1]<self.q.shape[-1]:
            self.tiled_arr=np.tile(self.W_norm, (1,1,1,self.n_scenarios))
            self.second_stage_data = np.concatenate((np.expand_dims(self.q, axis=1), self.tiled_arr), axis=1)
            #tiled_arr.shape
        # If only T,W, and/or h are random
        elif self.W_norm.shape[-1]>self.q.shape[-1]:
            self.tiled_arr=np.tile(self.q, (1,1,self.n_scenarios))
            self.second_stage_data = np.concatenate((np.expand_dims(self.tiled_arr, axis=1), self.W_norm), axis=1)

        # All are random
        else:
            self.second_stage_data = np.concatenate((np.expand_dims(self.q, axis=1), self.W_norm), axis=1)
            
        if self.second_stage_scaler is None:
            self.second_stage_scaler = MinMaxScaler(feature_range=(-1, 1))
            self.second_stage_scaler.fit(np.transpose(self.second_stage_data,axes=(0,3,2,1)).reshape(-1,self.second_stage_data.shape[1])) 

        # Scale seperately for each item
        for i in range(self.second_stage_data.shape[2]):
            for j in range(self.second_stage_data.shape[3]):
                self.second_stage_data[:, :, i, j] = self.second_stage_scaler.transform(self.second_stage_data[:, :, i, j])
    
    def update_second_stage_data(self, first_stage_decision = None):
        
        if first_stage_decision is not None:
            self.first_stage_decision = first_stage_decision
        else:
            self.generate_random_first_stage_decisions()
        # Ensure that we have a feasbile first stage solution
        
        #while not self.check_first_stage_feasibility():
        #    self.generate_random_first_stage_decisions()
        #    print('inf')
        self.update_remaining_h_ratio_dist()
        self.scale_second_stage_data()    
        
