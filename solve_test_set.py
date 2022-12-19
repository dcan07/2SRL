import platform
import sys

if platform.system()=='Windows':
    TwoStageRLpath = 'C:/Users/dy234/Documents/paper3/TwoStageRL'
    msppypath = 'C:/Users/dy234/Documents/paper3/msppy-master'
    testset_config_path='C:/Users/dy234/Documents/paper3/config_testset1'
    testset_config_path='C:/Users/dy234/Documents/paper3/config_testset0-3'
    testset_path='C:/Users/dy234/Documents/paper3/testset1/'
    
else: 
    TwoStageRLpath = '/home/d/dy234/paper3/TwoStageRL'
    msppypath = '/home/d/dy234/paper3/msppy-master'
    testset_config_path=str(sys.argv[1])
    testset_path=str(sys.argv[2])

    
if TwoStageRLpath not in sys.path:
    sys.path.append(TwoStageRLpath)
if msppypath not in sys.path:
    sys.path.append(msppypath)
    
import dataset
import pointer_network
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from matplotlib import pyplot as plt
import pickle as pkl
from baseline_solvers import solve2stageknapsackcplex, solve2stageknapsackgurobi, solve2stageknapsackSDDIP, solve2stageknapsackheuristic2, solve2stageknapsackheuristic
from sklearn.metrics import accuracy_score
import gc
from os import walk




def load_path(testset_config_path):

    
    with open(testset_config_path+ '.pkl', 'rb') as f:
        model_config_testset = pkl.load(f)
        
        model_config_testset['mean_first_stage_decision']=np.array([0.5])
        model_config_testset['remaining_h_ratio']=np.array([0.5])
    

        
    return  model_config_testset




def build_dataset(model_config_testset, first_stage_decision = None):
    
    
    test_dataset = dataset.TwoStage_MKP_Dataset( 5*model_config_testset['n_problems_test'], 
                                                 model_config_testset['n_firststage_variables'], 
                                                 model_config_testset['n_firststage_constraints'], 
                                                 model_config_testset['n_secondstage_variables'], 
                                                 model_config_testset['n_secondstage_constraints'], 
                                                 model_config_testset['n_scenarios'], 
                                                 model_config_testset['seed_test'], 
                                                 model_config_testset['first_stage_scaler'], 
                                                 model_config_testset['second_stage_scaler'], 
                                                 model_config_testset['random_q'], 
                                                 model_config_testset['random_T'], 
                                                 model_config_testset['random_W'], 
                                                 model_config_testset['random_h'], 
                                                 model_config_testset['mean_first_stage_decision'],
                                                 model_config_testset['remaining_h_ratio'])
    
    
        
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return test_dataloader




def solve_iterator(iterator, testset_path, **kwargs):
    
    gurobi_total_time = 0
    sddip_total_time = 0
    heur_total_time = 0
    
    ct = 0
    instance=0
    
    solved_instances = next(walk(testset_path), (None, None, []))[2]  # [] if no file
    solved_instances = [x for x in solved_instances if not x.startswith('.')]
    solved_instances = [int(name.split('.')[0]) for name in solved_instances]
        
    for _, _, c_tensor, A_tensor, b_tensor, q_tensor, T_tensor, W_tensor, h_tensor, _ in iterator:
        
        # If we have not consider instance in the test set
        if instance>max(solved_instances, default=-1):
        
            # One instance at a time
            assert c_tensor.shape[0]==1
            c = c_tensor[0].numpy()
            A = A_tensor[0].numpy()
            b = b_tensor[0].numpy()
            q = q_tensor[0].numpy()
            T = T_tensor[0].numpy()
            W = W_tensor[0].numpy()
            h = h_tensor[0].numpy()

            
            np.mean(q)
            try: 
                sddip_obj, sddip_time, sddip_vars_1, sddip_vars_2  = solve2stageknapsackSDDIP(c, A, b, q, T, W, h,logToConsole = 0, save_variables=False, max_stable_iterations=20, max_time=3600)
            except:
                print("Went wrong")
            else:
                print("Nothing went wrong")
                    
                gurobi_obj, gurobi_time, gurobi_vars_1, gurobi_vars_2, gurobi_gap, gurobi_bound = solve2stageknapsackgurobi(c, A, b, q, T, W, h, use_set_time = True, set_time = 3600)
                #print('done gurobi')
                #heur_obj0, heur_time, heur_vars_1, heur_vars_2  = solve2stageknapsackheuristic(c, A, b, q, T, W, h)
                heur_obj, heur_time, heur_vars_1, heur_vars_2  = solve2stageknapsackheuristic2(c, A, b, q, T, W, h)
                #print(heur_obj0-heur_obj)
                instance_result = [c, A, b, q, T, W, h, gurobi_obj, gurobi_time, gurobi_vars_1, gurobi_vars_2, gurobi_gap, gurobi_bound,sddip_obj, sddip_time, sddip_vars_1, sddip_vars_2,heur_obj, heur_time, heur_vars_1, heur_vars_2 ]
                with open(testset_path+str(instance)+ '.pkl', 'wb') as f:
                    pkl.dump(instance_result,f)
                solved_instances.append(instance)
                
                gurobi_total_time += gurobi_time
                sddip_total_time += sddip_time
                heur_total_time += heur_time
                ct+=1
                
                
        
        # We have consider this instance        
        #else:
            #Do nothing 
        # Either way increase the instance count
        instance += 1
        
        # Stop iteration if we solve enough instances:
        if len(solved_instances)>=20:
            print('gurobi time ',  gurobi_total_time/ct )
            print('sddip time ' ,sddip_total_time/ct )
            print('heur time ',  heur_total_time/ct )
            break
        


if __name__ == '__main__':
    model_config_testset= load_path(testset_config_path)
    test_dataloader = build_dataset( model_config_testset)
    solve_iterator(test_dataloader, testset_path)
    
   

    

