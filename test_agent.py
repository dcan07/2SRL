import platform
import sys

if platform.system()=='Windows':
    TwoStageRLpath = 'C:/Users/dy234/Documents/paper3/TwoStageRL'
    msppypath = 'C:/Users/dy234/Documents/paper3/msppy-master'
    trainset_config_path='C:/Users/dy234/Documents/paper3/config_trainset1'
    testset_config_path='C:/Users/dy234/Documents/paper3/config_testset1'
    testset_path='C:/Users/dy234/Documents/paper3/testset1/'
    agent2_config_path='C:/Users/dy234/Documents/paper3/config_set1_agent2_model0'
    agent2_path='C:/Users/dy234/Documents/paper3/set1_agent2_model0'
    agent1_config_path='C:/Users/dy234/Documents/paper3/config_set1_agent1_model0'
    agent1_path='C:/Users/dy234/Documents/paper3/set1_agent1_model0'
    result_path='C:/Users/dy234/Documents/paper3/set1_model0_result'
else: 
    TwoStageRLpath = '/home/d/dy234/paper3/TwoStageRL'
    msppypath = '/home/d/dy234/paper3/msppy-master'
    #sys.path.append('/home/d/dy234/msppy-master')
    trainset_config_path=str(sys.argv[1])
    testset_config_path=str(sys.argv[2])
    testset_path=str(sys.argv[3])
    agent2_config_path=str(sys.argv[4])
    agent2_path=str(sys.argv[5])
    agent1_config_path=str(sys.argv[6])
    agent1_path=str(sys.argv[7])
    result_path=str(sys.argv[8])

    
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
from baseline_solvers import solve2stageknapsackcplex, solve2stageknapsack, solve2stageknapsackSDDIP
from sklearn.metrics import accuracy_score
import gc
import os

def get_torch_memory():
    factor = 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / factor
    r = torch.cuda.memory_reserved(0) / factor
    a = torch.cuda.memory_allocated(0) / factor
    f = r-a
    print('Total mem: ','{:.2f}'.format(t), ' GB' )
    print('Reserved mem: ','{:.2f}'.format(r), ' GB' )
    print('Allocated mem: ','{:.2f}'.format(a), ' GB' )
    print('Free mem: ','{:.2f}'.format(f), ' GB' )
    print('\n')


def load_path(trainset_config_path, testset_config_path, agent2_config_path, agent1_config_path):
    
    with open(trainset_config_path+ '.pkl', 'rb') as f:
        model_config_trainset = pkl.load(f)
    
    with open(testset_config_path+ '.pkl', 'rb') as f:
        model_config_testset = pkl.load(f)
    
    with open(agent2_config_path+ '.pkl', 'rb') as f:
        model_config_agent2 = pkl.load(f)
        
    with open(agent1_config_path+ '.pkl', 'rb') as f:
        model_config_agent1 = pkl.load(f)
        
    return model_config_trainset, model_config_testset, model_config_agent2, model_config_agent1

def change_config_for_test(model_config_trainset, testset_config_path, model_config_testset, model_config_agent2, model_config_agent1):
    
    # Get the scaler form the trainset
    if model_config_testset['second_stage_scaler'] is None:
        model_config_testset['second_stage_scaler']= model_config_trainset['second_stage_scaler']
        model_config_testset['first_stage_scaler']= model_config_trainset['first_stage_scaler']
        model_config_testset['mean_first_stage_decision']= model_config_trainset['mean_first_stage_decision']
        model_config_testset['remaining_h_ratio']= model_config_trainset['remaining_h_ratio']
    
        with open(testset_config_path+ '.pkl', 'wb') as f:
            pkl.dump(model_config_testset,f)
    
    # Change form greedy search to stochastic decode
    model_config_agent2['decode_type'] = 'stochastic'
    model_config_agent1['decode_type'] = 'stochastic'
    
    if testset_config_path.endswith('0'):
        model_config_testset['stochastic_pass_count'] = 1000
    if testset_config_path.endswith('1'):
        model_config_testset['stochastic_pass_count'] = 1000
    if testset_config_path.endswith('2'):
        model_config_testset['stochastic_pass_count'] = 1000
    if testset_config_path.endswith('3'):
        model_config_testset['stochastic_pass_count'] = 5000
    if testset_config_path.endswith('4'):
        model_config_testset['stochastic_pass_count'] = 5000    
    if testset_config_path.endswith('5'):
        model_config_testset['stochastic_pass_count'] = 5000
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_agent2_model(model_config_agent2, agent2_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = pointer_network.Encoder(model_config_agent2['input_dim'], model_config_agent2['actor_enc_hid_dim'], model_config_agent2['actor_n_layers'], model_config_agent2['actor_n_directions'], model_config_agent2['dropout_lstm'], model_config_agent2['use_embedding'], model_config_agent2['emb_dim'])
    pointer = pointer_network.Pointer(model_config_agent2['actor_dec_hid_dim'], model_config_agent2['softmax_temperature'], model_config_agent2['logit_clip'])
    glimpse = pointer_network.Glimpse(model_config_agent2['actor_dec_hid_dim'])
    decoder = pointer_network.Decoder(model_config_agent2['input_dim'], model_config_agent2['actor_enc_hid_dim'], model_config_agent2['actor_dec_hid_dim'], model_config_agent2['actor_n_layers'], model_config_agent2['dropout_lstm'], pointer, model_config_agent2['use_glimpse'], model_config_agent2['n_glimpses'], glimpse, model_config_agent2['use_embedding'], model_config_agent2['emb_dim'])
    encoder_critic = pointer_network.Encoder(model_config_agent2['input_dim'], model_config_agent2['critic_enc_hid_dim'], model_config_agent2['critic_n_layers'], model_config_agent2['critic_n_directions'], model_config_agent2['dropout_lstm'], model_config_agent2['use_embedding'], model_config_agent2['emb_dim'])
    glimpse_critic = pointer_network.Glimpse(model_config_agent2['critic_dec_hid_dim'])
    
    actor = pointer_network.PointerNetwork( encoder, pointer, glimpse, decoder, model_config_agent2['decode_type'], device)
    critic = pointer_network.Critic(encoder_critic, glimpse_critic, model_config_agent2['n_glimpses'], model_config_agent2['dropout_linear'], device)
    
    print('device' , device)
    if device=='cuda':
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
    device_ids = [i for i in range(torch.cuda.device_count())]
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        actor = torch.nn.DataParallel(actor,device_ids=device_ids)
        critic = torch.nn.DataParallel(critic,device_ids=device_ids)
    
    if torch.cuda.is_available():
        actor.load_state_dict(torch.load(agent2_path+'_actor.pt'))
        critic.load_state_dict(torch.load(agent2_path+'_critic.pt'))
    else:
        actor.load_state_dict(torch.load(agent2_path+'_actor.pt',map_location=torch.device('cpu')))
        critic.load_state_dict(torch.load(agent2_path+'_critic.pt',map_location=torch.device('cpu')))

    
    
    
    print(f'The actor of agent 2 has {count_parameters(actor):,} trainable parameters')
    print(f'The critic of agent 2 has {count_parameters(critic):,} trainable parameters')
    return actor, critic

def build_agent1_model(model_config_agent1, agent1_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = pointer_network.Encoder(model_config_agent1['input_dim'], model_config_agent1['actor_enc_hid_dim'], model_config_agent1['actor_n_layers'], model_config_agent1['actor_n_directions'], model_config_agent1['dropout_lstm'], model_config_agent1['use_embedding'], model_config_agent1['emb_dim'])
    pointer = pointer_network.Pointer(model_config_agent1['actor_dec_hid_dim'], model_config_agent1['softmax_temperature'], model_config_agent1['logit_clip'])
    glimpse = pointer_network.Glimpse(model_config_agent1['actor_dec_hid_dim'])
    decoder = pointer_network.Decoder(model_config_agent1['input_dim'], model_config_agent1['actor_enc_hid_dim'], model_config_agent1['actor_dec_hid_dim'], model_config_agent1['actor_n_layers'], model_config_agent1['dropout_lstm'], pointer, model_config_agent1['use_glimpse'], model_config_agent1['n_glimpses'], glimpse, model_config_agent1['use_embedding'], model_config_agent1['emb_dim'])
    encoder_critic = pointer_network.Encoder(model_config_agent1['input_dim'], model_config_agent1['critic_enc_hid_dim'], model_config_agent1['critic_n_layers'], model_config_agent1['critic_n_directions'], model_config_agent1['dropout_lstm'], model_config_agent1['use_embedding'], model_config_agent1['emb_dim'])
    glimpse_critic = pointer_network.Glimpse(model_config_agent1['critic_dec_hid_dim'])
    
    actor = pointer_network.PointerNetwork( encoder, pointer, glimpse, decoder, model_config_agent1['decode_type'], device)
    critic = pointer_network.Critic(encoder_critic, glimpse_critic, model_config_agent1['n_glimpses'], model_config_agent1['dropout_linear'], device)
    
    print('device' , device)
    if device=='cuda':
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
    device_ids = [i for i in range(torch.cuda.device_count())]
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        actor = torch.nn.DataParallel(actor,device_ids=device_ids)
        critic = torch.nn.DataParallel(critic,device_ids=device_ids)
        
    if torch.cuda.is_available():
        actor.load_state_dict(torch.load(agent1_path+'_actor.pt'))
        critic.load_state_dict(torch.load(agent1_path+'_critic.pt'))
    else:
        actor.load_state_dict(torch.load(agent1_path+'_actor.pt',map_location=torch.device('cpu')))
        critic.load_state_dict(torch.load(agent1_path+'_critic.pt',map_location=torch.device('cpu')))
        print('Change model read name')
    
    
    print(f'The actor of agent 1 has {count_parameters(actor):,} trainable parameters')
    print(f'The critic of agent 1 has {count_parameters(critic):,} trainable parameters')
    return actor, critic


def read_testset(testset_path, n_problems):
    
    c_test, A_test, b_test, q_test, T_test, W_test, h_test = [], [], [], [], [], [], []    
    gurobi_obj, gurobi_time, gurobi_vars_1, gurobi_vars_2, gurobi_gap, gurobi_bound = [], [], [], [], [], []
    sddip_obj, sddip_time, sddip_vars_1, sddip_vars_2  = [], [], [], []
    heur_obj, heur_time, heur_vars_1, heur_vars_2  = [], [], [], []
    
    for filename in os.listdir(testset_path):
        with open(testset_path+filename, 'rb') as infile:
            instance = pkl.load(infile)
        c_test.append(instance[0])
        A_test.append(instance[1])
        b_test.append(instance[2])
        q_test.append(instance[3])
        T_test.append(instance[4])
        W_test.append(instance[5])
        h_test.append(instance[6])
        gurobi_obj.append(instance[7])
        gurobi_time.append(instance[8])
        gurobi_vars_1.append(instance[9])
        gurobi_vars_2.append(instance[10])
        gurobi_gap.append(instance[11])
        gurobi_bound.append(instance[12])
        sddip_obj.append(instance[13])
        sddip_time.append(instance[14])
        sddip_vars_1.append(instance[15])
        sddip_vars_2.append(instance[16])
        heur_obj.append(instance[17])
        heur_time.append(instance[18])
        heur_vars_1.append(instance[19])
        heur_vars_2.append(instance[20])
        
    c_test = np.array(c_test[:n_problems])
    A_test = np.array(A_test[:n_problems])
    b_test = np.array(b_test[:n_problems])
    q_test = np.array(q_test[:n_problems])
    T_test = np.array(T_test[:n_problems])
    W_test = np.array(W_test[:n_problems])
    h_test = np.array(h_test[:n_problems])
    gurobi_obj = np.array(gurobi_obj[:n_problems])
    gurobi_time = np.array(gurobi_time[:n_problems])
    gurobi_vars_1 = np.array(gurobi_vars_1[:n_problems])
    gurobi_vars_2 = np.array(gurobi_vars_2[:n_problems])
    gurobi_gap = np.array(gurobi_gap[:n_problems])
    gurobi_bound = np.array(gurobi_bound[:n_problems])
    sddip_obj = np.array(sddip_obj[:n_problems])
    sddip_time = np.array(sddip_time[:n_problems])
    sddip_vars_1 = np.array(sddip_vars_1[:n_problems])
    sddip_vars_2 = np.array(sddip_vars_2[:n_problems])
    heur_obj = np.array(heur_obj[:n_problems])
    heur_time = np.array(heur_time[:n_problems])
    heur_vars_1 = np.array(heur_vars_1[:n_problems])
    heur_vars_2 = np.array(heur_vars_2[:n_problems])
    
    return c_test, A_test, b_test, q_test, T_test, W_test, h_test, gurobi_obj, gurobi_time, gurobi_vars_1, gurobi_vars_2, gurobi_gap, gurobi_bound, sddip_obj, sddip_time, sddip_vars_1, sddip_vars_2, heur_obj, heur_time, heur_vars_1, heur_vars_2 

def build_dataset(c_test, A_test, b_test, q_test, T_test, W_test, h_test, model_config_testset, testset_config_path, first_stage_decision = None):
    
    
    test_dataset = dataset.TwoStage_MKP_TestDataset(c_test, A_test, b_test, q_test, T_test, W_test, h_test, 
                                                 model_config_testset['n_problems_test'], 
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
    
    if first_stage_decision is not None:
        test_dataset.first_stage_decision = first_stage_decision
        test_dataset.update_remaining_h()
        test_dataset.scale_second_stage_data()
        
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return test_dataloader









# test  module
def test_stage_1(actor_model1, critic_model1, iterator):
    
    mse = torch.nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    get_torch_memory()
    batch = 0 
    # Send bot agent 1 and 2 to device 
    #actor_model2.to(device)
    #critic_model2.to(device)
    actor_model1.to(device)
    critic_model1.to(device)
    
    # Agent 1 is in the training model
    actor_model1.eval()
    critic_model1.eval()
    
    # Agent 2 is in the testing mode
    #actor_model2.eval()
    #critic_model2.eval()
        
    actor_loss_seq = []
    critic_loss_seq = []
    
    actor_obj_values_1 = []
    critic_obj_values_1 = []
    #actor_obj_values_2 = []
    #critic_obj_values_2 = []
    
    selected_items_1_model = []
    selected_items_2_model = []
    #actor_obj_1_model = []
    #actor_obj_2_model = []
    
    agent1_time = []
    #agent2_time = []


    with torch.no_grad():
        
        #first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h  = next(iter(test_dataloader))  
        # Start with first stage decisions. Then rescale data to select second stage decisions
        for first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h in iterator:       
            get_torch_memory()
            # Start timeing
            start = time.time()
            # Forward pass for Agent 1
            src_1 = first_stage_data
            obj_1 = c
            lhs_1 = A
            rhs_1 = b
            
            # Src, lhs, rhs to device
            # Lhs, and rhs is needed for actor because of masks and feasibility
            src_1 = src_1.to(device)
            obj_1 = obj_1.to(device)
            lhs_1 = lhs_1.to(device) 
            rhs_1 = rhs_1.to(device)
            # Actor forward pass for stochastic decoding
            actual_value_list_1 = torch.empty(size = (model_config_testset['stochastic_pass_count'],src_1.shape[0],1))
            selected_items_list_1 = torch.empty(size = (model_config_testset['stochastic_pass_count'],*src_1[:,0].shape))
            log_likelihood_list_1 = torch.empty(size = (model_config_testset['stochastic_pass_count'],src_1.shape[0]))
            for ix in range(model_config_testset['stochastic_pass_count']):
                selected_items, log_likelihood = actor_model1(src_1, lhs_1, rhs_1)
                # Calculate true objective function value
                actual_value = pointer_network.calculate_obj(obj_1, selected_items)
                selected_items_list_1[ix] = selected_items
                log_likelihood_list_1[ix] = log_likelihood
                actual_value_list_1[ix] = actual_value
            #Calculate the max index
            max_ix = torch.argmax(actual_value_list_1, dim=0)
            # Get the values corresponding to max index
            actual_value_1 = torch.gather(actual_value_list_1.transpose(0,1).squeeze(-1), dim=1, index=max_ix).squeeze(-1)
            log_likelihood_1 = torch.gather(log_likelihood_list_1.transpose(0,1), dim=1, index=max_ix).squeeze(-1)
            max_ix = max_ix.repeat(1, src_1.shape[-1]).unsqueeze(-1)
            selected_items_1 = torch.gather(selected_items_list_1.permute(1,2,0), dim=2, index=max_ix).squeeze(-1)
            # Ensure that the selection is correct 
            #actual_value_temp = pointer_network.calculate_obj(obj, selected_items_1)
            #actual_value_temp.squeeze(-1) == actual_value_1
           
            # Pred from Critic 1
            # Critic forward pass
            pred_value_1 = critic_model1(src_1)
            
            # Critic 1 MSE loss
            critic_loss = mse(actual_value, pred_value_1)
            # Save critic 1 loss
            critic_loss_seq.append(critic_loss.item())
            
            # Save objective function values
            actor_obj_values_1.extend(actual_value_1.squeeze(-1).tolist())
            # Save objective function values from critic
            critic_obj_values_1.extend(pred_value_1.detach().squeeze(-1).tolist())
            # Save selected items
            selected_items_1_model.extend(selected_items_1.tolist())
            # Save obj values
            #actor_obj_1_model.extend(actual_value_1.tolist())
            
            end = time.time()
            
            # Save pred time
            agent1_time.extend([(end-start)/len(actual_value_1) for ic in range(len(actual_value_1))])
            
            batch += 1
        actor_obj_values_1 = np.array(actor_obj_values_1)
        critic_obj_values_1 = np.array(critic_obj_values_1)
        selected_items_1_model = np.array(selected_items_1_model)
        agent1_time = np.array(agent1_time)
    

    actor_model1.to('cpu')
    critic_model1.to('cpu')  
    del actor_model1, critic_model1  
    del first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h
    del actual_value, actual_value_list_1, log_likelihood_1, max_ix, selected_items_list_1, selected_items_1, pred_value_1, critic_loss_seq
    del critic_loss
    del src_1, lhs_1, rhs_1, obj_1
    get_torch_memory()
    torch.cuda.empty_cache()
    get_torch_memory()
    return actor_obj_values_1, critic_obj_values_1, selected_items_1_model, agent1_time



# test  module
def test_stage_2(actor_model2, critic_model2, iterator):
    
    mse = torch.nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    get_torch_memory()
    batch = 0 
    # Send bot agent 1 and 2 to device 
    actor_model2.to(device)
    critic_model2.to(device)
    #actor_model1.to(device)
    #critic_model1.to(device)
    
    # Agent 1 is in the training model
    #actor_model1.eval()
    #critic_model1.eval()
    
    # Agent 2 is in the testing mode
    actor_model2.eval()
    critic_model2.eval()
        
    actor_loss_seq = []
    critic_loss_seq = []
    
    #actor_obj_values_1 = []
    #critic_obj_values_1 = []
    actor_obj_values_2 = []
    critic_obj_values_2 = []
    
    #selected_items_1_model = []
    selected_items_2_model = []
    #actor_obj_1_model = []
    #actor_obj_2_model = []
    
    #agent1_time = []
    agent2_time = []


    with torch.no_grad():
        
        #first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h  = next(iter(test_dataloader))  
        # Start with first stage decisions. Then rescale data to select second stage decisions
        for first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h in iterator:       
            get_torch_memory()
            # Start timeing
            start = time.time()
            
            
            # Now we move on the agent 2 to get the second stage decisions
            # Reshape to not have scenarios
            # row indexes are Instance1Sce1,Instance1Sce2,Instance1Sce3,... 
            src_2 = second_stage_data.permute(0,3,1,2).reshape((-1,second_stage_data.shape[1],second_stage_data.shape[2]))
            # Check how scenario is distributed. It is Instance1Sce1,Instance1Sce2,Instance1Sce3,... 
            # We must also repeat all determinisic matrixes to have same shape as scenario ones
            # Because the shapes should be the same
            # if q is deterministic, extend like it has scenarios
            if q.shape[-1]==1:
                q = q.expand(-1, -1,second_stage_data.shape[-1])
            # if W is deterministic, extend like it has scenarios
            if W.shape[-1]==1:
                W = W.expand(-1, -1, -1,second_stage_data.shape[-1])
            # Now calculate the remaining h from the selected variables
            #lhs_sum = T*np.expand_dims(selected_items_1.unsqueeze(-1), axis=1)
            #lhs_sum = torch.sum(lhs_sum, axis=2)
            #remaining_h = h - lhs_sum
            # if remaining_h is deterministic, extend like it has scenarios
            if remaining_h.shape[-1]==1:
                remaining_h = remaining_h.expand(-1, -1,second_stage_data.shape[-1])

            # Reshape not to have scenarious
            # row indexes are Instance1Sce1,Instance1Sce2,Instance1Sce3,... 
            obj_2 = q.permute(0,2,1).reshape((-1,q.shape[1]))
            lhs_2 = W.permute(0,3,1,2).reshape((-1,W.shape[1],W.shape[2]))
            rhs_2 = remaining_h.permute(0,2,1).reshape((-1,remaining_h.shape[1]))
            
            # Src, lhs, rhs of second stage to device
            # Lhs, and rhs is needed for actor because of masks and feasibility
            src_2 = src_2.to(device)
            obj_2 = obj_2.to(device)
            lhs_2 = lhs_2.to(device) 
            rhs_2 = rhs_2.to(device)
            
            
            # Actor forward pass
            # Actor forward pass for stochastic decoding
            
            actual_value_list_2 = torch.empty(size = (model_config_testset['stochastic_pass_count'],src_2.shape[0],1))
            selected_items_list_2 = torch.empty(size = (model_config_testset['stochastic_pass_count'],*src_2[:,0].shape))
            log_likelihood_list_2 = torch.empty(size = (model_config_testset['stochastic_pass_count'],src_2.shape[0]))
            
            # in order to save memory during agent 1 training use half precision agent 2
            #if device=='cuda':
            #    actor_model2.half()
            #    src_2 = src_2.half()
            #    lhs_2 = lhs_2.half()
            #    rhs_2 = rhs_2.half()
            
            
            for ix in range(model_config_testset['stochastic_pass_count']):

                factor = 1000
                #If we have too many scenarios to fit GPU
                if model_config_testset['n_scenarios']>factor:
                    iters = int(model_config_testset['n_scenarios']/factor)
                    for iteration in range(iters):
                   
                        selected_items_iter, log_likelihood_iter = actor_model2(src_2[(iteration*factor):((iteration+1)*factor)], lhs_2[(iteration*factor):((iteration+1)*factor)], rhs_2[(iteration*factor):((iteration+1)*factor)])
                        
                        # Calculate true objective function value
                        actual_value = pointer_network.calculate_obj(obj_2[(iteration*factor):((iteration+1)*factor)], selected_items_iter)
                        selected_items_list_2[ix,(iteration*factor):((iteration+1)*factor)] = selected_items_iter
                        log_likelihood_list_2[ix,(iteration*factor):((iteration+1)*factor)] = log_likelihood_iter
                        actual_value_list_2[ix,(iteration*factor):((iteration+1)*factor)] = actual_value
                
                else:
                    selected_items, log_likelihood = actor_model2(src_2, lhs_2, rhs_2)
                
                
                    # Bring them back to float
                    #if device=='cuda':
                    #    selected_items = selected_items.to(torch.float32)
                    #    log_likelihood = log_likelihood.to(torch.float32)
                    
                    # Calculate true objective function value
                    actual_value = pointer_network.calculate_obj(obj_2, selected_items)
                    selected_items_list_2[ix] = selected_items
                    log_likelihood_list_2[ix] = log_likelihood
                    actual_value_list_2[ix] = actual_value

            # Reshape to have scenarios
            actual_value_list_2 = actual_value_list_2.reshape((actual_value_list_2.shape[0],second_stage_data.shape[0],second_stage_data.shape[-1]))
            log_likelihood_list_2 = log_likelihood_list_2.reshape((log_likelihood_list_2.shape[0],second_stage_data.shape[0],second_stage_data.shape[-1]))
            selected_items_list_2 = selected_items_list_2.reshape((selected_items_list_2.shape[0],second_stage_data.shape[0],second_stage_data.shape[-1],selected_items_list_2.shape[-1]))
            # Calculate the second stage obj value for all elements in batch
            actual_value_list_mean_2 = torch.mean(actual_value_list_2, axis=-1).unsqueeze(-1)
            
            # Now, for each scenario, find the max value independently
            # And then save them to a different tensor
            actual_value_max_list_2 = torch.empty(size = (second_stage_data.shape[0],second_stage_data.shape[-1]))
            log_likelihood_max_list_2 = torch.empty(size = (second_stage_data.shape[0],second_stage_data.shape[-1]))
            selected_items_max_list_2 = torch.empty(size = (second_stage_data.shape[0],second_stage_data.shape[-1],selected_items_list_2.shape[-1]))
           
            for ix in range(actual_value_list_2.shape[-1]):
                #Calculate the max index
                max_ix = torch.argmax(actual_value_list_2[:,:,ix], dim=0).unsqueeze(-1)
                max_ix.shape
                actual_value_list_2[:,:,ix].shape
                # Get the values corresponding to max index
                actual_value_2 = torch.gather(actual_value_list_2[:,:,ix].transpose(0,1).squeeze(-1), dim=1, index=max_ix).squeeze(-1)
                log_likelihood_2 = torch.gather(log_likelihood_list_2[:,:,ix].transpose(0,1), dim=1, index=max_ix).squeeze(-1)
                max_ix = max_ix.repeat(1, src_2.shape[-1]).unsqueeze(-1)
                selected_items_2 = torch.gather(selected_items_list_2[:,:,ix].permute(1,2,0), dim=2, index=max_ix).squeeze(-1)
                # Ensure that the selection is correct 
                #actual_value_temp = pointer_network.calculate_obj(obj_2.reshape((second_stage_data.shape[0],second_stage_data.shape[-1],-1))[:,ix], selected_items_2)
                #print(actual_value_temp.squeeze(-1) == actual_value_2)
                # Save to the nes tensor
                actual_value_max_list_2[:,ix] = actual_value_2
                selected_items_max_list_2[:,ix] = selected_items_2
                log_likelihood_max_list_2[:,ix] = log_likelihood_2
                
            
            
            # Critic 2 forward pass
            pred_value_2 = critic_model2(src_2)
            # Reshape to have scenarios
            pred_value_2 = pred_value_2.reshape((first_stage_data.shape[0],-1))
            # Calculate the second stage obj value for all elements in batch
            #pred_value_2_mean = torch.mean(pred_value_2, axis=-1).unsqueeze(-1)
            
            # Add selected items
            selected_items_2_model.extend(selected_items_max_list_2.numpy())
            # Add obj values from actor
            actor_obj_values_2.extend(actual_value_max_list_2.numpy())
            
            

            
            
            # Save objective function values
            #actor_obj_values_1.extend(actual_value_1.squeeze(-1).tolist())
            # Save objective function values from critic
            critic_obj_values_2.extend(pred_value_2.detach().cpu().numpy())
            # Save selected items
            #selected_items_1_model.extend(selected_items_1.tolist())
            # Save obj values
            #actor_obj_1_model.extend(actual_value_1.tolist())
            
            end = time.time()
            
            # Save pred time
            agent2_time.extend([(end-start)/len(actual_value_2) for ix in range(len(actual_value_2))])
            
            batch += 1
            
        actor_obj_values_2 = np.array(actor_obj_values_2)
        critic_obj_values_2 = np.array(critic_obj_values_2)
        selected_items_2_model = np.array(selected_items_2_model)
        agent2_time = np.array(agent2_time)
        
    actor_model2.to('cpu')
    critic_model2.to('cpu')
    del actor_model2, critic_model2  
    del first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h
    del src_2, lhs_2, rhs_2, obj_2
    del actual_value_max_list_2, selected_items_max_list_2, log_likelihood_max_list_2
    torch.cuda.empty_cache()
    return actor_obj_values_2, critic_obj_values_2, selected_items_2_model, agent2_time



def solve_random(iterator):
    

    
    selected_items_1 = []
    selected_items_2 = []
    obj_vals = []
    times = []


        
    #first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h  = next(iter(test_dataloader))  
    # Start with first stage decisions. Then rescale data to select second stage decisions
    for first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h in iterator:
          
        for i in range(c.shape[0]):
            
            random_start_time = time.time()
            c_instance = c[i].numpy()
            A_instance = A[i].numpy()
            b_instance = b[i].numpy()
            q_instance = q[i].numpy()
            T_instance = T[i].numpy()
            W_instance = W[i].numpy()
            h_instance = h[i].numpy()
            
            
            
            #first stage random
            x=np.zeros_like(c_instance)
            iter1 = True
            remaining_b = b_instance.copy()
            while iter1:
                ix = np.random.randint(0,c_instance.shape[0], size=1)
                
                # If all lhs is less than rhs
                if np.all(np.squeeze(A_instance[:,ix])<=remaining_b):
                    x[ix]=1
                    remaining_b -= np.squeeze(A_instance[:,ix])
                else:
                    iter1 = False
                        
            #second stage random
            n_secondstage_variables = len(W_instance[0])
            n_scenarios = max(len(q_instance[0]),len(T_instance[0,0]),len(W_instance[0,0]),len(h_instance[0])) 
            y=np.zeros(shape=(n_scenarios, n_secondstage_variables))
            h_index = [0 if h.shape[-1]==1 else ct for ct in range(n_scenarios)]
            T_index = [0 if T.shape[-1]==1 else ct for ct in range(n_scenarios)]
            W_index = [0 if W.shape[-1]==1 else ct for ct in range(n_scenarios)]
            for sc in range(n_scenarios):
                iter2 = True
                remaining_h = h_instance[:,h_index[sc]].copy()
                remaining_h -= np.sum(T_instance[:,:,T_index[sc]]*np.expand_dims(x,axis=0) ,axis=1)
                while iter2:
                    ix = np.random.randint(0,n_secondstage_variables, size=1)
                         
                    # If all lhs is less than rhs
                    if np.all(np.squeeze(W_instance[:,ix,W_index[sc]])<=remaining_h):
                        y[sc, ix]=1
                        remaining_h -= np.squeeze(W_instance[:,ix,W_index[sc]])
                    else:
                        iter2 = False
            # Now finalize
            random_end_time = time.time()
            
            # Objective function
            first_stage = np.sum(c_instance*x)
            # Known q
            if q_instance.shape[1]==1:
                second_stage  = np.sum([[q_instance[j,0]*y[s,j] for j in range(n_secondstage_variables)] for s in range(n_scenarios)])
            # Random q
            else:
                second_stage  = np.sum([[q_instance[j,s]*y[s,j] for j in range(n_secondstage_variables)] for s in range(n_scenarios)])
            random_obj = first_stage  + (second_stage/n_scenarios)
            random_time = random_end_time-random_start_time
            
            selected_items_1.append(x)
            selected_items_2.append(y)
            obj_vals.append(random_obj)
            times.append(random_time)
                
    selected_items_1 = np.array(selected_items_1)
    selected_items_2 = np.array(selected_items_2)
    obj_vals = np.array(obj_vals)
    times = np.array(times)
    
    return obj_vals, times, selected_items_1, selected_items_2
                             





def solve_pech(iterator):
    selected_items_1 = []
    selected_items_2 = []
    obj_vals = []
    times = []        
    #first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h  = next(iter(test_dataloader))  
    # Start with first stage decisions. Then rescale data to select second stage decisions
    for first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h in iterator:
          
        for instance in range(c.shape[0]):
            
            pech_start_time = time.time()
            c_instance = c[instance].numpy()
            A_instance = A[instance].numpy()
            b_instance = b[instance].numpy()
            q_instance = q[instance].numpy()
            T_instance = T[instance].numpy()
            W_instance = W[instance].numpy()
            h_instance = h[instance].numpy()
            n_first_variables = len(c_instance)
            n_secondstage_variables = len(W_instance[0])
            n_scenarios = max(len(q_instance[0]),len(T_instance[0,0]),len(W_instance[0,0]),len(h_instance[0])) 
            q_index = [0 if q.shape[-1]==1 else ct for ct in range(n_scenarios)]
            h_index = [0 if h.shape[-1]==1 else ct for ct in range(n_scenarios)]
            T_index = [0 if T.shape[-1]==1 else ct for ct in range(n_scenarios)]
            W_index = [0 if W.shape[-1]==1 else ct for ct in range(n_scenarios)]
            h_instance = np.tile(h_instance,(1,n_scenarios))
            
     
            #first stage random
            # If they are in the set, they are one 
            Ex=np.ones_like(c_instance)
            Ey=np.ones(shape=(n_scenarios, n_secondstage_variables))
            remaining_b = b_instance.copy()
            remaining_h = h_instance.copy()
            #return variables
            x=np.zeros_like(c_instance)
            y=np.zeros(shape=(n_scenarios, n_secondstage_variables))
            
            itervars=True
            while itervars:
                #Step 2
                effective_capacity_x = np.zeros_like(Ex)
                for i in range(n_first_variables):
                    if Ex[i]==1:
                        effective_capacity_x[i]=np.min(np.floor(remaining_b/A_instance[:,i]))
                        for s in range(n_scenarios):
                            value = np.min(np.floor(remaining_h[:,s]/T_instance[:,i,T_index[s]] ))
                            if value<effective_capacity_x[i]:
                                effective_capacity_x[i]=value
                effective_capacity_x = np.maximum(effective_capacity_x,0)
                
                effective_capacity_y = np.zeros_like(Ey)
                for s in range(n_scenarios):
                    for j in range(n_secondstage_variables):
                        if Ey[s,j]==1:
                            effective_capacity_y[s,j]=np.min(np.floor(remaining_h[:,s]/W_instance[:,j,W_index[s]]   ))
                effective_capacity_y = np.maximum(effective_capacity_y,0)            
                
                if np.mean(effective_capacity_x)==0 and np.mean(effective_capacity_y)==0:
                    break
                
                #Step 3
                c_value=0
                c_maxindex=None
                for i in range(n_first_variables):
                    if Ex[i]==1 and effective_capacity_x[i]>0:
                        if c_instance[i]>c_value:
                            c_value = c_instance[i]
                            c_maxindex = i
                            
                q_value = 0
                q_maxindex0,q_maxindex1 = None,None
                for s in range(n_scenarios):
                    for j in range(n_secondstage_variables):
                        if Ey[s,j]==1 and effective_capacity_y[s,j]>0:
                            if (q_instance[j,q_index[s]]/n_scenarios)>q_value:
                                q_value = (q_instance[j,q_index[s]]/n_scenarios)
                                q_maxindex0,q_maxindex1 = s,j
                                        
                                
                              
                if c_value>q_value:
                    #Add to decision variables
                    x[c_maxindex]=1
                    #Remove from possible
                    Ex[c_maxindex]=0
                    # Update remaining
                    remaining_b -= A_instance[:,c_maxindex]
                    remaining_h -= T_instance[:,c_maxindex,:]
                    #remaining_b -= np.squeeze(A_instance[:,c_maxindex])
                else:
                    #Add to decision variables
                    y[q_maxindex0,q_maxindex1]=1
                    #Remove from possible
                    Ey[q_maxindex0,q_maxindex1]=0
                    # Update remaining
                    remaining_h[:,q_maxindex0] -= W_instance[:,q_maxindex1,W_index[q_maxindex0]]
                

            
            # Objective function
            first_stage = np.sum(c_instance*x)
            # Known q
            if q_instance.shape[1]==1:
                second_stage  = np.sum([[q_instance[j,0]*y[s,j] for j in range(n_secondstage_variables)] for s in range(n_scenarios)])
            # Random q
            else:
                second_stage  = np.sum([[q_instance[j,s]*y[s,j] for j in range(n_secondstage_variables)] for s in range(n_scenarios)])
            pech_obj = first_stage  + (second_stage/n_scenarios)
            pech_end_time = time.time()
            pech_time = pech_end_time-pech_start_time
            
            selected_items_1.append(x)
            selected_items_2.append(y)
            obj_vals.append(pech_obj)
            times.append(pech_time)
                
    selected_items_1 = np.array(selected_items_1)
    selected_items_2 = np.array(selected_items_2)
    obj_vals = np.array(obj_vals)
    times = np.array(times)
    
    return obj_vals, times, selected_items_1, selected_items_2
                             
    
if __name__ == '__main__':
    model_config_trainset, model_config_testset, model_config_agent2, model_config_agent1 = load_path(trainset_config_path, testset_config_path, agent2_config_path, agent1_config_path)
    change_config_for_test(model_config_trainset, testset_config_path, model_config_testset, model_config_agent2, model_config_agent1)
    c_test, A_test, b_test, q_test, T_test, W_test, h_test, gurobi_obj, gurobi_time, gurobi_vars_1, gurobi_vars_2, gurobi_gap, gurobi_bound, sddip_obj, sddip_time, sddip_vars_1, sddip_vars_2, heur_obj, heur_time, heur_vars_1, heur_vars_2  = read_testset(testset_path,model_config_testset['n_problems_test'])
    
    agent2_actor, agent2_critic = build_agent2_model(model_config_agent2, agent2_path)
    agent1_actor, agent1_critic = build_agent1_model(model_config_agent1, agent1_path)
    test_dataloader = build_dataset(c_test, A_test, b_test, q_test, T_test, W_test, h_test, model_config_testset, testset_config_path)
    
   

    actor_obj_values_1, critic_obj_values_1, selected_items_1_model, agent1_time = test_stage_1(agent1_actor, agent1_critic, test_dataloader)
    gc.collect()
    test_dataloader = build_dataset(c_test, A_test, b_test, q_test, T_test, W_test, h_test, model_config_testset, testset_config_path, np.expand_dims(selected_items_1_model,axis=-1) )
    actor_obj_values_2, critic_obj_values_2, selected_items_2_model, agent2_time = test_stage_2(agent2_actor, agent2_critic, test_dataloader)

    gc.collect()
    actor_obj_values_2_mean = np.mean(actor_obj_values_2, axis=-1, keepdims = False)
    critic_obj_values_2_mean = np.mean(critic_obj_values_2, axis=-1, keepdims = False)
    

    


    actor_obj_values = actor_obj_values_1 + actor_obj_values_2_mean
    agent_time = agent1_time + agent2_time
    print('optgap ', abs(np.mean(100*(actor_obj_values-gurobi_obj)/gurobi_obj)))
    print('timeimp ', np.mean(gurobi_time/agent_time))
    print('accuracy first stage ',accuracy_score(gurobi_vars_1.flatten(), selected_items_1_model.flatten()))
    print('accuracy second stage ',accuracy_score(gurobi_vars_2.flatten(), selected_items_2_model.flatten()))
    
    random_obj, random_time, random_vars_1, random_vars_2 = solve_random(test_dataloader)
    print('random optgap ', abs(np.mean(100*(random_obj-gurobi_obj)/gurobi_obj)))
    pech_obj, pech_time, pech_vars_1, pech_vars_2 = solve_pech(test_dataloader)
    print('pech optgap ', abs(np.mean(100*(pech_obj-gurobi_obj)/gurobi_obj)))

    np.mean(random_vars_1)
    np.mean(random_vars_2)  
    np.mean(pech_vars_1)  
    np.mean(pech_vars_2)  
    
    # Write back the data with  heuristic
    with open(result_path+'.pkl', 'wb') as f:
        pkl.dump([actor_obj_values_1, critic_obj_values_1, selected_items_1_model, agent1_time,
                  actor_obj_values_2, critic_obj_values_2, selected_items_2_model, agent2_time,
                  gurobi_obj, gurobi_time, gurobi_vars_1, gurobi_vars_2, gurobi_gap, gurobi_bound,
                  sddip_obj, sddip_time, sddip_vars_1, sddip_vars_2,
                  heur_obj, heur_time, heur_vars_1, heur_vars_2,
                  random_obj, random_time, random_vars_1, random_vars_2,
                  pech_obj, pech_time, pech_vars_1, pech_vars_2],f)

    
