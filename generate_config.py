import platform
import sys
import os
if platform.system()=='Windows':
    TwoStageRLpath = 'C:/Users/dy234/Documents/paper3/TwoStageRL'
    msppypath = 'C:/Users/dy234/Documents/paper3/msppy-master'
    trainset_config_path='C:/Users/dy234/Documents/paper3/config_trainset1'
    testset_config_path='C:/Users/dy234/Documents/paper3/config_testset1'
    agent2_config_path='C:/Users/dy234/Documents/paper3/config_set1_agent2_model0'
    agent1_config_path='C:/Users/dy234/Documents/paper3/config_set1_agent1_model0'
else: 
    TwoStageRLpath = '/home/d/dy234/paper3/TwoStageRL'
    msppypath = '/home/d/dy234/paper3/msppy-master'
    #sys.path.append('/home/d/dy234/msppy-master')
    trainset_config_path=str(sys.argv[1])
    testset_config_path=str(sys.argv[2])
    agent2_config_path=str(sys.argv[3])
    agent1_config_path=str(sys.argv[4])    
if TwoStageRLpath not in sys.path:
    sys.path.append(TwoStageRLpath)
if msppypath not in sys.path:
    sys.path.append(msppypath)
    

import pickle as pkl




if trainset_config_path!='False':
    model_config_trainset={}
    model_config_trainset['n_secondstage_variables'] = 10
    model_config_trainset['n_secondstage_constraints'] = 5
    model_config_trainset['n_scenarios'] = 10
    model_config_trainset['n_firststage_variables'] = 10
    model_config_trainset['n_firststage_constraints'] = 5
    
    model_config_trainset['n_problems_train'] = 10000
    model_config_trainset['n_problems_valid'] = 100
    model_config_trainset['seed_train'] = 0
    model_config_trainset['seed_valid'] = 1
    model_config_trainset['first_stage_scaler'] = None
    model_config_trainset['second_stage_scaler'] = None
    
    model_config_trainset['random_q']=True
    model_config_trainset['random_T']=False
    model_config_trainset['random_W']=False # False for 0, True for 1 and 2
    model_config_trainset['random_h']=False
    model_config_trainset['mean_first_stage_decision']=None
    model_config_trainset['remaining_h_ratio'] = None
    
    
    
    print('Dataset config:')
    for key in model_config_trainset:
        print(key, ' : ', model_config_trainset[key])
        
    with open(trainset_config_path+ '.pkl', 'wb') as f:
        pkl.dump(model_config_trainset,f)



if testset_config_path!='False':
    model_config_testset={}
    model_config_testset['n_firststage_variables'] = 30
    model_config_testset['n_secondstage_variables'] = 30
    model_config_testset['n_firststage_constraints'] = 15
    model_config_testset['n_secondstage_constraints'] = 15
    model_config_testset['n_scenarios'] = 10
    
    model_config_testset['n_problems_test'] = 20
    model_config_testset['seed_test'] = 2
    model_config_testset['first_stage_scaler'] = None
    model_config_testset['second_stage_scaler'] = None
    model_config_testset['stochastic_pass_count'] = 20
    
    model_config_testset['random_q']=True
    model_config_testset['random_T']=False
    model_config_testset['random_W']=False
    model_config_testset['random_h']=False
    model_config_testset['mean_first_stage_decision']=None
    model_config_testset['remaining_h_ratio'] = None
    
    
    
    print('Dataset config:')
    for key in model_config_testset:
        print(key, ' : ', model_config_testset[key])
    
    if not os.path.exists(testset_config_path+ '.pkl'):
        
        with open(testset_config_path+ '.pkl', 'wb') as f:
            pkl.dump(model_config_testset,f)
    else:
        print('!!!\n\n!!!\n\n!!!\n\n!!!\n\n!!!\n\n!!!\n\n!!!\n\n!!!\n\n!!!\n\n!!!\n\n')
        print('Test set already exists')

if agent2_config_path!='False':
    setnumber = int(agent2_config_path.rsplit('_', 1)[0][-1])
    input_dim = 6 if setnumber==0 else (11 if setnumber==1 else 16)
    model_config_agent2 = {}
    
    # Agent 2 Actor
    model_config_agent2['actor_enc_hid_dim'] = 256#512
    model_config_agent2['actor_dec_hid_dim'] = 2*model_config_agent2['actor_enc_hid_dim']
    model_config_agent2['actor_n_layers'] = 2
    model_config_agent2['actor_n_directions'] = 2
    
    # Agent 2 Critic
    model_config_agent2['critic_enc_hid_dim'] = 64#256
    model_config_agent2['critic_dec_hid_dim'] = 2*model_config_agent2['critic_enc_hid_dim']
    model_config_agent2['critic_n_layers'] = 2
    model_config_agent2['critic_n_directions'] = 2
    
    
    model_config_agent2['bidirectional'] = True
    model_config_agent2['dropout_lstm'] = 0.4
    model_config_agent2['dropout_linear'] = 0.4
    model_config_agent2['softmax_temperature'] = 1.5
    model_config_agent2['logit_clip'] = True
    model_config_agent2['use_glimpse'] = True
    model_config_agent2['n_glimpses'] = 1 
    model_config_agent2['use_embedding'] = False
    model_config_agent2['emb_dim'] = 128
    model_config_agent2['decode_type'] = 'greedy'
    model_config_agent2['clip_value'] = 1
    model_config_agent2['early_stop_tolerance'] = 3000
    model_config_agent2['n_epochs'] = 10000
    model_config_agent2['batch_size'] = 2048
    model_config_agent2['input_dim'] = input_dim
    model_config_agent2['actor_learning_rate'] = 0.1
    model_config_agent2['critic_learning_rate'] = 0.1
    model_config_agent2['lr_factor'] = 0.5
    model_config_agent2['lr_patience'] = 2000
    model_config_agent2['use_training_permutation'] = True
    model_config_agent2['sample_scenario_for_training'] = True
    
    model_config_agent2['no_improvement_epoch_stoppping_tolerance'] = 3000
    
    print('\n\nAgent 2 config:')
    for key in model_config_agent2:
        print(key, ' : ', model_config_agent2[key])
        
    
    with open(agent2_config_path+ '.pkl', 'wb') as f:
        pkl.dump(model_config_agent2,f)

if agent1_config_path!='False':
    setnumber = int(agent1_config_path.rsplit('_', 1)[0][-1])
    input_dim = 6 if setnumber==0 else (11 if setnumber==1 else 16)
    model_config_agent1 = {}
    
    # Agent 1 Actor
    model_config_agent1['actor_enc_hid_dim'] = 256#33
    model_config_agent1['actor_dec_hid_dim'] = 2*model_config_agent1['actor_enc_hid_dim']
    model_config_agent1['actor_n_layers'] = 2
    model_config_agent1['actor_n_directions'] = 2
    
    # Agent 1 Critic
    model_config_agent1['critic_enc_hid_dim'] = 64#45
    model_config_agent1['critic_dec_hid_dim'] = 2*model_config_agent1['critic_enc_hid_dim']
    model_config_agent1['critic_n_layers'] = 2
    model_config_agent1['critic_n_directions'] = 2
    
    
    model_config_agent1['bidirectional'] = True
    model_config_agent1['dropout_lstm'] = 0.4
    model_config_agent1['dropout_linear'] = 0.4
    model_config_agent1['softmax_temperature'] = 1.5
    model_config_agent1['logit_clip'] = True
    model_config_agent1['use_glimpse'] = True
    model_config_agent1['n_glimpses'] = 1 
    model_config_agent1['use_embedding'] = False
    model_config_agent1['emb_dim'] = 128
    model_config_agent1['decode_type'] = 'greedy'
    model_config_agent1['clip_value'] = 1
    model_config_agent1['early_stop_tolerance'] = 3000
    model_config_agent1['n_epochs'] = 10000
    model_config_agent1['batch_size'] = 512
    model_config_agent1['input_dim'] = input_dim
    model_config_agent1['actor_learning_rate'] = 0.1
    model_config_agent1['critic_learning_rate'] = 0.1
    model_config_agent1['lr_factor'] = 0.5
    model_config_agent1['lr_patience'] = 2000
    model_config_agent1['use_training_permutation'] = True
    model_config_agent1['no_improvement_epoch_stoppping_tolerance'] = 3000
    
    print('\n\nAgent 1 config:')
    for key in model_config_agent1:
        print(key, ' : ', model_config_agent1[key])
        
    with open(agent1_config_path+ '.pkl', 'wb') as f:
        pkl.dump(model_config_agent1,f)
