import platform
import sys

if platform.system()=='Windows':
    TwoStageRLpath = 'C:/Users/dy234/Documents/paper3/TwoStageRL'
    msppypath = 'C:/Users/dy234/Documents/paper3/msppy-master'
    trainset_config_path='C:/Users/dy234/Documents/paper3/config_trainset1'
    agent2_config_path='C:/Users/dy234/Documents/paper3/config_set1_agent2_model0'
    agent2_path='C:/Users/dy234/Documents/paper3/set1_agent2_model0'
    agent1_config_path='C:/Users/dy234/Documents/paper3/config_set1_agent1_model0'
    agent1_path='C:/Users/dy234/Documents/paper3/set1_agent1_model0'
else: 
    TwoStageRLpath = '/home/d/dy234/paper3/TwoStageRL'
    msppypath = '/home/d/dy234/paper3/msppy-master'
    #sys.path.append('/home/d/dy234/msppy-master')
    trainset_config_path=str(sys.argv[1])
    agent2_config_path=str(sys.argv[2])
    agent2_path=str(sys.argv[3])
    agent1_config_path=str(sys.argv[4])
    agent1_path=str(sys.argv[5])

    
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
import gc


def load_path(trainset_config_path, agent2_config_path, agent1_config_path):
    
    with open(trainset_config_path+ '.pkl', 'rb') as f:
        model_config_trainset = pkl.load(f)
    
    with open(agent2_config_path+ '.pkl', 'rb') as f:
        model_config_agent2 = pkl.load(f)
        
    with open(agent1_config_path+ '.pkl', 'rb') as f:
        model_config_agent1 = pkl.load(f)
    
    if trainset_config_path.endswith('2'):
        model_config_agent2['batch_size'] = int(model_config_agent2['batch_size']/2)
        
    return model_config_trainset, model_config_agent2, model_config_agent1

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
        critic = torch.nn.DataParallel(actor,device_ids=device_ids)
    
    if torch.cuda.is_available():
        actor.load_state_dict(torch.load(agent2_path+'_actor.pt'))
        critic.load_state_dict(torch.load(agent2_path+'_critic.pt'))
    else:
        actor.load_state_dict(torch.load(agent2_path+'_actor.pt',map_location=torch.device('cpu')))
        critic.load_state_dict(torch.load(agent2_path+'_critic.pt',map_location=torch.device('cpu')))
        print('Change model read name')

    actor.eval() 
    critic.eval()
    
    
    print(f'The actor of agent 2 has {count_parameters(actor):,} trainable parameters')
    print(f'The critic of agent 2 has {count_parameters(critic):,} trainable parameters')
    return actor, critic

def build_agent1_model(model_config_agent1):

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
        critic = torch.nn.DataParallel(actor,device_ids=device_ids)
    
    
    print(f'The actor of agent 1 has {count_parameters(actor):,} trainable parameters')
    print(f'The critic of agent 1 has {count_parameters(critic):,} trainable parameters')
    return actor, critic

def build_dataset(model_config_trainset, trainset_config_path, model_config_agent1):
    train_dataset = dataset.TwoStage_MKP_Dataset(model_config_trainset['n_problems_train'], 
                                                 model_config_trainset['n_firststage_variables'], 
                                                 model_config_trainset['n_firststage_constraints'], 
                                                 model_config_trainset['n_secondstage_variables'], 
                                                 model_config_trainset['n_secondstage_constraints'], 
                                                 model_config_trainset['n_scenarios'], 
                                                 model_config_trainset['seed_train'], 
                                                 model_config_trainset['first_stage_scaler'], 
                                                 model_config_trainset['second_stage_scaler'], 
                                                 model_config_trainset['random_q'], 
                                                 model_config_trainset['random_T'], 
                                                 model_config_trainset['random_W'], 
                                                 model_config_trainset['random_h'], 
                                                 model_config_trainset['mean_first_stage_decision'],
                                                 model_config_trainset['remaining_h_ratio'])
    train_dataloader = DataLoader(train_dataset, batch_size=model_config_agent1['batch_size'], shuffle=True)
    model_config_trainset['second_stage_scaler']= train_dataset.second_stage_scaler
    model_config_trainset['first_stage_scaler']= train_dataset.first_stage_scaler
    model_config_trainset['mean_first_stage_decision'] = train_dataset.mean_first_stage_decision
    model_config_trainset['remaining_h_ratio'] = train_dataset.remaining_h_ratio
    with open(trainset_config_path+ '.pkl', 'wb') as f:
        pkl.dump(model_config_trainset,f)
    valid_dataset = dataset.TwoStage_MKP_Dataset(model_config_trainset['n_problems_valid'], 
                                                 model_config_trainset['n_firststage_variables'], 
                                                 model_config_trainset['n_firststage_constraints'], 
                                                 model_config_trainset['n_secondstage_variables'], 
                                                 model_config_trainset['n_secondstage_constraints'], 
                                                 model_config_trainset['n_scenarios'], 
                                                 model_config_trainset['seed_valid'], 
                                                 model_config_trainset['first_stage_scaler'], 
                                                 model_config_trainset['second_stage_scaler'], 
                                                 model_config_trainset['random_q'], 
                                                 model_config_trainset['random_T'], 
                                                 model_config_trainset['random_W'], 
                                                 model_config_trainset['random_h'], 
                                                 model_config_trainset['mean_first_stage_decision'],
                                                 model_config_trainset['remaining_h_ratio'])
    valid_dataloader = DataLoader(valid_dataset, batch_size=model_config_agent1['batch_size'], shuffle=False)
    
    return train_dataloader, valid_dataloader



def build_optimizers(actor, critic, model_config_agent1):
    
    actor_optimizer = torch.optim.Adam(actor.parameters(),lr = model_config_agent1['actor_learning_rate'] )
    #actor_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_optimizer, mode='max', factor=model_config_agent1['lr_factor'], patience=model_config_agent1['lr_patience'], threshold=0.0001, min_lr=1e-10, verbose=True)
    actor_lr_scheduler = torch.optim.lr_scheduler.StepLR(actor_optimizer, gamma=0.5, step_size=500)
    
    critic_optimizer = torch.optim.Adam(critic.parameters(),lr = model_config_agent1['critic_learning_rate'])
    #critic_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, mode='min', factor=model_config_agent1['lr_factor'], patience=model_config_agent1['lr_patience'], threshold=0.0001, min_lr=1e-10, verbose=True)
    critic_lr_scheduler = torch.optim.lr_scheduler.StepLR(critic_optimizer, gamma=0.5, step_size=500)
    
    return actor_optimizer, actor_lr_scheduler, critic_optimizer, critic_lr_scheduler

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




# Training module
def train(actor_model2, critic_model2, actor_model1, critic_model1, actor_optim, critic_optim, iterator, clip_grad, use_training_permutation, epoch, device):
    
    mse = torch.nn.MSELoss()
    
    batch = 0 

    

    # Agent 1 is in the training model
    actor_model1.train()
    critic_model1.train()
    
    
        
    actor_loss_seq = []
    critic_loss_seq = []
    
    actor_obj_values = []
    critic_obj_values = []



    #first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h  = next(iter(train_dataloader))  
    #for src, obj, lhs, rhs in iterator:
    for first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h in iterator:       
        
        # Forward pass for Agent 1
        # Send agent 1 to device 
        actor_model1.to(device)
        critic_model1.to(device)
        

        
        # Reorder items if using training permutation
        if use_training_permutation:
            permutation = torch.randperm(first_stage_data.shape[-1])
            src = first_stage_data[:,:,permutation]
            obj = c[:,permutation]
            lhs = A[:,:,permutation]
        rhs = b
    
        # Reset gradients
        actor_optim.zero_grad()
        critic_optim.zero_grad()
        
        
        # Src, lhs, rhs to device
        # Lhs, and rhs is needed for actor because of masks and feasibility
        src = src.to(device)
        obj = obj.to(device)
        lhs = lhs.to(device) 
        rhs = rhs.to(device)

        # Actor forward pass
        selected_items, log_likelihood = actor_model1(src, lhs, rhs)
        

        
        # Update Critic 1
        # Critic forward pass
        pred_value = critic_model1(src)
        # Calculate true objective function value
        actual_value = pointer_network.calculate_obj(obj, selected_items)
        # Critic 1 MSE loss
        critic_loss = mse(actual_value, pred_value)
        # Compute gradients of critic 1 
        critic_loss.backward()
        # Clip gradients of critic 1
        torch.nn.utils.clip_grad_norm_(critic_model1.parameters(), max_norm = clip_grad)
    	# Update parameters of critic 1
        critic_optim.step()
        # Save critic 1 loss
        critic_loss_seq.append(critic_loss.item())
        
        # Save objective function values
        actor_obj_values.extend(actual_value.squeeze(-1).tolist())
        # Save objective function values from critic
        critic_obj_values.extend(pred_value.detach().squeeze(-1).tolist())
        
        # Now we move on the agent 2 to get the second stage decisions
        # Send agent 2 to device and agent 1 to cpu to save cuda memory    
        src = src.to('cpu')
        obj = obj.to('cpu')
        lhs = lhs.to('cpu') 
        rhs = rhs.to('cpu')
        actor_model1.to('cpu')
        critic_model1.to('cpu')
        actor_model2.to(device)
        critic_model2.to(device)

        # free inside reserved
        torch.cuda.empty_cache()

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
        #print(T.shape,selected_items.shape)
        #print(selected_items.unsqueeze(-1).unsqueeze(1).shape)
        lhs_sum = T*selected_items.cpu().unsqueeze(-1).unsqueeze(1)#np.expand_dims(selected_items.unsqueeze(-1), axis=1)
        lhs_sum = torch.sum(lhs_sum, axis=2)
        remaining_h = h - lhs_sum
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
        # Agent 2 is in the testing mode
        actor_model2.eval()
        critic_model2.eval()

        
        # in order to save memory during agent 1 training use half precision agent 2
        if device=='cuda':
            actor_model2.half()
            src_2 = src_2.half()
            lhs_2 = lhs_2.half()
            rhs_2 = rhs_2.half()

        # Actor forward pass
        with torch.no_grad():
            selected_items_2, log_likelihood_2 = actor_model2(src_2, lhs_2, rhs_2)
        
        # Bring them back to float
        if device=='cuda':
            selected_items_2 = selected_items_2.to(torch.float32)
            log_likelihood_2 = log_likelihood_2.to(torch.float32)

        # Calculate the second stage actual obj with the second stage items
        actual_value_2 = pointer_network.calculate_obj(obj_2, selected_items_2)
        # Reshape to have scenarios
        actual_value_2 = actual_value_2.reshape((first_stage_data.shape[0],-1))
        # Calculate the second stage obj value for all elements in batch
        actual_value_2 = torch.mean(actual_value_2, axis=-1).unsqueeze(-1)
        
        # in order to save memory during agent 1 training use half precision agent 2
        if device=='cuda':
            critic_model2.half()
        # Critic 2 forward pass
        with torch.no_grad():
            pred_value_2 = critic_model2(src_2)
        if device=='cuda':    
            pred_value_2 = pred_value_2.to(torch.float32)
        # Reshape to have scenarios
        pred_value_2 = pred_value_2.reshape((first_stage_data.shape[0],-1))
        # Calculate the second stage obj value for all elements in batch
        pred_value_2 = torch.mean(pred_value_2, axis=-1).unsqueeze(-1)
        
        # Send agent 1 back to device and agent 2 to cpu to save cuda memory
        actor_model2.to('cpu')
        critic_model2.to('cpu')
        actor_model1.to(device)
        critic_model1.to(device)
        # free inside reserved
        torch.cuda.empty_cache()
        

        # Update Actor 1
        advantage = actual_value + actual_value_2 - pred_value.detach() - pred_value_2.detach()
        # Actor loss
        actor_loss = (advantage.squeeze(-1) * log_likelihood).mean()
        # Compute gradients of actor
        actor_loss.backward()
        # Clip gradients of actor
        torch.nn.utils.clip_grad_norm_(actor_model1.parameters(), max_norm = clip_grad)
    	   # Update parameters of actor
        actor_optim.step()
        # Save actor loss
        actor_loss_seq.append(actor_loss.item())
        
        # First stage or overall onjective to stop training?
        
        
        
    
        if batch % np.ceil(len(iterator)/10)== 0:
            print('epoch: {}, train_batch_id: {}, critic_loss: {}'.format(epoch, batch, critic_loss.item()))
        
        batch += 1
    
    del actor_model2, actor_model1, critic_model2, critic_model1
    del first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h
    del actor_loss, critic_loss
    del src_2, lhs_2, rhs_2, src, lhs, rhs
    torch.cuda.empty_cache()
    return actor_loss_seq, critic_loss_seq, sum(actor_loss_seq)/batch, sum(critic_loss_seq)/batch, actor_obj_values, critic_obj_values





# Validation module
def validate(actor_model2, critic_model2, actor_model1, critic_model1, actor_scheduler, critic_scheduler, iterator, epoch, device):
    
    mse = torch.nn.MSELoss()
    
    batch = 0 
    # Send bot agent 1 and 2 to device 
    actor_model2.to(device)
    critic_model2.to(device)
    actor_model1.to(device)
    critic_model1.to(device)
    
    # Agent 1 is in the training model
    actor_model1.eval()
    critic_model1.eval()
    
    # Agent 2 is in the testing mode
    actor_model2.eval()
    critic_model2.eval()
        
    actor_loss_seq = []
    critic_loss_seq = []
    
    actor_obj_values = []
    critic_obj_values = []


    with torch.no_grad():

        #first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h  = next(iter(train_dataloader))  
        #for src, obj, lhs, rhs in iterator:
        for first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h in iterator:       
            
            
            # Forward pass for Agent 1
            src = first_stage_data
            obj = c
            lhs = A
            rhs = b
            
            # Src, lhs, rhs to device
            # Lhs, and rhs is needed for actor because of masks and feasibility
            src = src.to(device)
            obj = obj.to(device)
            lhs = lhs.to(device) 
            rhs = rhs.to(device)
            # Actor forward pass
            selected_items, log_likelihood = actor_model1(src, lhs, rhs)
    
            # Pred from Critic 1
            # Critic forward pass
            pred_value = critic_model1(src)
            # Calculate true objective function value
            actual_value = pointer_network.calculate_obj(obj, selected_items)
            # Critic 1 MSE loss
            critic_loss = mse(actual_value, pred_value)
            # Save critic 1 loss
            critic_loss_seq.append(critic_loss.item())
            
            # Save objective function values
            actor_obj_values.extend(actual_value.squeeze(-1).tolist())
            # Save objective function values from critic
            critic_obj_values.extend(pred_value.detach().squeeze(-1).tolist())
            
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
            lhs_sum = T*selected_items.cpu().unsqueeze(-1).unsqueeze(1)#np.expand_dims(selected_items.unsqueeze(-1), axis=1)
            lhs_sum = torch.sum(lhs_sum, axis=2)
            remaining_h = h - lhs_sum
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
            
            # in order to save memory during agent 1 training use half precision agent 2
            if device=='cuda':
                actor_model2.half()
                src_2 = src_2.half()
                lhs_2 = lhs_2.half()
                rhs_2 = rhs_2.half()
                
            
            # Actor forward pass
            
            selected_items_2, log_likelihood_2 = actor_model2(src_2, lhs_2, rhs_2)
            
            # Bring them back to float
            if device=='cuda':
                selected_items_2 = selected_items_2.to(torch.float32)
                log_likelihood_2 = log_likelihood_2.to(torch.float32)
            # Calculate the second stage actual obj with the seco
            # Calculate the second stage actual obj with the second stage items
            actual_value_2 = pointer_network.calculate_obj(obj_2, selected_items_2)
            # Reshape to have scenarios
            actual_value_2 = actual_value_2.reshape((first_stage_data.shape[0],-1))
            # Calculate the second stage obj value for all elements in batch
            actual_value_2 = torch.mean(actual_value_2, axis=-1).unsqueeze(-1)
            
            
            # in order to save memory during agent 1 training use half precision agent 2
            if device=='cuda':
                critic_model2.half()
            
        
    
            # Critic 2 forward pass
            pred_value_2 = critic_model2(src_2)
            if device=='cuda':
                pred_value_2 = pred_value_2.to(torch.float32)
            # Reshape to have scenarios
            pred_value_2 = pred_value_2.reshape((first_stage_data.shape[0],-1))
            # Calculate the second stage obj value for all elements in batch
            pred_value_2 = torch.mean(pred_value_2, axis=-1).unsqueeze(-1)
    
            
            # First stage or overall onjective to stop training?
            
        
                    
            batch += 1
    
        print('epoch: {}, validation_critic_loss: {}'.format(epoch, sum(critic_loss_seq)/batch))
        print('epoch: {}, validation_actor_mean_obj: {}'.format(epoch, np.mean(actor_obj_values)))
        #actor_scheduler.step(np.mean(actor_obj_values))
        #critic_scheduler.step(sum(critic_loss_seq)/batch)
        actor_scheduler.step()
        critic_scheduler.step()

    del actor_model2, actor_model1, critic_model2, critic_model1  
    del first_stage_data, second_stage_data, c, A, b, q, T, W, h, remaining_h
    del critic_loss
    del src_2, lhs_2, rhs_2, src, lhs, rhs
    torch.cuda.empty_cache()
    return actor_loss_seq, critic_loss_seq, sum(actor_loss_seq)/batch, sum(critic_loss_seq)/batch, actor_obj_values, critic_obj_values

   
def early_stop_minimize(val_loss, no_improvement, tolerance = 100):
    #At least 2 of epoch should pass for comparison
    if len(val_loss) >= 2:        
        #If the val_loss has not decreased, increase no_improvement counter
        if val_loss[-1]>=val_loss[-2] :
            no_improvement[0] += 1
        #Else val_loss decreased, reset no_improvement counter
        else: 
            no_improvement[0] = 0
        
        #If the val_loss did not decrease for tolerance epochs, stop training
        if no_improvement[0]==tolerance:
            print('Early stopping at epoch ', len(val_loss))
            return 'early_stop'

def early_stop_maximize(val_loss, no_improvement, tolerance = 100):
    #At least 2 of epoch should pass for comparison
    if len(val_loss) >= 2:        
        #If the val_loss has not increased, increase no_improvement counter
        if val_loss[-1]<=val_loss[-2] :
            no_improvement[0] += 1
        #Else val_loss increased, reset no_improvement counter
        else: 
            no_improvement[0] = 0
        
        #If the val_loss did not increase for tolerance epochs, stop training
        if no_improvement[0]==tolerance:
            print('Early stopping at epoch ', len(val_loss))
            return 'early_stop'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_training_plot(train_actor_mean_obj_values_list, valid_actor_mean_obj_values_list, agent2_path):
    fig, ax = plt.subplots()
    epochs = range(1, len(train_actor_mean_obj_values_list) + 1)
    ax.plot(epochs, train_actor_mean_obj_values_list, 'g', label='Training obj')
    ax.plot(epochs, valid_actor_mean_obj_values_list, 'y', label='Validation obj')
    ax.set(title='Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    fig.savefig(agent2_path+'-loss.png', dpi=1000)
    plt.close(fig)   



def training_loop(actor_model2, critic_model2, actor_model1, critic_model1, actor_optimizer, actor_lr_scheduler, critic_optimizer, critic_lr_scheduler, train_dataloader, valid_dataloader, model_config_agent1, agent2_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #mse_loss = torch.nn.MSELoss()
    best_valid_obj = float('-inf')
    train_actor_loss_sequence = []
    train_critic_loss_sequence = []
    train_actor_mean_loss = []
    train_critic_mean_loss = []
    train_actor_obj_values_list = []
    train_critic_obj_values_list = []
    train_actor_mean_obj_values_list = []
    train_critic_mean_obj_values_list = []
    valid_actor_loss_sequence = []
    valid_critic_loss_sequence = []
    valid_actor_mean_loss = []
    valid_critic_mean_loss = []
    valid_actor_obj_values_list = []
    valid_critic_obj_values_list = []
    valid_actor_mean_obj_values_list = []
    valid_critic_mean_obj_values_list = []
    train_times = []
    no_improvement_epochs = [0]
    
    for epoch in range(1,model_config_agent1['n_epochs']):
        print('Running epoch :',epoch)
        if device=='cuda':
            get_torch_memory()
        start_time = time.time()
      
        i,j,k,l,m,n  = train(actor_model2, critic_model2, actor_model1, critic_model1, actor_optimizer, critic_optimizer, train_dataloader, model_config_agent2['clip_value'], model_config_agent2['use_training_permutation'], epoch, device)
        gc.collect()
        train_actor_loss_sequence.append(i)
        train_critic_loss_sequence.append(j)
        train_actor_mean_loss.append(k)
        train_critic_mean_loss.append(l)
        train_actor_obj_values_list.append(m)
        train_critic_obj_values_list.append(n)
        train_actor_mean_obj_values_list.append(np.mean(train_actor_obj_values_list[-1]))
        train_critic_mean_obj_values_list.append(np.mean(train_critic_obj_values_list[-1]))
        
        i,j,k,l,m,n = validate(actor_model2, critic_model2, actor_model1, critic_model1, actor_lr_scheduler, critic_lr_scheduler, valid_dataloader, epoch, device)
        gc.collect()
        valid_actor_loss_sequence.append(i)
        valid_critic_loss_sequence.append(j)
        valid_actor_mean_loss.append(k)
        valid_critic_mean_loss.append(l)
        valid_actor_obj_values_list.append(m)
        valid_critic_obj_values_list.append(n)
        valid_actor_mean_obj_values_list.append(np.mean(valid_actor_obj_values_list[-1]))
        valid_critic_mean_obj_values_list.append(np.mean(valid_critic_obj_values_list[-1]))
        
        
        print('Actor Learning rate: ',get_lr(actor_optimizer))
        print('Critic Learning rate: ',get_lr(critic_optimizer))
        
        end_time = time.time()
        
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        train_times.append(end_time - start_time)
        
        #Save best model
        if valid_actor_mean_obj_values_list[-1] > best_valid_obj:
            best_valid_obj = valid_actor_mean_obj_values_list[-1]
            torch.save(actor_model1.state_dict(), agent1_path+'_actor.pt')
            torch.save(critic_model1.state_dict(), agent1_path+'_critic.pt')
            
        print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Actor Loss: {train_actor_mean_loss[-1]:.2f} ')
        print(f'\t Val. Actor Loss: {valid_actor_mean_loss[-1]:.2f} ')
        print(f'\tTrain Critic Loss: {train_critic_mean_loss[-1]:.2f} ')
        print(f'\t Val. Critic Loss: {valid_critic_mean_loss[-1]:.2f} ')
        print(f'\tTrain Critic Obj: {train_actor_mean_obj_values_list[-1]:.2f} ')
        print(f'\t Val. Critic Obj: {valid_actor_mean_obj_values_list[-1]:.2f} ')
        
        #with open(agent1_path+'_trainstats.pkl', 'wb') as f:
        #     pkl.dump([train_actor_loss_sequence, train_critic_loss_sequence, train_actor_mean_loss, train_critic_mean_loss, train_actor_obj_values_list, train_critic_obj_values_list, train_actor_mean_obj_values_list, train_critic_mean_obj_values_list, valid_actor_loss_sequence, valid_critic_loss_sequence, valid_actor_mean_loss, valid_critic_mean_loss, valid_actor_obj_values_list, valid_critic_obj_values_list, valid_actor_mean_obj_values_list, valid_critic_mean_obj_values_list, train_times],f)

           
            
        flag=early_stop_maximize(valid_actor_mean_obj_values_list, no_improvement_epochs, int(model_config_agent1['early_stop_tolerance']/3) )  
        if flag =='early_stop':
            break
            

    
if __name__ == '__main__':
    model_config_trainset, model_config_agent2, model_config_agent1 = load_path(trainset_config_path, agent2_config_path, agent1_config_path)
    agent2_actor, agent2_critic = build_agent2_model(model_config_agent2, agent2_path)
    agent1_actor, agent1_critic = build_agent1_model(model_config_agent1)
    print('agent1_actor is cuda ',next(agent1_actor.parameters()).is_cuda)
    print('agent1_critic is cuda ',next(agent1_critic.parameters()).is_cuda)
    print('agent2_actor is cuda ',next(agent2_actor.parameters()).is_cuda)
    print('agent2_critic is cuda ',next(agent2_critic.parameters()).is_cuda)
    train_dataloader, valid_dataloader = build_dataset(model_config_trainset, trainset_config_path, model_config_agent1)
    actor_optimizer, actor_lr_scheduler, critic_optimizer, critic_lr_scheduler = build_optimizers(agent1_actor, agent1_critic, model_config_agent1)
    training_loop(agent2_actor, agent2_critic, agent1_actor, agent1_critic, actor_optimizer, actor_lr_scheduler, critic_optimizer, critic_lr_scheduler, train_dataloader, valid_dataloader, model_config_agent2, agent2_path)

  




    
