import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn.functional as F

class Greedy(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, masked_logits):
		return torch.argmax(torch.softmax(masked_logits, dim = -1), dim=-1)

class Stochastic(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, masked_logits):
		return torch.multinomial(torch.log_softmax(masked_logits, dim = -1).exp(), 1).long().squeeze(1)
        

class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, n_layers, n_directions, dropout_lstm, use_embedding, emb_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.n_layers = n_layers
        self.n_directions = n_directions
        self.bidirectional = True if self.n_directions==2 else False
        self.use_embedding = use_embedding
        self.emb_dim = emb_dim
        if self.use_embedding:
            self.input_to_LSTM_dim = self.emb_dim
        else:
            self.input_to_LSTM_dim = self.input_dim
        self.rnn = nn.LSTM(self.input_to_LSTM_dim, enc_hid_dim, n_layers, dropout = dropout_lstm, bidirectional=self.bidirectional)
        #rnn = nn.LSTM(input_dim, enc_hid_dim, n_layers, dropout = dropout_lstm, bidirectional = bidirectional)
        
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-0.08,0.08)
            if module.bias is not None:
                module.bias.data.zero_()


    
    def forward(self, src):
            
        #src = [batch_size, input_dim, len_seq]
        
        src = src.transpose(1,2).transpose(0,1)
            
        #src = [len_seq, batch_size, input_dim]
        
        outputs, (hidden, cell) = self.rnn(src)
        #outputs, (hidden, cell) = rnn(src)
       
        #outputs = [len_seq, batch_size, enc_hid_dim * n directions]
        #hidden = [n_layers * n_directions, batch_size, enc_hid_dim]
        #cell = [n_layers * n_directions, batch_size, enc_hid_dim]
        
        #outputs are always from the top hidden layers with both directions merged
        
        return outputs, hidden, cell
    

class Pointer(nn.Module):
    def __init__(self, dec_hid_dim, softmax_temperature, logit_clip):
        super(Pointer, self).__init__()
        self.Wref = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        self.Wq = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        self.softmax_temperature = softmax_temperature
        self.logit_clip = logit_clip
        
        #Working old pointer
        #self.W = nn.Linear(dec_hid_dim*2, dec_hid_dim, bias=False)
        #self.v = nn.Linear(dec_hid_dim,1, bias=False)
        #out = torch.tanh(self.W(torch.cat((encoder_hidden.permute(1, 0, 2),decoder_hidden.unsqueeze(1).expand(-1, encoder_hidden.shape[0], -1)), 2)))
        #return F.softmax(self.v(out).squeeze(-1),dim=1)
        
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-0.08,0.08)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, decoder_hidden, encoder_hidden, mask):
        #Inputs are the top hidden layers of encoder and decoder
        #decoder_hidden = [batch_size, dec_hid_dim]
        #encoder_hidden = [len_seq, batch_size, dec_hid_dim]
    
        #Logit clipping and softmax temperature
        out = torch.tanh(self.Wref(encoder_hidden.permute(1, 0, 2))+self.Wq(decoder_hidden.unsqueeze(1).expand(-1, encoder_hidden.shape[0], -1))).clamp(min=-self.logit_clip, max=self.logit_clip) 
        # out = [batch_size, len_seq, dec_hid_dim]
        logits = self.v(out).squeeze(-1)
        # logits = [batch_size, len_seq]
        #Masking here
        masked_logits = logits - (1e8*mask)
        # masked_logits = [batch_size, len_seq] 
        #out = torch.tanh(torch.cat((self.Wref(encoder_hidden.permute(1, 0, 2)),self.Wq(decoder_hidden.unsqueeze(1).expand(-1, encoder_hidden.shape[0], -1))), 2))
        return F.softmax(masked_logits / self.softmax_temperature,dim=1), masked_logits
    
class Glimpse(nn.Module):
    def __init__(self, dec_hid_dim):
        super(Glimpse, self).__init__()
        self.Wref_g = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        self.Wq_g = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        self.v_g = nn.Linear(dec_hid_dim, 1, bias=False)
        
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-0.08,0.08)
            if module.bias is not None:
                module.bias.data.zero_()

 
    def forward(self, decoder_hidden, encoder_hidden, mask=0):
        #Inputs are the top hidden layers of encoder and decoder
        #decoder_hidden = [batch_size, dec_hid_dim]
        #encoder_hidden = [len_seq, batch_size, dec_hid_dim]
        out = torch.tanh(self.Wref_g(encoder_hidden.permute(1, 0, 2))+self.Wq_g(decoder_hidden.unsqueeze(1).expand(-1, encoder_hidden.shape[0], -1)))
        # out = [batch_size, len_seq, dec_hid_dim]
        logits = self.v_g(out).squeeze(-1)
        # logits = [batch_size, len_seq]
        #Masking here
        #masked_logits = logits - (1e8*mask)
        masked_logits = logits - (torch.tensor(1e8,dtype=torch.float32)*mask)
        #masked_logits = masked_logits.to(decoder_hidden.dtype)
        # masked_logits = [batch_size, len_seq]
        #out = torch.tanh(torch.cat((self.Wref(encoder_hidden.permute(1, 0, 2)),self.Wq(decoder_hidden.unsqueeze(1).expand(-1, encoder_hidden.shape[0], -1))), 2))
        scores =  F.softmax(masked_logits ,dim=1).unsqueeze(1)
        # We must outout with the correct dtype since we might be using half precision
        scores =  scores.to(decoder_hidden.dtype)
        # scores = [batch_size, 1, len_seq]
        return torch.bmm(scores, encoder_hidden.permute(1, 0, 2)).squeeze(1)
    
class Decoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout_lstm, pointer, use_glimpse, n_glimpses, glimpse, use_embedding, emb_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.n_layers = n_layers
        self.activation=nn.ReLU()  
        self.emb_dim = emb_dim
        if use_embedding:
            self.input_to_LSTM_dim = emb_dim
        else:
            self.input_to_LSTM_dim = input_dim
        self.rnn = nn.LSTM(self.input_to_LSTM_dim, dec_hid_dim, n_layers, dropout = dropout_lstm, bidirectional=False)
        #self.rnn = nn.LSTM(input_dim, dec_hid_dim, n_layers, dropout = dropout_lstm, bidirectional=False)
        self.pointer = pointer
        self.glimpse = glimpse
        self.use_glimpse = use_glimpse
        self.n_glimpses = n_glimpses
        if not self.use_glimpse:
            self.n_glimpses = 0
        
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-0.08,0.08)
            if module.bias is not None:
                module.bias.data.zero_()
 
    
    def forward(self, input_decoder, encoder_outputs, hidden, cell, mask):     
        #input_decoder = [batch_size, input_dim]
        #hidden = [n_layers, batch_size, dec_hid_dim]
        #cell = [n_layers, batch_size, dec_hid_dim]
        
        input_decoder = input_decoder.unsqueeze(0)
        #input_decoder = [1, batch_size, input_dim]
        output, (hidden, cell) = self.rnn(input_decoder, (hidden, cell))

        #output = [1, batch_size, dec_hid_dim]
        #hidden = [n_layers, batch_size, dec_hid_dim]
        #cell = [n_layers, batch_size, dec_hid_dim]

        output = output.squeeze(0)
        
        #output = [batch_size, dec_hid_dim]
        # Might change here to glimpse to not the have the mask
        for _ in range(self.n_glimpses):
            output = self.glimpse(output, encoder_outputs, mask)
 
                
        #Calculate the probabilities using pointer mechanism
        probabilities, masked_logits = self.pointer(output, encoder_outputs, mask)

        # probabilities = [batch_size, len_seq]
        # masked_logits = [batch_size, len_seq]
        
        assert not torch.isnan(probabilities).any()
       
       
        return probabilities, masked_logits, hidden, cell


class PointerNetwork(nn.Module):
    def __init__(self, encoder, pointer, glimpse, decoder, decode_type, device):
        super().__init__()
        
        self.encoder = encoder
        self.pointer = pointer
        self.glimpse = glimpse
        self.decoder = decoder
        self.device = device
        self.sigmoid=nn.Sigmoid()
        # A linear layer for embeddings if using embeddings
        if self.encoder.use_embedding:
            #self.emb_layer = nn.Linear(in_features=self.encoder.input_dim, out_features=self.encoder.emb_dim, bias=False)
            self.emb_layer = nn.Conv2d(in_channels=1, out_channels=self.encoder.emb_dim, kernel_size=(1,self.encoder.input_dim), stride=1)
        # Identity layer just passes on layer if not using embeddings
        else:
            self.emb_layer = nn.Identity()
            
        if decode_type == 'greedy':
            self.select_item = Greedy()
            print('Initializing greedy decoder')
        elif decode_type == 'stochastic':
            self.select_item = Stochastic()
            print('Initializing stochastic decoder')
        else:
            print('Invalid decode_type')
        
    def forward(self, src, lhs, rhs):
        
        #src = [batch_size, input_dim, len_seq] (len_seq is n_items)
        #lhs = [batch_size, n_constraints, len_seq]
        lhs = lhs.transpose(1,2)
        #lhs = [batch_size, len_seq, n_constraints]
        #rhs = [batch_size, n_constraints]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = src.shape[0]
        len_seq = src.shape[2] # This is the number of items
        steps = len_seq# This is number of steps of decoder
        
        
        #Embedding layer
        src = self.emb_layer(src)
            
        #tensor to store the selected items
        selected_items = torch.zeros((batch_size, len_seq), device = self.device, dtype=src.dtype)
        #tensor to store the masked logits for decoding sequence
        masked_logits_seq = torch.zeros(size = (batch_size, steps, len_seq), device = self.device, dtype=src.dtype)
        #tensor to store the log prob for decoding sequence
        log_prob_seq = torch.zeros(size = (batch_size, steps, len_seq), device = self.device, dtype=src.dtype)
        #tensor to store selected_items for decoding sequence as sequence
        selected_items_seq = torch.zeros(size = (batch_size, steps), device = self.device, dtype = torch.int64)
        
        # tensor to store the stopping info of problems in the batch
        # Takes value 1 if the problem is not finished, takes value 0 otherwise
        # "Anti"mask for problem info
        remaining_problems = torch.ones((batch_size,1) ,device = self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_outputs, hidden, cell = self.encoder(src)
#enc_outputs.shape  
#hidden_reshaped.shape
        #hidden= torch.cat((torch.cat([hidden[0,:, :], hidden[1,:,:]], dim=1).unsqueeze(0) ,torch.cat([hidden[2,:, :], hidden[3,:,:]], dim=1).unsqueeze(0)),dim=0).to(torch.float)
        #cell=torch.cat((torch.cat([cell[0,:, :], cell[1,:,:]], dim=1).unsqueeze(0) ,torch.cat([cell[2,:, :], cell[3,:,:]], dim=1).unsqueeze(0)),dim=0).to(torch.float)
        #hidden_reshaped, cell_reshaped= hidden,cell
        # Probability mask initialized as all zeros, i.e., no masking
        mask = torch.zeros((batch_size, len_seq), device = self.device, dtype=src.dtype)

        hidden_reshaped=torch.empty(size=(self.encoder.n_layers,batch_size,self.decoder.dec_hid_dim), device=self.device, dtype=src.dtype)
        cell_reshaped=torch.empty(size=(self.encoder.n_layers,batch_size,self.decoder.dec_hid_dim), device=self.device, dtype=src.dtype)
        for idx in range(self.encoder.n_layers):
            hidden_reshaped[idx,:,:]=torch.cat([hidden[(2*idx),:, :], hidden[((2*idx)+1),:,:]], dim=1).unsqueeze(0)
            cell_reshaped[idx,:,:]=torch.cat([cell[(2*idx),:, :], cell[((2*idx)+1),:,:]], dim=1).unsqueeze(0)

        #first input to the decoder is the <sos> tokens
        input_to_decoder = torch.zeros_like(src[:,:,0])
        # input_to_decoder = [batch_size, input_dim]
        
        #print(hidden_reshaped)

        for t in range(steps):

            # Forward pass on the decoder to receive probabilities of predictions
            probabilities, masked_logits, hidden_reshaped, cell_reshaped = self.decoder(input_to_decoder, enc_outputs, hidden_reshaped, cell_reshaped, mask)
                                                                                      
            #print(probabilities[0,:])
            
            # Keep track of logits
            masked_logits_seq[:,t,:] = masked_logits
            # Choose the max index
            # Here is greedy
            #max_index = torch.argmax(probabilities, dim=-1) 
            # Here is stochastic
            #max_index = torch.multinomial(torch.log_softmax(masked_logits, dim = -1).exp(), 1).long().squeeze(1)
            max_index = self.select_item(masked_logits)
            
            #max_index = [batch_size]
            index_tensor = max_index.unsqueeze(-1).unsqueeze(-1).repeat(1, self.decoder.input_to_LSTM_dim, 1)
            #index_tensor = [batch_size, input_dim, 1]
            input_to_decoder = torch.gather(src, dim=2, index=index_tensor).squeeze(-1)
            # input_to_decoder = [batch_size, input_dim]


            selected_items_seq[:,t] = max_index 
            log_prob_seq[:,t,:] = torch.log_softmax(masked_logits, dim = -1)*remaining_problems
            
            
            # Update the mask
            current_decision = torch.zeros((batch_size,len_seq), device = self.device).scatter_(dim = 1, index = max_index.unsqueeze(1), value = 1)
            # Mask works in 2 ways:
            # 1) Can not select alread selected_items
            # 2) Can not select items that will make problem infeasible
            selected_items += current_decision*remaining_problems
            # remaining_weight is rhs-weights*selected_items
            #selected_items, lhs, rhs = torch.from_numpy(x),torch.from_numpy(a),torch.from_numpy(b)
            # This is max formulation for knapsack
            remaining_weight = rhs - torch.bmm(selected_items.unsqueeze(1), lhs).squeeze(1)
            # Feasible by weight have value 1 if iitem weight for that constraint is less than remaining weight, otherwise it is 0
            feasible_by_weight = torch.gt(remaining_weight.unsqueeze(1), lhs)
            # feasible items must be feasible by weight and it must not be a priorly selected item
            feasible_items = torch.logical_and(feasible_by_weight,1-selected_items.unsqueeze(2))
            # if an item is infeasible in a single constraint, then it shoud be impossible for all constraints, therefore select min
            # feasible items have value 1 which is opposite of mask
            feasible_items = torch.min(feasible_items, dim=-1).values.long() #Might change here with torch.amin
            # mask has vale 1 for masked (infeasible) items, invert feasible items
            mask = 1-feasible_items
            
            # update remaining problems info. Takes value 1 if the problem is not finished, takes value 0 otherwise
            # 1 means that there are at least some feasible items, therefore we can continue 
            # 0 means that all values in the feasible_items are zero, therefore this problem has no feasible item that we can use
            remaining_problems = torch.max(feasible_items, dim=-1).values.long().unsqueeze(-1)
            
            # we should stop decoding loop if we have no remaining problems left in batch
            if torch.max(remaining_problems) == torch.tensor(0):
                break
        #Only return up to the processed step. No need to retun empty parts of the tensors
        masked_logits_seq = masked_logits_seq[:,:(t+1),:]
        selected_items_seq = selected_items_seq[:,:(t+1)] 
        log_prob_seq = log_prob_seq[:,:(t+1),:]
        log_likelihood = self.calculate_log_likelihood(log_prob_seq,selected_items_seq)
        
        return selected_items, log_likelihood
    def calculate_log_likelihood(self, seq_log_prob, seq_selected_items):
        #https://github.com/Rintarooo/TSP_DRL_PtrNet/blob/master/actor.py
        '''args:
            seq_log_prob: [batch, steps, len_seq], log probs as sequence
            seq_selected_items: [batch, steps], predicted items in a sequence
            return: [batch,]
        '''
        selected_seq_log_prob = torch.gather(input = seq_log_prob, dim = 2, index = seq_selected_items[:,:,None])
        #selected_seq_log_prob is sequence of selected items's log probs. [batch, steps, 1]
        return torch.sum(selected_seq_log_prob.squeeze(-1), 1)


def calculate_obj(c, x):
    # Calculate the objective function value to use in reward as batch
    # c =[batch_size, len_seq]
    # x =[batch_size, len_seq]
    return torch.bmm(c.unsqueeze(1), x.unsqueeze(2)).squeeze(-1)
    


#encoder_critic = Encoder(input_dim, enc_hid_dim, n_layers, n_directions, dropout_lstm, use_embedding, emb_dim)
#glimpse_critic = Glimpse(dec_hid_dim)
class Critic(nn.Module):
    def __init__(self, encoder, glimpse, n_glimpses, dropout_linear, device):
        super().__init__()
        
        self.encoder = encoder
        self.glimpse = glimpse
        self.hidden = self.encoder.enc_hid_dim*self.encoder.n_directions 
        self.decoder = nn.Sequential(
                        nn.Linear(self.hidden, self.hidden),
                        nn.ReLU(),
                        nn.Dropout(dropout_linear),
                        nn.Linear(self.hidden, 1)
                        )
        
        self.n_glimpses = n_glimpses
        self.device = device
        # A linear layer for embeddings if using embeddings
        if self.encoder.use_embedding:
            self.emb_layer = nn.Linear(self.encoder.input_dim, self.encoder.emb_dim, bias=False)
        # Identity layer just passes on layer if not using embeddings
        else:
            self.emb_layer = nn.Identity()

        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-0.08,0.08)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, src):     
        #src = [batch_size, input_dim, len_seq]
        
        batch_size = src.shape[0]
        len_seq = src.shape[2] # This is the number of items
        
        #Embedding layer
        src = self.emb_layer(src)
                   
        encoder_outputs, hidden, cell = self.encoder(src)

        #encoder_outputs = [len_seq, batch_size, hidden]
        # hidden is enc_hid_dim * n_directions
        #hidden = [n_layers * n_directions, batch_size, enc_hid_dim]
        #cell = [n_layers * n_directions, batch_size, enc_hid_dim]
        
        decoder_hidden  = torch.cat([hidden[-2,:, :], hidden[-1,:,:]], dim=1)
        #decoder_hidden = [batch_size, hidden]

        # No mask
        mask = torch.zeros((batch_size, len_seq), device = self.device)
        
        #Attention as glimpse
        for _ in range(self.n_glimpses):
            output = self.glimpse(decoder_hidden, encoder_outputs, mask)
            #output = [batch_size, hidden]
            
        assert not torch.isnan(output).any()
        
        value = self.decoder(output)

        return value
