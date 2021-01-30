import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trainer import AgentTrainer
from trainer.maddpg.replay_buffer import ReplayBuffer
from trainer.maddpg.distributions import SoftCategoricalPdType


def _make_update_exp(source, target):
    """ Use values of parameters from the source model to update values of parameters from the target model. Each update just change values of paramters from the target model slightly, which aims to provide relative stable evaluation. Note that structures of the two models should be the same. 
    
    Parameters
    ----------
    source : torch.nn.Module
        The model which provides updated values of parameters.
    target : torch.nn.Module
        The model which receives updated values of paramters to update itself.
    """
    polyak = 1e-2
    for tgt, src in zip(target.named_parameters(recurse=True), source.named_parameters(recurse=True)):
        assert src[0] == tgt[0] # The identifiers should be the same
        tgt[1].data = polyak * src[1].data + (1.0 - polyak) * tgt[1].data


def _compute_q_input_size(obs_shape_n, act_space_n, index, local_q=False):
    """ Compute the size of input for q_network.
    
    Parameters
    ----------
    obs_shape_n : list
        Each element 'i' is the size of observation belonging to the corresponding agent 'i'.
    act_space_n : list
        Each element 'i' is the action space of the corresponding agent 'i'.
    index : int
        The index of the current agent, which is used to identified different agents. 
    local_q : bool, optional
        Use the joint observation and action if True, otherwise only the local observation and action. By default False
    
    Returns
    -------
    int
        The size of input for q_network.
    """
    input_size = 0
    if local_q:
        input_size += obs_shape_n[index]
        input_size += act_space_n[index].n
    else:
        input_size += np.sum(obs_shape_n)
        input_size += np.sum([act_space.n for act_space in act_space_n])

    return input_size


def _get_q_input(obs_n, act_n, index, local_q=False):
    """ Return the input for q_network.
    
    Parameters
    ----------
    obs_n : list
        Each element is a tensor with size [batch_size, num_local_observation].
    act_n : list
        Each element is a tensor with size [batch_size, num_local_action]
    index : int
        The index of the current agent, which is used to identified different agents. 
    local_q : bool, optional
        Use the joint observation and action if True, otherwise only the local observation and action. By default False.
    
    Returns
    -------
    Tensor
        The size is [batch_size, num_local_observation + num_local_action] if local_q==True, otherwise [batch_size, num_joint_observation + num_joint_action]
    """
    if local_q:
        return torch.cat([obs_n[index], act_n[index]], dim=1)
    else:
        _obs_n = torch.cat(obs_n, dim=1)
        return torch.cat((torch.cat(tuple(obs_n), dim=1), torch.cat(tuple(act_n), dim=1)), dim=1)

def _get_action(p_network, obs_n, index=None, device='cpu'):
    """ Return the action computed by p_network.
    
    Parameters
    ----------
    p_network : torch.nn.Module
        The model which is used to compute action taken by the agent.
    obs_n : list
        Each element is a tensor with size [batch_size, num_local_observation].
    index : index, optional
        The index of the current agent, which is used to identified different agents. By default None
    
    Returns
    -------
    Tensor
        The tensor with  size [num_action, ].
    """
    if index is None:   # During execution
        p_input = obs_n
    else:               # During training
        p_input = obs_n[index]
    p = p_network(p_input)
    num_act = p.size()[1]
    act_pdtype = SoftCategoricalPdType(num_act)
    act_pd = act_pdtype.pdfromflat(p, device)
    act = act_pd.sample()

    return act


def _p_train(p_network, q_network, obs_n, act_n, index, local_q=False, device='cpu'):
    """ Define the way for training p_network.
    
    Parameters
    ----------
    p_network : torch.nn.Module
        The target model to be trained.
    q_network : torch.nn.Module
        The model which evalutes how well the p_network performs.
    obs_n : list
        Each element is a tensor with size [batch_size, num_local_observation].
    act_n : list
        Each element is a tensor with size [batch_size, num_local_action]
    index : int
        The index of the current agent, which is used to identified different agents.
    local_q : bool, optional
        Use the joint observation and action if True, otherwise only the local observation and action. By default False.
    
    Returns
    -------
    tuple
        The loss for p_network, the output of q_network. All are tensors.
    """

    # Compute p_network
    p_input = obs_n[index]
    cur_p = p_network(p_input)              # -->[batch_size, num_local_action]

    # Compute q_network
    num_act = cur_p.size()[1]
    act_pdtype = SoftCategoricalPdType(num_act)
    cur_act_pd = act_pdtype.pdfromflat(cur_p, device=device)

    _act_n = []
    _log_p = []
    for i in range(len(act_n)):
        if i != index:
            p = p_network(obs_n[i])
            act_pd = act_pdtype.pdfromflat(p, device=device)
            act_i = act_pd.sample()
            _act_n.append(torch.tensor(act_i.cpu().clone().detach().numpy()).to(device))
            _log_p.append(torch.mean(cur_act_pd.logp(act_n[i])))
        else:
            _act_n.append(cur_act_pd.sample())
    
    q_input = _get_q_input(obs_n, _act_n, index, local_q)   
    q_value = q_network(q_input)        # -->[batch_size, 1]

    # Compute loss for p_network
    pg_loss = -torch.mean(q_value)
    p_reg = torch.mean(torch.pow(cur_p, 2))
    # infer_loss = 0
    # for log_p in _log_p:
    #     infer_loss -= log_p
    # loss = pg_loss + p_reg * 1e-3 + 1e-2 * infer_loss
    loss = pg_loss + p_reg * 1e-3

    return loss, q_value                # All are tensors

def infer_train(p_network, obs_n, act_n, index, device='cpu', beta=1e-2):
    p_input = obs_n[index]
    cur_p = p_network(p_input) 
    num_act = cur_p.size()[1]
    act_pdtype = SoftCategoricalPdType(num_act)
    cur_act_pd = act_pdtype.pdfromflat(cur_p, device=device)

    _log_p = []
    for i in range(len(act_n)):
        if i != index:
            _log_p.append(torch.mean(cur_act_pd.logp(act_n[i])))

    infer_loss = 0
    for log_p in _log_p:
        infer_loss -= log_p
    
    infer_loss *= beta
    return infer_loss


def _q_train(q_network, target_q_network, obs_n, act_n, reward, obs_next_n, act_next_n, index, local_q=False, num_sample=1, gamma=0.95, done=False, device='cpu'):
    """ Define the way for training q_network.
    
    Parameters
    ----------
    q_network : torch.nn.Module
        The target model to be trained.
    target_q_network : torch.nn.Module
        The model which computes relatively stable Q-Value.
    obs_n : list
        Each element is a tensor with size [batch_size, num_local_observation].
    act_n : list
        Each element is a tensor with size [batch_size, num_local_action].
    reward : float
        The immediate reward after action taken.
    obs_next_n : [type]
        Each element is a tensor with size [batch_size, num_local_observation], representing observation after action taken.
    act_next_n : [type]
        Each element is a tensor with size [batch_size, num_local_action].
    index : int
        The index of the current agent, which is used to identified different agents.
    local_q : bool, optional
        Use the joint observation and action if True, otherwise only the local observation and action. By default False.
    num_sample : int, optional
        Use multiple samples to obtain relatively stable evaluation, by default 1
    gamma : float, optional
        The discount factor, by default 0.95
    done : bool, optional
        Whether task has finished or not, by default False
    
    Returns
    -------
    tuple
        The loss for q_network, the output of q_network and the output of target_q_network. All are tensors.
    """    
    # Compute q_network
    q_input = _get_q_input(obs_n, act_n, index, local_q)
    q_value = q_network(q_input)[:,0]

    # Compute target_q_network
    target_q = 0.0
    for i in range(num_sample):
        target_q_input = _get_q_input(obs_next_n, act_next_n, index, local_q)
        target_q_next = target_q_network(target_q_input)
        if device == 'cpu':
            target_q_next = target_q_next.clone().detach().numpy()[:,0]
        else:
            target_q_next = target_q_next.cpu().clone().detach().numpy()[:,0]
        target_q += reward + gamma * (1.0 - done) * target_q_next
    target_q /= num_sample
    target_q = torch.tensor(target_q, dtype=torch.float32).to(device)

    # Compute loss for q_network
    q_loss = torch.mean(torch.pow(q_value - target_q, 2))
    q_reg = torch.mean(torch.pow(q_value, 2))
    loss = q_loss  + q_reg * 1e-3

    return loss, q_value, target_q      # All are tensors


class _MLP(nn.Module):
    def __init__(self, input=69, output=5, num_units=64):
        super(_MLP, self).__init__()
        self.fc1 = nn.Linear(input, num_units)
        self.fc2 = nn.Linear(num_units, num_units)
        # self.fc3 = nn.Linear(num_units, num_units)
        # self.fc4 = nn.Linear(num_units, output)
        self.fc3 = nn.Linear(num_units, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        x = self.fc3(x)

        return x


class MADDPGMergeAgentTrainer(AgentTrainer):
    def __init__(self, name, obs_shape_n, act_space_n, agent_index, args, local_q=False):
        """ Initialize an instance for MADPPGAgentTrainer.
        
        Parameters
        ----------
        AgentTrainer : Class
            The base class that define a universal structure of trainers.
        name : str
            The name of this trainer, such as in the form of 'agent_i' where i represents the current index.
        obs_shape_n : list
            Each element 'i' is the size of observation belonging to the corresponding agent 'i'.
        act_space_n : list
            Each element 'i' is the action space of the corresponding agent 'i'.
        agent_index : int
            The index of the current agent, which is used to identified different agents.
        args : Namespace
            The object which contains necessary data for training.
        local_q : bool, optional
             Use the joint observation and action if True, otherwise only the local observation and action. By default False.
        """
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.local_q = local_q
        self.num_act = act_space_n[agent_index].n

        # Define main networks
        # Local observation of the agent
        p_input_size = obs_shape_n[agent_index][0]
        p_output_size = act_space_n[agent_index].n  # Action taken by the agent
        q_input_size = _compute_q_input_size(obs_shape_n, act_space_n, agent_index, local_q=local_q)
        q_output_size = 1   # For Q-Value only

        self.p_network = _MLP(input=p_input_size, output=p_output_size, num_units=args.num_units).to(args.device)
        self.q_network = _MLP(input=q_input_size, output=q_output_size, num_units=args.num_units).to(args.device)
        self.target_p_network = _MLP(input=p_input_size, output=p_output_size, num_units=args.num_units).to(args.device)
        self.target_q_network = _MLP(input=q_input_size, output=q_output_size, num_units=args.num_units).to(args.device)

        self.p_optimizer = optim.Adam(self.p_network.parameters(), lr=args.lr)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr)

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        """ Compute the action to be taken.
        
        Parameters
        ----------
        obs : np.array
            The local observation of the current agent, with size [observation, ]
        
        Returns
        -------
        np.array
            The action to be taken.
        """
        _obs = torch.tensor(obs[None], dtype=torch.float32).to(self.args.device)
        action = _get_action(self.p_network, _obs, device=self.args.device)
        if self.args.device == 'cpu':
            action = action.clone().detach().numpy()[0]
        else:
            action = action.cpu().clone().detach().numpy()[0]
        return action

    def experience(self, obs, act, reward, new_obs, done, terminal):
        """ Add new experience into the replay buffer.
        
        Parameters
        ----------
        obs : np.array
            The local observation of the current agent.
        act : np.array
            The local action taken by the current agent.
        reward : [type]
            The immediate reward after action taken.
        new_obs : np.array
            The local observation of the current agent after action taken.
        done : bool
            Whether task has finished or not.
        terminal : bool
            Whether one episode ends or not.
        """
        self.replay_buffer.add(obs, act, reward, new_obs, float(done))

    def preupdate(self):
        """ Reset the index of the replay buffer for sampling.
        """
        self.replay_sample_index = None

    def update(self, agents, t):
        """ Update trainable parameters of the trainer.
        
        Parameters
        ----------
        agents : list
            It contains all the agents and each element is an instance of MADDPGAgentTrainer.
        t : int
            The amount of steps taken for training.
        """
        # Replay buffer is not large enough
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            return {'p_loss': np.inf, 'q_loss': np.inf, 'q-value': np.nan, 'target_q_value': np.nan}
        if not t % 100 == 0:  # Only update every 100 steps
            return {'p_loss': np.inf, 'q_loss': np.inf, 'q-value': np.nan, 'target_q_value': np.nan}

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)

        latest_index = self.replay_buffer.make_latest_index(self.args.update_gap)
        latest_obs_n = []
        latest_act_n = []
        for i in range(self.n):
            obs, act, _, _, _ = agents[i].replay_buffer.sample_index(latest_index)
            latest_obs_n.append(torch.from_numpy(obs).float().to(self.args.device))
            latest_act_n.append(torch.from_numpy(act).float().to(self.args.device))
        infer_loss = infer_train(self.p_network, latest_obs_n, latest_act_n, index=self.agent_index, device=self.args.device, beta=self.args.beta)
        self.p_optimizer.zero_grad()
        infer_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.p_network.parameters(), self.args.grad_norm_clip)
        self.p_optimizer.step()

        # Collect a batch of replay samples from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, reward, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(torch.from_numpy(obs).float().to(self.args.device))
            obs_next_n.append(torch.from_numpy(obs_next).float().to(self.args.device))
            act_n.append(torch.from_numpy(act).float().to(self.args.device))
        obs, act, reward, obs_next, done = self.replay_buffer.sample_index(index)

        act_next_n = []
        for i in range(self.n):
            act_next_i = _get_action(agents[i].target_p_network, obs_next_n, index=i, device=self.args.device)
            act_next_i = torch.tensor(act_next_i.cpu().clone().detach().numpy()).to(self.args.device)
            act_next_n.append(act_next_i)
        

        # Compute q_loss
        q_loss, q_value, target_q_value = _q_train(self.q_network, self.target_q_network, obs_n, act_n, reward, obs_next_n, act_next_n, self.agent_index, gamma=self.args.gamma, done=done, device=self.args.device)
        
        # Compute p_loss
        p_loss, _ = _p_train(self.p_network, self.q_network, obs_n, act_n, self.agent_index, self.local_q, device=self.args.device)

        # Debug
        # print('p_loss: {}'.format(p_loss))
        
        # BP for q_loss and update q_network
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.p_network.parameters(), self.args.grad_norm_clip)
        self.q_optimizer.step()

        # BP for p_loss and update p_network
        self.p_optimizer.zero_grad()
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.args.grad_norm_clip)
        self.p_optimizer.step()

        # Update target_p_network and taget_q_network
        _make_update_exp(self.p_network, self.target_p_network)
        _make_update_exp(self.q_network, self.target_q_network)

        info = {'p_loss':p_loss, 'q_loss': q_loss, 'q-value': torch.mean(q_value), 'target_q_value':torch.mean(target_q_value)}

        return info

    def get_state_dict(self):
        """Return all the trainable parameters.
        
        Returns
        -------
        dict
            The dictionary which contains all traiable parameters.
        """
        state_dict = {
            'p_network': self.p_network.state_dict(), 'target_p_network': self.target_p_network.state_dict(),
            'q_network': self.q_network.state_dict(), 'target_q_network': self.target_q_network.state_dict(),
            'p_optimizer': self.p_optimizer.state_dict(), 'q_optimizer': self.q_optimizer.state_dict()
            }
        return state_dict
    
    def restore_state(self, ckpt):
        """Restore all the trainable parameters.
        
        Parameters
        ----------
        ckpt : dict
            Contain all information for restoring trainable parameters.
        """
        self.p_network.load_state_dict(ckpt['p_network'])
        self.q_network.load_state_dict(ckpt['q_network'])
        self.target_p_network.load_state_dict(ckpt['target_p_network'])
        self.target_q_network.load_state_dict(ckpt['target_q_network'])
        self.p_optimizer.load_state_dict(ckpt['p_optimizer'])
        self.q_optimizer.load_state_dict(ckpt['q_optimizer'])

    def eval(self):
        """ Switch all models inside the trainer into eval mode.
        """
        self.p_network.eval()
        self.q_network.eval()
        self.target_p_network.eval()
        self.target_q_network.eval()
    
    def train(self):
        """ Switch all models inside the trainer into train mode.
        """
        self.p_network.train()
        self.q_network.train()
        self.target_p_network.train()
        self.target_q_network.train()


