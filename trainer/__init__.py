class AgentTrainer(object):
    def __init__(self, name, model, obs_shape, act_space, args):
        raise NotImplemented()

    def action(self, obs):
        raise NotImplemented()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def update(self, agents):
        raise NotImplemented()

    def get_state_dict(self):
        """ Return all the trainable parameters.
        
        Raises
        ------
        NotImplemented
            Not implemented error.
        """
        raise NotImplemented()

    def restore_state(self, ckpt):
        """ Restore all the trainable parameters.
        
        Parameters
        ----------
        ckpt : dict
            Contain all information for restoring trainable parameters.
        
        Raises
        ------
        NotImplemented
            Not implemented error.
        """
        raise NotImplemented()

    def eval(self):
        """ Switch all models inside the trainer into eval mode.
        
        Raises
        ------
        NotImplemented
            Not implemented error.
        """
        raise NotImplemented()

    def train(self):
        """ Switch all models inside the trainer into train mode.
        
        Raises
        ------
        NotImplemented
            Not implemented error
        """
        raise NotImplemented()