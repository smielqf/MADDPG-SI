import os
import torch


#################### Save and Restore models #############

def generate_checkpoint_path(directory, num_episodes):
    """ Genrate the path of checkpoints in the same form.
    
    Parameters
    ----------
    directory : str
        The directory which contains checkpoints
    num_episode : int
        The index which is used to identified different checkpoints.
    
    Returns
    -------
    str
        The path of checkpoints.
    """
    path = os.path.join(directory, 'ckpt_{}.pt'.format(num_episodes))
    return path

def save_state(trainers, path, info=None):
    """ Save trainable parameters of trainers.
    
    Parameters
    ----------
    trainers : list or tuple
        Each element of the list is an instance that contains trainable parameters of networks.
    path : str
        Where the saved file should be placed, such as "dir/xxxx.pt".
    info : dict, optional
        Other information to be saved, by default None and for futural use.
    """

    save_dict = {}
    for i, agent in enumerate(trainers):
        state_dict = agent.get_state_dict()
        save_dict['agent_{}'.format(i)] = state_dict
    torch.save(save_dict, path)

def restore_state(trainers, path, info=None, map_location='cpu'):
    """ Restore trainable parameters of trainers.
    
    Parameters
    ----------
    trainers : list or tuple
        Each element of the list is an instance that contains trainable parameters of networks.
    path : str
        Where the saved file should be placed, such as "dir/xxxx.pt".
    info : dict, optional
        Other information to be saved, by default None and for futural use.
    """
    checkpoint = torch.load(path, map_location=map_location)
    for i, agent in enumerate(trainers):
        ckpt = checkpoint['agent_{}'.format(i)]
        agent.restore_state(ckpt)

def latest_checkpoint(path):
    """ Return the latest checkpoint and the corresponding index.
    
    Parameters
    ----------
    path : str
        The directory which contains all checkpoints.
    
    Returns
    -------
    tuple
        The latest checkpoint and the corresponding index.
    """
    import re
    latest_index = 0
    _latest_checkpoint = ''
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            matched = re.match(r'ckpt_([0-9]+.pt)', f)
            if matched:
                strings = matched.groups()
                for s in strings:
                    _matched = re.match(r'[0-9]+', s)
                    if latest_index < int(_matched.group()):
                        latest_index = int(_matched.group())
                        _latest_checkpoint = os.path.join(path, f)
    
    return _latest_checkpoint, latest_index

    ############## Choose device ###########################
def get_device(use_gpu=False, gpuid=0):
    if use_gpu and torch.cuda.is_available():
        device = "cuda:{}".format(gpuid)
    else:
        device = "cpu"
    
    return device
