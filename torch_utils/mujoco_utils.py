
def apply_conditioning(x, conditions, observation_dim=None):
    if observation_dim is None:
        observation_dim = x.shape[-1]
    for t, val in conditions.items():
        val = val.to(x.device)
        x[:, t, :observation_dim] = val.clone()
    return x

def apply_action_condition(x, action, observation_dim=None):
    if observation_dim is None:
        observation_dim = x.shape[-1]
    x[..., observation_dim:] = action.clone()
    return x


