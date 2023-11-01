import torch
from configs import g_conf


def Action_nospeed_L1(params):
    B = params['action_output'].shape[0]  # batch_size

    # SingleFrame model - we only take into account the last frame's action
    actions_loss_mat = torch.abs(params['action_output'][:,-1,:] - params['targets_action'][-1])  # (B, 2)

    steer_loss = actions_loss_mat[:, 0] * params['variable_weights']['actions']['steer']
    steer_loss = torch.sum(steer_loss) / B

    if g_conf.ACCELERATION_AS_ACTION:
        acceleration_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['acceleration']
        acceleration_loss = torch.sum(acceleration_loss) / B

        loss = steer_loss + acceleration_loss

        return loss, steer_loss, acceleration_loss

    else:
        throttle_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['throttle']
        brake_loss = actions_loss_mat[:, 2] * params['variable_weights']['actions']['brake']
        throttle_loss = torch.sum(throttle_loss) / B
        brake_loss = torch.sum(brake_loss) / B

        loss = steer_loss + throttle_loss + brake_loss

        return loss, steer_loss, throttle_loss, brake_loss

def Loss(loss):

    if loss=='Action_nospeed_L1':
        return Action_nospeed_L1

    else:
        raise NotImplementError(" The loss of this model type has not yet defined ")
