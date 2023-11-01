
import torch
from configs import g_conf
from collections import OrderedDict

import matplotlib.pyplot as plt
import os

class CIL_multiview_Evaluator(object):
    """
    Evaluate
    """
    def __init__(self, name):
        self.reset()
        self.name = name

    def reset(self):
        self._action_batch_errors_mat = 0
        self._total_num = 0
        self._metrics = {}
        self.steers =[]
        self.accelerations=[]
        self.gt_steers =[]
        self.gt_accelerations=[]

    def process(self, action_outputs, targets_action):
        """
        Compute the errors sum for the outputs and targets of the neural network in val dataset
        """
        B = action_outputs.shape[0]
        self._total_num += B
        action_outputs = action_outputs[:, -1, -len(g_conf.TARGETS):]
        self.steers += list(action_outputs[:,0].detach().cpu().numpy())
        self.accelerations += list(action_outputs[:, 1].detach().cpu().numpy())
        self.gt_steers += list(targets_action[-1][:, 0].detach().cpu().numpy())
        self.gt_accelerations += list(targets_action[-1][:, 1].detach().cpu().numpy())
        actions_loss_mat_normalized = torch.clip(action_outputs, -1, 1) - targets_action[-1] # (B, len(g_conf.TARGETS))

        # unnormalize the outputs and targets to compute actual error
        if g_conf.ACCELERATION_AS_ACTION:
            self._action_batch_errors_mat += torch.abs(actions_loss_mat_normalized)  # [-1, 1]

        else:
            pass

    def evaluate(self, current_epoch, dataset_name):
        self.metrics_compute(self._action_batch_errors_mat)
        results = OrderedDict({self.name: self._metrics})

        plt.figure()
        W, H = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches([4.0*W, H])
        plt.plot(range(len(self.gt_accelerations)), self.gt_accelerations, color = 'green')
        plt.plot(range(len(self.accelerations)), self.accelerations, color = 'blue')
        plt.ylim([-1.2, 1.2])
        plt.xlabel('frame id')
        plt.ylabel('')
        plt.savefig(os.path.join(g_conf.EXP_SAVE_PATH, 'acc_'+dataset_name+'_epoch'+str(current_epoch)+'.jpg'))
        plt.close()

        plt.figure()
        W, H = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches([4.0*W, H])
        plt.plot(range(len(self.gt_steers)), self.gt_steers, color = 'green')
        plt.plot(range(len(self.steers)), self.steers, color = 'blue')
        plt.ylim([-1.2, 1.2])
        plt.xlabel('frame id')
        plt.ylabel('')
        plt.savefig(os.path.join(g_conf.EXP_SAVE_PATH, 'steer_'+dataset_name+'_epoch'+str(current_epoch)+'.jpg'))
        plt.close()
        return results

    def metrics_compute(self, action_errors_mat):

        self._metrics.update({'MAE_steer': torch.sum(action_errors_mat, 0)[0] / self._total_num})
        if g_conf.ACCELERATION_AS_ACTION:
            self._metrics.update({'MAE_acceleration': torch.sum(action_errors_mat, 0)[1] / self._total_num})
        else:
            pass
        self._metrics.update({'MAE': torch.sum(action_errors_mat) / self._total_num})


