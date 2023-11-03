import os

# Because the launcher runs this program,
# change the root directory to the corrent
script_dir = os.path.dirname(__file__)
root_dir, current_dir = os.path.split(script_dir)
if current_dir == 'train':
    os.chdir(root_dir)

import numpy as np
import glob
from omegaconf import OmegaConf
import argparse
import torch
from tqdm.auto import tqdm

from models.CILv2_multiview import g_conf, merge_with_yaml, CIL_multiview_actor_critic

from train.utils import extract_model_data_tensors_no_device, forward_actor_critic
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

# suppress rllib's warnings
import logging
logging.getLogger("ray.rllib").setLevel(logging.ERROR)

os.environ['MASTER_ADDR'] = 'localhost'

if os.name == 'nt':
    torch.distributed.init_process_group(backend='gloo')

def main(args):
    if os.path.sep in args.config:
        conf_file = args.config
    else:
        conf_file = os.path.join(*'./train/configs'.split('/'), args.config)

    conf = OmegaConf.load(conf_file)

    LEARNING_RATE = conf.LEARNING_RATE
    EPOCHS = conf.EPOCHS
    MODEL_NAME = conf.MODEL_NAME
    SAVE_PATH = os.path.join(*conf.SAVE_PATH.split('/'), MODEL_NAME)

    data_path = os.path.join(*conf.data_path.split('/'))
    train_dataset_names = conf.train_dataset_names
    val_dataset_names = conf.val_dataset_names
    batch_size = conf.batch_size
    num_workers = conf.num_workers

    config = ProjectConfiguration(project_dir=".", logging_dir="runs")
    accelerator = Accelerator(log_with="tensorboard", project_config=config)
    accelerator.init_trackers("policy_head_traning")
    device = accelerator.device

    path_to_yaml = os.path.join(*'./models/CILv2_multiview/_results/Ours/Town12346_5/CILv2.yaml'.split('/'))
    merge_with_yaml(path_to_yaml)
    model = CIL_multiview_actor_critic(g_conf)
    # model.forward = model.forward2

    # load the checkpoint
    checkpoints = glob.glob(os.path.join(SAVE_PATH, f"{MODEL_NAME}_*.pth"))
    if len(checkpoints) > 0 and not args.clean:
        accelerator.print(f'Loading checkpoint: {max(checkpoints)}')
        checkpoint = torch.load(max(checkpoints))
        checkpoint_model = checkpoint['model']
        checkpoint_optimizer = checkpoint['optimizer']
        checkpoint_scheduler = checkpoint['scheduler']
        epoch_start = checkpoint['epoch']
    else:
        accelerator.print('No checkpoint found')
        checkpoint_file = os.path.join(*'./models/CILv2_multiview/_results/Ours/Town12346_5/checkpoints/CILv2_multiview_attention_40.pth'.split('/'))
        checkpoint = torch.load(checkpoint_file, map_location=device)['model']
        checkpoint = {k[7:] if k.startswith('_model.') else k:v for k, v in checkpoint.items()}
        checkpoint_model = {k:v for k, v in checkpoint.items() if not k.startswith('action_output.layers.2.0')}
        checkpoint_optimizer = None
        checkpoint_scheduler = None
        epoch_start = 0

    # load state dict
    model.load_state_dict(checkpoint_model, strict=False)

    model.eval()
    model.action_output.train()

    # freeze weights
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the last layer
    for param in model.action_output.parameters():
        param.requires_grad = True


    from models.CILv2_multiview import make_data_loader2

    train_loader, val_loader = make_data_loader2(
        "transfer",
        data_path,
        train_dataset_names,
        batch_size,
        val_dataset_names,
        num_workers=num_workers,
    )

    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.action_output.parameters(), lr=LEARNING_RATE)
    if checkpoint_optimizer: optimizer.load_state_dict(checkpoint_optimizer)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    if checkpoint_scheduler: scheduler.load_state_dict(checkpoint_scheduler)

    os.makedirs(SAVE_PATH, exist_ok=True)
    min_loss = float('inf')

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    target_names = ['steer', 'acceleration']

    bar = tqdm(range(epoch_start, EPOCHS), desc='Epochs', disable=not accelerator.is_local_main_process)
    for epoch in bar:
        #####################################################
        # Training
        #####################################################
        model.module.action_output.train()

        loss_list = []
        steer_loss_list = []
        acceleration_loss_list = []
        for data in train_loader:
            src_images, src_directions, src_speed, target = extract_model_data_tensors_no_device(
                data,
                target_names,
                g_conf.DATA_USED,
            )

            steer_out, acceleration_out = forward_actor_critic(
                model,
                (src_images, src_directions, src_speed),
            )
            steer, acceleration = target[:, 0], target[:, 1]

            steer_loss = criterion(steer_out, steer)
            acceleration_loss = criterion(acceleration_out, acceleration)

            loss = steer_loss + acceleration_loss
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            loss_list.append(loss.item())
            steer_loss_list.append(steer_loss.item())
            acceleration_loss_list.append(acceleration_loss.item())
        loss = np.mean(loss_list)
        steer_loss = np.mean(steer_loss_list)
        acceleration_loss = np.mean(acceleration_loss_list)

        # log values
        accelerator.log({"Loss/train": loss,
                         "SteerLoss/train": steer_loss,
                         "AccelerationLoss/train": acceleration_loss,
                         }, step=epoch)

        #####################################################
        # Validation
        #####################################################
        model.module.action_output.eval()

        loss_list = []
        steer_loss_list = []
        acceleration_loss_list = []
        for data in val_loader:
            src_images, src_directions, src_speed, target = extract_model_data_tensors_no_device(
                data,
                target_names,
                g_conf.DATA_USED,
            )

            with torch.no_grad():
                steer_out, acceleration_out = forward_actor_critic(
                    model,
                    (src_images, src_directions, src_speed),
                )
                steer, acceleration = target[:, 0], target[:, 1]
                steer_loss = criterion(steer_out, steer)
                acceleration_loss = criterion(acceleration_out, acceleration)

                loss = steer_loss + acceleration_loss
            loss_list.append(loss.item())
            steer_loss_list.append(steer_loss.item())
            acceleration_loss_list.append(acceleration_loss.item())

        loss = np.mean(loss_list)
        steer_loss = np.mean(steer_loss_list)
        acceleration_loss = np.mean(acceleration_loss_list)

        # log values
        accelerator.log({"Loss/eval": loss,
                         "SteerLoss/eval": steer_loss,
                         "AccelerationLoss/eval": acceleration_loss,
                         }, step=epoch)

        scheduler.step(loss)

        if accelerator.is_main_process:
            if loss < min_loss:
                torch.save({
                    'epoch': epoch,
                    'model': accelerator.unwrap_model(model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    }, os.path.join(SAVE_PATH, f'{MODEL_NAME}_{epoch}.pth'))
        min_loss = min(loss, min_loss)

    torch.save({
        'model': accelerator.unwrap_model(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, os.path.join(SAVE_PATH, f'{MODEL_NAME}_final.pth'))

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train the policy head of the model with IL.',
    )
    parser.add_argument('-c', '--config',
                        default='IL_CIL_multiview_actor_critic.yaml',
                        help='Filename or whole path to the config',
                        type=str,
                        )
    parser.add_argument('--clean',
                        action="store_true",
                        help='Do not load a checkpoint if any exits.',
                        )
    parser.add_argument('--cpu',
                        action="store_true",
                        help='Set torch device to "cpu"',
                        )
    args = parser.parse_args()

    main(args)

