import os
import numpy as np
import glob
from omegaconf import OmegaConf
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.CILv2_multiview import g_conf, merge_with_yaml, CIL_multiview_actor_critic, make_data_loader2

from train.utils import extract_model_data_tensors, forward_actor_critic, set_seed, get_lr

# suppress rllib's warnings
import logging
logging.getLogger("ray.rllib").setLevel(logging.ERROR)


def main(args):
    if os.path.sep in args.config:
        conf_file = args.config
    else:
        conf_file = os.path.join(*'./train/configs'.split('/'), args.config)

    conf = OmegaConf.load(conf_file)

    set_seed(conf.seed)

    LEARNING_RATE = conf.LEARNING_RATE
    EPOCHS = conf.EPOCHS
    MODEL_NAME = conf.MODEL_NAME
    SAVE_PATH = os.path.join(*conf.SAVE_PATH.split('/'), MODEL_NAME)

    data_path = os.path.join(*conf.data_path.split('/'))
    train_dataset_names = [os.path.join(*name.split('/')) for name in conf.train_dataset_names]
    val_dataset_names = [os.path.join(*name.split('/')) for name in conf.val_dataset_names]
    batch_size = conf.batch_size
    num_workers = conf.num_workers

    if args.cpu:
        device = 'cpu'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path_to_yaml = os.path.join(*'./models/CILv2_multiview/_results/Ours/Town12346_5/CILv2.yaml'.split('/'))
    merge_with_yaml(path_to_yaml)
    model = CIL_multiview_actor_critic(g_conf)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        class MyDataParallel(torch.nn.DataParallel):
            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.module, name)

        model = MyDataParallel(model)

    # load the checkpoint
    checkpoints = glob.glob(os.path.join(SAVE_PATH, f"{MODEL_NAME}_*.pth"))
    if len(checkpoints) > 0 and not args.clean:
        print(f'Loading checkpoint: {max(checkpoints)}')
        checkpoint = torch.load(max(checkpoints))
        checkpoint_model = checkpoint['model']
        checkpoint_optimizer = checkpoint['optimizer']
        checkpoint_scheduler = checkpoint['scheduler']
        epoch_start = checkpoint['epoch']
    else:
        print('No checkpoint found')
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

    writer = SummaryWriter(comment=f"{MODEL_NAME}_IL")

    os.makedirs(SAVE_PATH, exist_ok=True)
    min_loss = float('inf')

    target_names = ['steer', 'acceleration']

    for epoch in tqdm(range(epoch_start, EPOCHS), desc='Epochs'):
        #####################################################
        # Training
        #####################################################
        model.action_output.train()

        writer.add_scalar("Learning rate", get_lr(optimizer), epoch)

        loss_list = []
        steer_loss_list = []
        acceleration_loss_list = []
        for data in train_loader:
            src_images, src_directions, src_speed, target = extract_model_data_tensors(
                data,
                target_names,
                g_conf.DATA_USED,
                device,
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
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            steer_loss_list.append(steer_loss.item())
            acceleration_loss_list.append(acceleration_loss.item())

        loss = np.mean(loss_list)
        steer_loss = np.mean(steer_loss_list)
        acceleration_loss = np.mean(acceleration_loss_list)

        # log values
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("SteerLoss/train", steer_loss, epoch)
        writer.add_scalar("AccelerationLoss/train", acceleration_loss, epoch)


        #####################################################
        # Validation
        #####################################################
        model.action_output.eval()

        loss_list = []
        steer_loss_list = []
        acceleration_loss_list = []
        for data in val_loader:
            src_images, src_directions, src_speed, target = extract_model_data_tensors(
                data,
                target_names,
                g_conf.DATA_USED,
                device,
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
        writer.add_scalar("Loss/eval", loss, epoch)
        writer.add_scalar("SteerLoss/eval", steer_loss, epoch)
        writer.add_scalar("AccelerationLoss/eval", acceleration_loss, epoch)

        scheduler.step(loss)

        if loss < min_loss:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                }, os.path.join(SAVE_PATH, f'{MODEL_NAME}_{epoch}.pth'))
        min_loss = min(loss, min_loss)

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        }, os.path.join(SAVE_PATH, f'{MODEL_NAME}_final.pth'))
    writer.close()

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

