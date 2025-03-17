import os
import glob
import torch
import traceback
from src.utils import loading, multigpu
import pdb

class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None,reload_data=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)
        self.reload_data = reload_data
        self.first_run_flag_reload = True


    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None


    def train(self, max_epochs, load_latest=False, fail_safe=True, reload_data=False):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        epoch = -1
        num_tries = 10
        for i in range(num_tries):
            try:
                if load_latest:
                    self.load_checkpoint()
                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch
                    if self.reload_data is not None:
                        if not self.first_run_flag_reload:
                            dataset = self.reload_data['dataset']
                            dataset_ = dataset['dls'](cfd=dataset['cfd'],samples_per_videos=dataset['samples_per_videos'],num_test_frames=dataset['num_test_frames'], num_train_frames=dataset['num_train_frames'],label_function_params=dataset['label_function_params'])
                            loader= self.reload_data['dataloader']
                            loader_ = loader['loader'](loader['name'],dataset_,training =loader['training'], num_workers=loader['num_workers'],stack_dim=loader['stack_dim'], batch_size=loader['batch_size'])
                            self.loaders =[loader_]
                            print('Reloaded Data...')
                        else:
                            # self.first_run_flag_reload = False
                            self.first_run_flag_reload = True   # Dont reload
                            print('Not Reloading Data..')

                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    if self._checkpoint_dir:
                        self.save_checkpoint()
                        pass
            except:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')


    def train_epoch(self):
        raise NotImplementedError


    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            # 'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            # 'net_info': getattr(net, 'info', None),
            # 'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            # 'stats': self.stats,
            # 'settings': self.settings
        }


        directory = '{}'.format(self._checkpoint_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)


    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}_ep*.pth.tar'.format(self._checkpoint_dir, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, 
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        # checkpoint_dict = loading.torch_load_legacy(checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()

        if ignore_fields is None:
            ignore_fields = ['settings']

        # Never load the scheduler. It exists in older checkpoints.
        # Add epoch here, if force reset to epoch 0
        # ignore_fields.extend(['epoch','lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net_keys = net.state_dict().keys()
                loaded_keys = checkpoint_dict[key].keys()
                weights= {}
                for nkey in net_keys:
                    if nkey in loaded_keys:
                        weights[nkey]=checkpoint_dict[key][nkey]
                    else:
                        weights[nkey]=net.state_dict()[nkey]
                        print('Skipping Loading... Key in Network Exist and Not found in Loaded: {}'.format(nkey))
                try:
                    net.load_state_dict(weights)
                except:
                    print('Mismatch in model weights... Loading what can be loaded')
                    nw= {}
                    for key in net.state_dict().keys():
                        if weights[key].shape == net.state_dict()[key].shape:
                            nw[key]=weights[key]
                        else:
                            nw[key]=net.state_dict()[key]
                            print('Key not loaded... {}'.format(key))
                    net.load_state_dict(nw)
            elif key == 'optimizer':
                try:
                    self.optimizer.load_state_dict(checkpoint_dict[key])
                except:
                    print('Couldnt Load Optimizer States')
            else:
                setattr(self, key, checkpoint_dict[key])


        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch

        print('Loaded Model...')
        return True
