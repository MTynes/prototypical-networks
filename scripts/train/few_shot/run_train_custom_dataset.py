import argparse
import sys
from train_custom_dataset import main


class ExecutionArgumentList:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train prototypical networks')

        # data args
        default_dataset = 'custom' # or 'omniglot'
        parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS',
                            help="data set name (default: {:s})".format(default_dataset))
        dataset_path='/content/eeg_sz_spectrograms/gen_data_20s_70pct_overlap_-_high_nfft_all_channels_sml'
        parser.add_argument('--data.data_path', type=str, default=dataset_path, metavar='DSPATH',
                          help="dataset path {:s}".format(dataset_path))
        labels_path= dataset_path
        parser.add_argument('--data.labels_path', type=str, default=labels_path, metavar='LABELSPATH',
                          help="location of the label .txt files".format(labels_path))

        default_split = 'vinyals'
        parser.add_argument('--data.split', type=str, default=default_split, metavar='SP',
                            help="Split name (default: {:s})".format(default_split))
        parser.add_argument('--data.way', type=int, default=60, metavar='WAY',
                            help="Number of classes per episode (default: 60)")
        parser.add_argument('--data.shot', type=int, default=5, metavar='SHOT',
                            help="Number of support examples per class (default: 5)")
        parser.add_argument('--data.query', type=int, default=5, metavar='QUERY',
                            help="Number of query examples per class (default: 5)")
        parser.add_argument('--data.test_way', type=int, default=5, metavar='TESTWAY',
                            help="Number of classes per episode in test. 0 means same as data.way (default: 5)")
        parser.add_argument('--data.test_shot', type=int, default=0, metavar='TESTSHOT',
                            help="Number of support examples per class in test. 0 means same as data.shot (default: 0)")
        parser.add_argument('--data.test_query', type=int, default=15, metavar='TESTQUERY',
                            help="Number of query examples per class in test. 0 means same as data.query (default: 15)")
        parser.add_argument('--data.train_episodes', type=int, default=100, metavar='NTRAIN',
                            help="Number of train episodes per epoch (default: 100)")
        parser.add_argument('--data.test_episodes', type=int, default=100, metavar='NTEST',
                            help="Number of test episodes per epoch (default: 100)")
        parser.add_argument('--data.trainval', action='store_true', help="Run in train+validation mode (default: False)")
        parser.add_argument('--data.sequential', action='store_true', help="Use sequential sampler instead of episodic (default: False)")
        parser.add_argument('--data.cuda', action='store_true', help="Run in CUDA mode (default: False)")

        # model args
        default_model_name = 'protonet_conv'
        parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                            help="Model name (default: {:s})".format(default_model_name))
        parser.add_argument('--model.x_dim', type=str, default='4,28,28', metavar='XDIM',
                            help="Dimensionality of input images in format 'channels, width, height' (default: '4,28,28')")
        parser.add_argument('--model.hid_dim', type=int, default=64, metavar='HIDDIM',
                            help="Dimensionality of hidden layers (default: 64)")
        parser.add_argument('--model.z_dim', type=int, default=64, metavar='ZDIM',
                            help="Dimensionality of input images (default: 64)")

        # train args
        parser.add_argument('--train.epochs', type=int, default=10000, metavar='NEPOCHS',
                            help='number of epochs to train (default: 10000)')
        parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                            help='optimization method (default: Adam)')
        parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.0001)')
        parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
                            help='number of epochs after which to decay the learning rate')
        default_weight_decay = 0.0
        parser.add_argument('--train.weight_decay', type=float, default=default_weight_decay, metavar='WD',
                            help="weight decay (default: {:f})".format(default_weight_decay))
        parser.add_argument('--train.patience', type=int, default=300, metavar='PATIENCE',
                            help='number of epochs to wait before validation improvement (default: 300)')

        # log args
        default_fields = 'loss,acc'
        parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                            help="fields to monitor during training (default: {:s})".format(default_fields))
        default_exp_dir = 'results'
        parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                            help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))
        self.parser = parser

    def get_arguments(self):
        args = vars(self.parser.parse_args())
        return args


if __name__ == "__main__":
    print(sys.argv)
    eal = ExecutionArgumentList()
    meta_args = eal.get_arguments()
    main(meta_args)
