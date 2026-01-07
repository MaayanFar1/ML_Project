import argparse
import pathlib


class Args(argparse.ArgumentParser):
    def __init__(self, ):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # data param
        self.add_argument('--csv-file',
                          default="/home/tomerweiss/aromatic/Polycyclic-Aromatic-Prediction-and-Design/data/dft-data-8678.csv",
                          type=str,
                          help='Path to the csv files which contain the molecules names and target features.')
        self.add_argument('--xyz-root', default='/home/tomerweiss/aromatic/Polycyclic-Aromatic-Prediction-and-Design/'
                                                'data/dft-data-8678-xyzs/', type=str,
                          help='Path to the folder which contains the xyz files.')

        # task param
        self.add_argument('--target_features', default='Erel_eV', type=str,
                          help='list of the names of the target features in the csv file - can be multiple targets seperated with commas'
                               '[HOMO_eV, LUMO_eV, GAP_eV, Dipmom_Debye, Etot_eV, Etot_pos_eV,'
                               'Etot_neg_eV, aEA_eV, aIP_eV, Erel_eV]')
        self.add_argument('--sample-rate', type=float, default=1.,
                          help='Fraction of total molecules to include in the datasets')

        # training param
        self.add_argument('--name', type=str, default='Erel',
                        help="Experiment name - name of the dir for logs and trained model")
        self.add_argument('--restore', type=bool, default=None,
                help="If set will load the model according to name.")
        self.add_argument('--rings_graph', type=bool, default=True,
                          help='Select if we use graph of rings or graph of atoms.')
        self.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
        self.add_argument('--num_epochs', type=int, default=200,  help="Number of epochs")
        self.add_argument('--transform', type=bool, default=False,
                          help='Adding rotation transform as augmentation during training.')
        self.add_argument('--normalize', type=bool, default=True,
                          help='Normalize the targets.')

        self.add_argument('--batch-size', type=int, default=64,  help='The size of the batch.')

        # Model parameters
        self.add_argument('--model', type=str, default='SE3Transformer',
                            help="String name of model")
        self.add_argument('--num_layers', type=int, default=6,
                            help="Number of equivariant layers")
        self.add_argument('--num_degrees', type=int, default=6,
                            help="Number of irreps {0,1,...,num_degrees-1}")
        self.add_argument('--num_channels', type=int, default=16,
                            help="Number of channels in middle layers")
        self.add_argument('--num_nlayers', type=int, default=0,
                            help="Number of layers for nonlinearity")
        self.add_argument('--div', type=float, default=4,
                            help="Low dimensional embedding fraction")
        self.add_argument('--pooling', type=str, default='avg',
                            help="Choose from avg or max")
        self.add_argument('--head', type=int, default=1,
                            help="Number of attention heads")

        self.add_argument('--num-workers', type=int, default=16, help='Number of workers for each dataloader.')

        # Logging
        self.add_argument('--save_dir', type=str, default="summary/", help="Directory name to save models and logs")

