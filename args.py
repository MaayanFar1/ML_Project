import argparse
import pathlib


class Args(argparse.ArgumentParser):
    def __init__(self, ):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # data param
        self.add_argument('--csv-file',
                          default="/home/maayanfarkash/proj/PBHs-design/Compas1/compas-1x.csv",
                          type=str,
                          help='Path to the csv files which contain the molecules names and target features.')
        self.add_argument('--xyz-root', default='/home/maayanfarkash/proj/PBHs-design/Compas1/pahs-cata-34072-xyz/', type=str,
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
        self.add_argument("--dp", type=eval, default=True, help="Data parallelism")
        self.add_argument("--n_layers", type=int, default=12, help="number of layers")
        self.add_argument("--nf", type=int, default=196, help="number of layers")
        self.add_argument("--tanh", type=eval, default=True)
        self.add_argument("--attention", type=eval, default=True)
        self.add_argument("--coords_range", type=float, default=4)
        self.add_argument("--norm_constant", type=float, default=1)
        self.add_argument("--normalization_factor", type=float, default=1)

        self.add_argument('--num-workers', type=int, default=8, help='Number of workers for each dataloader.')

        # Logging
        self.add_argument('--save_dir', type=str, default="summary/", help="Directory name to save models and logs")

