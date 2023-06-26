import argparse


def settings():
    parser = argparse.ArgumentParser()

    # public parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed. Default is 0.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')

    parser.add_argument('--in_file', default="dataset1/LDA.edgelist",    # positive sample
                        help='Path to data. e.g., data/LDA.edgelist')

    parser.add_argument('--neg_sample', default="dataset1/non_LDA.edgelist",     # negative sample
                        help='Path to data. e.g., data/LDA.edgelist')

    parser.add_argument('--validation_type', default="5_cv1", choices=['5_cv1', '5_cv2'],
                        help='Initial cross_validation type. Default is 5_cv1.')

    parser.add_argument('--task_type', default="LDA", choices=['LDA', 'MDA','LMI'],
                        help='Initial prediction task type. Default is LDA.')

    parser.add_argument('--feature_type', type=str, default='normal', choices=['one_hot', 'uniform', 'normal', 'position'],
                        help='Initial node feature type. Default is position.')

    # Training settings
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate. Default is 5e-4.')    #

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate. Default is 0.5.')

    parser.add_argument('--weight_decay', default=5e-4,
                        help='Weight decay (L2 loss on parameters) Default is 5e-4.')

    parser.add_argument('--batch', type=int, default=25,
                        help='Batch size. Default is 25.')

    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train. Default is 50.')

    parser.add_argument('--loss_ratio1', type=float, default=1,
                        help='Ratio of task1. Default is 1')

    parser.add_argument('--loss_ratio2', type=float, default=0.5,
                        help='Ratio of task2. Default is 0.5')

    parser.add_argument('--loss_ratio3', type=float, default=0.5,
                        help='Ratio of task3. Default is 0.5')

    parser.add_argument('--loss_ratio4', type=float, default=0.5,
                        help='Ratio of task4. Default is 0.5')

    # model parameter setting
    parser.add_argument('--dimensions', type=int, default=256,
                        help='dimensions of feature d. Default is 256 (LDA, MDA tasks) 512 (LMI task).')

    parser.add_argument('--hidden1', default=128,
                        help='Embedding dimension of encoder layer 1 for CSGLMD. Default is d/2.')

    parser.add_argument('--hidden2', default=64,
                        help='Embedding dimension of encoder layer 2 for CSGLMD. Default is d/4.')

    parser.add_argument('--decoder1', default=512,
                        help='NEmbedding dimension of decoder layer 1 for CSGLMD. Default is 512.')

    args = parser.parse_args()

    return args
