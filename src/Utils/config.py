import argparse
import os

def parse_args():
    desc = "Vision Transformer"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training')

    parser.add_argument('--bs', type=int, default=64, help='batch size')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for Adam optimizer')

    parser.add_argument('--ps', type=int, default=16, help='Path_size')

    parser.add_argument('--im_s', type=int, default=224, help='Image size')

    parser.add_argument('--emb_dim', type=int, default=128, help='embedding_dimension_size')

    parser.add_argument('--mlp_dim', type=int, default=128, help=' multilayer_perceptron_dimension_size')

    parser.add_argument('--num_heads', type=int, default=16, help='number of heads')

    parser.add_argument('--num_layers', type=int, default=10, help='numuber of layers')

    parser.add_argument('--num_classes', type=int, help='number_of_classes')

    parser.add_argument('--at_d_r', type=float, default=0.0, help='Dropout_rate_for_for Attention')

    parser.add_argument('--train_data', type=str,help='path_to_train_data')

    parser.add_argument('--test_data', type=str,help='path_to_test_data')

    parser.add_argument('--msp', type=str,help='path_to_save _model')

    parser.add_argument('--logs', type=str,help='path_to_logs')

    return parser.parse_args()