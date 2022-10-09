import argparse


def get_config():
    parser = argparse.ArgumentParser()
    '''Base'''
    parser.add_argument('--num_classes',type=int,default='int')
    parser.add_argument('--model_name', type=str, default='bert', choices=['bert', 'roberta', 'glove'])
    '''Optimization'''
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)

    '''Environment'''
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
