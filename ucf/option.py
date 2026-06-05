import argparse

parser = argparse.ArgumentParser(description='VAD')

parser.add_argument('--rgb_list', default='list/ucf-i3d.list', help='list of rgb features ')
parser.add_argument('--test_rgb_list', default='list/ucf-i3d-test.list', help='list of test rgb features ')
parser.add_argument('--gt', type=str, default='list/gt-ucf.npy', help='Path to ground truth file')
parser.add_argument('--model_name', type=str, default='best_model_ucf', help='Model name for saving')
parser.add_argument('--cuda', type=int, default=0, help='CUDA device ID')
parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--bias', type=int, default=0)

parser.add_argument('--max_sequence_length', type=int, default=200, help='maximum sequence length during training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--max_epoch', type=int, default=100, help='Maximum number of epochs')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
parser.add_argument('--seed', type=int, default=71239, help='Random seed')
parser.add_argument('--relu_rate', type=float, default=0.0, help='leaky relu')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--gamma', type=float, default=0.01)
parser.add_argument('--tau', type=float, default=1.3)
parser.add_argument('--g', type=float, default=0.02)
parser.add_argument('--ratioK', type=float, default=0.1)
#  clip list
parser.add_argument('--clip_list', default='list/ucf-clip-train-10crop.list', help='list of clip features ')
parser.add_argument('--test_clip_list', default='list/ucf-clip-test-10crop.list', help='list of test clip features ')
