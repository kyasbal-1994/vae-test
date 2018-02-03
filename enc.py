import chainer
from chainer import cuda, training
from chainer import training
from chainer.training import extensions, Trainer
import numpy as np
import net
import argparse
import matplotlib
# Disable interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_image(x, filename):
    fig, ax = plt.subplots(20, 20, figsize=(20, 20), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        ai.imshow(xi.reshape(28, 28), cmap='gray')
    fig.savefig(filename)
    plt.close('all')

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--out', '-o', type=str, default='./result/',
                        help='dir to save snapshots.')
    parser.add_argument('--interval', '-i', type=int,
                        default=5, help='interval of save images.')
    parser.add_argument
    args = parser.parse_args()

    batchsize = args.batchsize
    n_epoch = args.epoch
    n_latent = args.dimz

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Prepare dataset
    print('load MNIST dataset')

    model = net.VAE(784, n_latent, 500)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    # Setup optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist(withlabel=False)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    # 訓練結果をロード
    chainer.serializers.load_npz(f'./result/vae/snapshot_epoch_20', trainer)
    # 400個のランダムな正規分布の乱数ベクトルを生成
    z = chainer.Variable(np.random.normal(
        0, 1, (400, n_latent)).astype(np.float32))
    # 乱数を元にデコード
    x = model.decode(z)
    save_image(x.data,"C.png")

main()
