import argparse
import numpy
import pylearn2.utils.serial
import pylearn2.utils.environ
import pieceout.iclr.diamond.hyperparameters
import pieceout.iclr.mnist08.hyperparameters
import pieceout.iclr.mnist17.hyperparameters
import pieceout.iclr.mnist18.hyperparameters
import pieceout.iclr.mnist23.hyperparameters

def main():
    parser = argparse.ArgumentParser(description="Run dropout network experiments.")
    parser.add_argument("dataset", choices=['diamond', 'mnist23', 'mnist18',
                                            'mnist17', 'mnist08'], type=str,
                        help="The dataset settings to use for "
                             "hyperparameters.")
    parser.add_argument("config", type=str,
                        help="YAML config file.")
    parser.add_argument("start_seed", type=int, help="RNG seed.")
    parser.add_argument("-E", "--end_seed", type=int, required=False,
                        help="Evaluate every from START_SEED up to this "
                             "one, inclusive.")
    args = parser.parse_args()

    start_seed = args.start_seed
    if args.end_seed:
        assert args.start_seed < args.end_seed
        end_seed = args.end_seed
    else:
        end_seed = args.start_seed

    for seed in range(start_seed, end_seed + 1):
        # Put the seed in the environment.
        pylearn2.utils.environ.putenv('SEED', str(seed))

        # Instantiate an RNG with seed.
        rng = numpy.random.RandomState(seed)

        # Grab the hyperparameter scheme for this dataset.
        mod = getattr(pieceout.iclr, args.dataset).hyperparameters

        # Iterate through them and put them in the environment.
        for key, value in mod.get_hyperparameters(rng).iteritems():
            pylearn2.utils.environ.putenv(key, str(value))

        train_obj = pylearn2.utils.serial.load_train_file(args.config)
        train_obj.main_loop()
        del train_obj


if __name__ == "__main__":
    main()
