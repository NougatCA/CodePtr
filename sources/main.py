
from config import logger
import train
import eval


def _train():
    logger.info('*' * 100)
    logger.info('Initializing the training environment')

    training = train.Train()

    logger.info('Training environment are initialized successfully')
    logger.info('*' * 100)

    return training.run()


def _test(model, vocab):
    logger.info('*' * 100)
    logger.info('Initializing the testing environment')

    testing = eval.Test(model, vocab)

    logger.info('Testing environment are initialized successfully')
    logger.info('*' * 100)

    testing.run_test()


if __name__ == '__main__':
    best_model, train_vocab = _train()
    _test(best_model, train_vocab)
