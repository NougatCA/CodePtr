import os

import config
import train
import eval
import predict


def _train(vocab_file_path=None, model_file_path=None):
    print('\nStarting the training process......\n')

    if vocab_file_path:
        source_vocab_path, code_vocab_path, ast_vocab_path, nl_vocab_path = vocab_file_path
        print('Vocabulary will be built by given file path.')
        print('\tSource code vocabulary path:\t', os.path.join(config.vocab_dir, source_vocab_path))
        print('\tTokenized source code vocabulary path:\t', os.path.join(config.vocab_dir, code_vocab_path))
        print('\tAst of code vocabulary path:\t', os.path.join(config.vocab_dir, ast_vocab_path))
        print('\tCode comment vocabulary path:\t', os.path.join(config.vocab_dir, nl_vocab_path))
    else:
        print('Vocabulary will be built according to dataset.')

    if model_file_path:
        print('Model will be built by given state dict file path:', os.path.join(config.model_dir, model_file_path))
    else:
        print('Model will be created by program.')

    print('\nInitializing the training environments......\n')
    train_instance = train.Train(vocab_file_path=vocab_file_path, model_file_path=model_file_path)
    print('Environments built successfully.\n')
    print('Size of train dataset:', train_instance.train_dataset_size)

    source_oov_rate = 1 - train_instance.source_vocab_size / train_instance.origin_source_vocab_size
    code_oov_rate = 1 - train_instance.code_vocab_size / train_instance.origin_code_vocab_size
    nl_oov_rate = 1 - train_instance.nl_vocab_size / train_instance.origin_nl_vocab_size

    print('\nSize of source code vocabulary:', train_instance.origin_source_vocab_size,
          '->', train_instance.source_vocab_size)
    print('Source code OOV rate: {:.2f}%'.format(source_oov_rate * 100))
    print('\nSize of tokenized source code vocabulary:', train_instance.origin_code_vocab_size,
          '->', train_instance.code_vocab_size)
    print('Tokenized source code OOV rate: {:.2f}%'.format(code_oov_rate * 100))
    print('\nSize of ast of code vocabulary:', train_instance.ast_vocab_size)
    print('\nSize of code comment vocabulary:', train_instance.origin_nl_vocab_size, '->', train_instance.nl_vocab_size)
    print('Code comment OOV rate: {:.2f}%'.format(nl_oov_rate * 100))
    config.logger.info('Size of train dataset:{}'.format(train_instance.train_dataset_size))
    config.logger.info('Size of source code vocabulary: {} -> {}'.format(
        train_instance.origin_source_vocab_size, train_instance.source_vocab_size))
    config.logger.info('Source source code OOV rate: {:.2f}%'.format(source_oov_rate * 100))
    config.logger.info('Size of tokenized source code vocabulary: {} -> {}'.format(
        train_instance.origin_code_vocab_size, train_instance.code_vocab_size))
    config.logger.info('Source code OOV rate: {:.2f}%'.format(code_oov_rate * 100))
    config.logger.info('Size of ast of code vocabulary: {}'.format(train_instance.ast_vocab_size))
    config.logger.info('Size of code comment vocabulary: {} -> {}'.format(
        train_instance.origin_nl_vocab_size, train_instance.nl_vocab_size))
    config.logger.info('Code comment OOV rate: {:.2f}%'.format(nl_oov_rate * 100))

    if config.validate_during_train:
        print('\nValidate every', config.validate_every, 'batches and each epoch.')
        print('Size of validation dataset:', train_instance.eval_instance.dataset_size)
        config.logger.info('Size of validation dataset: {}'.format(train_instance.eval_instance.dataset_size))

    print('\nStart training......\n')
    config.logger.info('Start training.')
    best_model = train_instance.run_train()
    print('\nTraining is done.')
    config.logger.info('Training is done.')

    # writer = SummaryWriter('runs/CodePtr')
    # for _, batch in enumerate(train_instance.train_dataloader):
    #     batch_size = len(batch[0][0])
    #     writer.add_graph(train_instance.model, (batch, batch_size, train_instance.nl_vocab))
    #     break
    # writer.close()

    return best_model


def _test(model):
    print('\nInitializing the test environments......')
    test_instance = eval.Test(model)
    print('Environments built successfully.\n')
    print('Size of test dataset:', test_instance.dataset_size)
    config.logger.info('Size of test dataset: {}'.format(test_instance.dataset_size))

    config.logger.info('Start Testing.')
    print('\nStart Testing......')
    test_instance.run_test()
    print('Testing is done.')


def _predict(model):
    print('\nInitializing the predict environments......')
    predict_instance = predict.Predict(model)
    print('Environments built successfully.\n')
    predict_instance.run_predict()


if __name__ == '__main__':
    best_model_dict = _train()
    _test(best_model_dict)
    # # _test(os.path.join('20200604_230516', 'model_valid-loss-3.2370_epoch-1_batch--1.pt'))
    # _predict('model_valid-loss-3.3545_epoch-1_batch--1.pt')

    # import models
    #
    # model = models.Model(source_vocab_size=30000,
    #                      code_vocab_size=30000,
    #                      ast_vocab_size=58,
    #                      nl_vocab_size=20000)
    # total_1 = sum([param.nelement() for param in model.parameters()])
    # total_2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('  + Number of params: %.2fM' % (total_1 / 1e6))
    # print('  + Number of params: %.2fM' % (total_2 / 1e6))
