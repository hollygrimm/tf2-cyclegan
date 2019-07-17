import sys
from utils.utils import get_args, process_config, create_dirs
from data_loader.cyclegan_data_loader import CycleGANDataLoader
from models.cyclegan_model import CycleGANModel
#from trainers.cyclegan_trainer import CycleGANModelTrainer

def main():
    try:
        args = get_args()
        config, log_dir, checkpoint_dir = process_config(args.config)
    except:
        print('missing or invalid arguments')
        print('Unexpected error:', sys.exc_info()[0])

    # create the experiment directories
    create_dirs([log_dir, checkpoint_dir])

    print('Create the data generator')
    data_loader = CycleGANDataLoader(config)

    print(next(iter(data_loader.train_a)))
    print(next(iter(data_loader.train_b)))

    print('Create the model')
    model = CycleGANModel(config, config['weights_path'])
    print('model ready loading data now')

    # print('Create the trainer')
    # trainer = CycleGANModelTrainer(model.model, data_loader.get_train_data(), data_loader.get_val_data(), config, log_dir, checkpoint_dir)

    # print('Start training the model.')
    # trainer.train()


def infer():
    # get json configuration filepath from the run argument
    # process the json configuration file
    try:
        config = 'input_params_for_inference.json'
        config, _, _ = process_config(config)
    except:
        print('missing or invalid arguments')
        print('Unexpected error:', sys.exc_info()[0])

    print('Create the data generator')
    data_loader = CycleGANDataLoader(config)

    # print('Create the model')
    # model = CycleGANModel(config)
    # print('model ready loading data now')

    # print('Create the trainer')
    # trainer = CycleGANModelTrainer(model.model, data_loader.get_train_data(), data_loader.get_val_data(), config, '', '')

    # print('Infer.')
    # trainer.predict()

if __name__ == '__main__':
    main()