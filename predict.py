import sys
from utils.utils import get_args, process_config, create_dirs
from data_loader.cyclegan_data_loader import CycleGANDataLoader
from models.cyclegan_model import CycleGANModel
from trainers.cyclegan_trainer import CycleGANModelTrainer


def predict():
    # get json configuration filepath from the run argument
    # process the json configuration file
    try:
        args = get_args()
        config, log_dir, _, _, predict_dir = process_config(args.config)
    except:
        print('missing or invalid arguments')
        print('Unexpected error:', sys.exc_info()[0])

    # create the experiment directories
    create_dirs([predict_dir])

    print('Create the data generator')
    data_loader = CycleGANDataLoader(config)

    print('Create the model')
    model = CycleGANModel(config)
    print('model ready loading data now')

    print('Create the trainer')
    trainer = CycleGANModelTrainer(model, data_loader.train_a, data_loader.train_b, data_loader.test_a, data_loader.test_b, config, log_dir, config['trained_checkpoint_dir'], predict_dir)

    print('Infer.')
    trainer.predict()

if __name__ == '__main__':
    predict()