import os
from House_gan_sdpl.utils import (GenerateTfrecord, HouseGan)

here = os.path.dirname(os.path.abspath(__file__))

HParams = {
    'input_size': 2,
    'output_size': 2,
    'num_class': 10,
    'latent': 32,
    'batch': 1,
    'img_size': [299, 299],
    'num_variations': None,
    'epochs': 1000,
    'generator_lr': 0.0001,
    'discriminator_lr': 0.0001,
    'num_process': 2,
    'decay_steps': 20000,
    'decay_rate': 0.96,
    'model_path': 'data/models/',
    'plt_path': 'data/models/',
    'log_path': 'data/models/',
    'train_data': 'data/train.record',
    'test_data': 'data/train.record'
}

def main():
    ################### 이미지 생성 명령 ###################
    # a = GenerateImage(
    #     csv_file='F:/PersonalProject/Gangsoo/rooms.CSV',
    #     save_dir='F:/PersonalProject/Gangsoo/data/'
    # )
    # a.generateImg()
    ################## Tfrecord 생성 명령 ###################
    # house_data = GenerateTfrecord(
    #     csv_file='data/rooms_data.CSV',
    #     save_dir='data/')
    # house_data.createTfrecord('train')

    ################### 학습 명령 ###################
    house_gan = HouseGan(HParams)
    # house_gan.training()
    # house_gan.validation()
    house_gan.test()

if __name__ == '__main__':
    main()