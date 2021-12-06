from utils import (HouseGan, GenerateImage, GenerateTfrecord)

HParams = {
    'num_class': 10,
    'latent': 32,
    'batch': 1,
    'img_size': [299, 299],
    'num_variations': None,
    'epochs': 10,
    'generator_lr': 0.0001,
    'discriminator_lr': 0.0001,
    'num_process': 2,
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
    ################### Tfrecord 생성 명령 ###################
    # house_data = GenerateTfrecord(
    #     csv_file='F:/PersonalProject/Gangsoo/rooms.CSV',
    #     save_dir='F:/PersonalProject/Gangsoo/data/')
    # house_data.createTfrecord('train')

    ################### 학습 명령 ###################
    house_gan = HouseGan(HParams)
    house_gan.training()


if __name__ == '__main__':
  main()