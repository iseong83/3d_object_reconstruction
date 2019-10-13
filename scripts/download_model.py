import os
import sys

MODEL_DIR = './models'
def download_models():
    LINK = 'https://shapenetv1.s3-us-west-2.amazonaws.com/model/model.tar'
    os.system('wget -c {0} -P {1}'.format(LINK, MODEL_DIR))



def main():
    archive = 'model.tar'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if os.path.exists(os.path.join(MODEL_DIR, archive)):
        print ('{} exit. Please check'.format(archive))
        sys.exit(1)
    download_models()
    os.system("tar -xvf {0} -C {1}".format(os.path.join(MODEL_DIR,archive), MODEL_DIR))


if __name__ == '__main__':
    main()
