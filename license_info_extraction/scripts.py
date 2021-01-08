import fire
import os 
from main import predict

def predict_all(load_path, dir_path, save_dir_path):
    _list = os.walk(dir_path)
    for root, _, files in _list:
        for file in files:
            file_path = os.path.join(root, file)
            print('current file: ', file_path)
            save_path = os.path.join(save_dir_path, file)
            if not os.path.exists(save_path):
                predict(load_path=load_path, file_path=file_path, save_path=save_path)
            
if __name__ == '__main__':
    fire.Fire()