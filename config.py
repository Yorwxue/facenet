import os


program_path, _ = os.path.split(os.path.realpath(__file__))
dataset_path = '/csist/clliao/dataset/face_recognition_dataset/MsCelebV1/align/'

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)
