## Generate training/validation/testing jsons/pkls 

Steps to generate MMDET style train/val/test jsons/pkls:

1. Run `python generate_training_data/generate_per_scene_sync_info.py --data-root ./TruckDrive/ --sync-info-root ./TruckDrive/sensor_sync --multiprocessing` to syncronize LiDARs, Ousters, Radars, Cameras, Annotations, Global Poses.
2. Create and activate the environment as explained in `dataset_viewer/README.md`
3. Run `python  generate_training_data/generate_mmdet_annotations.py --data-root ./TruckDrive/ --sync-info-root ./TruckDrive/sensor_sync --metainfo-path metainfo.json --output-root ./TruckDrive/mmdet_annotations --multiprocessing` to generate mmdet style training and validation Jsons containing samples, labels and calibration matrixes.
