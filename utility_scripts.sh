docker run -it --runtime nvidia --shm-size=64G --gpus "device=3" --ipc=host  --volume /home/bartek/PHYSIONET-2025/data/CODE-15/micro:/data/input  --volume /home/bartek/PHYSIONET-2025/data/CODE-15/micro:/data/test  --volume /home/bartek/PHYSIONET-2025/data/outputs:/data/output  --volume /home/bartek/PHYSIONET-2025/models:/models image:latest 
docker run -it --runtime nvidia --shm-size=64G --gpus "device=3" --ipc=host  --volume /home/bartek/PHYSIONET-2025/training_data/all_files:/data/input  --volume /home/bartek/PHYSIONET-2025/test_files:/data/test  --volume /home/bartek/PHYSIONET-2025/test_outputs:/data/output  --volume /home/bartek/PHYSIONET-2025/models:/models image:latest

nohup python train_model.py -d /data/input/ -m /models/ &


python run_model.py -d /data/test/ -m /models/ -o /data/output/


python evaluate_model.py -d /data/test/ -o /data/output/
