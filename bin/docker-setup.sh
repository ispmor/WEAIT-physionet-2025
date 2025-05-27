docker build -t image .
docker run -it --runtime nvidia --shm-size=64G --gpus "device=3" --ipc=host  --volume /home/bartek/PHYSIONET-2025/training_data/all_files:/data/input  --volume /home/bartek/PHYSIONET-2025/test_files:/data/test  --volume /home/bartek/PHYSIONET-2025/test_outputs:/data/output  --volume /home/bartek/PHYSIONET-2025/models:/models image:latest
