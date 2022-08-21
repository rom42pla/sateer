#!/bin/bash
dataset=$1

if [ ! -d $(pwd)/condor_logs ]
then
    mkdir $(pwd)/condor_logs
    echo "created $(pwd)/condor_logs"
fi

printf "universe = docker
docker_image = visionlabsapienza/workgroup:container-08082022
executable = /bin/python3
arguments = $(pwd)/ablation_simple.py --dataset_type=$dataset --dataset_path=data/$dataset --checkpoints_path=./checkpoints --batch_size=768 --windows_size=2 --windows_stride=2 --seed=42
output = $(pwd)/condor_logs/out_\$(ClusterId)
error = $(pwd)/condor_logs/err_\$(ClusterId)
log = $(pwd)/condor_logs/log_\$(ClusterId)
request_cpus = 1
request_gpus = 1
request_memory = 64G
request_disk = 100G
+MountData1=TRUE
+MountData2=FALSE
+MountHomes=FALSE
queue 1" > run.sub

cat run.sub
condor_submit run.sub
rm run.sub
condor_q
