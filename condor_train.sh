#!/bin/bash
printf "universe = docker
docker_image = visionlabsapienza/workgroup:container-08082022
executable = /bin/python3
arguments = /data1/visionlab/train.py $*
output = ./condor_logs/out_\$(ClusterId)
error = ./condor_logs/err_\$(ClusterId)
log = ./condor_logs/err_\$(ClusterId)
request_cpus = 1
request_gpus = 1
request_memory = 64G
request_disk = 100G
+MountData1=TRUE
+MountData2=FALSE
+MountHomes=FALSE
queue 1" > run.sub
#echo 'universe = docker' > $project_path/run.sub
#echo 'docker_image = visionlabsapienza/workgroup:container-08082022' >> $project_path/run.sub
#echo 'executable = /bin/python3' >> $project_path/run.sub
#echo 'arguments = /data1/visionlab/train.py ' $arguments >> $project_path/run.sub
#echo 'output = ./CondorLogs/out.$(ClusterId).$(ProcId)' >> $project_path/run.sub
#echo 'error = ./CondorLogs/err.$(ClusterId).$(ProcId)' >> $project_path/run.sub
#echo 'log = ./CondorLogs/log.$(ClusterId).$(ProcId)' >> $project_path/run.sub
#echo 'request_cpus = 1' >> $project_path/run.sub
#echo 'request_gpus = 1' >> $project_path/run.sub
#echo 'request_memory = 64G' >> $project_path/run.sub
#echo 'request_disk = 100G' >> $project_path/run.sub
#echo '+MountData1=TRUE' >> $project_path/run.sub
#echo '+MountData2=FALSE' >> $project_path/run.sub
#echo '+MountHomes=FALSE' >> $project_path/run.sub
#echo 'queue 1' >> $project_path/run.sub
#
condor_submit run.sub
rm run.sub
condor_q