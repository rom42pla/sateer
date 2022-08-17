script_starter="bash cluster_train.sh"
shared_params="--checkpoints_path=./checkpoints"

deap_command="$script_starter deap data/deap $shared_params"
$deap_command --batch_size=1024 --windows_size=1 --windows_stride=1
$deap_command --batch_size=1024 --windows_size=2 --windows_stride=1
$deap_command --batch_size=256 --windows_size=10 --windows_stride=2

dreamer_command="$script_starter dreamer data/dreamer $shared_params"
$dreamer_command --batch_size=1024 --windows_size=1 --windows_stride=1
$dreamer_command --batch_size=1024 --windows_size=2 --windows_stride=1
$dreamer_command --batch_size=1024 --windows_size=2 --windows_stride=2
$dreamer_command --batch_size=256 --windows_size=10 --windows_stride=2

seed_command="$script_starter seed data/seed $shared_params"
$seed_command --batch_size=1024 --windows_size=1 --windows_stride=1
$seed_command --batch_size=1024 --windows_size=2 --windows_stride=2

amigos_command="$script_starter amigos data/amigos $shared_params"
$amigos_command --batch_size=1024 --windows_size=1 --windows_stride=1
$amigos_command --batch_size=1024 --windows_size=2 --windows_stride=2

