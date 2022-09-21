script_starter="bash condor_train.sh"
shared_params="--seed=42 --checkpoints_path=./checkpoints/weights --validation=simple"

deap_command="$script_starter deap data/deap --spectrogram_time_masking_perc=0 --spectrogram_frequency_masking_perc=0 --disable_flipping $shared_params"
#$deap_command --batch_size=1024 --windows_size=2 --windows_stride=1 --model_name="eegst_deap_base"
$deap_command --batch_size=1024 --windows_size=2 --windows_stride=1 --hidden_size=256 --num_encoders=1 --num_decoders=1 --model_name="eegst_deap_small"

dreamer_command="$script_starter dreamer data/dreamer --spectrogram_time_masking_perc=0 --spectrogram_frequency_masking_perc=0 --mel_window_size=1 --mel_window_stride=0.05 --disable_flipping $shared_params"
#$dreamer_command --batch_size=1024 --windows_size=2 --windows_stride=1 --model_name="eegst_dreamer_base"
$dreamer_command --batch_size=1024 --windows_size=2 --windows_stride=1 --hidden_size=256 --num_encoders=1 --num_decoders=1 --model_name="eegst_dreamer_small"

seed_command="$script_starter seed data/seed --spectrogram_time_masking_perc=0 --spectrogram_frequency_masking_perc=0 --disable_flipping $shared_params"
#$seed_command --batch_size=1024 --windows_size=2 --windows_stride=1 --model_name="eegst_seed_base"
$seed_command --batch_size=1024 --windows_size=2 --windows_stride=1 --hidden_size=256 --num_encoders=1 --num_decoders=1 --model_name="eegst_seed_small"

amigos_command="$script_starter amigos data/amigos --num_decoders=4 --spectrogram_time_masking_perc=0 --spectrogram_frequency_masking_perc=0 --disable_flipping $shared_params"
#$amigos_command --batch_size=1024 --windows_size=2 --windows_stride=1 --model_name="eegst_amigos_base"
$amigos_command --batch_size=1024 --windows_size=2 --windows_stride=1 --hidden_size=256 --num_encoders=1 --num_decoders=1 --model_name="eegst_amigos_small"