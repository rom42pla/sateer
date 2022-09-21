script_starter="bash condor_train.sh"
shared_params="--seed=42 --checkpoints_path=./checkpoints/training/with_user_embeddings"

dreamer_command="$script_starter dreamer data/dreamer --spectrogram_time_masking_perc=0 --spectrogram_frequency_masking_perc=0 --mel_window_size=1 --mel_window_stride=0.05 --disable_flipping $shared_params"
$dreamer_command --batch_size=1024 --windows_size=2 --windows_stride=2