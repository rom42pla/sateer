dataset=$1
command="bash condor_ablation.sh --dataset_type=$dataset --dataset_path=data/$dataset --seed=42 --grid_search --checkpoints_path=./checkpoints --batch_size=768 --windows_size=2 --windows_stride=2"

$command --study_name=mels --test_mels
$command --study_name=mel_window --test_mel_window_size ----test_mel_window_stride

$command --study_name=users_embeddings --test_users_embeddings
$command --study_name=encoder_only --test_encoder_only
$command --study_name=architecture --test_hidden_size --test_num_layers
$command --study_name=positional_embedding_type --test_positional_embedding_type
$command --study_name=regularization --test_noise_strength --test_dropout_p

$command --study_name=signal_augmentation --test_shifting --test_cropping --test_flipping
$command --study_name=spectrogram_augmentation_time --test_spectrogram_time_masking_perc
$command --study_name=spectrogram_augmentation_frequency --test_spectrogram_frequency_masking_perc

