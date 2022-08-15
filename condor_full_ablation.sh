command="bash cluster_ablation.sh --dataset_type=deap --dataset_path=data/deap --grid_search --checkpoints_path=./checkpoints --batch_size=768 --windows_size=2 --windows_stride=2"

$command --study_name=mels --mels
$command --study_name=mel_window --mel_window_size --mel_window_stride

$command --study_name=users_embeddings --users_embeddings
$command --study_name=encoder_only --encoder_only
$command --study_name=architecture --hidden_size --num_layers
$command --study_name=positional_embedding_type --positional_embedding_type
$command --study_name=regularization --noise_strength --dropout_p

$command --study_name=signal_augmentation --shifting --cropping --flipping
$command --study_name=spectrogram_augmentation --spectrogram_time_masking_perc --spectrogram_frequency_masking_perc

