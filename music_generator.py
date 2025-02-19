import note_seq
from magenta.models.music_vae import TrainedModel
import tensorflow as tf

# Initialize the pre-trained MusicVAE model (make sure to download the appropriate checkpoint)
music_vae = TrainedModel(
    model_name='cat-mel_2bar_big',
    batch_size=4,
    checkpoint_dir_or_path='path_to_checkpoint'  # Replace with your checkpoint path
)

def generate_music():
    # Generate a 2-bar melody (32 time steps)
    generated_sequence = music_vae.sample(n=1, length=32, temperature=1.0)[0]
    note_seq.sequence_proto_to_midi_file(generated_sequence, 'generated.mid')
    print("Music generated and saved as generated.mid")

if __name__ == '__main__':
    generate_music()
