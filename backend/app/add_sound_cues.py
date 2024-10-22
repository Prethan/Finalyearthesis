import pandas as pd
import os
import re

class AddSoundCues:
    def __init__(self):
        # Get the directory of the current script
        current_directory = os.path.dirname(__file__)

        # Construct the path to the original dataset CSV file 
        self.sound_cues_file = os.path.join(current_directory, 'datasets', 'sound_cues.csv')
        
        self.sound_cues = self.load_sound_cues()

    # Function to load sound cues from CSV file using Pandas
    def load_sound_cues(self):
        df = pd.read_csv(self.sound_cues_file)
        sound_cues = dict(zip(df['sound_source'], df['sound_cue']))
        return sound_cues

    # Function to identify sound source
    def identify_sound_source(self, dialogue):
        """
        Identifies the sound source in the given dialogue.

        Args:
            dialogue: A string containing the dialogue to be analyzed.

        Returns:
            sound_source: The identified sound source, if found. None otherwise.
        """

        for sound in self.sound_cues.keys():
            # Using regular expression to match whole words only
            pattern = r'\b{}\b'.format(re.escape(sound))
            if re.search(pattern, dialogue.lower()):
                return sound
        return None  # No sound source found

    # Function to add sound cue 
    def add_sound_cue(self, dialogue):
        """
        Adds a sound cue to the given dialogue if a sound source is identified.

        Args:
            dialogue: A string containing the dialogue to which the sound cue will be added.

        Returns:
            dialogue_with_cue: The dialogue with the added sound cue, if a sound source is identified. Otherwise, returns the original dialogue.
        """

        sound_source = self.identify_sound_source(dialogue)
        if sound_source:
            cue = self.sound_cues[sound_source]
            return dialogue + "\n" + "\n[SFX : " + cue + "]"
        else:
            return dialogue
        
        