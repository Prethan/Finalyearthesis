import re
import openai
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import spacy
from add_sound_cues import AddSoundCues
import os

openai.api_key = "sk-Sl6Mfub7CR6bMS4-fwiS5GMHAAh5iqq16J-inNU_23T3BlbkFJ4xrNKRXfWRyFdYGUIDiEPRM7a9AH21hhq5qDhxOTIA"

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the directory where the fine-tuned BERT model is saved
model_dir = os.path.join(current_directory, 'finetune_models', 'ft_bert_sentiment_analysis')

# Load fine-tuned BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=3)  # 3 labels: positive, negative, neutral

class AddMonologues:
    
    def identify_emotions_and_dialogues(self, text):
        """
        Identify speakers, emotions, and dialogues from the given text.
        
        Args:
            text (str): The text containing dialogue with optional speaker names and emotions.
        
        Returns:
            list: A list of dictionaries, each containing 'speaker', 'emotion', and 'dialogue' keys.
        """

        if not text or not isinstance(text, str):
            print("*** ValueError *** \nMethod Name: identify_emotions_and_dialogues() \nError: Input text must be a non-empty string.")
            raise ValueError("\nMethod Name: identify_emotions_and_dialogues() \nError: Input text must be a non-empty string.")

        # Initialize list to store identified elements
        elements = []
        
        # Regular expression pattern to match speaker, emotion, and dialogue
        pattern = r'^([A-Z]+(?: [A-Z]+)*)?:\n(?:\((.*?)\))?\n?(.*)$'
        
        # Find all matches in the text
        try:
            matches = re.findall(pattern, text, re.MULTILINE)
        except Exception as e:
            print(f"*** ValueError *** \nMethod Name: identify_emotions_and_dialogues() \nError: Error occurred while matching pattern: {e}")
            raise ValueError(f"\nMethod Name: identify_emotions_and_dialogues() \nError: Error occurred while matching pattern: {e}")
        
        # Iterate through each match
        for match in matches:
            speaker = match[0].strip() if match[0] else "NARRATOR"
            emotion = match[1] if match[1] else "None"
            dialogue = match[2].strip()
            elements.append({'speaker': speaker, 'emotion': emotion, 'dialogue': dialogue})
        
        return elements

    # Function to analyze sentiment using fine-tuned BERT sentiment analysis model
    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of the given text using the fine-tuned BERT sentiment analysis model.

        Args:
            text: The text for which sentiment analysis is to be performed.

        Returns:
            sentiment: The predicted sentiment label (either "negative", "neutral", or "positive").
        """

        try:
            # Tokenize input text
            inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)

            # Forward pass through BERT model
            with torch.no_grad():
                outputs = model(**inputs)

            # Get predicted sentiment label
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

            # Map predicted class index to sentiment label
            sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = sentiment_labels[predicted_class]

            return sentiment

        except Exception as e:
            print(f"*** Exception *** \nMethod Name: analyze_sentiment() \nError: Error occurred while analyzing sentiment: {e}")
            raise Exception(f"\nMethod Name: analyze_sentiment() \nError: Error occurred while analyzing sentiment: {e}")



    def extract_entities(self, text):
        """
        Extracts named entities from text using spaCy.

        Args:
            text (str): The input text.

        Returns:
            list: A list of tuples containing named entities and their labels.
        """
        try:
            # Load an English NLP model with NER
            nlp = spacy.load("en_core_web_lg") 
        except Exception as e:
            print(f"*** RuntimeError *** \nMethod Name: extract_entities() \nError: Error loading spaCy model: {e}")
            raise RuntimeError(f"\nMethod Name: extract_entities() \nError: Error loading spaCy model: {e}")

        if not text:
            print(f"*** ValueError *** \nMethod Name: extract_entities() \nError: Empty input text.")
            raise ValueError("\nMethod Name: extract_entities() \nError: Empty input text.")

        try:
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            return entities
        except Exception as e:
            print(f"*** ValueError *** \nMethod Name: extract_entities() \nError: Error processing text: {e}")
            raise RuntimeError(f"\nMethod Name: extract_entities() \nError: Error processing text: {e}")



    def generate_inner_monologue(self, element, scene, temp):
        """
        Generates a small inner monologue for the character after their dialogue.

        Args:
            element (dict): A dictionary containing information about the character's dialogue.
                            It should have keys 'speaker', 'dialogue', and 'emotion'.
            scene (str): The scene description to consider when generating the inner monologue.
            temp (float): The temperature parameter for the GPT-3 API.

        Returns:
            str: The generated inner monologue for the character.
        """

        if not element['speaker'].strip() or not element['dialogue'].strip() or not element['emotion'].strip() or not scene.strip():
            print("*** ValueError *** \nMethod Name: generate_inner_monologue() \nError: Element, speaker, dialogue, emotion, and scene must not be empty.")
            raise ValueError("\nMethod Name: generate_inner_monologue() \nError: Element, speaker, dialogue, emotion, and scene must not be empty.")

        inner_thought_prompt = f'''
        Your task is to generate a small and simple inner monologue for the character ```{element['speaker']}``` after this dialogue: ```{element['dialogue']}```. 
        Consider the emotion of the character when delivering the dialogue ```{element['emotion']}``` and the scene ```{scene}``` when generating the inner monologue. 
        The maximum length of the generated inner monologue should not exceed 50 words."
        '''

        try:
            response = openai.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=inner_thought_prompt,
                temperature=temp,
                max_tokens=256
            )

            inner_thought = response.choices[0].text.strip()
        except Exception as e:
            print(f"*** Exception *** \nMethod Name: generate_inner_monologue() \nError: Error generating inner monologue: {e}")
            raise Exception(f"\nMethod Name: generate_inner_monologue() \nError: Error generating inner monologue: {e}")

        return inner_thought
    
    
    def add_inner_monologues(self, elements, scene, temp):
        """
        Adds inner monologues for characters based on the sentiment, entities, and emotion of their dialogues.

        Args:
            elements (list): A list of dictionaries containing information about each character's dialogue.
                             Each dictionary should have keys 'speaker', 'dialogue', 'emotion'.
            scene (str): The scene description to consider when generating the inner monologues.
            temp (float): The temperature parameter for the GPT-3 API.

        Returns:
            list: A list of inner monologues generated for each character. If no inner monologue is needed,
                  None is appended to the list for that character.
        """

        if not elements or not scene.strip():
            print("*** ValueError *** \nMethod Name: add_inner_monologues() \nError: Elements and scene must not be empty.")
            raise ValueError("\nMethod Name: add_inner_monologues() \nError: Elements and scene must not be empty.")

        inner_monologues = []
        for element in elements:
            dialogue = element['dialogue']
            sentiment = self.analyze_sentiment(dialogue) 
            entities = self.extract_entities(dialogue)    

            prompt = f'''
            Your task is to decide whether an inner monologue should be added after the following dialogue: ```{dialogue}```? 
            The sentiment of the dialogue is: ```{sentiment}```. 
            The entities identified from the dialogue is: ```{entities}```. 
            The emotion of the character at the time of the dialogue is :```{element['emotion']}```. 
            Consider the sentiment, entities, and the emotion of the dialogue when making the decision. 
            The response should be either 'yes' or 'no'.  
            '''

            try:
                response = openai.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=64
                )

                should_add_inner_dialogue = response.choices[0].text.strip().lower()
                if should_add_inner_dialogue == 'yes' and element['speaker'] != 'NARRATOR':
                    inner_monologues.append(self.generate_inner_monologue(element, scene, temp))
                else:
                    inner_monologues.append(None)  # No inner monologue needed
            except Exception as e:
                print(f"*** Exception *** \nMethod Name: add_inner_monologues() \nError: Error adding inner monologue: {e}")
                raise Exception(f"\nMethod Name: add_inner_monologues() \nError: Error adding inner monologue: {e}")

        return inner_monologues
    
    def final_script(self, elements, inner_monologues):
        """
        Generates the final script including character dialogues and inner monologues.

        Args:
            elements (list): A list of dictionaries containing information about each character's dialogue.
                             Each dictionary should have keys 'speaker', 'dialogue', 'emotion'.
            inner_monologues (list): A list of inner monologues generated for each character.

        Returns:
            str: The final script including character dialogues and inner monologues.
        """

        if not elements or not inner_monologues:
            print("*** ValueError *** \nMethod Name: final_script() \nError: Elements and inner monologues must not be empty.")
            raise ValueError("\nMethod Name: final_script() \nError: Elements and inner monologues must not be empty.")
        
        script = ""

        for element, inner_monologue in zip(elements, inner_monologues):
            speaker = element['speaker']
            emotion = element['emotion'] if element['emotion'] else "None"
            dialogue = element['dialogue']
            dialogue = AddSoundCues().add_sound_cue(dialogue)

            if emotion == "None":
                if inner_monologue:
                    script += f"{speaker}: \n{dialogue}\n"
                    script += f"\nV.O. ({speaker.title()}) {inner_monologue}\n\n"
                else:
                    script += f"{speaker}: \n{dialogue}\n\n"
            else:
                if inner_monologue:
                    script += f"{speaker}: \n({emotion}) \n{dialogue}\n"
                    script += f"\nV.O. ({speaker.title()}) {inner_monologue}\n\n"
                else:
                    script += f"{speaker}: \n({emotion}) \n{dialogue}\n\n"

        return script