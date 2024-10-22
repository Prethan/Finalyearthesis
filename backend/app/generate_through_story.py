import openai
import nltk
import spacy

# Set OpenAI API key
openai.api_key = "sk-Sl6Mfub7CR6bMS4-fwiS5GMHAAh5iqq16J-inNU_23T3BlbkFJ4xrNKRXfWRyFdYGUIDiEPRM7a9AH21hhq5qDhxOTIA"

# Download nltk resources
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

class GenerateThroughStory:

    # Generate enhanced scene descriptions based on the story
    def generate_heading_and_descriptions(self, story, genre, tone, temp):
        """
        Generates a scene heading and description based on the provided story, genre, tone, and temperature.

        Args:
            story (str): The story prompt to base the scene on.
            genre (str): The specified genre for the scene.
            tone (str): The specified tone for the scene.
            temp (float): The temperature parameter for the GPT-3 API.

        Returns:
            list: A list containing a dictionary with keys "heading" and "description",
                  representing the generated scene heading and description.
        """

        if not story.strip() or not genre.strip() or not tone.strip():
            print("*** ValueError *** \nMethod Name: generate_heading_and_descriptions() \nError: Story, genre, and tone must not be empty.")
            raise ValueError("\nMethod Name: generate_heading_and_descriptions() \nError: Story, genre, and tone must not be empty.")

        # Generate scene heading
        heading_prompt = f'''
        Your task is to generate a scene heading based on the story ```{story}```. 
        Focus on the specified genre ```{genre}``` and tone ```{tone}```. 
        Maximum length of the scene heading should be 20 words.
        It should be in the following format : INT./EXT. LOCATION - DAY/NIGHT. 
        '''

        try:
            heading_response = openai.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=heading_prompt,
                temperature=temp,
                n=1,
                max_tokens=128
            )

            heading = heading_response.choices[0].text.strip()
        except Exception as e:
            print(f"*** Exception *** \nMethod Name: generate_heading_and_descriptions() \nError: Error generating scene heading: {e}")
            raise Exception(f"\nMethod Name: generate_heading_and_descriptions() \nError: Error generating scene heading: {e}")

        # Generate scene description
        description_prompt = f'''
        Your task is to generate a scene description based on the story ```{story}```. 
        This is the scene heading ```{heading}```. 
        Focus on the specified genre ```{genre}``` and tone ```{tone}``` when generating the scene description. 
        Enhance the scene with sensory details for a richer experience. 
        Only return the scene description without the scene heading as a single paragraph. 
        Maximum length of the paragraph should be 100 words.
        '''

        try:
            description_response = openai.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=description_prompt,
                temperature=temp,
                n=1,
                max_tokens=512
            )

            description = description_response.choices[0].text.strip()
        except Exception as e:
            print(f"*** Exception *** \nMethod Name: generate_heading_and_descriptions() \nError: Error generating scene description: {e}")
            raise Exception(f"\nMethod Name: generate_heading_and_descriptions() \nError: Error generating scene description: {e}")

        scene = [{"heading": heading, "description": description}]
        return scene

    # Function to identify characters and generate character descriptions
    def generate_character_descriptions(self, story, character, temp):
        """
        Generates a character description based on the provided story and character name.

        Args:
            story (str): The story prompt to use as a reference.
            character (str): The name of the character for whom the description is to be generated.
            temp (float): The temperature parameter for the GPT-3 API.

        Returns:
            str: The generated character description.
        """

        if not story.strip() or not character.strip():
            print("*** ValueError *** \nMethod Name: generate_character_descriptions() \nError: Story and character must not be empty.")
            raise ValueError("\nMethod Name: generate_character_descriptions() \nError: Story and character must not be empty.")

        prompt = f'''
        Your task is to generate the character description for the following character ```{character}```. 
        Return the character description as a single paragraph. 
        Maximum length of the paragraph should be 100 words. 
        Take the following story as a reference when generating the character description \n```{story}```\n
        '''

        try:
            model = "gpt-3.5-turbo-instruct"
            response = openai.completions.create(
                model=model,
                prompt=prompt,
                temperature=temp,
                max_tokens=512
            )

            character_description = response.choices[0].text.strip()
        except Exception as e:
            print(f"*** Exception *** \nMethod Name: generate_character_descriptions() \nError: Error generating character description: {e}")
            raise Exception(f"\nMethod Name: generate_character_descriptions() \nError: Error generating character description: {e}")

        return character_description

    # Function to generate character dialogues
    def generate_dialogues(self, story, scene, character_list, character_descriptions, genre, tone, temp):
        """
        Modifies existing character dialogues in the story or generates new dialogues for the characters listed.

        Args:
            story (str): The story where dialogues are to be modified or new dialogues generated.
            scene (str): The scene description to take into consideration.
            character_list (list): List of characters for whom dialogues are to be modified or generated.
            character_descriptions (str): The descriptions of the characters to be considered when generating dialogues.
            genre (str): The genre of the story.
            tone (str): The tone of the story.
            temp (float): The temperature parameter for the GPT-3 API.

        Returns:
            str: The modified or generated dialogues.
        """

        if not story.strip() or not scene.strip() or not character_list or not character_descriptions or not genre.strip() or not tone.strip():
            print("*** ValueError *** \nMethod Name: generate_dialogues() \nError: Story, Scene, character list, character descriptions, genre and tone must not be empty.")
            raise ValueError("\nMethod Name: generate_dialogues() \nError: Story, Scene, character list, character descriptions, genre and tone must not be empty.")

        prompt = f'''
        Your task is to modify character dialogues included in the story: ```{story}``` or generate more suitable and interactive character dialogues for the characters listed in ```{character_list}```. 
        Take the scene description: ```{scene}``` and character descriptions: ```{character_descriptions}``` into consideration when generating character dialogues. 
        The dialogues generated should be relevant to the plot of the story given and should reflect the genre ```{genre}``` and the tone ```{tone}```. 
        There should be a narrator for the narrating so narrative elements that guide the delivery, tone, and pacing of the narration should be included when generating dialogues. 
        No voiceovers must be there and no scene descriptions unless narrated by the narrator. 
        The maximum length of the generated dialogues should be at least 1000 words but should not exceed 1000 words.
        The dialogues should be in standard screenplay conventions, including:
        - Character names in all caps when introducing dialogue
        - Dialogue blocks indented and centered
        - Parentheticals for character actions or expressions

        Example format:
        JOHN:
        (smiling)
        I think this just might be my masterpiece.   
        '''

        try:
            model = "gpt-3.5-turbo-instruct"
            response = openai.completions.create(
                model=model,
                prompt=prompt,
                temperature=temp,
                max_tokens=2048
            )

            dialogue = response.choices[0].text.strip()
        except Exception as e:
            print(f"*** Exception *** \nMethod Name: generate_dialogues() \nError: Error generating dialogues: {e}")
            raise Exception(f"\nMethod Name: generate_dialogues() \nError: Error generating dialogues: {e}")

        return dialogue
