from flask import Flask, request, jsonify 
from flask_cors import CORS
from generate_through_story import GenerateThroughStory
from add_monologues import AddMonologues

app = Flask(__name__)
CORS(app)


@app.route('/generate_script', methods=['POST'])
def generate_script():
    """
    Generates descriptions for characters, scene, and dialogues based on the provided story, genre, tone,
    characters, and temperature from the incoming JSON request.

    Returns:
        jsonify: A JSON response containing the generated chapter heading, chapter description, script,
                and character descriptions.

    Raises:
        KeyError: If any required field is missing from the incoming JSON request.
        ValueError: If any required field in the JSON request is empty.
        Exception: If there's an error during the generation process.
    """

    # Retrieve data from the incoming JSON request.
    try:
        story = str(request.json['story'])
        genre = str(request.json['genre'])
        tone = str(request.json['tone'])
        characters = str(request.json['characters'])
        temperature = float(request.json['temperature'])
    except KeyError:
        return jsonify({'error': 'Missing required fields in the JSON request'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid value for temperature'}), 400

    # Split the string of characters into a list, removing any extra spaces around each character.
    characters_list = [character.strip() for character in characters.split(',')]

    # Validate if all required fields are provided
    if not story or not genre or not tone or not characters_list:
        return jsonify({'error': 'Please provide story, genre, tone, and characters'}), 400

    try:
        # Generate descriptions for each character
        characters_and_descriptions = {}
        for character in characters_list:
            character_description = GenerateThroughStory().generate_character_descriptions(story, character, temperature)
            characters_and_descriptions[character] = character_description

        # Convert the dictionary into a list of dictionaries
        character_descriptions_list = [{"character": character, "description": description} for character, description in characters_and_descriptions.items()]

        # Generate scene heading and description
        chapter = GenerateThroughStory().generate_heading_and_descriptions(story, genre, tone, temperature)
        chapter_heading = chapter[0]["heading"]
        chapter_description = chapter[0]["description"]

        # Generate dialogues
        dialogues = GenerateThroughStory().generate_dialogues(story, chapter_description, characters_list, character_descriptions_list, genre, tone, temperature)
    except Exception as e:
        return jsonify({'error': f'Error generating dialogues: {str(e)}'}), 500

    try:
        # Identify emotions in dialogues
        elements = AddMonologues().identify_emotions_and_dialogues(dialogues)
    except Exception as e:
        return jsonify({'error': f'Error identifying emotions: {str(e)}'}), 500

    try:
        # Add inner monologues
        inner_monologues = AddMonologues().add_inner_monologues(elements, chapter_description, temperature)

        # Generate final script
        final_script = AddMonologues().final_script(elements, inner_monologues)
    except Exception as e:
        return jsonify({'error': f'Error adding inner monologues: {str(e)}'}), 500

    response = {
        'chapter_heading': chapter_heading,
        'chapter_description': chapter_description,
        'script': final_script,
        'character_descriptions': character_descriptions_list
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
