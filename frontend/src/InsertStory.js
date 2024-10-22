import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './InsertStory.css';
import Progressbar from './ProgressBar'; // Assuming SimpleProgressbar.js is in the same directory
import Navbar from './Navbar';


const InsertStory = () => {
    const [story, setStory] = useState('');
    const [genre, setGenre] = useState('');
    const [tone, setTone] = useState('');
    const [characters, setCharacters] = useState('');
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();
    const [creativity, setCreativity] = useState(1);

    const [characterCount, setCharacterCount] = useState(0);

    const handleCountChange = (e) => {
        const value = e.target.value;
        setStory(value);
        setCharacterCount(value.length);
    };


    // Define handleCreativityChange function
    const handleCreativityChange = (e) => {
        const value = parseInt(e.target.value);
        setCreativity(value); // Update the creativity state with the raw value from the range input
    }

    // Map the creativity state value to the desired output value for your API call or other purposes
    const mappedCreativity = (() => {
        switch (creativity) {
            case 1:
                return 0.3;
            case 2:
                return 0.4;
            case 3:
                return 0.5;
            case 4:
                return 0.6;
            case 5:
                return 0.7;
            default:
                return 0.5;
        }
    })();

    const handleSubmit = (e) => {
        e.preventDefault();
        setLoading(true); // Set loading state to true when form is submitted
        axios.post('http://localhost:5000/generate_script', {
            story: story,
            genre: genre,
            tone: tone,
            characters: characters,
            temperature: mappedCreativity,
        })
            .then(response => {
                navigate('/generate-script', { state: { result: response.data } });
            })
            
            .catch(error => {
                if (error.response) {
                    // Server responded with an error status code
                    if (error.response.status === 400) {
                        navigate('/error400');
                    } else if (error.response.status === 500) {
                        navigate('/error500');
                    } else {
                      console.error('Error:', error.response.data.error);
                      // Handle other errors if needed
                    }
                }
            })
            .finally(() => {
                setLoading(false); // Set loading state back to false when request is completed
            });
    };

    const setPageTitle = (title) => {
        document.title = title;
    };

    useEffect(() => {
        setPageTitle('Scriptify');
    }, []);

    return (
        <div className="insert-story">
            <Navbar />
            <div className="display-section">
                <form onSubmit={handleSubmit}>
                    <div className="display-boxes jsGridView">
                        <div className="display-box-wrapper">
                            <h1>Let's start generating your script!</h1>
                            <p>Unlock the magic of your imagination! Begin crafting your script by typing or pasting your book chapter into the text area below. The maximum number of characters allowed is 2000 ({characterCount}/2000)</p>
                            <textarea className="script-textarea" rows="13" placeholder="Type or paste your storybook chapter here..." value={story} onChange={handleCountChange} maxLength={2000} required></textarea>                         
                            {loading ? (
                                <div className="progress-bar">
                                    <p> <Progressbar/> Generating... </p>
                                </div>
                            ) : (
                                <button className="submit-btn">Generate Script</button>
                            )}
                        </div>

                        <div className="display-box-wrapper">
                            <div className="customization">
                                <div className="custom-input characters">
                                    <label>Characters</label> <br></br>
                                    <textarea className="script-text" rows="2" placeholder="Enter characters seperated by a comma..." value={characters} onChange={(e) => setCharacters(e.target.value)} maxLength={500} required></textarea>
                                </div>
                                <div className="custom-input">
                                    <label>Genre</label><br></br>
                                    <select className="script-select" value={genre} onChange={(e) => setGenre(e.target.value)} required>
                                        <option value="Comedy">Comedy</option>
                                        <option value="Drama">Drama</option>
                                        <option value="Horror">Horror</option>
                                        <option value="Romance">Romance</option>
                                        <option value="Fiction">Fiction</option>
                                        <option value="Mystery">Mystery</option>
                                    </select>
                                </div>
                                <div className="custom-input">
                                    <label>Tone</label><br></br>
                                    <select className="script-select" value={tone} onChange={(e) => setTone(e.target.value)} required>
                                        <option value="Humorous">Humorous</option>
                                        <option value="Optimistic">Optimistic</option>
                                        <option value="Romantic">Romantic</option>
                                        <option value="Mysterious">Mysterious</option>
                                        <option value="Suspenseful">Suspenseful</option>
                                        <option value="Serious">Serious</option>
                                    </select>
                                </div>
                                <div class="range-container">
                                    <label for="creativity">Creativity Level : <span>{creativity}</span></label>
                                    <input type="range" id="creativity" name="creativity" value={creativity} onChange={handleCreativityChange} min="1" max="5" required />
                                </div>
                            </div>

                        </div>
                    </div>
                </form>
            </div>
        </div>
    );
}

export default InsertStory;
