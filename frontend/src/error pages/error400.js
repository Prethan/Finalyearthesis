import React, { useEffect } from 'react';
import './error.css';
import Navbar from '../Navbar';

const Error400 = () => {
    // Define the setPageTitle function within the component
    const setPageTitle = (title) => {
        document.title = title;
    };
    useEffect(() => {
        setPageTitle('Audio Craft');
    }, []);
    return (
        <div className="error">
            <Navbar/>
            <div className="display-section">
                <h1>400</h1>
                <h2>You've Sent a Bad Request...</h2>
                <p>Error code : 400</p>
            </div>
        </div>
    );
}

export default Error400;