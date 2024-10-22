import React, { useEffect } from 'react';
import './error.css';
import Navbar from '../Navbar';

const Error500 = () => {
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
                <h1>Shoot!</h1>
                <h2>Well this is unexpected...</h2>
                <p>Error code : 500</p>
                <p>An error has occured! Try to refresh this page or feel free to contact us if the issue persists.</p>
            </div>
        </div>
    );
}

export default Error500;