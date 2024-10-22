import React, { useEffect } from 'react';
import './error.css';
import Navbar from '../Navbar';

const Error404 = () => {
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
                <h1>404</h1>
                <h2>Page Not Found</h2>
                <p>Error code : 404</p>
                <p>Sorry we couldn't find the page you were looking for!</p>
            </div>
        </div>
    );
}

export default Error404;