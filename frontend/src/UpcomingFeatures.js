import React, { useEffect } from 'react';
import './UpcomingFeatures.css';
import Navbar from './Navbar';
import customize from './images/customize.png';
import headphones from './images/headphones.png';

const UpcomingFeatures = () => {
    // Define the setPageTitle function within the component
    const setPageTitle = (title) => {
        document.title = title;
    };
    useEffect(() => {
        setPageTitle('Scriptify');
    }, []);
    return (
        <div className="upcoming-features">
            <Navbar/>
            <div className="display-section">
                <div className="heading_container">
                    <h2>UPCOMING FEATURES</h2>
                    <p>Unlock the future of storytelling with our upcoming features, designed to elevate your scripts and immerse your audience in unforgettable experiences.</p>
                </div>
                <div className="row">
                    <div className="box">
                        <div className="img-box">
                            <img src={customize} alt="Description" />
                        </div>
                        <div className="detail-box">
                            <h5>Expanded Customization</h5>
                            <p>Gain more control over your scripts with new features. Fine-tune character dialogue, adjust pacing, and shape narrative structure to match your unique storytelling style.</p>
                        </div>
                    </div>

                    <div className="box">
                        <div className="img-box">
                            <img src={headphones} alt="Description" />
                        </div>
                        <div className="detail-box">
                            <h5>Audio Book Conversion</h5>
                            <p>Elevate your scripts into audio books. Professional narration, sound effects, and music integration bring your stories to life, captivating listeners and immersing them in your narrative world.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default UpcomingFeatures;