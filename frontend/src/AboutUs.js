import React, { useEffect } from 'react';
import './AboutUs.css'
import mission from './images/mission.png';
import vision from './images/vision.png';
import values from './images/values.png';
import Navbar from './Navbar';

const AboutUs = () => {
    // Define the setPageTitle function within the component
    const setPageTitle = (title) => {
        document.title = title;
    };
    useEffect(() => {
        setPageTitle('Scriptify');
    }, []);
    return (

        <div className="aboutus">
            <Navbar/>
            <div className="display-section">
                <div className="heading_container">
                    <h2>ABOUT US</h2>
                    <p>Welcome to Scriptify! We are passionate storytellers and innovators dedicated to revolutionizing the way narratives are experienced.</p>
                    <p>Scriptify was born out of a shared love for storytelling and technology. Frustrated by the limitations of traditional storytelling mediums, we set out to create a platform that harnesses the power of audio to bring stories to life in ways never before imagined. </p>
                    <p>Here's a glimpse into who we are and what drives us:</p>
                </div>
                <div className="row">
                    <div className="box">
                        <div className="img-box">
                            <img src={mission} alt="Description" />
                        </div>
                        <div className="detail-box">
                            <h5>Our Mission</h5>
                            <p>To empower storytellers by transforming their narratives into immersive audio experiences through cutting-edge technology.</p>
                        </div>
                    </div>

                    <div className="box">
                        <div className="img-box">
                            <img src={vision} alt="Description" />
                        </div>
                        <div className="detail-box">
                            <h5>Our Vision</h5>
                            <p>To revolutionize the way stories are shared and experienced, creating a world where every tale finds its voice in captivating audio form.</p>
                        </div>
                    </div>
                    <div className="box">
                        <div className="img-box">
                            <img src={values} alt="Description" />
                        </div>
                        <div className="detail-box">
                            <h5>Our Values</h5>
                            <p>At AudioCraft, we value integrity, innovation, collaboration, excellence, and impact.</p>
                        </div>
                    </div>
                </div>

            </div>
        </div>

    );
}

export default AboutUs;