import React, { useEffect } from 'react';
import { Link } from "react-router-dom";
import './Home.css';
import Navbar from './Navbar';
import arrow from './images/arrow.png';
import upload from './images/upload.png';
import customize from './images/customize.png';
import download from './images/download.png';

const Home = () => {
    // Define the setPageTitle function within the component
    const setPageTitle = (title) => {
        document.title = title;
    };
    useEffect(() => {
        setPageTitle('Scriptify');
    }, []);
    return (
        <div className="home-page">
            <div className="home">
                <Navbar/>
                <div className="detail-box">
                    <h1>Transforming Stories into Engaging Scripts</h1>
                    <p>Are you an aspiring writer, a seasoned novelist, or simply someone with a story to tell? Look no further! At Scriptify, we're passionate about transforming your stories into captivating scripts that come to life.</p>
                    <div className="btn-box">
                        <Link to="/insert-story" className="btn1">Generate Script</Link>
                    </div>
                </div>
                <div className="down">
                    <a href="#use">
                        <img src={arrow} alt="Slide Down" />
                    </a>
                </div>
            </div>

            <section className="use_section layout_padding" id="use">
                <div className="use_container">
                    <div className="container ">
                        <div className="heading_container">
                            <h2>HOW TO USE</h2>
                            <p>Get started today and transform your written story into a captivatingscript with just a few simple steps!</p>
                        </div>
                        <div className="row">
                            <div className="box">
                                <div className="img-box">
                                    <img src={upload} alt="Upload" />
                                </div>
                                <div className="detail-box">
                                    <h5>Upload Your Story</h5>
                                    <p>Begin by uploading your written story or script to our platform, providing any specific instructions or preferences you have.</p>
                                </div>
                            </div>

                            <div className="box">
                                <div className="img-box">
                                    <img src={customize} alt="Description" />
                                </div>
                                <div className="detail-box">
                                    <h5>Customize Settings</h5>
                                    <p>Explore our intuitive interface to customize script settings such as character dialogue, plot development, pacing, and additional elements, tailoring the script to suit your narrative style.</p>
                                </div>
                            </div>
                            <div className="box">
                                <div className="img-box">
                                    <img src={download} alt="Description" />
                                </div>
                                <div className="detail-box">
                                    <h5>Review and Download</h5>
                                    <p>Once your script is generated, review the preview to ensure it meets your expectations. Then, simply download the finalized script ready for sharing, publishing, or further refinement.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <section className="footer_section">
                <div className="container">
                    <p><Link to="/">Scriptify</Link> by Prethan Ramesh | <span id="displayYear">2024</span> &copy;</p>
                </div>
            </section>

        </div>
    );
}

export default Home;
