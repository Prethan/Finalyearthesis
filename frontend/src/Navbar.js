import React from 'react';
import './Navbar.css';
import { Link } from 'react-router-dom';
// import logo from './images/logo-no-background.png';

const Navbar = () => {
    return (
        <div className="navigation-bar">
            <header>
                <nav className="navbar">
                    {/* <div className="logo">
                        <a className="navbar-brand" href="index.html">
                            <img src={logo} alt="Logo" />
                        </a>
                    </div> */}
                    <div className="navbar-right">
                        <div className="nav-item"><Link to="/" className="nav-link"> HOME </Link></div>
                        <div className="nav-item"><Link to="/insert-story" className="nav-link"> CREATE SCRIPT </Link></div>
                        <div className="nav-item"><Link to="/upcoming-features" className="nav-link"> UPCOMING FEATURES </Link></div>
                        <div className="nav-item"><Link to="/about" className="nav-link"> ABOUT US </Link></div>
                        <div className="nav-item"><Link to="/contact" className="nav-link"> CONTACT US </Link></div>
                    </div>
                </nav>
            </header>
        </div>
    );
};

export default Navbar;