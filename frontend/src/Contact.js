import React, { useEffect } from 'react';
import './Contact.css';
import Navbar from './Navbar';

const Contact = () => {
    // Define the setPageTitle function within the component
    const setPageTitle = (title) => {
        document.title = title;
    };
    useEffect(() => {
        setPageTitle('Scriptify');
    }, []);
    return (
        <div className="contact">
            <Navbar/>
            <div className="display-section-contact">
                <h1>Keep in touch!</h1>
                <form action="#">
                    <div className="form-group">
                        <input type="text" className="form-style" name="fullname" placeholder="Full name" required />
                    </div>
                    <div className="form-group">
                        <input type="email" className="form-style" name="email" placeholder="Email" required />
                    </div>
                    <div className="form-group">
                        <textarea rows="5" className="form-textarea" name="message" placeholder="Message..." required></textarea>
                    </div>
                    <input type="submit" value="Send Your Feedback" />
                </form>

            </div>
        </div>

    );
}

export default Contact;
