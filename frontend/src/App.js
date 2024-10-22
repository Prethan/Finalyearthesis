import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './App.css'; // Import your global CSS file
import Home from './Home'; // Import the Home component
import AboutUs from './AboutUs'; // Import the AboutUs component
import UpcomingFeatures from './UpcomingFeatures'; // Import the UpcomingFeatures component
import InsertStory from './InsertStory'; // Import the InsertStory component
import Contact from './Contact'; // Import the Contact component
import GenerateScript from './GenerateScript'; // Import the GenerateScript component
import Error404 from './error pages/error404';
import Error500 from './error pages/error500';
import Error400 from './error pages/error400';

function App() {
  return (
    <BrowserRouter>
      <Routes>
          <Route path="/" element={<Home />}></Route>
          <Route path="/insert-story" element={<InsertStory />}></Route>
          <Route path="/generate-script" element={<GenerateScript />}></Route>
          <Route path="/upcoming-features" element={<UpcomingFeatures />}></Route>
          <Route path="/about" element={<AboutUs />}></Route>
          <Route path="/contact" element={<Contact />}></Route>
          <Route path='*' element={<Error404 />}/>
          <Route path='/error500' element={<Error500 />}/>
          <Route path='/error400' element={<Error400 />}/>
      </Routes>
    </BrowserRouter>
  );
}

export default App;