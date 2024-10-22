import React, { useState, useEffect } from 'react';
import { CircularProgressbar } from 'react-circular-progressbar'
import 'react-circular-progressbar/dist/styles.css';

const Progressbar = () => {
  const [progress, setProgress] = useState(10); // Initial progress

  useEffect(() => {
    const intervalId = setInterval(() => {
      if (progress < 100) {
        setProgress((prevProgress) => prevProgress + 10);
      } else {
        clearInterval(intervalId); // Stop interval when progress reaches 100
      }
    }, 2000); // Update progress every 2 seconds

    return () => clearInterval(intervalId); // Cleanup function to clear interval on unmount
  }, [progress]);

  return (
    <CircularProgressbar
      value={progress}
      styles={{
        root: { width: 50, height: 50 },
        path: { stroke: '#ffeba7', strokeWidth: 6 },
        circle: { strokeWidth: 6 },
      }}
    />
  );
};

export default Progressbar;
