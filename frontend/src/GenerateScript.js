import React, { useEffect, useState } from 'react';
import { useLocation } from "react-router-dom";
import { PDFDownloadLink, Page, Text, View, Document, StyleSheet } from '@react-pdf/renderer';
import './GenerateScript.css';
import Navbar from './Navbar';

const styles = StyleSheet.create({
    page: {
        flexDirection: 'row',
        backgroundColor: '#fcf0c7'
    },
    section: {
        fontSize: 13,
        margin: 30,
        padding: 30,
        flexGrow: 1
    },
    heading: {
        fontWeight: 'bold',
        textAlign: 'center',
        marginBottom: 20
    }
});

const GenerateScript = () => {
    const location = useLocation();
    const result = location.state && location.state.result;
    const chapterHeading = result && result.chapter_heading;
    const chapterDescription = result && result.chapter_description;
    const script = result && result.script;
    const characterDescriptions = result && result.character_descriptions;

    const [activeTab, setActiveTab] = useState('chapter'); // State to track active tab

    const setPageTitle = (title) => {
        document.title = title;
    };

    useEffect(() => {
        setPageTitle('Audio Craft');
    }, []);

    // Function to handle tab click
    const openTab = (tabName) => {
        setActiveTab(tabName);
    };

    return (
        <div className="generate-script">
            <Navbar />
            <div className="display-section">
                <div className="display-boxes jsGridView">
                    <div className="display-box-wrapper">
                        <div className="tabs">
                            <div className="tab-container">
                                <button className={`tab-link ${activeTab === 'chapter' ? 'active' : ''}`} onClick={() => openTab('chapter')}>Chapter</button>
                                <button className={`tab-link ${activeTab === 'characters' ? 'active' : ''}`} onClick={() => openTab('characters')}>Characters</button>
                            </div>
                            <div id="chapter" className={`tabcontent tab-content ${activeTab === 'chapter' ? 'active' : ''}`}>
                                <div>
                                    <h3>{chapterHeading}</h3>
                                    <p>{chapterDescription}</p>
                                </div>
                            </div>

                            <div id="characters" className={`tabcontent tab-content ${activeTab === 'characters' ? 'active' : ''}`}>
                                {characterDescriptions && characterDescriptions.map((character, index) => (
                                    <div key={index}>
                                        <h3>{character.character}</h3>
                                        <p>{character.description}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div className="buttons-container">
                            {/* PDF Download Link */}
                            <PDFDownloadLink document={<MyDocument chapterHeading={chapterHeading} script={script} />} fileName="script.pdf">
                              {({ blob, url, loading, error }) =>
                                <button>Download Script</button>
                              }
                            </PDFDownloadLink>
                        </div>
                    </div>
                    <div className="display-box-wrapper">
                        <div className="container">
                            <p>{script}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// Custom PDF Document component
const MyDocument = ({ chapterHeading, script }) => (
    <Document>
        <Page size="A4" style={styles.page}>
            <View style={styles.section}>
                <Text style={styles.heading}>{chapterHeading}</Text>
                <Text>{script}</Text>
            </View>
        </Page>
    </Document>
);

export default GenerateScript;

