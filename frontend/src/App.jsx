import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  ArcElement,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Doughnut, Line } from 'react-chartjs-2';
import { Send, Mic, Search, MoreVertical, Paperclip, Smile } from 'lucide-react';

ChartJS.register(
  ArcElement,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE = 'http://127.0.0.1:8000';

const emotionColors = {
  // Positive
  joy: '#ffdd00', excitement: '#ffd700', love: '#ff69b4', pride: '#ffa500', 
  admiration: '#ffd700', amusement: '#ffb6c1', approval: '#98fb98', caring: '#ffc0cb',
  desire: '#ff1493', gratitude: '#32cd32', optimism: '#00fa9a', relief: '#00ffff',
  // Negative
  sadness: '#4facfe', anger: '#ff0844', fear: '#667eea', disappointment: '#4682b4',
  disapproval: '#708090', disgust: '#a8c0ff', embarrassment: '#ffc0cb', grief: '#2f4f4f',
  nervousness: '#dda0dd', remorse: '#8b4513', annoyance: '#ff4500',
  // Ambiguous
  surprise: '#fda085', confusion: '#d3d3d3', curiosity: '#eee8aa', realization: '#f0e68c', questioning: '#add8e6',
  // Neutral
  neutral: '#94a3b8',
  // Extras
  urgency: '#ff4500'
};

const emotionEmojis = {
  joy: '😊', excitement: '🤩', love: '❤️', pride: '😌', 
  admiration: '🙌', amusement: '😂', approval: '👍', caring: '🥰',
  desire: '😍', gratitude: '🙏', optimism: '✨', relief: '😮‍💨',
  sadness: '😢', anger: '😠', fear: '😨', disappointment: '😞',
  disapproval: '👎', disgust: '🤢', embarrassment: '😳', grief: '💔',
  nervousness: '😬', remorse: '😔', annoyance: '🙄',
  surprise: '😲', confusion: '😕', curiosity: '🤔', realization: '💡',
  neutral: '😐', questioning: '❓', urgency: '⚠️'
};

const macroColors = {
  Positive: '#10b981',
  Negative: '#ef4444',
  Ambiguous: '#f59e0b',
  Neutral: '#64748b'
};

function App() {
  const [messages, setMessages] = useState([]);
  const [inputVal, setInputVal] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [activeAnalysisId, setActiveAnalysisId] = useState(null);
  
  const chatEndRef = useRef(null);

  // Initialize Speech Recognition
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = SpeechRecognition ? new SpeechRecognition() : null;

  useEffect(() => {
    setMessages([{
      id: 1,
      sender: 'bot',
      text: "Hello! 🤖 I am your Advanced Emotion Bot. Type to me or send a voice note. I'm ready to analyze your 28 deep emotional states. ✨",
      emotions: [],
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }]);
  }, []);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (textOverride = null) => {
    const textToSend = textOverride || inputVal;
    if (!textToSend.trim()) return;

    if (!textOverride) setInputVal('');

    const newUserMsg = {
      id: Date.now(),
      sender: 'user',
      text: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    
    setMessages(prev => [...prev, newUserMsg]);
    setIsLoading(true);

    try {
      const [predRes, explainRes] = await Promise.all([
        axios.post(`${API_BASE}/predict`, { text: textToSend }),
        axios.post(`${API_BASE}/explain`, { text: textToSend })
      ]);

      const data = predRes.data;
      const explanation = explainRes.data.important_words || {};

      const primaryEmotion = data.primary_emotion;
      const intensity = data.intensity;
      const macroSentiment = data.macro_sentiment;
      const empathyResponse = data.empathy_response;
      const labels = data.emotions.map(e => {
        const titleCase = String(e).charAt(0).toUpperCase() + String(e).slice(1);
        const emoji = emotionEmojis[e.toLowerCase()] || '';
        return `${titleCase} ${emoji}`.trim();
      });
      
      const newBotMsg = {
        id: Date.now() + 1,
        sender: 'bot',
        text: empathyResponse,
        macro_sentiment: macroSentiment,
        raw_emotion: primaryEmotion,
        emotions: labels,
        intensity: intensity,
        confidences: data.confidences,
        explanation: explanation,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, newBotMsg]);
      setActiveAnalysisId(newBotMsg.id);

    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        sender: 'bot',
        text: "Make sure the backend is running and the new model finishes downloading.",
        isError: true,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    }
    setIsLoading(false);
  };

  const handleMicClick = () => {
    if (!recognition) {
      alert("Your browser doesn't support Speech Recognition. Please try Google Chrome.");
      return;
    }

    if (isRecording) {
      recognition.stop();
      setIsRecording(false);
      return;
    }

    recognition.lang = 'en-US';
    recognition.start();
    setIsRecording(true);

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInputVal(transcript);
      setIsRecording(false);
      handleSend(transcript); 
    };

    recognition.onerror = (event) => {
      if (event.error !== 'no-speech') {
        alert(`Microphone error: ${event.error}. Please ensure microphone permissions are granted and try using Google Chrome.`);
      }
      setIsRecording(false);
    };
    recognition.onend = () => setIsRecording(false);
  };

  const trackedResponses = messages.filter(m => m.sender === 'bot' && m.raw_emotion);
  
  const emotionCounts = {};
  const macroCounts = { Positive: 0, Negative: 0, Ambiguous: 0, Neutral: 0 };

  trackedResponses.forEach(m => {
    emotionCounts[m.raw_emotion] = (emotionCounts[m.raw_emotion] || 0) + 1;
    if (m.macro_sentiment && macroCounts[m.macro_sentiment] !== undefined) {
      macroCounts[m.macro_sentiment]++;
    }
  });

  const doughnutData = {
    labels: Object.keys(emotionCounts).map(l => l.charAt(0).toUpperCase() + l.slice(1)),
    datasets: [{
      data: Object.values(emotionCounts),
      backgroundColor: Object.keys(emotionCounts).map(l => emotionColors[l] || '#fff'),
      borderWidth: 0,
      hoverOffset: 4
    }]
  };

  const sentimentData = {
    labels: Object.keys(macroCounts),
    datasets: [{
      data: Object.values(macroCounts),
      backgroundColor: [macroColors.Positive, macroColors.Negative, macroColors.Ambiguous, macroColors.Neutral],
      borderWidth: 0,
      hoverOffset: 4
    }]
  };

  const commonChartOptions = {
    responsive: true,
    cutout: '70%',
    plugins: {
      legend: { display: false }
    }
  };

  return (
    <div className="wa-app-container">


      {/* CENTER CHAT AREA */}
      <div className="wa-chat-area">
        {/* Chat Header */}
        <div className="wa-chat-header">
          <div className="header-left" onClick={() => setShowSidebar(!showSidebar)} style={{cursor:'pointer'}}>
            <div className="chat-dp bot-dp"></div>
            <div className="chat-title-info">
              <h2>Advanced Emotion Bot</h2>
              <span>Online • Engine: roberta-base-go_emotions</span>
            </div>
          </div>
          <div className="header-icons">
            <Search size={20} title="Toggle Dashboard" onClick={() => setShowSidebar(!showSidebar)} style={{cursor:'pointer'}} />
            <MoreVertical size={20} />
          </div>
        </div>

        {/* Chat Main Window (Background pattern simulation) */}
        <div className="wa-messages-container">
          <div className="date-badge">Today</div>
          {messages.map(msg => (
            <div key={msg.id} className={`wa-message-row ${msg.sender === 'user' ? 'wa-out' : 'wa-in'}`}>
              <div className="wa-message-bubble">
                <div className="wa-message-text">{msg.text}</div>
                <div className="wa-message-time">
                  {msg.timestamp}
                  {msg.sender === 'bot' && msg.raw_emotion && (
                     <span className="history-btn" 
                           onClick={() => { setActiveAnalysisId(msg.id); setShowSidebar(true); }} 
                           title="Show Analysis in Dashboard" 
                           style={{cursor: 'pointer', marginLeft: '6px', opacity: activeAnalysisId === msg.id ? 1 : 0.6}}>
                       📊
                     </span>
                  )}
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="wa-message-row wa-in">
              <div className="wa-message-bubble typing">
                 <span className="dot"></span><span className="dot"></span><span className="dot"></span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Bar */}
        <div className="wa-input-bar">
          <div className="wa-input-wrap">
            <input 
              type="text" 
              placeholder="Type a message"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            />
          </div>
          {inputVal.trim() ? (
            <div className="wa-send-btn" onClick={() => handleSend()}>
               <Send size={20} color="#fff" />
            </div>
          ) : (
            <div className={`wa-voice-btn ${isRecording ? 'recording' : ''}`} onClick={handleMicClick}>
               <Mic size={24} color="#fff" />
            </div>
          )}
        </div>
      </div>

      {/* RIGHT SIDEBAR - ANALYTICS DASHBOARD */}
      {showSidebar && (
        <div className="wa-dashboard-sidebar">
          <div className="wa-sidebar-header">
            <h3>Emotion Dashboard</h3>
          </div>
          
          <div className="dashboard-content">
            <div className="dash-card">
               <h4>Conversation Sentiment</h4>
               <p className="dash-sub">Macro-level Positivity vs Negativity</p>
               {trackedResponses.length > 0 ? (
                 <div className="chart-box">
                    <Doughnut data={sentimentData} options={commonChartOptions} />
                 </div>
               ) : <div className="empty-state">No data yet</div>}
               <div className="dash-legend">
                 {Object.entries(macroColors).map(([k, v]) => (
                    <div key={k} className="legend-item"><span className="dot" style={{background: v}}></span>{k}</div>
                 ))}
               </div>
            </div>

            <div className="dash-card">
               <h4>28-Emotion Spectrum</h4>
               <p className="dash-sub">Granular breakdown of detected sub-emotions</p>
               {trackedResponses.length > 0 ? (
                 <div className="chart-box">
                    <Doughnut data={doughnutData} options={commonChartOptions} />
                 </div>
               ) : <div className="empty-state">No data yet</div>}
            </div>
            
            {activeAnalysisId && messages.find(m => m.id === activeAnalysisId) && (
              <div className="dash-card info-card">
                 <h4 style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                    Message Analysis
                    <span style={{fontSize: '0.75rem', opacity: 0.6}}>ID: {activeAnalysisId}</span>
                 </h4>
                 
                 {(() => {
                   const activeResponse = messages.find(m => m.id === activeAnalysisId);
                   return (
                     <>
                       <div className="wa-emotion-tags" style={{marginTop: '10px'}}>
                         <span className="macro-tag" style={{backgroundColor: macroColors[activeResponse.macro_sentiment] + '33', color: macroColors[activeResponse.macro_sentiment]}}>
                           {activeResponse.macro_sentiment} {activeResponse.intensity && `• ${activeResponse.intensity}`}
                         </span>
                         {activeResponse.emotions?.map((e, idx) => (
                            <span key={idx} className="micro-tag" style={{borderColor: emotionColors[activeResponse.raw_emotion]}}>{e}</span>
                         ))}
                       </div>
                       {activeResponse.explanation && Object.keys(activeResponse.explanation).length > 0 && (
                         <div className="wa-explanation-wrap" style={{marginTop: '15px'}}>
                            <div className="wa-exp-title">Explainability Drop:</div>
                            {Object.entries(activeResponse.explanation).map(([word, score]) => (
                              score > 0 ? (
                                <span key={word} className="wa-exp-word" style={{ opacity: Math.min(1, 0.4 + (score/100)) }}>
                                  {word}
                                </span>
                              ) : null
                            ))}
                         </div>
                       )}
                     </>
                   );
                 })()}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
