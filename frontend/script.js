// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;

// DOM elements
let chatMessages, chatInput, sendButton, totalCourses, courseTitles, newChatButton;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    newChatButton = document.getElementById('newChatButton');

    setupEventListeners();
    createNewSession();
    loadCourseStats();
});

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // New chat
    newChatButton.addEventListener('click', startNewChat);


    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();
        
        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        // Replace loading message with error
        loadingMessage.remove();
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    const displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);
    
    let html = `<div class="message-content">${displayContent}</div>`;
    
    if (sources && sources.length > 0) {
        const sourcesHtml = groupSourcesByCourse(sources).map(group => {
            if (group.lessons.length === 0) {
                // No lesson breakdown - render as a single standalone pill
                const text = escapeHtml(group.course);
                return group.link
                    ? `<a href="${group.link}" target="_blank" rel="noopener noreferrer" class="source-pill">${text}</a>`
                    : `<span class="source-pill source-pill-plain">${text}</span>`;
            }

            const lessonPills = group.lessons.map(lesson => {
                const label = escapeHtml(lesson.label);
                return lesson.link
                    ? `<a href="${lesson.link}" target="_blank" rel="noopener noreferrer" class="source-lesson-pill">${label}</a>`
                    : `<span class="source-lesson-pill source-pill-plain">${label}</span>`;
            }).join('');

            return `
                <div class="source-group">
                    <div class="source-course-name">${escapeHtml(group.course)}</div>
                    <div class="source-lesson-row">${lessonPills}</div>
                </div>
            `;
        }).join('');

        html += `
            <details class="sources-collapsible">
                <summary class="sources-header">Sources <span class="sources-count">${sources.length}</span></summary>
                <div class="sources-content">${sourcesHtml}</div>
            </details>
        `;
    }
    
    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageId;
}

// Group flat source list ("Course - Lesson N") into per-course entries with
// deduped, numerically-sorted lesson chips, so a course isn't repeated per lesson.
function groupSourcesByCourse(sources) {
    const lessonSuffix = / - Lesson (\d+)$/;
    const groups = new Map();
    const order = [];

    sources.forEach(source => {
        const match = source.text.match(lessonSuffix);
        const course = match ? source.text.slice(0, match.index) : source.text;
        const lessonNumber = match ? parseInt(match[1], 10) : null;

        if (!groups.has(course)) {
            groups.set(course, { course, link: source.link, lessons: [], lessonNumbers: new Set() });
            order.push(course);
        }
        const group = groups.get(course);

        if (lessonNumber === null) {
            // Prefer a course-level link if we encounter one
            if (!group.link) group.link = source.link;
            return;
        }

        if (!group.lessonNumbers.has(lessonNumber)) {
            group.lessonNumbers.add(lessonNumber);
            group.lessons.push({ label: `Lesson ${lessonNumber}`, link: source.link, number: lessonNumber });
        }
    });

    return order.map(course => {
        const group = groups.get(course);
        group.lessons.sort((a, b) => a.number - b.number);
        return group;
    });
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
}

// Start a new chat: clear the current session server-side, then reset the UI
async function startNewChat() {
    if (currentSessionId) {
        try {
            await fetch(`${API_URL}/session/${currentSessionId}`, { method: 'DELETE' });
        } catch (error) {
            console.error('Error clearing session:', error);
        }
    }
    chatInput.disabled = false;
    sendButton.disabled = false;
    createNewSession();
    chatInput.focus();
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${title}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}