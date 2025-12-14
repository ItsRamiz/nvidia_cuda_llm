// DOM Elements
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const charCount = document.getElementById('charCount');
const welcomeMessage = document.getElementById('welcomeMessage');
const newChatBtn = document.getElementById('newChatBtn');

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    charCount.textContent = this.value.length;
});

// Handle Shift+Enter for new line, Enter to send
messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Send button click
sendBtn.addEventListener('click', sendMessage);

// New chat button
newChatBtn.addEventListener('click', function() {
    clearChat();
});

// Format message text (handle basic markdown formatting)
function formatMessage(text) {
    if (!text) return '';
    
    // Escape HTML to prevent XSS
    const escapeHtml = (str) => {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    };
    
    let formatted = escapeHtml(text);
    
    // Convert **bold** to <strong>
    formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    
    // Convert `inline code` to <code>
    formatted = formatted.replace(/`([^`\n]+)`/g, '<code>$1</code>');
    
    // Convert newlines to <br>
    formatted = formatted.replace(/\n/g, '<br>');
    
    // Handle bullet points - convert * item or - item to list items
    // This is a simple approach that works for most cases
    formatted = formatted.replace(/(<br>)?[\*\-]\s+([^<]+)(?=<br>|$)/g, '<li>$2</li>');
    
    // Wrap consecutive <li> tags in <ul>
    formatted = formatted.replace(/(<li>.*?<\/li>(?:<br>)?)+/g, (match) => {
        return '<ul>' + match.replace(/<br><\/li>/g, '</li>').replace(/<br>/g, '') + '</ul>';
    });
    
    return formatted;
}

// Add message to chat
function addMessage(content, isUser = false) {
    // Hide welcome message if visible
    if (welcomeMessage && !welcomeMessage.classList.contains('hidden')) {
        welcomeMessage.classList.add('hidden');
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = isUser ? '👤' : '🤖';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    if (isUser) {
        bubble.textContent = content;
    } else {
        bubble.innerHTML = formatMessage(content);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(bubble);
    
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv;
}

// Show typing indicator
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '🤖';
    
    const dots = document.createElement('div');
    dots.className = 'typing-dots';
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        dots.appendChild(dot);
    }
    
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(dots);
    
    messagesContainer.appendChild(typingDiv);
    scrollToBottom();
}

// Remove typing indicator
function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Scroll to bottom of messages
function scrollToBottom() {
    messagesContainer.scrollTo({
        top: messagesContainer.scrollHeight,
        behavior: 'smooth'
    });
}

// Send message to API
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message) {
        return;
    }
    
    // Disable input and send button
    messageInput.disabled = true;
    sendBtn.disabled = true;
    
    // Add user message to chat
    addMessage(message, true);
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    charCount.textContent = '0';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        if (response.ok) {
            // Add assistant response
            addMessage(data.answer || 'No response received.');
        } else {
            // Show error message
            addMessage(`Error: ${data.error || 'Failed to get response'}`);
        }
    } catch (error) {
        removeTypingIndicator();
        addMessage(`Error: ${error.message}`);
        console.error('Error:', error);
    } finally {
        // Re-enable input and send button
        messageInput.disabled = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

// Clear chat
function clearChat() {
    messagesContainer.innerHTML = '';
    if (welcomeMessage) {
        welcomeMessage.classList.remove('hidden');
        messagesContainer.appendChild(welcomeMessage);
    }
    messageInput.value = '';
    charCount.textContent = '0';
}

// Initialize
messageInput.focus();

