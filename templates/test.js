
        let conversations = [
            {
                id: 1,
                date: "2023-10-15 14:30",
                user_message: "Once upon a time, in the heart of an enchanted forest, there lived a magnificent unicorn named Starlight. This magical creature was not only beautiful but also kind-hearted and wise beyond all others. In those days, a terrible drought had struck Eldoria, causing crops to wither away before their very eyes and water sources to run dry at every turn.",
                ai_message: "That a really amazing story, thank you. Once upon a time, in the heart of an enchanted forest, there lived a magnificent unicorn named Starlight. This magical creature was not only beautiful but also kind-hearted and wise beyond all others. In those days, a terrible drought had struck Eldoria, causing crops to wither away before their very eyes and water so. Once upon a time, in the heart of an enchanted forest, there lived a magnificent unicorn named Starlight. This magical creature was not only beautiful but also kind-hearted and wise beyond all others. In those days, a terrible drought had struck Eldoria, causing crops to wither away before their very eyes and water so. Benjamin Sinzore.",
                starred: true
            },
            {
                id: 2,
                date: "2023-10-15 15:45",
                user_message: "Tell me a long bed-time story",
                ai_message: "Once upon a time, in the magical kingdom of Eldoria, there was an enchanted forest filled with tall trees and sparkling streams. The villagers who lived nearby would often gather at the edge of the forest to listen to stories told by the wisest creatures. One evening, as the sun began its descent into the horizon, a gentle breeze whispered secrets through the leaves of the ancient trees. In this peaceful scene stood an old owl named Zephyr, who had been watching over Eldoria for many years",
                starred: false
            },
            {
                id: 3,
                date: "2023-10-16 09:20",
                user_message: "Can you explain quantum computing in simple terms?",
                ai_message: "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. The devices that perform quantum computations are known as quantum computers. They are believed to be able to solve certain computational problems, such as integer factorization (which underlies RSA encryption), substantially faster than classical computers.",
                starred: true
            }
        ];
        let currentFilter = localStorage.getItem('conversationFilter') || 'all';
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        function formatDate(dateString) {
            const date = new Date(dateString);
            return date.toLocaleDateString('en-US', { 
                year: 'numeric', 
                month: 'short', 
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        }
        const MAX_PREVIEW_LENGTH = 80;
        function getPreviewText(text) {
            const firstSentenceMatch = text.match(/^[^.!?]*[.!?](?=\s|$)|^[^.!?]+/);
            let firstSentence = firstSentenceMatch ? firstSentenceMatch[0] : text;
            if (firstSentence.length <= MAX_PREVIEW_LENGTH) {
                return firstSentence;
            } else {
                return firstSentence.substring(0, MAX_PREVIEW_LENGTH).trim() + '…';
            }
        }
        function renderConversations(filter = currentFilter) {
            const mainContent = document.querySelector('.main-content');
            mainContent.innerHTML = '';
            let filteredConvs = conversations;
            if (filter === 'starred') {
                filteredConvs = conversations.filter(c => c.starred);
            } else if (filter === 'recent') {
                const now = new Date();
                const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
                filteredConvs = conversations.filter(c => new Date(c.date) > yesterday);
            }
            if (filteredConvs.length === 0) {
                let message = '';
                let emoji = '';
                switch(filter) {
                    case 'starred':
                        message = 'No starred conversations yet';
                        emoji = '⭐';
                        break;
                    case 'recent':
                        message = 'No recent conversations';
                        emoji = '🕒';
                        break;
                    default:
                        message = 'No conversations yet';
                        emoji = '💬';
                }
                const emptyState = document.createElement('div');
                emptyState.style.cssText = `
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 300px;
                    text-align: center;
                    color: #6b7280;
                `;
                emptyState.innerHTML = `
                    <div style="font-size: 4rem; margin-bottom: 1rem;">${emoji}</div>
                    <div style="font-size: 1.25rem; font-weight: 500; margin-bottom: 0.5rem;">${message}</div>
                    <div style="font-size: 0.875rem;">Start a new conversation to see it here</div>
                `;
                mainContent.appendChild(emptyState);
                return;
            }
            filteredConvs.forEach(conv => {
                const fullUser = escapeHtml(conv.user_message);
                const fullAi = escapeHtml(conv.ai_message);
                const previewUser = escapeHtml(getPreviewText(conv.user_message));
                const previewAi = escapeHtml(getPreviewText(conv.ai_message));
                const card = document.createElement('div');
                card.className = 'conversation-card';
                card.dataset.fullUser = fullUser;
                card.dataset.fullAi = fullAi;
                card.dataset.previewUser = previewUser;
                card.dataset.previewAi = previewAi;
                card.dataset.id = conv.id;
                const isStarred = conv.starred;
                card.innerHTML = `
                    <div class="flex justify-between items-start mb-4">
                        <div class="text-sm text-gray-500">${formatDate(conv.date)}</div>
                        <div class="text-xs bg-gray-100 px-2 py-1 rounded">Conversation #${conv.id}</div>
                    </div>
                    <div class="conversation-sections">
                        <div class="message-section user-section">
                            <div class="message-header">
                                <div class="icon-container user-icon">
                                    <svg xmlns="http://www.w3.org/2000/svg  " class="icon-svg" viewBox="0 0 20 20" fill="currentColor">
                                        <path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd" />
                                    </svg>
                                </div>
                                <div class="text-blue-700 font-semibold message-label">User:</div>
                            </div>
                            <div class="message-content">
                                <div class="message-text">${previewUser}</div>
                            </div>
                        </div>
                        <div class="message-section ai-section">
                            <div class="message-header">
                                <div class="icon-container ai-icon">
                                    <svg xmlns="http://www.w3.org/2000/svg  " class="icon-svg" viewBox="0 0 20 20" fill="currentColor">
                                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                                    </svg>
                                </div>
                                <div class="text-green-700 font-semibold message-label">AI Companion:</div>
                            </div>
                            <div class="message-content">
                                <div class="message-text">${previewAi}</div>
                            </div>
                        </div>
                    </div>
                    <div class="action-icons">
                        <div class="action-icon star-icon ${isStarred ? 'active' : ''}" data-id="${conv.id}">
                            <svg xmlns="http://www.w3.org/2000/svg  " viewBox="0 0 24 24" fill="currentColor">
                                <path fill-rule="evenodd" d="M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.007 5.404.433c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.433 2.082-5.006z" clip-rule="evenodd" />
                            </svg>
                        </div>
                        <div class="action-icon expand-icon" data-id="${conv.id}">
                            <svg xmlns="http://www.w3.org/2000/svg  " viewBox="0 0 24 24" fill="currentColor">
                                <path fill-rule="evenodd" d="M15 3.75a.75.75 0 01.75-.75h4.5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0V5.56l-3.97 3.97a.75.75 0 11-1.06-1.06l3.97-3.97h-2.69a.75.75 0 01-.75-.75zm-12 0A.75.75 0 013.75 3h4.5a.75.75 0 010 1.5H5.56l3.97 3.97a.75.75 0 01-1.06 1.06L4.5 5.56v2.69a.75.75 0 01-1.5 0v-4.5zm11.47 14.78a.75.75 0 111.06-1.06l3.97 3.97v-2.69a.75.75 0 011.5 0v4.5a.75.75 0 01-1.5 0v-4.5a.75.75 0 010-1.5h4.5a.75.75 0 010 1.5h-2.69l-3.97-3.97zm-4.94-1.06a.75.75 0 010 1.06L5.56 19.5h2.69a.75.75 0 010 1.5h-4.5a.75.75 0 01-.75-.75v-4.5a.75.75 0 011.5 0v2.69l3.97-3.97a.75.75 0 011.06 0z" clip-rule="evenodd" />
                            </svg>
                        </div>
                    </div>
                `;
                mainContent.appendChild(card);
            });
            document.querySelectorAll('.star-icon').forEach(icon => {
                icon.addEventListener('click', function() {
                    const id = parseInt(this.getAttribute('data-id'), 10);
                    const conv = conversations.find(c => c.id === id);
                    if (conv) {
                        conv.starred = !conv.starred;
                        this.classList.toggle('active');
                    }
                    renderConversations(currentFilter);
                });
            });
            document.querySelectorAll('.expand-icon').forEach(icon => {
                icon.addEventListener('click', function() {
                    const id = this.getAttribute('data-id');
                    const card = this.closest('.conversation-card');
                    const isExpanded = card.classList.contains('expanded');
                    card.classList.toggle('expanded');
                    const userContent = card.querySelector('.user-section .message-text');
                    const aiContent = card.querySelector('.ai-section .message-text');
                    if (!isExpanded) {
                        userContent.innerHTML = card.dataset.fullUser;
                        aiContent.innerHTML = card.dataset.fullAi;
                        this.querySelector('svg').innerHTML = `<path fill-rule="evenodd" d="M4.5 12a.75.75 0 01.75-.75h13.5a.75.75 0 010 1.5H5.25a.75.75 0 01-.75-.75z" clip-rule="evenodd" />`;
                    } else {
                        userContent.innerHTML = card.dataset.previewUser;
                        aiContent.innerHTML = card.dataset.previewAi;
                        this.querySelector('svg').innerHTML = `<path fill-rule="evenodd" d="M15 3.75a.75.75 0 01.75-.75h4.5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0V5.56l-3.97 3.97a.75.75 0 11-1.06-1.06l3.97-3.97h-2.69a.75.75 0 01-.75-.75zm-12 0A.75.75 0 013.75 3h4.5a.75.75 0 010 1.5H5.56l3.97 3.97a.75.75 0 01-1.06 1.06L4.5 5.56v2.69a.75.75 0 01-1.5 0v-4.5zm11.47 14.78a.75.75 0 111.06-1.06l3.97 3.97v-2.69a.75.75 0 011.5 0v4.5a.75.75 0 01-1.5 0v-4.5a.75.75 0 010-1.5h4.5a.75.75 0 010 1.5h-2.69l-3.97-3.97zm-4.94-1.06a.75.75 0 010 1.06L5.56 19.5h2.69a.75.75 0 010 1.5h-4.5a.75.75 0 01-.75-.75v-4.5a.75.75 0 011.5 0v2.69l3.97-3.97a.75.75 0 011.06 0z" clip-rule="evenodd" />`;
                    }
                });
            });
        }
        document.addEventListener('DOMContentLoaded', () => {
            const filterButtons = document.querySelectorAll('.filter-btn');
            filterButtons.forEach(btn => {
                const filter = btn.getAttribute('data-filter');
                if (filter === currentFilter) {
                    btn.classList.add('active');
                }
            });
            filterButtons.forEach(btn => {
                btn.addEventListener('click', () => {
                    filterButtons.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    currentFilter = btn.getAttribute('data-filter');
                    localStorage.setItem('conversationFilter', currentFilter);
                    renderConversations(currentFilter);
                });
            });
            renderConversations(currentFilter);
        });

