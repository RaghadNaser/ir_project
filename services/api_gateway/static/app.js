/**
 * Professional IR System - Advanced Interface Controller
 * Handles all client-side operations with error handling and performance monitoring
 */

class IRSystemInterface {
    constructor() {
        this.searchInProgress = false;
        this.serviceUrls = {
            main: 'http://localhost:8000',
            topicDetection: 'http://localhost:8006',
            querySuggestion: 'http://localhost:8010',
            vectorStore: 'http://localhost:8008',
            embedding: 'http://localhost:8004',
            hybrid: 'http://localhost:8005',
            tfidf: 'http://localhost:8003',
            preprocessing: 'http://localhost:8002'
        };
        
        this.serviceStatus = {};
        this.performanceMetrics = {
            searchTimes: [],
            errorCount: 0,
            requestCount: 0
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.checkServiceStatus();
        this.setupPerformanceMonitoring();
        this.loadUserPreferences();
        
        // Auto-refresh status every 30 seconds
        setInterval(() => this.checkServiceStatus(), 30000);
    }
    
    setupEventListeners() {
        // Search form submission
        const searchForm = document.getElementById('search-form');
        if (searchForm) {
            searchForm.addEventListener('submit', (e) => this.handleSearch(e));
        }
        
        // Service checkboxes
        const serviceCheckboxes = [
            { id: 'enable-topic-detection', config: 'topic-config' },
            { id: 'enable-query-suggestion', config: 'suggestion-config' },
            { id: 'enable-vector-store', config: 'vector-config' }
        ];
        
        serviceCheckboxes.forEach(service => {
            const checkbox = document.getElementById(service.id);
            const config = document.getElementById(service.config);
            
            if (checkbox && config) {
                checkbox.addEventListener('change', () => {
                    config.classList.toggle('active', checkbox.checked);
                    this.saveUserPreferences();
                });
            }
        });
        
        // Status refresh
        const refreshBtn = document.getElementById('refresh-status');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.checkServiceStatus());
        }
        
        // Real-time search suggestions
        const queryInput = document.getElementById('query');
        if (queryInput) {
            queryInput.addEventListener('input', this.debounce((e) => {
                this.handleRealTimeSearch(e.target.value);
            }, 300));
        }
    }
    
    async handleSearch(e) {
        e.preventDefault();
        
        if (this.searchInProgress) {
            this.showNotification('Search in progress...', 'warning');
            return;
        }
        
        this.searchInProgress = true;
        const startTime = performance.now();
        
        try {
            const formData = new FormData(e.target);
            const searchParams = {
                query: formData.get('query'),
                dataset: formData.get('dataset'),
                topK: parseInt(formData.get('top_k')),
                representation: formData.get('representation')
            };
            
            // Validate input
            if (!this.validateSearchParams(searchParams)) {
                return;
            }
            
            // Show loading state
            this.showLoadingState();
            
            // Perform searches
            const results = await this.performAllSearches(searchParams);
            
            // Display results
            this.displayResults(results);
            
            // Update performance metrics
            const endTime = performance.now();
            this.updatePerformanceMetrics(endTime - startTime);
            
            this.showNotification('Search completed successfully', 'success');
            
        } catch (error) {
            console.error('Search error:', error);
            this.handleError(error, 'Search failed');
        } finally {
            this.searchInProgress = false;
            this.hideLoadingState();
        }
    }
    
    validateSearchParams(params) {
        if (!params.query || params.query.trim().length < 2) {
            this.showNotification('Please enter a search query with at least 2 characters', 'error');
            return false;
        }
        
        if (params.topK < 1 || params.topK > 100) {
            this.showNotification('Results count must be between 1 and 100', 'error');
            return false;
        }
        
        return true;
    }
    
    async performAllSearches(params) {
        const results = { main: null, additional: [] };
        
        // Main search
        results.main = await this.performMainSearch(params);
        
        // Additional services
        const additionalPromises = [];
        
        if (document.getElementById('enable-topic-detection')?.checked) {
            additionalPromises.push(this.performTopicDetection(params));
        }
        
        if (document.getElementById('enable-query-suggestion')?.checked) {
            additionalPromises.push(this.performQuerySuggestion(params));
        }
        
        if (document.getElementById('enable-vector-store')?.checked) {
            additionalPromises.push(this.performVectorSearch(params));
        }
        
        if (additionalPromises.length > 0) {
            results.additional = await Promise.allSettled(additionalPromises);
        }
        
        return results;
    }
    
    async performMainSearch(params) {
        const response = await this.makeRequest(`${this.serviceUrls.main}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: params.query,
                dataset: params.dataset,
                top_k: params.topK,
                representation: params.representation
            })
        });
        
        return response;
    }
    
    async performTopicDetection(params) {
        const maxTopics = parseInt(document.getElementById('topic-max-topics')?.value || 5);
        const minScore = parseFloat(document.getElementById('topic-min-score')?.value || 0.1);
        
        const response = await this.makeRequest(`${this.serviceUrls.topicDetection}/detect-topics`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: params.query,
                dataset: params.dataset,
                max_topics: maxTopics,
                min_relevance_score: minScore
            })
        });
        
        return { type: 'topic', data: response };
    }
    
    async performQuerySuggestion(params) {
        const method = document.getElementById('suggestion-method')?.value || 'hybrid';
        const count = parseInt(document.getElementById('suggestion-count')?.value || 8);
        
        const response = await this.makeRequest(`${this.serviceUrls.querySuggestion}/suggest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: params.query,
                dataset: params.dataset,
                method: method,
                top_k: count
            })
        });
        
        return { type: 'suggestion', data: response };
    }
    
    async performVectorSearch(params) {
        // First get embedding for the query
        const embeddingResponse = await this.makeRequest(`${this.serviceUrls.embedding}/embed`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: params.query,
                dataset: params.dataset
            })
        });
        
        // Then search with vector store
        const indexType = document.getElementById('vector-index-type')?.value || 'auto';
        
        const response = await this.makeRequest(`${this.serviceUrls.vectorStore}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset: params.dataset,
                query_vector: embeddingResponse.embedding,
                top_k: params.topK,
                index_type: indexType
            })
        });
        
        return { type: 'vector', data: response };
    }
    
    async makeRequest(url, options) {
        this.performanceMetrics.requestCount++;
        
        try {
            const response = await fetch(url, {
                ...options,
                timeout: 30000
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            this.performanceMetrics.errorCount++;
            throw error;
        }
    }
    
    displayResults(results) {
        const resultsPanel = document.getElementById('results-panel');
        const mainResults = document.getElementById('main-results');
        const additionalResults = document.getElementById('additional-results');
        
        if (!resultsPanel || !mainResults || !additionalResults) return;
        
        // Show results panel
        resultsPanel.style.display = 'block';
        resultsPanel.classList.add('fade-in');
        
        // Display main results
        const mainResultsHtml = this.generateMainResultsHtml(results.main);
        mainResults.innerHTML = mainResultsHtml;
        
        // Update results count
        const resultsCount = document.getElementById('results-count');
        const resultsTime = document.getElementById('results-time');
        
        if (resultsCount) {
            resultsCount.innerHTML = `<i class="fas fa-list-ol"></i> Found ${results.main?.results?.length || 0} results`;
        }
        
        if (resultsTime && results.main?.execution_time) {
            resultsTime.innerHTML = `<i class="fas fa-clock"></i> ${(results.main.execution_time * 1000).toFixed(1)}ms`;
        }
        
        // Display additional results
        const additionalResultsHtml = this.generateAdditionalResultsHtml(results.additional);
        additionalResults.innerHTML = additionalResultsHtml;
    }
    
    generateMainResultsHtml(results) {
        if (!results?.results || results.results.length === 0) {
            return '<div class="error">No results found for your query.</div>';
        }
        
        let html = '<table class="results-table">';
        html += '<thead><tr><th>#</th><th>Document ID</th><th>Score</th><th>Action</th></tr></thead>';
        html += '<tbody>';
        
        results.results.forEach((result, index) => {
            const docId = Array.isArray(result) ? result[0] : result.doc_id;
            const score = Array.isArray(result) ? result[1] : result.score;
            
            html += `<tr>
                <td>${index + 1}</td>
                <td><a href="/document/${results.dataset || 'argsme'}/${docId}" target="_blank" class="doc-link">${docId}</a></td>
                <td>${score.toFixed(4)}</td>
                <td>
                    <button onclick="irSystem.viewDocument('${docId}')" class="btn-secondary">View</button>
                    <button onclick="irSystem.copyDocId('${docId}')" class="btn-secondary">Copy</button>
                </td>
            </tr>`;
        });
        
        html += '</tbody></table>';
        return html;
    }
    
    generateAdditionalResultsHtml(results) {
        if (!results || results.length === 0) return '';
        
        let html = '';
        
        results.forEach(result => {
            if (result.status === 'fulfilled') {
                const { type, data } = result.value;
                
                if (type === 'topic') {
                    html += this.generateTopicResultsHtml(data);
                } else if (type === 'suggestion') {
                    html += this.generateSuggestionResultsHtml(data);
                } else if (type === 'vector') {
                    html += this.generateVectorResultsHtml(data);
                }
            } else {
                html += `<div class="result-card">
                    <div class="error">
                        <i class="fas fa-exclamation-triangle"></i>
                        Service temporarily unavailable: ${result.reason}
                    </div>
                </div>`;
            }
        });
        
        return html;
    }
    
    generateTopicResultsHtml(data) {
        if (!data.detected_topics || data.detected_topics.length === 0) {
            return '<div class="result-card"><div class="error">No topics detected.</div></div>';
        }
        
        let html = '<div class="result-card">';
        html += '<div class="result-card-header">';
        html += '<i class="fas fa-brain"></i>';
        html += '<span class="result-card-title">Detected Topics</span>';
        html += '</div>';
        
        data.detected_topics.forEach(topic => {
            html += `<span class="topic-tag" onclick="irSystem.searchTopic('${topic.topic}')">
                ${topic.topic} (${(topic.relevance_score * 100).toFixed(1)}%)
            </span>`;
        });
        
        html += '</div>';
        return html;
    }
    
    generateSuggestionResultsHtml(data) {
        if (!data.suggestions || data.suggestions.length === 0) {
            return '<div class="result-card"><div class="error">No suggestions available.</div></div>';
        }
        
        let html = '<div class="result-card">';
        html += '<div class="result-card-header">';
        html += '<i class="fas fa-lightbulb"></i>';
        html += '<span class="result-card-title">Smart Suggestions</span>';
        html += '</div>';
        
        data.suggestions.forEach(suggestion => {
            html += `<div class="suggestion-item" onclick="irSystem.useQuery('${suggestion.query}')">
                <strong>${suggestion.query}</strong>
                <div style="font-size: 0.9em; color: #7f8c8d;">
                    Score: ${(suggestion.score * 100).toFixed(1)}% | Type: ${suggestion.type}
                </div>
            </div>`;
        });
        
        html += '</div>';
        return html;
    }
    
    generateVectorResultsHtml(data) {
        let html = '<div class="result-card">';
        html += '<div class="result-card-header">';
        html += '<i class="fas fa-rocket"></i>';
        html += '<span class="result-card-title">Vector Search Performance</span>';
        html += '</div>';
        
        html += `<div class="success">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div>
                    <strong>Index Type:</strong> ${data.index_type}<br>
                    <strong>Search Time:</strong> ${(data.search_time * 1000).toFixed(1)}ms
                </div>
                <div>
                    <strong>Results:</strong> ${data.total_results}<br>
                    <strong>Vector Dimension:</strong> ${data.query_vector_dim}
                </div>
            </div>
        </div>`;
        
        html += '</div>';
        return html;
    }
    
    async checkServiceStatus() {
        const statusElement = document.getElementById('service-status');
        if (!statusElement) return;
        
        statusElement.innerHTML = '<div class="loading"><div class="spinner"></div><span>Checking services...</span></div>';
        
        const services = Object.entries(this.serviceUrls).map(([name, url]) => ({
            name: name.charAt(0).toUpperCase() + name.slice(1),
            url: url,
            key: name
        }));
        
        const statusPromises = services.map(async service => {
            try {
                const response = await fetch(`${service.url}/health`, { 
                    method: 'GET',
                    signal: AbortSignal.timeout(5000)
                });
                
                const isOnline = response.ok;
                this.serviceStatus[service.key] = isOnline;
                
                // Update individual status indicators
                const statusIndicator = document.getElementById(`${service.key}-status`);
                if (statusIndicator) {
                    statusIndicator.className = `status-indicator ${isOnline ? 'status-online' : 'status-offline'}`;
                }
                
                return {
                    name: service.name,
                    status: isOnline ? 'online' : 'offline',
                    url: service.url,
                    key: service.key
                };
            } catch (error) {
                this.serviceStatus[service.key] = false;
                
                const statusIndicator = document.getElementById(`${service.key}-status`);
                if (statusIndicator) {
                    statusIndicator.className = 'status-indicator status-offline';
                }
                
                return {
                    name: service.name,
                    status: 'offline',
                    url: service.url,
                    key: service.key
                };
            }
        });
        
        const statuses = await Promise.all(statusPromises);
        this.displayServiceStatus(statuses);
    }
    
    displayServiceStatus(statuses) {
        const statusElement = document.getElementById('service-status');
        if (!statusElement) return;
        
        let html = '<div class="service-status-grid">';
        
        statuses.forEach(service => {
            const statusClass = service.status === 'online' ? 'online' : 'offline';
            const statusIcon = service.status === 'online' ? 'fas fa-check-circle' : 'fas fa-times-circle';
            
            html += `<div class="service-status-item ${statusClass}">
                <i class="${statusIcon}"></i>
                <h4>${service.name}</h4>
                <p>${service.status.toUpperCase()}</p>
                <small>${service.url}</small>
            </div>`;
        });
        
        html += '</div>';
        
        // Add performance metrics
        html += `<div class="result-card" style="margin-top: 20px;">
            <div class="result-card-header">
                <i class="fas fa-chart-line"></i>
                <span class="result-card-title">Performance Metrics</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                <div>
                    <strong>Total Requests:</strong><br>
                    ${this.performanceMetrics.requestCount}
                </div>
                <div>
                    <strong>Error Rate:</strong><br>
                    ${this.performanceMetrics.requestCount > 0 ? 
                        ((this.performanceMetrics.errorCount / this.performanceMetrics.requestCount) * 100).toFixed(1) : 0}%
                </div>
                <div>
                    <strong>Avg Search Time:</strong><br>
                    ${this.performanceMetrics.searchTimes.length > 0 ? 
                        (this.performanceMetrics.searchTimes.reduce((a, b) => a + b, 0) / this.performanceMetrics.searchTimes.length).toFixed(1) : 0}ms
                </div>
            </div>
        </div>`;
        
        statusElement.innerHTML = html;
    }
    
    // Utility methods
    showLoadingState() {
        const resultsPanel = document.getElementById('results-panel');
        const resultsCount = document.getElementById('results-count');
        
        if (resultsPanel) resultsPanel.style.display = 'block';
        if (resultsCount) {
            resultsCount.innerHTML = '<div class="loading"><div class="spinner"></div><span>Searching...</span></div>';
        }
    }
    
    hideLoadingState() {
        // Loading state is hidden when results are displayed
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'times' : 'info'}"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
    
    viewDocument(docId) {
        const dataset = document.getElementById('dataset')?.value || 'argsme';
        window.open(`/document/${dataset}/${docId}`, '_blank');
    }
    
    copyDocId(docId) {
        navigator.clipboard.writeText(docId).then(() => {
            this.showNotification('Document ID copied to clipboard', 'success');
        }).catch(() => {
            this.showNotification('Failed to copy document ID', 'error');
        });
    }
    
    useQuery(query) {
        const queryInput = document.getElementById('query');
        if (queryInput) {
            queryInput.value = query;
            const searchForm = document.getElementById('search-form');
            if (searchForm) {
                searchForm.dispatchEvent(new Event('submit'));
            }
        }
    }
    
    searchTopic(topic) {
        const queryInput = document.getElementById('query');
        if (queryInput) {
            queryInput.value = topic;
            const searchForm = document.getElementById('search-form');
            if (searchForm) {
                searchForm.dispatchEvent(new Event('submit'));
            }
        }
    }
    
    setupPerformanceMonitoring() {
        // Monitor performance
        window.addEventListener('error', (e) => {
            this.performanceMetrics.errorCount++;
            console.error('Global error:', e);
        });
        
        window.addEventListener('unhandledrejection', (e) => {
            this.performanceMetrics.errorCount++;
            console.error('Unhandled promise rejection:', e);
        });
    }
    
    updatePerformanceMetrics(searchTime) {
        this.performanceMetrics.searchTimes.push(searchTime);
        
        // Keep only last 10 search times
        if (this.performanceMetrics.searchTimes.length > 10) {
            this.performanceMetrics.searchTimes.shift();
        }
    }
    
    handleError(error, context) {
        console.error(`Error in ${context}:`, error);
        this.showNotification(`${context}: ${error.message}`, 'error');
    }
    
    saveUserPreferences() {
        const preferences = {
            topicDetection: document.getElementById('enable-topic-detection')?.checked,
            querySuggestion: document.getElementById('enable-query-suggestion')?.checked,
            vectorStore: document.getElementById('enable-vector-store')?.checked,
            dataset: document.getElementById('dataset')?.value,
            representation: document.getElementById('representation')?.value,
            topK: document.getElementById('top_k')?.value
        };
        
        localStorage.setItem('irSystemPreferences', JSON.stringify(preferences));
    }
    
    loadUserPreferences() {
        const saved = localStorage.getItem('irSystemPreferences');
        if (!saved) return;
        
        try {
            const preferences = JSON.parse(saved);
            
            // Restore checkbox states
            if (preferences.topicDetection) {
                const checkbox = document.getElementById('enable-topic-detection');
                if (checkbox) {
                    checkbox.checked = true;
                    document.getElementById('topic-config')?.classList.add('active');
                }
            }
            
            if (preferences.querySuggestion) {
                const checkbox = document.getElementById('enable-query-suggestion');
                if (checkbox) {
                    checkbox.checked = true;
                    document.getElementById('suggestion-config')?.classList.add('active');
                }
            }
            
            if (preferences.vectorStore) {
                const checkbox = document.getElementById('enable-vector-store');
                if (checkbox) {
                    checkbox.checked = true;
                    document.getElementById('vector-config')?.classList.add('active');
                }
            }
            
            // Restore form values
            if (preferences.dataset) {
                const dataset = document.getElementById('dataset');
                if (dataset) dataset.value = preferences.dataset;
            }
            
            if (preferences.representation) {
                const representation = document.getElementById('representation');
                if (representation) representation.value = preferences.representation;
            }
            
            if (preferences.topK) {
                const topK = document.getElementById('top_k');
                if (topK) topK.value = preferences.topK;
            }
            
        } catch (error) {
            console.error('Failed to load user preferences:', error);
        }
    }
    
    handleRealTimeSearch(query) {
        // Real-time search suggestions (future implementation)
        if (query.length < 3) return;
        
        // This could be implemented to show live suggestions
        console.log('Real-time search for:', query);
    }
    
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// Initialize the system when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.irSystem = new IRSystemInterface();
});

// Add notification styles
const notificationStyles = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        display: flex;
        align-items: center;
        gap: 10px;
        max-width: 300px;
    }
    
    .notification.success {
        background: #27ae60;
    }
    
    .notification.error {
        background: #e74c3c;
    }
    
    .notification.warning {
        background: #f39c12;
    }
    
    .notification.info {
        background: #3498db;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;

// Inject notification styles
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet); 