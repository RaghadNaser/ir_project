/**
 * Simplified IR System Interface
 * Handles only basic search functionality without optional services
 */

// Simple search function
async function performSearch() {
    const form = document.getElementById('search-form');
    const formData = new FormData(form);
    
    const searchParams = {
        query: formData.get('query'),
        dataset: formData.get('dataset'),
        representation: formData.get('representation'),
        top_k: parseInt(formData.get('top_k')) || 10
    };
    
    // Validate input
    if (!searchParams.query || searchParams.query.trim().length < 2) {
        showMessage('Please enter a search query with at least 2 characters', 'error');
        return;
    }
    
    // Show loading
    showLoading(true);
    
    try {
        // Submit form normally - let the backend handle it
        form.submit();
    } catch (error) {
        console.error('Search error:', error);
        showMessage('Search failed: ' + error.message, 'error');
        showLoading(false);
    }
}

// Show/hide loading state
function showLoading(show) {
    const button = document.querySelector('button[type="submit"]');
    const spinner = document.getElementById('loading-spinner');
    
    if (show) {
        button.disabled = true;
        button.textContent = 'جارٍ البحث...';
        if (spinner) spinner.style.display = 'block';
    } else {
        button.disabled = false;
        button.textContent = 'بحث';
        if (spinner) spinner.style.display = 'none';
    }
}

// Show message
function showMessage(message, type = 'info') {
    // Create or update message div
    let messageDiv = document.getElementById('message');
    if (!messageDiv) {
        messageDiv = document.createElement('div');
        messageDiv.id = 'message';
        messageDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 4px;
            z-index: 1000;
            max-width: 300px;
        `;
        document.body.appendChild(messageDiv);
    }
    
    // Set message and style based on type
    messageDiv.textContent = message;
    messageDiv.className = `message ${type}`;
    
    if (type === 'error') {
        messageDiv.style.backgroundColor = '#f8d7da';
        messageDiv.style.color = '#721c24';
        messageDiv.style.border = '1px solid #f5c6cb';
    } else if (type === 'success') {
        messageDiv.style.backgroundColor = '#d4edda';
        messageDiv.style.color = '#155724';
        messageDiv.style.border = '1px solid #c3e6cb';
    } else {
        messageDiv.style.backgroundColor = '#d1ecf1';
        messageDiv.style.color = '#0c5460';
        messageDiv.style.border = '1px solid #bee5eb';
    }
    
    messageDiv.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        messageDiv.style.display = 'none';
    }, 5000);
}

// Simple document interaction functions
function viewDocument(docId) {
    window.open(`/document/${docId}`, '_blank');
}

function copyDocId(docId) {
    navigator.clipboard.writeText(docId).then(() => {
        showMessage('Document ID copied to clipboard', 'success');
    }).catch(() => {
        showMessage('Failed to copy Document ID', 'error');
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add form submission handler
    const form = document.getElementById('search-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            performSearch();
        });
    }
    
    // Remove service status checking and optional services
    console.log('Simplified IR System Interface loaded');
}); 