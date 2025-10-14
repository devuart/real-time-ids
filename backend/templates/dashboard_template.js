/**
 * Deep Learning System Dashboard - JavaScript Module
 * Template-Based Architecture v3.0
 * 
 * This module handles all dashboard JavaScript functionality including:
 * - Template data injection and processing
 * - Interactive UI components and animations
 * - Modal dialogs and data exports
 * - Progress bar animations with template data
 * - Tab navigation system
 * - Keyboard shortcuts and accessibility
 * - Quick Actions functionality
 * - Notification system
 * 
 * Architecture Notes:
 * - All JavaScript functionality is contained in this external module
 * - No inline JavaScript in dashboard_base.html
 * - Template variables are processed and used for dynamic content
 * - Event handlers check for module availability before execution
 */

// GLOBAL CONFIGURATION & DATA MANAGEMENT

/**
 * Dashboard configuration object containing template variables,
 * settings, and state management
 */
const DashboardConfig = {
    // Template variables for data injection (replaced by template renderer)
    templateVars: {
        reproducibilityScore: '$reproducibility_score',
        loggingCompliance: '$logging_compliance',
        ramPercent: '$ram_percent',
        storageUsedPercent: '$storage_used_percent',
        systemHealthScore: '$system_health_score',
        totalEntries: '$total_entries',
        dateStr: '$date_str',
        timeStr: '$time_str'
    },
    
    // Dashboard behavior settings
    settings: {
        animationDuration: 500,
        progressBarAnimationDelay: 100,
        modalZIndex: 1000,
        keyboardShortcutsEnabled: true,
        autoRefreshInterval: null // Set to milliseconds if auto-refresh needed
    },
    
    // UI state management
    state: {
        activeTab: null,
        modalOpen: false,
        progressBarsAnimated: false
    }
};

// Global data objects (injected by template renderer)
let dashboardData = {};
let currentSystemStatus = {};

// TEMPLATE DATA INJECTION & PROCESSING

/**
 * Initialize dashboard with template data
 * Called by template renderer to inject data and start UI initialization
 * 
 * @param {Object} dashboardDataObj - Complete dashboard data with entries
 * @param {Object} currentStatusObj - Current system status data
 */
function initializeDashboardData(dashboardDataObj, currentStatusObj) {
    dashboardData = dashboardDataObj || {};
    currentSystemStatus = currentStatusObj || {};
    
    console.log('Dashboard data initialized:', {
        entriesCount: dashboardData.entries?.length || 0,
        currentHealth: currentSystemStatus.system_health_score || 0
    });
    
    // Initialize UI after data is available
    initializeUI();
}

/**
 * Process template variables and convert them to usable values
 * Handles both template placeholders and direct numeric values
 * 
 * @returns {Object} Processed template variables with numeric values
 */
function processTemplateVariables() {
    const processed = {};
    
    for (const [key, templateVar] of Object.entries(DashboardConfig.templateVars)) {
        // Extract numeric value from template variable (remove $ and convert to number)
        if (typeof templateVar === 'string' && templateVar.startsWith('$')) {
            // Template variables will be replaced by actual values during rendering
            processed[key] = templateVar; // Keep as-is for template processing
        } else {
            processed[key] = parseFloat(templateVar) || 0;
        }
    }
    
    return processed;
}

// PROGRESS BAR MANAGEMENT & ANIMATIONS

/**
 * Set progress bar widths using template data with enhanced animations
 * Processes template variables and applies smooth animations to progress bars
 */
function setProgressBarWidths() {
    const progressBars = {
        'progress-fill-reproducibility': DashboardConfig.templateVars.reproducibilityScore,
        'progress-fill-logging': DashboardConfig.templateVars.loggingCompliance,
        'progress-fill-ram': DashboardConfig.templateVars.ramPercent,
        'progress-fill-storage': DashboardConfig.templateVars.storageUsedPercent
    };
    
    let delay = 0;
    
    for (const [className, templateValue] of Object.entries(progressBars)) {
        const elements = document.querySelectorAll('.' + className);
        
        elements.forEach(element => {
            setTimeout(() => {
                // Handle both template variables and direct values
                let width = templateValue;
                if (typeof templateValue === 'string' && templateValue.startsWith('$')) {
                    // Template variable - use as percentage
                    width = templateValue.replace('$', '');
                }
                
                const numericWidth = parseFloat(width) || 0;
                element.style.width = numericWidth + '%';
                
                // Add smooth transition animation
                element.style.transition = 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
                
                // Update data attribute for CSS ::after content
                element.setAttribute('data-value', `${numericWidth.toFixed(1)}%`);
                
            }, delay);
            
            delay += DashboardConfig.settings.progressBarAnimationDelay;
        });
    }
    
    DashboardConfig.state.progressBarsAnimated = true;
}

/**
 * Animate progress bars with staggered scale effect
 * Creates a visual loading animation for all progress bars
 */
function animateProgressBars() {
    const progressFills = document.querySelectorAll('.progress-fill');
    
    progressFills.forEach((fill, index) => {
        fill.style.transform = 'scaleX(0)';
        fill.style.transformOrigin = 'left center';
        
        setTimeout(() => {
            fill.style.transform = 'scaleX(1)';
            fill.style.transition = 'transform 0.8s ease-out';
        }, index * 200);
    });
}

// TAB NAVIGATION SYSTEM

/**
 * Enhanced tab navigation with state management and URL updates
 * Handles tab switching, button activation, and tab-specific initialization
 * 
 * @param {string} tabName - Name/ID of the tab to show
 * @param {Event} event - Optional click event for button identification
 */
function showTab(tabName, event = null) {
    try {
        // Remove active class from all content and buttons
        const contents = document.querySelectorAll('.tab-content');
        const buttons = document.querySelectorAll('.tab-button');
        
        contents.forEach(content => content.classList.remove('active'));
        buttons.forEach(button => button.classList.remove('active'));
        
        // Activate selected tab
        const selectedContent = document.getElementById(tabName);
        if (selectedContent) {
            selectedContent.classList.add('active');
            DashboardConfig.state.activeTab = tabName;
            
            // Activate corresponding button
            if (event && event.target) {
                event.target.classList.add('active');
            } else {
                // Find button by aria-controls attribute
                buttons.forEach(button => {
                    if (button.getAttribute('aria-controls') === tabName) {
                        button.classList.add('active');
                    }
                });
            }
            
            // Trigger tab-specific initialization
            onTabActivated(tabName);
            
            // Update URL hash for bookmarking (optional)
            if (history.replaceState) {
                history.replaceState(null, null, `#${tabName}`);
            }
            
        } else {
            console.warn(`Tab content not found for: ${tabName}`);
        }
        
    } catch (error) {
        console.error('Error switching tabs:', error);
    }
}

/**
 * Handle tab-specific initialization and updates
 * Called when a tab becomes active to perform any necessary setup
 * 
 * @param {string} tabName - Name of the activated tab
 */
function onTabActivated(tabName) {
    switch (tabName) {
        case 'history':
            refreshHistoryView();
            break;
        case 'performance':
            updatePerformanceCharts();
            break;
        case 'hardware':
            updateHardwareStatus();
            break;
        default:
            // No special initialization needed for other tabs
            break;
    }
    
    // Update Quick Actions highlights to reflect active tab
    highlightActiveQuickAction();
}

/**
 * Initialize first tab on page load
 * Sets up the default active tab and checks for URL hash navigation
 */
function initializeFirstTab() {
    const firstTab = document.querySelector('.tab-button');
    const firstContent = document.querySelector('.tab-content');
    
    if (firstTab && firstContent) {
        firstTab.classList.add('active');
        firstContent.classList.add('active');
        DashboardConfig.state.activeTab = firstContent.id;
    }
    
    // Check for URL hash to activate specific tab
    const hash = window.location.hash.substr(1);
    if (hash && document.getElementById(hash)) {
        showTab(hash);
    }
}

// QUICK ACTIONS FUNCTIONALITY

/**
 * Update Quick Actions JavaScript module status indicator
 * Updates the UI to show that the external JavaScript module is loaded
 */
function updateQuickActionsJsModuleStatus() {
    const statusElement = document.getElementById('quickActionsJsStatus');
    const jsModuleStatus = document.getElementById('jsModuleStatus');
    
    if (statusElement) {
        statusElement.textContent = 'External Module ✅';
        statusElement.className = 'loaded status-success';
    }
    
    if (jsModuleStatus) {
        jsModuleStatus.textContent = 'External Module ✅';
        jsModuleStatus.className = 'value status-success';
    }
    
    console.log('JavaScript module status updated');
}

/**
 * Enhanced keyboard shortcuts handler including Quick Actions shortcuts
 * Handles all keyboard navigation and shortcuts throughout the dashboard
 * 
 * @param {KeyboardEvent} event - The keyboard event
 */
function handleKeyboardShortcuts(event) {
    // Skip if typing in input fields or with modifier keys
    if (event.target.tagName.toLowerCase() === 'input' || 
        event.target.tagName.toLowerCase() === 'textarea' ||
        event.target.isContentEditable ||
        event.ctrlKey || event.altKey || event.metaKey) {
        return;
    }
    
    // Handle modal close with Escape key
    const modal = document.getElementById('dashboardModal') || document.querySelector('.modal-overlay');
    if (modal && event.key.toLowerCase() === 'escape') {
        event.preventDefault();
        closeModal();
        return;
    }
    
    // Don't handle other shortcuts when modal is open
    if (modal) return;
    
    const key = event.key.toLowerCase();
    const handledKeys = ['h', 'p', 'c', 'd', 'r', 'e'];
    
    if (handledKeys.includes(key)) {
        event.preventDefault();
        
        // Add visual feedback to corresponding elements
        highlightQuickActionCard(key);
        
        // Execute the corresponding action
        const actions = {
            'h': () => showTab('history'),
            'p': () => showTab('performance'),
            'c': () => showTab('configuration'),
            'd': () => showTab('dependencies'),
            'r': () => showTab('reports'),
            'e': () => exportDashboardData()
        };
        
        if (actions[key]) {
            actions[key]();
        }
    }
}

/**
 * Highlight corresponding quick action card when keyboard shortcut is used
 * Provides visual feedback for keyboard navigation
 * 
 * @param {string} key - The pressed key
 */
function highlightQuickActionCard(key) {
    const tabMapping = {
        'h': 'history',
        'p': 'performance', 
        'c': 'configuration',
        'd': 'dependencies',
        'r': 'reports'
    };
    
    // Highlight tab-related cards
    if (tabMapping[key]) {
        const card = document.querySelector('[data-tab="' + tabMapping[key] + '"]');
        if (card) {
            // Create highlight effect with smooth animation
            card.style.transform = 'translateY(-8px) scale(1.02)';
            card.style.borderLeftColor = '#4CAF50';
            card.style.boxShadow = '0 12px 30px rgba(76, 175, 80, 0.3)';
            card.style.background = 'linear-gradient(135deg, #e8f5e8 0%, #f3e5f5 100%)';
            
            // Reset styles after animation
            setTimeout(() => {
                card.style.transform = '';
                card.style.borderLeftColor = '';
                card.style.boxShadow = '';
                card.style.background = '';
            }, 400);
        }
    }
    
    // Highlight export button for 'e' key
    if (key === 'e') {
        const exportBtn = document.querySelector('.primary-export');
        if (exportBtn) {
            exportBtn.style.transform = 'translateY(-3px) scale(1.05)';
            exportBtn.style.boxShadow = '0 8px 20px rgba(33, 150, 243, 0.4)';
            
            setTimeout(() => {
                exportBtn.style.transform = '';
                exportBtn.style.boxShadow = '';
            }, 300);
        }
    }
}

/**
 * Enhanced card interactions for Quick Actions
 * Sets up hover effects, keyboard navigation, and click analytics
 */
function enhanceQuickActionCardInteractions() {
    const quickActionCards = document.querySelectorAll('.quick-action-card');
    
    quickActionCards.forEach(card => {
        const tabName = card.getAttribute('data-tab');
        
        // Add keyboard navigation support
        card.addEventListener('keydown', event => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                card.click();
            }
        });
        
        // Enhanced hover effects with keyboard shortcut tooltips
        card.addEventListener('mouseenter', function() {
            this.style.borderLeftColor = '#1976d2';
            
            // Update tooltip with keyboard shortcut information
            const shortcutMap = {
                'history': 'H',
                'performance': 'P',
                'configuration': 'C',
                'dependencies': 'D'
            };
            
            if (shortcutMap[tabName]) {
                this.setAttribute('title', `Click to open ${tabName} tab or press '${shortcutMap[tabName]}'`);
            }
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.borderLeftColor = '#2196f3';
        });
        
        // Click analytics tracking
        card.addEventListener('click', () => {
            console.log('Quick Action clicked: ' + tabName);
            
            // Optional analytics tracking if available
            if (typeof window.analytics !== 'undefined') {
                window.analytics.track('quick_action_click', {
                    tab: tabName,
                    method: 'click'
                });
            }
        });
    });
}

/**
 * Enhanced export button interactions with visual effects
 * Sets up click handlers, ripple effects, and keyboard support for export buttons
 */
function enhanceQuickActionExportButtons() {
    const exportButtons = document.querySelectorAll('.quick-export-btn, .export-button');
    
    exportButtons.forEach(button => {
        button.addEventListener('click', function(event) {
            const buttonType = this.classList.contains('primary-export') ? 'complete_data' :
                              this.classList.contains('secondary-export') ? 'print' : 'reports';
            
            console.log('Export action: ' + buttonType);
            
            // Add visual feedback effects
            createRippleEffect(this, event);
            
            // Scale animation for button press feedback
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
            
            // Optional analytics tracking
            if (typeof window.analytics !== 'undefined') {
                window.analytics.track('export_action', {
                    type: buttonType,
                    method: 'click'
                });
            }
        });
        
        // Add keyboard support for accessibility
        button.addEventListener('keydown', event => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                button.click();
            }
        });
    });
}

/**
 * Create ripple effect animation for button clicks
 * Provides Material Design-style visual feedback for button interactions
 * 
 * @param {HTMLElement} button - The button element
 * @param {MouseEvent} event - The click event for positioning
 */
function createRippleEffect(button, event) {
    const rect = button.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;
    
    const ripple = document.createElement('div');
    ripple.style.cssText = `position: absolute; border-radius: 50%; background: rgba(255, 255, 255, 0.6); transform: scale(0); animation: ripple 0.6s linear; pointer-events: none; left: ${x}px; top: ${y}px; width: ${size}px; height: ${size}px;`;
    
    // Add ripple animation keyframes if not already present
    if (!document.querySelector('#ripple-keyframes')) {
        const style = document.createElement('style');
        style.id = 'ripple-keyframes';
        style.textContent = '@keyframes ripple { to { transform: scale(2); opacity: 0; } }';
        document.head.appendChild(style);
    }
    
    button.style.position = 'relative';
    button.style.overflow = 'hidden';
    button.appendChild(ripple);
    
    // Clean up ripple element after animation
    setTimeout(() => {
        ripple.remove();
    }, 600);
}

/**
 * Highlight active quick actions based on current tab
 * Updates Quick Action cards to reflect the currently active tab
 */
function highlightActiveQuickAction() {
    const activeTab = document.querySelector('.tab-button.active');
    if (!activeTab) return;
    
    const activeTabName = activeTab.getAttribute('aria-controls');
    const quickActionCards = document.querySelectorAll('.quick-action-card');
    
    quickActionCards.forEach(card => {
        const cardTab = card.getAttribute('data-tab');
        
        if (cardTab === activeTabName) {
            card.classList.add('active');
        } else {
            card.classList.remove('active');
        }
    });
}

// MODAL DIALOG SYSTEM

/**
 * Enhanced modal system for initialization details
 * Creates and displays detailed information modals for dashboard entries
 * 
 * @param {number} entryIndex - Index of the entry to display details for
 */
function viewInitializationDetails(entryIndex) {
    try {
        const entries = dashboardData.entries || [];
        
        if (entryIndex < 0 || entryIndex >= entries.length) {
            showNotification('Invalid entry index: ' + entryIndex, 'error');
            return;
        }
        
        const entry = entries[entryIndex];
        const entryData = entry.data || {};
        
        const modalContent = createInitializationDetailsModal(entry, entryData, entryIndex);
        showModal(modalContent);
        
    } catch (error) {
        console.error('Error viewing initialization details:', error);
        showNotification('Error loading initialization details', 'error');
    }
}

/**
 * Create initialization details modal content with color-coded status
 * Generates HTML content for the initialization details modal
 * 
 * @param {Object} entry - The entry object
 * @param {Object} entryData - The entry data
 * @param {number} entryIndex - Index of the entry
 * @returns {string} HTML content for the modal
 */
function createInitializationDetailsModal(entry, entryData, entryIndex) {
    // Color coding based on status and values
    const statusColor = entryData.status === 'success' ? '#4CAF50' : '#F44336';
    const healthColor = entryData.health_score > 80 ? '#4CAF50' : 
                       entryData.health_score > 60 ? '#FF9800' : '#F44336';
    const cudaColor = entryData.cuda_available ? '#4CAF50' : '#F44336';
    const perfMonColor = entryData.performance_monitoring ? '#4CAF50' : '#F44336';
    const memMgmtColor = entryData.memory_management ? '#4CAF50' : '#FF9800';
    
    return `
        <div class="modal-content" style="max-width: 800px; max-height: 80vh; overflow-y: auto; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
            <div class="modal-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #2196f3;">
                <h2 style="margin: 0; color: #1976d2;">Initialization Details</h2>
                <button onclick="window.DashboardJS.closeModal()" class="modal-close-btn" style="background: #f44336; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer;">Close</button>
            </div>
            
            <div class="modal-body" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <div class="info-section">
                    <h3 style="color: #1976d2; margin-bottom: 10px;">Basic Information</h3>
                    <p><strong>Timestamp:</strong> ${entry.timestamp || 'Unknown'}</p>
                    <p><strong>Status:</strong> <span style="color: ${statusColor};">${(entryData.status || 'unknown').toUpperCase()}</span></p>
                    <p><strong>Duration:</strong> ${entryData.duration_seconds || 0} seconds</p>
                    <p><strong>Health Score:</strong> <span style="color: ${healthColor};">${entryData.health_score || 0}%</span></p>
                </div>
                
                <div class="info-section">
                    <h3 style="color: #1976d2; margin-bottom: 10px;">System Resources</h3>
                    <p><strong>CUDA Available:</strong> <span style="color: ${cudaColor};">${entryData.cuda_available ? 'Yes' : 'No'}</span></p>
                    <p><strong>Model Variants:</strong> ${entryData.model_variants || 0}</p>
                    <p><strong>Reproducibility Score:</strong> ${entryData.reproducibility_score || 0}%</p>
                    <p><strong>Logging Compliance:</strong> ${entryData.logging_compliance || 0}%</p>
                </div>
            </div>
            
            <div class="advanced-features">
                <h3 style="color: #1976d2; margin-bottom: 10px;">Advanced Features</h3>
                <p><strong>Performance Monitoring:</strong> <span style="color: ${perfMonColor};">${entryData.performance_monitoring ? 'Available' : 'Not Available'}</span></p>
                <p><strong>Memory Management:</strong> <span style="color: ${memMgmtColor};">${entryData.memory_management ? 'Comprehensive' : 'Basic'}</span></p>
            </div>
            
            <div class="modal-actions" style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #eee;">
                <button onclick="window.DashboardJS.exportEntry(${entryIndex})" class="action-btn" style="background: #2196f3; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin-right: 10px;">Export This Entry</button>
                <button onclick="compareWithCurrent(${entryIndex})" class="action-btn" style="background: #FF9800; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer;">Compare with Current</button>
            </div>
        </div>
    `;
}

/**
 * Generic modal display function with animation
 * Creates and displays modal dialogs with fade-in animation
 * 
 * @param {string} content - HTML content for the modal
 */
function showModal(content) {
    if (DashboardConfig.state.modalOpen) {
        closeModal(); // Close existing modal first
    }
    
    const modalOverlay = document.createElement('div');
    modalOverlay.id = 'dashboardModal';
    modalOverlay.className = 'modal-overlay';
    modalOverlay.style.cssText = `
        position: fixed; 
        top: 0; 
        left: 0; 
        width: 100%; 
        height: 100%; 
        background: rgba(0,0,0,0.5); 
        display: flex; 
        justify-content: center; 
        align-items: center; 
        z-index: ${DashboardConfig.settings.modalZIndex};
        opacity: 0;
        transition: opacity 0.3s ease;
    `;
    modalOverlay.innerHTML = content;
    
    document.body.appendChild(modalOverlay);
    DashboardConfig.state.modalOpen = true;
    
    // Animate modal appearance
    setTimeout(() => {
        modalOverlay.style.opacity = '1';
    }, 10);
    
    // Close modal on outside click
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            closeModal();
        }
    });
}

/**
 * Close modal with fade-out animation
 * Closes the currently open modal dialog with smooth animation
 */
function closeModal() {
    const modal = document.getElementById('dashboardModal');
    if (modal) {
        modal.style.opacity = '0';
        setTimeout(() => {
            if (modal.parentNode) {
                modal.parentNode.removeChild(modal);
            }
            DashboardConfig.state.modalOpen = false;
        }, 300);
    }
}

// DATA EXPORT FUNCTIONS

/**
 * Export individual entry data as JSON file
 * Downloads a specific initialization entry as a JSON file
 * 
 * @param {number} entryIndex - Index of the entry to export
 */
function exportEntryData(entryIndex) {
    try {
        const entries = dashboardData.entries || [];
        if (entryIndex >= 0 && entryIndex < entries.length) {
            const entry = entries[entryIndex];
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(entry, null, 2));
            
            const downloadLink = document.createElement('a');
            downloadLink.setAttribute("href", dataStr);
            downloadLink.setAttribute("download", `initialization_entry_${entryIndex}_${new Date().toISOString().split('T')[0]}.json`);
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            
            showNotification('Entry data exported successfully!', 'success');
        } else {
            showNotification('Invalid entry index for export', 'error');
        }
    } catch (error) {
        console.error('Error exporting entry data:', error);
        showNotification('Error exporting entry data', 'error');
    }
}

/**
 * Export complete dashboard data including all entries and metadata
 * Downloads comprehensive dashboard data as JSON file
 */
function exportDashboardData() {
    try {
        const exportData = {
            generated_at: new Date().toISOString(),
            dashboard_metadata: dashboardData.metadata || {},
            current_status: currentSystemStatus,
            all_entries: dashboardData.entries || [],
            export_type: 'complete_dashboard_export',
            export_version: '3.0'
        };
        
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportData, null, 2));
        
        const downloadLink = document.createElement('a');
        downloadLink.setAttribute("href", dataStr);
        downloadLink.setAttribute("download", `complete_dashboard_export_${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
        
        showNotification('Complete dashboard data exported successfully!', 'success');
        
    } catch (error) {
        console.error('Error exporting dashboard data:', error);
        showNotification('Error exporting dashboard data', 'error');
    }
}

/**
 * Print dashboard functionality
 * Triggers browser print dialog for dashboard
 */
function printDashboard() {
    window.print();
}

// UTILITY FUNCTIONS

/**
 * Show notification to user with auto-dismiss
 * Displays temporary notification messages with different styles based on type
 * 
 * @param {string} message - The notification message
 * @param {string} type - Type of notification (info, success, error)
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'success' ? '#4CAF50' : type === 'error' ? '#f44336' : '#2196f3'};
        color: white;
        border-radius: 4px;
        z-index: ${DashboardConfig.settings.modalZIndex + 100};
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transform: translateX(400px);
        transition: transform 0.3s ease;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Animate notification in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 10);
    
    // Auto remove after 3 seconds with fade out
    setTimeout(() => {
        notification.style.transform = 'translateX(400px)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

/**
 * Toggle section visibility
 * Generic function to show/hide dashboard sections
 * 
 * @param {string} sectionId - ID of the section to toggle
 */
function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.style.display = section.style.display === 'none' ? 'block' : 'none';
    }
}

/**
 * Refresh history view (placeholder for future enhancement)
 * Updates the history tab content with latest data
 */
function refreshHistoryView() {
    console.log('Refreshing history view...');
    // Future implementation: Update charts, reload data, etc.
}

/**
 * Update performance charts (placeholder for future enhancement)
 * Refreshes performance metrics and charts in the performance tab
 */
function updatePerformanceCharts() {
    console.log('Updating performance charts...');
    // Future implementation: Update chart data, refresh benchmarks, etc.
}

/**
 * Update hardware status (placeholder for future enhancement)
 * Refreshes hardware information and status indicators
 */
function updateHardwareStatus() {
    console.log('Updating hardware status...');
    // Future implementation: Poll hardware status, update GPU info, etc.
}

// KEYBOARD SHORTCUTS & ACCESSIBILITY

/**
 * Initialize keyboard shortcuts system
 * Sets up global keyboard event listeners for dashboard navigation
 */
function initializeKeyboardShortcuts() {
    if (!DashboardConfig.settings.keyboardShortcutsEnabled) return;
    
    // Add the comprehensive keyboard handler
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

// INITIALIZATION & SETUP

/**
 * Initialize all dashboard functionality
 * Main initialization function called when the dashboard loads
 */
function initializeUI() {
    try {
        console.log('Initializing dashboard UI...');
        
        // Update Quick Actions JavaScript module status
        updateQuickActionsJsModuleStatus();
        
        // Initialize tab system
        initializeFirstTab();
        
        // Set up progress bars with delay for smooth loading animation
        setTimeout(() => {
            setProgressBarWidths();
            animateProgressBars();
        }, 100);
        
        // Initialize keyboard shortcuts
        initializeKeyboardShortcuts();
        
        // Initialize Quick Actions functionality
        initializeQuickActions();
        
        // Initialize other UI components
        initializeTooltips();
        initializeAnimations();
        
        console.log('Dashboard UI initialized successfully');
        
    } catch (error) {
        console.error('Error initializing dashboard UI:', error);
    }
}

/**
 * Initialize Quick Actions specific functionality
 * Sets up all Quick Actions related features and interactions
 */
function initializeQuickActions() {
    console.log('Initializing Quick Actions functionality...');
    
    // Enhanced interactions for Quick Action cards
    enhanceQuickActionCardInteractions();
    
    // Enhanced interactions for export buttons
    enhanceQuickActionExportButtons();
    
    // Set up tab change detection for highlighting active quick actions
    document.addEventListener('click', event => {
        if (event.target.classList.contains('tab-button') || 
            event.target.closest('.tab-button')) {
            setTimeout(highlightActiveQuickAction, 100);
        }
    });
    
    // Initial highlight of active quick actions
    setTimeout(highlightActiveQuickAction, 200);
    
    // Update JS module status after delay to ensure everything is loaded
    setTimeout(updateQuickActionsJsModuleStatus, 1000);
    
    console.log('Quick Actions functionality initialized');
}

/**
 * Initialize tooltips (placeholder for future enhancement)
 * Sets up tooltip functionality for dashboard elements
 */
function initializeTooltips() {
    // Future implementation: Setup tooltip library, configure tooltips, etc.
}

/**
 * Initialize animations (placeholder for future enhancement)
 * Sets up additional animations and transitions for the dashboard
 */
function initializeAnimations() {
    // Future implementation: Setup scroll animations, hover effects, etc.
}

// DOM READY & MODULE EXPORT

/**
 * DOM Content Loaded event handler
 * Waits for DOM to be ready and data injection to complete before initializing
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, waiting for data injection...');
    
    // If data is already available (injected inline), initialize immediately
    if (typeof window.dashboardDataInjected !== 'undefined' && window.dashboardDataInjected) {
        initializeUI();
    }
    
    // Otherwise, UI will be initialized when initializeDashboardData is called
});

// GLOBAL EXPORTS FOR TEMPLATE SYSTEM

/**
 * Export dashboard functionality to global window object
 * Makes all dashboard functions available for template system and HTML onclick handlers
 */
window.DashboardJS = {
    // Core initialization and data management
    initialize: initializeDashboardData,
    
    // Navigation and UI functions
    showTab: showTab,
    viewDetails: viewInitializationDetails,
    closeModal: closeModal,
    toggleSection: toggleSection,
    
    // Export and utility functions
    exportData: exportDashboardData,
    exportEntry: exportEntryData,
    printDashboard: printDashboard,
    
    // Configuration and state
    config: DashboardConfig,
    
    // Quick Actions specific functions
    quickActions: {
        updateStatus: updateQuickActionsJsModuleStatus,
        highlight: highlightActiveQuickAction,
        refresh: () => {
            updateQuickActionsJsModuleStatus();
            highlightActiveQuickAction();
        }
    }
};