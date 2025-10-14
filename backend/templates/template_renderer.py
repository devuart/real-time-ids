import json
import sys
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from string import Template
import logging

logger = logging.getLogger(__name__)

class DashboardTemplateRenderer:
    """
    A dedicated class for rendering dashboard HTML templates with clear separation 
    between data processing and presentation logic.
    
    Now works with external JavaScript module (dashboard_template.js) for enhanced
    maintainability and clean architecture separation. Fully harmonized with
    dashboard_styles.css for comprehensive styling support.
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the template renderer.
        
        Args:
            template_dir: Path to templates directory. If None, uses default location.
        """
        self.template_dir = template_dir or Path(__file__).parent
        self.css_available = self._check_css_availability()
        self.javascript_module_available = self._check_javascript_module()
        
    def _check_css_availability(self) -> bool:
        """Check if the external CSS file is available."""
        css_path = self.template_dir / "dashboard_styles.css"
        available = css_path.exists()
        
        if available:
            logger.info("External CSS file (dashboard_styles.css) found and available")
        else:
            logger.warning("External CSS file not found, templates will use fallback styling")
            
        return available
        
    def _check_javascript_module(self) -> bool:
        """Check if the external JavaScript module is available."""
        js_path = self.template_dir / "dashboard_template.js"
        available = js_path.exists()
        
        if available:
            logger.info("External JavaScript module (dashboard_template.js) found and available")
        else:
            logger.warning("External JavaScript module not found, will use inline JavaScript fallback")
            
        return available
        
    def _load_template(self, template_name: str) -> str:
        """Load a template file and return its content."""
        template_path = self.template_dir / template_name
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Template file not found: {template_path}")
            raise
    
    def _prepare_javascript_data_external(self, dashboard_json_file_data: Dict, compact_status: Dict) -> str:
        """
        Prepare JavaScript data for external module approach.
        This creates the data injection script that initializes the external JavaScript module.
        """
        try:
            dashboard_data_js = json.dumps(dashboard_json_file_data, default=str, indent=2)
            current_status_js = json.dumps(compact_status, default=str, indent=2)
            
            return f"""
        // Data injection for external JavaScript module (dashboard_template.js)
        // This script initializes the DashboardJS module with real data
        
        // Set flag indicating data has been injected
        window.dashboardDataInjected = true;
        
        // Initialize dashboard when both DOM and external JS module are ready
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('DOM ready, initializing dashboard with external JS module...');
            
            // Check if external module is loaded
            if (typeof window.DashboardJS !== 'undefined') {{
                console.log('External DashboardJS module found, initializing...');
                try {{
                    window.DashboardJS.initialize(
                        {dashboard_data_js},
                        {current_status_js}
                    );
                    console.log('Dashboard initialization complete');
                }} catch (error) {{
                    console.error('Error initializing external DashboardJS module:', error);
                }}
            }} else {{
                console.warn('External DashboardJS module not found, data injection delayed');
                
                // Fallback: Wait a bit longer and try again
                setTimeout(function() {{
                    if (typeof window.DashboardJS !== 'undefined') {{
                        console.log('External DashboardJS module loaded after delay, initializing...');
                        try {{
                            window.DashboardJS.initialize(
                                {dashboard_data_js},
                                {current_status_js}
                            );
                            console.log('Dashboard initialization complete (delayed)');
                        }} catch (error) {{
                            console.error('Error in delayed initialization:', error);
                        }}
                    }} else {{
                        console.error('External DashboardJS module failed to load completely');
                        // Enhanced error handling for missing external module
                        const statusElement = document.getElementById('jsModuleStatus');
                        if (statusElement) {{
                            statusElement.textContent = 'Module Failed to Load';
                            statusElement.className = 'value status-error';
                        }}
                    }}
                }}, 1000);
            }}
        }});
        
        // Store data globally as backup (for debugging or fallback access)
        window.dashboardDataBackup = {dashboard_data_js};
        window.currentSystemStatusBackup = {current_status_js};
        
        // Enhanced error handling for external module integration
        window.addEventListener('error', function(event) {{
            if (event.filename && event.filename.includes('dashboard_template.js')) {{
                console.error('Error in external JavaScript module:', event.error);
                const statusElement = document.getElementById('jsModuleStatus');
                if (statusElement) {{
                    statusElement.textContent = 'Module Error';
                    statusElement.className = 'value status-warning';
                }}
            }}
        }});
        """
        except Exception as e:
            logger.error(f"Failed to prepare JavaScript data for external module: {e}")
            return """
        console.error('Failed to prepare dashboard data for external module');
        window.dashboardDataInjected = false;
        
        // Set error status
        document.addEventListener('DOMContentLoaded', function() {
            const statusElement = document.getElementById('jsModuleStatus');
            if (statusElement) {
                statusElement.textContent = 'Data Injection Failed';
                statusElement.className = 'value status-error';
            }
        });
        """
    
    def _prepare_javascript_data_inline_fallback(self, dashboard_json_file_data: Dict, compact_status: Dict) -> str:
        """
        Prepare JavaScript data for inline fallback approach.
        Used when external JavaScript module is not available.
        Enhanced to work with dashboard_styles.css classes.
        """
        try:
            dashboard_data_js = json.dumps(dashboard_json_file_data, default=str, indent=2)
            current_status_js = json.dumps(compact_status, default=str, indent=2)
            
            return f"""
        // Inline JavaScript fallback (external module not available)
        // Enhanced with dashboard_styles.css class support
        const dashboardData = {dashboard_data_js};
        const currentSystemStatus = {current_status_js};
        
        // Enhanced tab switching with CSS animations
        function showTab(tabName, event) {{
            console.log('Switching to tab:', tabName);
            
            try {{
                // Hide all tab contents with fade animation
                var contents = document.querySelectorAll('.tab-content');
                contents.forEach(function(content) {{
                    content.classList.remove('active');
                }});
                
                // Remove active class from all buttons
                var buttons = document.querySelectorAll('.tab-button');
                buttons.forEach(function(button) {{
                    button.classList.remove('active');
                }});
                
                // Show selected tab content with animation
                var selectedContent = document.getElementById(tabName);
                if (selectedContent) {{
                    selectedContent.classList.add('active');
                }} else {{
                    console.warn('Tab content not found:', tabName);
                    return;
                }}
                
                // Set active button with enhanced styling
                if (event && event.target) {{
                    var button = event.target.closest('.tab-button');
                    if (button) {{
                        button.classList.add('active');
                    }}
                }} else {{
                    // Find the button that corresponds to this tab
                    var targetButton = document.querySelector(`[aria-controls="${{tabName}}"]`);
                    if (targetButton) {{
                        targetButton.classList.add('active');
                    }}
                }}
                
                console.log('Tab switched to:', tabName);
            }} catch (error) {{
                console.error('Error in tab switching:', error);
            }}
        }}
        
        // Enhanced data export with better user feedback
        function exportDashboardData() {{
            console.log('Exporting dashboard data...');
            
            try {{
                // Show loading feedback
                showNotification('Preparing export...', 'info');
                
                // Create comprehensive export data structure
                const exportData = {{
                    generated_at: new Date().toISOString(),
                    dashboard_metadata: {{
                        version: '3.0',
                        architecture: 'template-based',
                        css_harmony: 'dashboard_styles.css',
                        date: dashboardData.metadata?.date || 'unknown',
                        time: dashboardData.metadata?.time || 'unknown',
                        total_entries: dashboardData.metadata?.total_entries || 0
                    }},
                    current_status: currentSystemStatus,
                    all_entries: dashboardData.entries || [],
                    system_info: {{
                        css_available: {str(self.css_available).lower()},
                        js_module_available: false,
                        fallback_mode: true
                    }},
                    export_type: 'complete_dashboard_export',
                    export_version: '3.0'
                }};
                
                // Create and download file with timestamp
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportData, null, 2));
                const downloadLink = document.createElement('a');
                downloadLink.setAttribute("href", dataStr);
                downloadLink.setAttribute("download", `dashboard_export_${{timestamp}}.json`);
                
                // Temporarily add to document, click, and remove
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
                
                // Show success notification
                showNotification('Dashboard data exported successfully!', 'success');
            }} catch (error) {{
                console.error('Export failed:', error);
                showNotification('Export failed: ' + error.message, 'error');
            }}
        }}
        
        // Enhanced initialization details viewer with modal support
        function viewInitializationDetails(entryIndex) {{
            console.log('Viewing details for entry:', entryIndex);
            
            try {{
                const entries = dashboardData.entries || [];
                if (entryIndex >= 0 && entryIndex < entries.length) {{
                    const entry = entries[entryIndex];
                    const entryData = entry.data || {{}};
                    
                    // Create modal content with styled formatting
                    const modalContent = `
                        <div class="modal-overlay" id="detailsModal">
                            <div class="modal-content" style="padding: 30px; max-width: 600px;">
                                <div class="modal-header">
                                    <h3 style="color: #1976d2; margin: 0;">Initialization Details</h3>
                                    <button class="modal-close-btn" onclick="closeModal()">✕</button>
                                </div>
                                <div class="modal-body">
                                    <div class="metric-grid" style="grid-template-columns: 1fr;">
                                        <div class="metric-card">
                                            <div class="metric-title">Timestamp</div>
                                            <div class="metric-value" style="font-size: 1.2em;">${{entry.timestamp || 'Unknown'}}</div>
                                        </div>
                                        <div class="metric-card">
                                            <div class="metric-title">Status & Performance</div>
                                            <div class="metric-description">
                                                <span>Status:</span>
                                                <span class="value status-${{entryData.status === 'success' ? 'success' : entryData.status === 'warning' ? 'warning' : 'error'}}">${{(entryData.status || 'unknown').toUpperCase()}}</span>
                                            </div>
                                            <div class="metric-description">
                                                <span>Health Score:</span>
                                                <span class="value">${{entryData.health_score || 0}}%</span>
                                            </div>
                                            <div class="metric-description">
                                                <span>Duration:</span>
                                                <span class="value">${{entryData.duration_seconds || 0}} seconds</span>
                                            </div>
                                        </div>
                                        <div class="metric-card">
                                            <div class="metric-title">System Configuration</div>
                                            <div class="metric-description">
                                                <span>CUDA Available:</span>
                                                <span class="value status-${{entryData.cuda_available ? 'success' : 'error'}}">${{entryData.cuda_available ? 'Yes' : 'No'}}</span>
                                            </div>
                                            <div class="metric-description">
                                                <span>Model Variants:</span>
                                                <span class="value">${{entryData.model_variants || 0}}</span>
                                            </div>
                                            <div class="metric-description">
                                                <span>Reproducibility:</span>
                                                <span class="value">${{entryData.reproducibility_score || 0}}%</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div style="margin-top: 20px;">
                                        <h4 style="color: #1976d2;">Raw Data</h4>
                                        <pre style="background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; max-height: 300px;">${{JSON.stringify(entryData, null, 2)}}</pre>
                                    </div>
                                </div>
                                <div class="modal-actions">
                                    <button class="action-btn" onclick="exportSingleEntry(${{entryIndex}})">Export This Entry</button>
                                    <button class="action-btn" onclick="closeModal()" style="background: #666;">Close</button>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // Add modal to document
                    document.body.insertAdjacentHTML('beforeend', modalContent);
                    
                    // Add escape key handler
                    document.addEventListener('keydown', function escHandler(e) {{
                        if (e.key === 'Escape') {{
                            closeModal();
                            document.removeEventListener('keydown', escHandler);
                        }}
                    }});
                    
                }} else {{
                    showNotification('Invalid entry index: ' + entryIndex, 'error');
                }}
            }} catch (error) {{
                console.error('Error viewing details:', error);
                showNotification('Error viewing details: ' + error.message, 'error');
            }}
        }}
        
        // Modal management functions
        function closeModal() {{
            const modal = document.getElementById('detailsModal');
            if (modal) {{
                modal.remove();
            }}
        }}
        
        function exportSingleEntry(entryIndex) {{
            try {{
                const entries = dashboardData.entries || [];
                if (entryIndex >= 0 && entryIndex < entries.length) {{
                    const entry = entries[entryIndex];
                    const exportData = {{
                        exported_at: new Date().toISOString(),
                        entry_index: entryIndex,
                        entry_data: entry,
                        export_type: 'single_entry_export'
                    }};
                    
                    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportData, null, 2));
                    const downloadLink = document.createElement('a');
                    downloadLink.setAttribute("href", dataStr);
                    downloadLink.setAttribute("download", `entry_${{entryIndex}}_${{timestamp}}.json`);
                    document.body.appendChild(downloadLink);
                    downloadLink.click();
                    document.body.removeChild(downloadLink);
                    
                    showNotification('Entry exported successfully!', 'success');
                    closeModal();
                }}
            }} catch (error) {{
                console.error('Single entry export failed:', error);
                showNotification('Export failed: ' + error.message, 'error');
            }}
        }}
        
        // Section toggle functionality with animations
        function toggleSection(sectionId) {{
            const section = document.getElementById(sectionId);
            if (section) {{
                if (section.style.display === 'none') {{
                    section.style.display = 'block';
                    section.style.animation = 'fadeInUp 0.3s ease';
                }} else {{
                    section.style.animation = 'fadeOut 0.3s ease';
                    setTimeout(() => {{
                        section.style.display = 'none';
                    }}, 300);
                }}
            }}
        }}
        
        // Enhanced notification system
        function showNotification(message, type = 'info') {{
            const container = document.getElementById('notificationContainer') || createNotificationContainer();
            
            const notification = document.createElement('div');
            notification.className = `notification notification-${{type}}`;
            notification.textContent = message;
            
            container.appendChild(notification);
            
            // Auto-remove after 5 seconds with fade animation
            setTimeout(() => {{
                if (notification.parentNode === container) {{
                    notification.style.animation = 'slideOut 0.3s ease';
                    setTimeout(() => {{
                        if (notification.parentNode === container) {{
                            container.removeChild(notification);
                        }}
                    }}, 300);
                }}
            }}, 5000);
        }}
        
        function createNotificationContainer() {{
            const container = document.createElement('div');
            container.id = 'notificationContainer';
            container.className = 'notification-container';
            document.body.appendChild(container);
            return container;
        }}
        
        // Progress bar width setter with animations
        function setProgressBarWidths() {{
            const progressBars = {{
                'progress-fill-reproducibility': currentSystemStatus.reproducibility_score || 0,
                'progress-fill-logging': currentSystemStatus.logging_compliance || 0,
                'progress-fill-ram': currentSystemStatus.ram_percent || 0,
                'progress-fill-storage': currentSystemStatus.storage_used_percent || 0
            }};
            
            for (const [className, width] of Object.entries(progressBars)) {{
                const elements = document.querySelectorAll('.' + className);
                elements.forEach(element => {{
                    const numericWidth = parseFloat(width) || 0;
                    // Animate from 0 to target width
                    element.style.width = '0%';
                    setTimeout(() => {{
                        element.style.width = numericWidth + '%';
                    }}, 100);
                }});
            }}
        }}
        
        // Enhanced DOM initialization with CSS harmony
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('Dashboard initialized with inline fallback JavaScript and CSS harmony support');
            
            try {{
                // Set up first tab as active with enhanced styling
                var firstButton = document.querySelector('.tab-button');
                var firstContent = document.querySelector('.tab-content');
                if (firstButton && firstContent) {{
                    firstButton.classList.add('active');
                    firstContent.classList.add('active');
                    console.log('First tab activated');
                }}
                
                // Initialize progress bars with animation
                setTimeout(setProgressBarWidths, 200);
                
                // Set module status
                const jsModuleStatus = document.getElementById('jsModuleStatus');
                if (jsModuleStatus) {{
                    jsModuleStatus.textContent = 'Inline Fallback';
                    jsModuleStatus.className = 'value status-warning';
                }}
                
                // Add keyboard shortcuts support
                document.addEventListener('keydown', function(event) {{
                    if (event.target.tagName.toLowerCase() === 'input' || 
                        event.target.tagName.toLowerCase() === 'textarea' ||
                        event.target.isContentEditable ||
                        event.ctrlKey || event.altKey || event.metaKey) {{
                        return;
                    }}
                    
                    const key = event.key.toLowerCase();
                    const shortcuts = {{
                        'h': 'history',
                        'p': 'performance',
                        'c': 'configuration',
                        'd': 'dependencies',
                        'r': 'reports'
                    }};
                    
                    if (shortcuts[key]) {{
                        event.preventDefault();
                        showTab(shortcuts[key]);
                    }} else if (key === 'e') {{
                        event.preventDefault();
                        exportDashboardData();
                    }}
                }});
                
                // Add click handlers for modal overlays
                document.addEventListener('click', function(event) {{
                    if (event.target.classList.contains('modal-overlay')) {{
                        closeModal();
                    }}
                }});
                
                // Initialize tooltips and hover effects
                initializeInteractiveElements();
                
                console.log('Dashboard fallback initialization complete');
            }} catch (error) {{
                console.error('Error during fallback initialization:', error);
                showNotification('Error during initialization: ' + error.message, 'error');
            }}
        }});
        
        // Interactive elements initialization
        function initializeInteractiveElements() {{
            // Add hover effects to metric cards
            const metricCards = document.querySelectorAll('.metric-card');
            metricCards.forEach(card => {{
                card.addEventListener('mouseenter', function() {{
                    this.style.transform = 'translateY(-5px)';
                }});
                card.addEventListener('mouseleave', function() {{
                    this.style.transform = 'translateY(0)';
                }});
            }});
            
            // Add click handlers for quick action cards
            const quickActionCards = document.querySelectorAll('.quick-action-card');
            quickActionCards.forEach(card => {{
                const tabName = card.getAttribute('data-tab');
                if (tabName) {{
                    card.addEventListener('click', function() {{
                        showTab(tabName);
                    }});
                }}
            }});
        }}
        
        // Global error handler
        window.addEventListener('error', function(event) {{
            console.error('Global JavaScript error:', event.error);
            showNotification('An error occurred: ' + event.error.message, 'error');
        }});
        
        // Export functions for external access
        window.DashboardFallback = {{
            showTab: showTab,
            exportData: exportDashboardData,
            viewDetails: viewInitializationDetails,
            showNotification: showNotification,
            closeModal: closeModal,
            toggleSection: toggleSection,
            version: '3.0.0-fallback'
        }};
        """
        except Exception as e:
            logger.error(f"Failed to prepare inline fallback JavaScript: {e}")
            return """
        console.error('Failed to prepare inline fallback JavaScript');
        
        // Minimal fallback functionality
        function showTab(tabName) {
            console.log('Minimal fallback - switching to tab:', tabName);
            var contents = document.querySelectorAll('.tab-content');
            contents.forEach(c => c.classList.remove('active'));
            var buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(b => b.classList.remove('active'));
            
            var content = document.getElementById(tabName);
            if (content) content.classList.add('active');
            
            var button = document.querySelector(`[aria-controls="${tabName}"]`);
            if (button) button.classList.add('active');
        }
        
        function exportDashboardData() {
            alert('Export functionality not available in minimal fallback mode');
        }
        
        function viewInitializationDetails(index) {
            alert('Details view not available in minimal fallback mode');
        }
        """
    
    def _generate_gpu_info_html(self, cuda_info: Dict) -> str:
        """Generate GPU information HTML section with enhanced CSS styling."""
        if not cuda_info.get('available'):
            return """
            <div class="metric-card">
                <div class="metric-title">CUDA Status</div>
                <div class="metric-value status-error">Not Available</div>
                <div class="metric-description">CUDA is not available on this system</div>
                <div style="margin-top: 15px; padding: 15px; background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); border-radius: 6px; border-left: 4px solid #f44336;">
                    <strong>Impact:</strong> GPU acceleration unavailable. System will fall back to CPU processing for deep learning operations.
                </div>
            </div>
            """
        
        gpu_info_html = f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-title">CUDA Version</div>
                <div class="metric-value">{cuda_info.get('cuda_version', 'unknown')}</div>
                <div class="metric-description">CUDA Runtime Version</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">cuDNN Version</div>
                <div class="metric-value">{cuda_info.get('cudnn_version', 'unknown')}</div>
                <div class="metric-description">Deep Learning Library Version</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">GPU Count</div>
                <div class="metric-value">{cuda_info.get('gpu_count', 0)}</div>
                <div class="metric-description">Available CUDA Devices</div>
            </div>
        </div>
        """
        
        # Add individual GPU details with enhanced styling
        for i, gpu in enumerate(cuda_info.get('gpus', [])):
            memory_usage_html = ""
            memory_stats_html = ""
            
            if 'current_usage' in gpu:
                usage = gpu['current_usage']
                memory_percent = usage.get('percent_allocated', 0)
                
                # Determine memory usage color
                if memory_percent > 80:
                    usage_color = '#f44336'  # Red for high usage
                    usage_class = 'status-error'
                elif memory_percent > 60:
                    usage_color = '#ff9800'  # Orange for moderate usage
                    usage_class = 'status-warning'
                else:
                    usage_color = '#4caf50'  # Green for low usage
                    usage_class = 'status-success'
                
                memory_usage_html = f"""
                <div class="progress-bar" style="margin: 15px 0;">
                    <div class="progress-fill" 
                         style="width: {memory_percent}%; background: linear-gradient(90deg, {usage_color}, {usage_color}aa);" 
                         data-value="{usage.get('allocated_mb', 0):.0f}MB ({memory_percent:.1f}%)"
                         aria-label="GPU Memory Usage: {memory_percent:.1f}%"></div>
                </div>
                """
                
                memory_stats_html = f"""
                <div class="metric-description">
                    <span>Memory Usage:</span>
                    <span class="value {usage_class}">{usage.get('allocated_mb', 0):.0f}MB / {gpu.get('memory_mb', 0):.0f}MB</span>
                </div>
                <div class="metric-description">
                    <span>Utilization:</span>
                    <span class="value {usage_class}">{memory_percent:.1f}%</span>
                </div>
                """
            
            # GPU temperature and power if available
            thermal_info = ""
            if 'temperature' in gpu or 'power_usage' in gpu:
                thermal_info = "<div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee;'>"
                if 'temperature' in gpu:
                    temp = gpu['temperature']
                    temp_class = 'status-error' if temp > 80 else 'status-warning' if temp > 70 else 'status-success'
                    thermal_info += f"""
                    <div class="metric-description">
                        <span>Temperature:</span>
                        <span class="value {temp_class}">{temp}°C</span>
                    </div>
                    """
                if 'power_usage' in gpu:
                    power = gpu['power_usage']
                    thermal_info += f"""
                    <div class="metric-description">
                        <span>Power Usage:</span>
                        <span class="value">{power}W</span>
                    </div>
                    """
                thermal_info += "</div>"
            
            gpu_info_html += f"""
            <div class="entry gpu-device-entry">
                <h4>GPU {i}: {gpu.get('name', 'Unknown GPU')}</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">
                    <div>
                        <div class="metric-description">
                            <span>Total Memory:</span>
                            <span class="value">{gpu.get('memory_gb', 0):.1f}GB</span>
                        </div>
                        <div class="metric-description">
                            <span>Compute Capability:</span>
                            <span class="value">{gpu.get('compute_capability', 'unknown')}</span>
                        </div>
                        <div class="metric-description">
                            <span>Multi-Processors:</span>
                            <span class="value">{gpu.get('multi_processor_count', 'N/A')}</span>
                        </div>
                    </div>
                    <div>
                        {memory_stats_html}
                    </div>
                </div>
                {memory_usage_html}
                {thermal_info}
                <div style="margin-top: 15px; padding: 12px; background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%); border-radius: 6px; border-left: 4px solid #2196f3;">
                    <strong>Status:</strong> <span class="status-success">Ready for Deep Learning Operations</span>
                </div>
            </div>
            """
        
        return gpu_info_html
    
    def _generate_performance_html(self, performance: Dict) -> str:
        """Generate performance metrics HTML section with enhanced styling."""
        if not performance:
            return """
            <div class="metric-card">
                <div class="metric-title">Performance Metrics</div>
                <div class="metric-value status-warning">Not Available</div>
                <div class="metric-description">No performance metrics collected</div>
                <div style="margin-top: 15px; padding: 15px; background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); border-radius: 6px; border-left: 4px solid #ff9800;">
                    <strong>Recommendation:</strong> Run performance benchmarks to assess system capabilities for deep learning workloads.
                </div>
            </div>
            """
        
        # Get summary info safely
        summary = performance.get('summary', {})
        overall_capability = summary.get('overall_capability', 'unknown').lower()
        
        # Enhanced capability classification
        if overall_capability == 'high':
            status_class = 'status-success'
            capability_color = '#4CAF50'
            capability_icon = '🚀'
            capability_description = 'Excellent performance for deep learning workloads'
        elif overall_capability == 'medium':
            status_class = 'status-warning'
            capability_color = '#FF9800'
            capability_icon = '⚡'
            capability_description = 'Good performance for most deep learning tasks'
        else:
            status_class = 'status-error'
            capability_color = '#F44336'
            capability_icon = '⚠️'
            capability_description = 'Limited performance - consider hardware upgrades'
        
        # Generate CPU baseline information
        cpu_baseline_html = ""
        if 'cpu' in performance.get('baselines', {}):
            cpu_baseline = performance['baselines']['cpu']
            cpu_gflops = cpu_baseline.get('gflops', 0)
            
            # Classify CPU performance
            if cpu_gflops > 50:
                cpu_class = 'status-success'
                cpu_rating = 'Excellent'
            elif cpu_gflops > 20:
                cpu_class = 'status-warning'
                cpu_rating = 'Good'
            else:
                cpu_class = 'status-error'
                cpu_rating = 'Limited'
            
            cpu_baseline_html = f"""
            <div class="entry performance-benchmark-entry">
                <h4>🖥️ CPU Performance Benchmark</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 15px 0;">
                    <div class="metric-description">
                        <span>Matrix Size:</span>
                        <span class="value">{cpu_baseline.get('matrix_size', 'unknown')}</span>
                    </div>
                    <div class="metric-description">
                        <span>Computation Time:</span>
                        <span class="value">{cpu_baseline.get('computation_time', 0):.4f}s</span>
                    </div>
                    <div class="metric-description">
                        <span>Performance:</span>
                        <span class="value {cpu_class}">{cpu_gflops:.2f} GFLOPS</span>
                    </div>
                    <div class="metric-description">
                        <span>Rating:</span>
                        <span class="value {cpu_class}">{cpu_rating}</span>
                    </div>
                </div>
                
                <div class="progress-bar" style="margin: 15px 0;">
                    <div class="progress-fill" 
                         style="width: {min(cpu_gflops, 100)}%; background: linear-gradient(90deg, {'#4CAF50' if cpu_class == 'status-success' else '#FF9800' if cpu_class == 'status-warning' else '#F44336'}, {'#81C784' if cpu_class == 'status-success' else '#FFB74D' if cpu_class == 'status-warning' else '#EF5350'});"
                         data-value="{cpu_gflops:.1f} GFLOPS"
                         aria-label="CPU Performance: {cpu_gflops:.1f} GFLOPS"></div>
                </div>
            </div>
            """
        
        # Generate GPU benchmark information
        gpu_benchmarks_html = ""
        if 'gpu' in performance.get('baselines', {}):
            gpu_data = performance['baselines']['gpu']
            
            for gpu_name, gpu_perf in gpu_data.items():
                if isinstance(gpu_perf, dict) and 'gflops' in gpu_perf:
                    gpu_gflops = gpu_perf.get('gflops', 0)
                    gpu_time = gpu_perf.get('computation_time', 0)
                    
                    # Classify GPU performance (GPUs typically have much higher GFLOPS)
                    if gpu_gflops > 1000:
                        gpu_class = 'status-success'
                        gpu_rating = 'Excellent'
                    elif gpu_gflops > 500:
                        gpu_class = 'status-warning'
                        gpu_rating = 'Good'
                    else:
                        gpu_class = 'status-error'
                        gpu_rating = 'Limited'
                    
                    gpu_benchmarks_html += f"""
                    <div class="entry performance-benchmark-entry">
                        <h4>🎮 {gpu_name} Performance</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 15px 0;">
                            <div class="metric-description">
                                <span>Computation Time:</span>
                                <span class="value">{gpu_time:.4f}s</span>
                            </div>
                            <div class="metric-description">
                                <span>Performance:</span>
                                <span class="value {gpu_class}">{gpu_gflops:.2f} GFLOPS</span>
                            </div>
                            <div class="metric-description">
                                <span>Rating:</span>
                                <span class="value {gpu_class}">{gpu_rating}</span>
                            </div>
                            <div class="metric-description">
                                <span>Speedup vs CPU:</span>
                                <span class="value status-success">{(gpu_gflops / max(performance.get('baselines', {}).get('cpu', {}).get('gflops', 1), 1)):.1f}x</span>
                            </div>
                        </div>
                        
                        <div class="progress-bar" style="margin: 15px 0;">
                            <div class="progress-fill" 
                                 style="width: {min(gpu_gflops / 20, 100)}%; background: linear-gradient(90deg, {'#4CAF50' if gpu_class == 'status-success' else '#FF9800' if gpu_class == 'status-warning' else '#F44336'}, {'#81C784' if gpu_class == 'status-success' else '#FFB74D' if gpu_class == 'status-warning' else '#EF5350'});"
                                 data-value="{gpu_gflops:.1f} GFLOPS"
                                 aria-label="GPU Performance: {gpu_gflops:.1f} GFLOPS"></div>
                        </div>
                    </div>
                    """
        
        # Performance summary card
        summary_card = f"""
        <div class="metric-card" style="border-left-color: {capability_color};">
            <div class="metric-title">System Performance Overview</div>
            <div class="metric-value {status_class}" style="display: flex; align-items: center; gap: 10px;">
                <span>{capability_icon}</span>
                <span>{overall_capability.upper()}</span>
            </div>
            <div class="metric-description" style="margin: 15px 0; font-style: italic;">
                {capability_description}
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-top: 15px;">
                <div class="metric-description">
                    <span>CPU:</span>
                    <span class="value">{summary.get('cpu_performance', 'unknown')}</span>
                </div>
                <div class="metric-description">
                    <span>Memory:</span>
                    <span class="value">{summary.get('memory_performance', 'unknown')}</span>
                </div>
                <div class="metric-description">
                    <span>I/O:</span>
                    <span class="value">{summary.get('io_performance', 'unknown')}</span>
                </div>
                <div class="metric-description">
                    <span>GPU:</span>
                    <span class="value">{summary.get('gpu_performance', 'N/A')}</span>
                </div>
            </div>
        </div>
        """
        
        return f"""
        <div class="stats-grid">
            {summary_card}
        </div>
        
        <div style="margin: 30px 0;">
            <h3 style="color: #1976d2; display: flex; align-items: center; gap: 10px;">
                📊 Detailed Benchmark Results
            </h3>
            <div style="margin-top: 20px;">
                {cpu_baseline_html}
                {gpu_benchmarks_html}
            </div>
        </div>
        
        <div style="margin: 30px 0; padding: 20px; background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%); border-radius: 8px; border-left: 4px solid #2196f3;">
            <h4 style="color: #1976d2; margin-top: 0;">🎯 Performance Recommendations</h4>
            <ul style="margin: 10px 0; color: #666;">
                <li><strong>For Training:</strong> {'GPU acceleration recommended for optimal performance' if gpu_benchmarks_html else 'Consider adding GPU acceleration for faster training'}</li>
                <li><strong>For Inference:</strong> {'Current setup supports real-time inference' if overall_capability in ['high', 'medium'] else 'Consider performance optimizations for production use'}</li>
                <li><strong>Memory Usage:</strong> Monitor GPU memory usage during large model training</li>
                <li><strong>Optimization:</strong> {'Performance is optimal for current configuration' if overall_capability == 'high' else 'Consider hardware upgrades or optimization techniques'}</li>
            </ul>
        </div>
        """
    
    def _generate_dependencies_tables(self, detailed_versions: Dict) -> tuple[str, str]:
        """Generate core and optional dependencies HTML tables with enhanced styling."""
        core_deps_html = "<tr><td colspan='3' style='text-align: center; color: #666; font-style: italic;'>No dependency information available</td></tr>"
        optional_deps_html = "<tr><td colspan='2' style='text-align: center; color: #666; font-style: italic;'>No optional dependencies information available</td></tr>"
        
        if not detailed_versions:
            return core_deps_html, optional_deps_html
        
        core_deps_rows = []
        optional_deps_rows = []
        
        for name, info in detailed_versions.items():
            if not isinstance(info, dict):
                continue
                
            if info.get('required', False):
                # Core dependencies with enhanced status indicators
                compatible = info.get('compatible', False)
                version = info.get('version', 'unknown')
                
                if compatible:
                    status_class = 'status-success'
                    status_text = '✅ Compatible'
                    status_icon = '🟢'
                else:
                    status_class = 'status-error'
                    status_text = '❌ Incompatible'
                    status_icon = '🔴'
                
                # Add version-specific recommendations
                recommendations = ""
                if not compatible and name.lower() in ['torch', 'pytorch', 'tensorflow']:
                    recommendations = f"<br><small style='color: #666;'>Consider upgrading to compatible version</small>"
                elif name.lower() == 'cuda' and not compatible:
                    recommendations = f"<br><small style='color: #666;'>GPU acceleration unavailable</small>"
                
                core_deps_rows.append(f"""
                <tr style="transition: background-color 0.2s ease;">
                    <td style="font-weight: 600;">{name}</td>
                    <td style="font-family: monospace; color: #1976d2;">{version}{recommendations}</td>
                    <td><span class="{status_class}" style="display: flex; align-items: center; gap: 5px;">{status_icon} {status_text}</span></td>
                </tr>
                """)
            else:
                # Optional dependencies
                available = info.get('available', False)
                
                if available:
                    status_class = 'status-success'
                    status_text = '✅ Available'
                    status_icon = '🟢'
                    version_info = f" (v{info.get('version', 'unknown')})" if 'version' in info else ""
                else:
                    status_class = 'status-warning'
                    status_text = '⚠️ Not Available'
                    status_icon = '🟡'
                    version_info = ""
                
                # Add feature impact information
                feature_impact = ""
                if not available:
                    if name.lower() in ['matplotlib', 'plotly']:
                        feature_impact = "<br><small style='color: #666;'>Visualization features limited</small>"
                    elif name.lower() in ['jupyter', 'ipython']:
                        feature_impact = "<br><small style='color: #666;'>Notebook features unavailable</small>"
                    elif name.lower() in ['tensorboard']:
                        feature_impact = "<br><small style='color: #666;'>Training monitoring limited</small>"
                
                optional_deps_rows.append(f"""
                <tr style="transition: background-color 0.2s ease;">
                    <td style="font-weight: 600;">{name}{version_info}{feature_impact}</td>
                    <td><span class="{status_class}" style="display: flex; align-items: center; gap: 5px;">{status_icon} {status_text}</span></td>
                </tr>
                """)
        
        if core_deps_rows:
            core_deps_html = "".join(core_deps_rows)
        if optional_deps_rows:
            optional_deps_html = "".join(optional_deps_rows)
        
        return core_deps_html, optional_deps_html
    
    def _generate_quick_actions_html(self, 
                                dashboard_json_file_data: Dict, 
                                performance: Dict, 
                                config_info: Dict, 
                                detailed_versions: Dict) -> str:
        """
        Generate the Quick Actions section HTML with enhanced CSS styling.
        Following the same pattern as _generate_gpu_info_html() - returns complete HTML section
        with embedded styling that can be imported directly as $quick_actions_html.
        """
        try:
            # Calculate performance status values with enhanced logic
            performance_available = bool(performance and performance.get('baselines'))
            if performance_available:
                summary = performance.get('summary', {})
                overall_capability = summary.get('overall_capability', 'unknown').lower()
                if overall_capability == 'high':
                    performance_status = 'Excellent'
                    performance_status_color = '#4CAF50'
                    performance_status_message = 'System optimized for deep learning workloads'
                elif overall_capability == 'medium':
                    performance_status = 'Good'
                    performance_status_color = '#8BC34A'
                    performance_status_message = 'Suitable for most deep learning tasks'
                else:
                    performance_status = 'Limited'
                    performance_status_color = '#FF9800'
                    performance_status_message = 'Consider hardware optimizations'
            else:
                performance_status = 'Setup Required'
                performance_status_color = '#FF9800'
                performance_status_message = 'Run performance benchmarks to assess capabilities'

            # Calculate presets count values with validation
            available_presets = config_info.get('available_presets', [])
            available_presets_count = len(available_presets) if isinstance(available_presets, list) else 0
            presets_count_color = '#4CAF50' if available_presets_count > 0 else '#FF9800'

            # Calculate dependencies count values with enhanced analysis
            core_dependencies_count = 0
            dependencies_status = 'Unknown'
            dependencies_count_color = '#666'

            if detailed_versions and isinstance(detailed_versions, dict):
                core_deps = [d for d in detailed_versions.values() 
                            if isinstance(d, dict) and d.get('required', False)]
                core_dependencies_count = len(core_deps)

                if core_dependencies_count > 0:
                    compatible_count = sum(1 for d in core_deps if d.get('compatible', False))
                    compatibility_rate = compatible_count / core_dependencies_count

                    if compatibility_rate >= 0.9:
                        dependencies_status = 'Excellent'
                        dependencies_count_color = '#4CAF50'
                    elif compatibility_rate >= 0.7:
                        dependencies_status = 'Good'
                        dependencies_count_color = '#8BC34A'
                    else:
                        dependencies_status = 'Issues'
                        dependencies_count_color = '#F44336'
                else:
                    dependencies_status = 'No Core Deps'
                    dependencies_count_color = '#FF9800'

            # Generate recent initialization cards HTML with enhanced styling
            recent_initialization_cards = self._generate_recent_initialization_cards(dashboard_json_file_data)

            # Determine which JavaScript functions to use
            tab_function = "window.DashboardJS.showTab" if self.javascript_module_available else "showTab"
            export_data_function = "window.DashboardJS.exportData" if self.javascript_module_available else "exportDashboardData"
            print_function = "window.DashboardJS.printDashboard" if self.javascript_module_available else "window.print"

            # Generate the complete Quick Actions HTML section with embedded styling
            return f"""
            <!-- Quick Actions Section with Enhanced Embedded Styling -->
            <style>
            /* Quick Actions Enhanced Styling - Embedded CSS */
            .quick-actions-section {{
                position: relative;
                margin: 30px 0;
                background: linear-gradient(135deg, rgba(233, 101, 0, 0.02) 0%, rgba(255, 152, 0, 0.05) 100%);
                border-radius: 16px;
                padding: 30px;
                border-left: 6px solid #e65100;
                box-shadow: 0 8px 32px rgba(233, 101, 0, 0.1);
                overflow: hidden;
            }}

            .quick-actions-section::before {{
                content: '';
                position: absolute;
                top: -50%;
                right: -50%;
                width: 100%;
                height: 100%;
                background: radial-gradient(circle, rgba(233, 101, 0, 0.08) 0%, transparent 70%);
                z-index: 0;
                animation: quickActionsPulse 4s ease-in-out infinite;
            }}

            @keyframes quickActionsPulse {{
                0%, 100% {{ transform: scale(1) rotate(0deg); opacity: 0.3; }}
                50% {{ transform: scale(1.1) rotate(5deg); opacity: 0.1; }}
            }}

            .quick-actions-section > * {{
                position: relative;
                z-index: 2;
            }}

            .quick-actions-section h2 {{
                color: #e65100 !important;
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 25px;
                font-size: 1.8em;
                font-weight: 700;
                text-shadow: 0 2px 4px rgba(233, 101, 0, 0.1);
            }}

            .quick-actions-section h2::before {{
                content: '🚀';
                font-size: 1.2em;
                animation: quickActionsRocket 3s ease-in-out infinite;
            }}

            @keyframes quickActionsRocket {{
                0%, 100% {{ transform: translateY(0) rotate(0deg); }}
                25% {{ transform: translateY(-5px) rotate(-2deg); }}
                50% {{ transform: translateY(-8px) rotate(0deg); }}
                75% {{ transform: translateY(-5px) rotate(2deg); }}
            }}

            /* Quick Action Cards Enhanced Styling */
            .quick-actions-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin: 25px 0;
            }}

            .quick-action-card {{
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border-radius: 12px;
                padding: 24px;
                border-left: 5px solid #2196f3;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                cursor: pointer;
                position: relative;
                overflow: hidden;
            }}

            .quick-action-card::before {{
                content: '';
                position: absolute;
                top: -2px;
                left: -2px;
                right: -2px;
                bottom: -2px;
                background: linear-gradient(45deg, transparent 30%, rgba(33, 150, 243, 0.2) 50%, transparent 70%);
                z-index: -1;
                transition: opacity 0.3s ease;
                opacity: 0;
                border-radius: 14px;
            }}

            .quick-action-card:hover {{
                transform: translateY(-8px) scale(1.02);
                box-shadow: 0 12px 35px rgba(33, 150, 243, 0.2);
                border-left-color: #1976d2;
            }}

            .quick-action-card:hover::before {{
                opacity: 1;
            }}

            .quick-action-card:focus {{
                outline: 3px solid rgba(33, 150, 243, 0.5);
                outline-offset: 2px;
            }}

            .quick-action-card .metric-title {{
                font-size: 1.1em;
                font-weight: 600;
                color: #1976d2;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}

            .quick-action-card .metric-value {{
                font-size: 1.8em;
                font-weight: 800;
                margin: 10px 0;
                text-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }}

            .quick-action-card .metric-description {{
                color: #666;
                font-size: 0.95em;
                line-height: 1.4;
            }}

            .quick-action-card .metric-description > div {{
                margin-top: 8px;
                font-weight: 600;
                font-size: 0.85em;
            }}

            /* Status-specific coloring */
            .quick-actions-section .metric-value.performance-status {{
                color: {performance_status_color} !important;
                background: linear-gradient(135deg, {performance_status_color}15, {performance_status_color}25);
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                position: relative;
            }}

            .quick-actions-section .metric-value.performance-status::after {{
                content: '';
                position: absolute;
                right: -15px;
                top: 50%;
                transform: translateY(-50%);
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: {performance_status_color};
                box-shadow: 0 0 0 3px {performance_status_color}25;
                animation: statusIndicatorPulse 2s infinite;
            }}

            .quick-actions-section .metric-value.presets-count {{
                color: {presets_count_color} !important;
                background: linear-gradient(135deg, {presets_count_color}15, {presets_count_color}25);
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                position: relative;
            }}

            .quick-actions-section .metric-value.presets-count::after {{
                content: '';
                position: absolute;
                right: -15px;
                top: 50%;
                transform: translateY(-50%);
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: {presets_count_color};
                box-shadow: 0 0 0 3px {presets_count_color}25;
                animation: statusIndicatorPulse 2s infinite;
            }}

            .quick-actions-section .metric-value.dependencies-count {{
                color: {dependencies_count_color} !important;
                background: linear-gradient(135deg, {dependencies_count_color}15, {dependencies_count_color}25);
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                position: relative;
            }}

            .quick-actions-section .metric-value.dependencies-count::after {{
                content: '';
                position: absolute;
                right: -15px;
                top: 50%;
                transform: translateY(-50%);
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: {dependencies_count_color};
                box-shadow: 0 0 0 3px {dependencies_count_color}25;
                animation: statusIndicatorPulse 2s infinite;
            }}

            @keyframes statusIndicatorPulse {{
                0%, 100% {{ transform: translateY(-50%) scale(1); opacity: 1; }}
                50% {{ transform: translateY(-50%) scale(1.2); opacity: 0.7; }}
            }}

            /* Keyboard Shortcuts Styling */
            .quick-action-shortcut {{
                position: absolute;
                top: 15px;
                right: 15px;
                background: linear-gradient(135deg, rgba(33, 150, 243, 0.1) 0%, rgba(33, 150, 243, 0.2) 100%);
                padding: 6px 10px;
                border-radius: 8px;
                border: 1px solid rgba(33, 150, 243, 0.3);
                backdrop-filter: blur(10px);
            }}

            .quick-action-shortcut kbd {{
                background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                font-weight: 600;
                box-shadow: 0 2px 6px rgba(33, 150, 243, 0.3);
                border: none;
            }}

            /* Recent Initializations Styling */
            .recent-initializations-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}

            .recent-init-card {{
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
                transition: all 0.3s ease;
                cursor: pointer;
                position: relative;
                overflow: hidden;
            }}

            .recent-init-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(135deg, transparent 0%, rgba(76, 175, 80, 0.05) 100%);
                opacity: 0;
                transition: opacity 0.3s ease;
                pointer-events: none;
            }}

            .recent-init-card:hover::before {{
                opacity: 1;
            }}

            .recent-init-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
            }}

            /* Export Actions Styling */
            .export-actions {{
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                align-items: center;
                margin: 20px 0;
            }}

            .quick-export-btn {{
                background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                font-size: 0.95em;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
                display: flex;
                align-items: center;
                gap: 8px;
                position: relative;
                overflow: hidden;
            }}

            .quick-export-btn::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                transition: left 0.5s ease;
            }}

            .quick-export-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4);
            }}

            .quick-export-btn:hover::before {{
                left: 100%;
            }}

            .quick-export-btn.primary-export {{
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
            }}

            .quick-export-btn.primary-export:hover {{
                box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
            }}

            .quick-export-btn.secondary-export {{
                background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
                box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
            }}

            .quick-export-btn.secondary-export:hover {{
                box-shadow: 0 6px 20px rgba(255, 152, 0, 0.4);
            }}

            .quick-export-btn.tertiary-export {{
                background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%);
                box-shadow: 0 4px 15px rgba(156, 39, 176, 0.3);
            }}

            .quick-export-btn.tertiary-export:hover {{
                box-shadow: 0 6px 20px rgba(156, 39, 176, 0.4);
            }}

            .button-icon {{
                font-size: 1.1em;
            }}

            /* Keyboard Shortcuts Section */
            .keyboard-shortcuts-section {{
                margin: 25px 0;
                padding: 20px;
                background: linear-gradient(135deg, rgba(233, 101, 0, 0.08) 0%, rgba(255, 152, 0, 0.12) 100%);
                border-radius: 12px;
                border-left: 4px solid #ff9800;
                backdrop-filter: blur(10px);
            }}

            .shortcuts-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }}

            .shortcut-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px 12px;
                background: rgba(255, 255, 255, 0.5);
                border-radius: 8px;
                transition: all 0.2s ease;
            }}

            .shortcut-item:hover {{
                background: rgba(255, 255, 255, 0.8);
                transform: translateY(-1px);
            }}

            .shortcut-item kbd {{
                background: linear-gradient(135deg, #e65100 0%, #d84315 100%);
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                font-weight: 600;
                box-shadow: 0 2px 6px rgba(233, 81, 0, 0.3);
                border: none;
                min-width: 24px;
                text-align: center;
            }}

            .shortcut-item span {{
                font-size: 0.85em;
                color: #5d4037;
                font-weight: 600;
            }}

            /* Module Status Indicator */
            .quick-actions-section h2 span {{
                font-size: 0.55em;
                background: rgba(233, 101, 0, 0.1);
                padding: 6px 12px;
                border-radius: 12px;
                font-weight: normal;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(233, 101, 0, 0.2);
            }}

            /* Responsive Design */
            @media (max-width: 768px) {{
                .quick-actions-grid {{
                    grid-template-columns: 1fr;
                    gap: 15px;
                }}
                
                .recent-initializations-grid {{
                    grid-template-columns: 1fr;
                    gap: 15px;
                }}
                
                .export-actions {{
                    flex-direction: column;
                    align-items: stretch;
                }}
                
                .quick-export-btn {{
                    width: 100%;
                    justify-content: center;
                }}
                
                .shortcuts-grid {{
                    grid-template-columns: repeat(3, 1fr);
                    gap: 10px;
                }}
            }}
            </style>

            <div class="analysis-section quick-actions-section">
                <h2>
                    Quick Actions
                    <span>
                        JavaScript: <span id="quickActionsJsStatus">Loading...</span>
                    </span>
                </h2>
                
                <!-- Main Action Cards Grid -->
                <div class="metric-grid quick-actions-grid">
                    <div class="metric-card quick-action-card" 
                        data-tab="history"
                        onclick="{tab_function}('history', event)"
                        tabindex="0"
                        role="button"
                        aria-label="View initialization history">
                        <div class="metric-title">📊 Initialization History</div>
                        <div class="metric-value" style="color: #1976d2;">{dashboard_json_file_data['metadata']['total_entries']}</div>
                        <div class="metric-description">
                            <span>Total entries available</span>
                            <div style="margin-top: 5px; font-size: 0.85em; color: #666;">
                                View detailed history and trends
                            </div>
                        </div>
                        <div class="quick-action-shortcut">
                            <kbd>H</kbd>
                        </div>
                    </div>
                    
                    <div class="metric-card quick-action-card" 
                        data-tab="performance"
                        onclick="{tab_function}('performance', event)"
                        tabindex="0"
                        role="button"
                        aria-label="View performance benchmarks">
                        <div class="metric-title">⚡ Performance Benchmarks</div>
                        <div class="metric-value performance-status">{performance_status}</div>
                        <div class="metric-description">
                            <span>System performance testing</span>
                            <div style="margin-top: 5px; font-size: 0.85em; color: #666;">
                                {performance_status_message}
                            </div>
                        </div>
                        <div class="quick-action-shortcut">
                            <kbd>P</kbd>
                        </div>
                    </div>
                    
                    <div class="metric-card quick-action-card" 
                        data-tab="configuration"
                        onclick="{tab_function}('configuration', event)"
                        tabindex="0"
                        role="button"
                        aria-label="Manage configurations">
                        <div class="metric-title">⚙️ Configuration Manager</div>
                        <div class="metric-value presets-count">{available_presets_count}</div>
                        <div class="metric-description">
                            <span>Available configuration presets</span>
                            <div style="margin-top: 5px; font-size: 0.85em; color: #666;">
                                Manage system configurations
                            </div>
                        </div>
                        <div class="quick-action-shortcut">
                            <kbd>C</kbd>
                        </div>
                    </div>
                    
                    <div class="metric-card quick-action-card" 
                        data-tab="dependencies"
                        onclick="{tab_function}('dependencies', event)"
                        tabindex="0"
                        role="button"
                        aria-label="Check dependency status">
                        <div class="metric-title">📦 Dependency Status</div>
                        <div class="metric-value dependencies-count">{core_dependencies_count}</div>
                        <div class="metric-description">
                            <span>Core dependencies - {dependencies_status}</span>
                            <div style="margin-top: 5px; font-size: 0.85em; color: #666;">
                                Check framework compatibility
                            </div>
                        </div>
                        <div class="quick-action-shortcut">
                            <kbd>D</kbd>
                        </div>
                    </div>
                </div>
                
                <!-- Recent Initializations Section -->
                <div style="margin-top: 30px;">
                    <h3 style="color: #e65100; display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                        🕒 Recent Initializations
                        <span style="font-size: 0.7em; background: rgba(233, 101, 0, 0.1); padding: 4px 12px; border-radius: 12px; font-weight: normal; backdrop-filter: blur(10px);">
                            Last 4 Entries
                        </span>
                    </h3>
                    
                    <div class="metric-grid recent-initializations-grid">
                        {recent_initialization_cards}
                    </div>
                    
                    <!-- Fallback message when no recent initializations -->
                    <div id="no-recent-initializations" style="display: none;" class="sr-only">
                        <div class="metric-card" style="text-align: center; padding: 30px;">
                            <div class="metric-title" style="color: #666;">No Previous Initializations</div>
                            <div class="metric-value" style="font-size: 1.5em; color: #999;">N/A</div>
                            <div class="metric-description" style="color: #666;">
                                This appears to be the first initialization today.<br>
                                Previous entries will appear here after subsequent runs.
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Quick Export Actions -->
                <div style="margin-top: 25px; padding-top: 20px; border-top: 2px solid rgba(233, 101, 0, 0.2);">
                    <h4 style="color: #e65100; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                        📥 Export Options
                    </h4>
                    <div class="export-actions">
                        <button class="quick-export-btn primary-export" 
                                onclick="{export_data_function}()" 
                                title="Export complete dashboard data including all historical entries"
                                aria-label="Export complete data">
                            <span class="button-icon">💾</span>
                            Export Complete Data
                        </button>
                        
                        <button class="quick-export-btn secondary-export" 
                                onclick="{print_function}()" 
                                title="Print or save current dashboard view as PDF"
                                aria-label="Print dashboard">
                            <span class="button-icon">🖨️</span>
                            Print Dashboard
                        </button>
                        
                        <button class="quick-export-btn tertiary-export" 
                                onclick="{tab_function}('reports', event)" 
                                title="View all reports and export options"
                                aria-label="View all reports">
                            <span class="button-icon">📄</span>
                            View All Reports
                        </button>
                    </div>
                    
                    <!-- Keyboard Shortcuts Help -->
                    <div class="keyboard-shortcuts-section">
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px; color: #e65100; font-weight: 600;">
                            ⌨️ Keyboard Shortcuts
                        </div>
                        <div class="shortcuts-grid">
                            <div class="shortcut-item">
                                <kbd>H</kbd>
                                <span>History</span>
                            </div>
                            <div class="shortcut-item">
                                <kbd>P</kbd>
                                <span>Performance</span>
                            </div>
                            <div class="shortcut-item">
                                <kbd>C</kbd>
                                <span>Configuration</span>
                            </div>
                            <div class="shortcut-item">
                                <kbd>D</kbd>
                                <span>Dependencies</span>
                            </div>
                            <div class="shortcut-item">
                                <kbd>R</kbd>
                                <span>Reports</span>
                            </div>
                            <div class="shortcut-item">
                                <kbd>E</kbd>
                                <span>Export</span>
                            </div>
                            <div class="shortcut-item">
                                <kbd>ESC</kbd>
                                <span>Close Modal</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """

        except Exception as e:
            logger.error(f"Error generating quick actions HTML: {e}")
            return self._generate_quick_actions_fallback(dashboard_json_file_data, performance, config_info, detailed_versions)

    def _generate_recent_initialization_cards(self, dashboard_json_file_data: Dict) -> str:
        """
        Generate HTML for recent initialization cards with enhanced CSS styling.
        Updated to work harmoniously with dashboard_styles.css animations and colors.
        """
        recent_entries = dashboard_json_file_data.get('entries', [])
        
        if len(recent_entries) <= 1:
            return """
            <div class="metric-card" style="text-align: center; padding: 30px; border-left-color: #FF9800;">
                <div class="metric-title" style="color: #666; display: flex; align-items: center; justify-content: center; gap: 10px;">
                    📊 No Previous Initializations
                </div>
                <div class="metric-value" style="font-size: 1.5em; color: #999;">N/A</div>
                <div class="metric-description" style="color: #666; text-align: center; margin-top: 15px;">
                    This appears to be the first initialization today.<br>
                    Previous entries will appear here after subsequent runs.
                    <div style="margin-top: 15px; padding: 12px; background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); border-radius: 6px; color: #e65100; font-weight: 600;">
                        ℹ️ Tip: Run the system multiple times to see initialization history trends
                    </div>
                </div>
            </div>
            """
        
        # Get the last 4 entries (excluding current one if it's already there)
        display_entries = recent_entries[-4:-1] if len(recent_entries) > 4 else recent_entries[:-1]
        display_entries.reverse()  # Show most recent first
        
        cards_html = ""
        
        for i, entry in enumerate(display_entries):
            entry_data = entry.get('data', {})
            entry_time = entry.get('timestamp', 'Unknown time')
            
            # Parse time for display
            try:
                parsed_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                display_time = parsed_time.strftime('%H:%M:%S')
                display_date = parsed_time.strftime('%m/%d')
            except:
                display_time = entry_time.split('T')[1][:8] if 'T' in entry_time else entry_time
                display_date = entry_time.split('T')[0] if 'T' in entry_time else 'Unknown'
            
            # Extract status and health data
            entry_status = entry_data.get('status', 'unknown')
            entry_health = entry_data.get('health_score', 0)
            entry_duration = entry_data.get('duration_seconds', 0)
            entry_models = entry_data.get('model_variants', 0)
            entry_cuda = entry_data.get('cuda_available', False)
            entry_repro_score = entry_data.get('reproducibility_score', 0)
            entry_logging_score = entry_data.get('logging_compliance', 0)
            
            # Determine status styling with enhanced colors
            if entry_status.lower() == 'success':
                status_class = "history-status-success"
                status_display = "SUCCESS"
                status_color = "#4CAF50"
                status_bg = "linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)"
            elif entry_status.lower() in ['warning', 'partial']:
                status_class = "history-status-warning"
                status_display = "WARNING"
                status_color = "#FF9800"
                status_bg = "linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%)"
            else:
                status_class = "history-status-error"
                status_display = "ERROR"
                status_color = "#F44336"
                status_bg = "linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%)"
            
            # Determine health color with gradient
            if entry_health > 80:
                health_color = "#4CAF50"
                health_bg = "linear-gradient(135deg, #4CAF50, #81C784)"
            elif entry_health > 60:
                health_color = "#FF9800"
                health_bg = "linear-gradient(135deg, #FF9800, #FFB74D)"
            else:
                health_color = "#F44336"
                health_bg = "linear-gradient(135deg, #F44336, #EF5350)"
            
            # Performance indicators
            perf_indicators = []
            if entry_cuda:
                perf_indicators.append("🚀 GPU")
            else:
                perf_indicators.append("🖥️ CPU")
            
            if entry_repro_score > 80:
                perf_indicators.append("🔒 Secure")
            if entry_logging_score > 80:
                perf_indicators.append("📝 Logged")
            
            perf_indicator_text = " ".join(perf_indicators)
            
            # Determine JavaScript function to use
            onclick_handler = f"window.DashboardJS.viewDetails({len(recent_entries) - len(display_entries) + i})" if self.javascript_module_available else f"viewInitializationDetails({len(recent_entries) - len(display_entries) + i})"
            
            cards_html += f"""
            <div class="metric-card recent-init-card" 
                style="cursor: pointer; transition: all 0.3s ease; border-left-color: {status_color}; position: relative;" 
                onclick="{onclick_handler}"
                onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.15)'"
                onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(0,0,0,0.08)'">
                
                <!-- Animated background overlay -->
                <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: {status_bg}; opacity: 0; transition: opacity 0.3s ease; border-radius: 10px; pointer-events: none;"
                    onmouseenter="this.style.opacity='0.1';" onmouseleave="this.style.opacity='0';"></div>
                
                <div style="position: relative; z-index: 1;">
                    <div class="metric-title" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="display: flex; align-items: center; gap: 8px;">
                            🕐 {display_date} at {display_time}
                        </span>
                        <span class="{status_class}" style="font-size: 0.75em; padding: 4px 10px; border-radius: 12px; font-weight: 600;">{status_display}</span>
                    </div>
                    
                    <div class="metric-value" style="font-size: 1.8em; background: {health_bg}; background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;">
                        {entry_health:.1f}% Health
                    </div>
                    
                    <div class="metric-description" style="margin-top: 15px;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 10px;">
                            <div style="font-size: 0.85em; display: flex; align-items: center; gap: 4px;">
                                ⏱️ <strong>Duration:</strong> {entry_duration:.2f}s
                            </div>
                            <div style="font-size: 0.85em; display: flex; align-items: center; gap: 4px;">
                                🔄 <strong>Models:</strong> {entry_models}
                            </div>
                            <div style="font-size: 0.85em; display: flex; align-items: center; gap: 4px;">
                                {'🟢' if entry_cuda else '🔴'} <strong>CUDA:</strong> <span style="color: {'#4CAF50' if entry_cuda else '#F44336'};">
                                    {'Available' if entry_cuda else 'Not Available'}
                                </span>
                            </div>
                            <div style="font-size: 0.85em; display: flex; align-items: center; gap: 4px;">
                                🎯 <strong>Repro:</strong> <span style="color: {health_color};">{entry_repro_score:.1f}%</span>
                            </div>
                        </div>
                        
                        <div style="margin-top: 12px; padding: 8px 12px; background: rgba(33, 150, 243, 0.05); border-radius: 6px; border-left: 3px solid {status_color};">
                            <div style="font-size: 0.8em; color: #666; display: flex; align-items: center; justify-content: space-between;">
                                <span style="font-weight: 600;">{perf_indicator_text}</span>
                                <span style="color: #2196f3;">Click for details →</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
        
        return cards_html

    def _generate_quick_actions_fallback(self, dashboard_json_file_data, performance, config_info, detailed_versions) -> str:
        """
        Enhanced fallback quick actions generation if template file is missing.
        Updated to use appropriate JavaScript calls based on module availability and harmonized with CSS.
        """
        recent_cards_html = self._generate_recent_initialization_cards(dashboard_json_file_data)
        
        # Determine which JavaScript functions to use
        tab_function = "window.DashboardJS.showTab" if self.javascript_module_available else "showTab"
        export_data_function = "window.DashboardJS.exportData" if self.javascript_module_available else "exportDashboardData"
        
        # Calculate enhanced metrics
        performance_available = bool(performance and performance.get('baselines'))
        if performance_available:
            summary = performance.get('summary', {})
            overall_capability = summary.get('overall_capability', 'unknown').lower()
            if overall_capability == 'high':
                performance_status = 'Excellent'
                performance_color = '#4CAF50'
                performance_message = 'System optimized for deep learning workloads'
            elif overall_capability == 'medium':
                performance_status = 'Good'
                performance_color = '#8BC34A'
                performance_message = 'Suitable for most deep learning tasks'
            else:
                performance_status = 'Limited'
                performance_color = '#FF9800'
                performance_message = 'Consider hardware optimizations'
        else:
            performance_status = 'Setup Required'
            performance_color = '#FF9800'
            performance_message = 'Run performance benchmarks to assess capabilities'
        
        available_presets_count = len(config_info.get('available_presets', []))
        presets_color = '#4CAF50' if available_presets_count > 0 else '#FF9800'
        
        # Calculate dependencies status
        core_deps_count = 0
        deps_status = 'Unknown'
        deps_color = '#666'
        
        if detailed_versions:
            core_deps = [d for d in detailed_versions.values() 
                        if isinstance(d, dict) and d.get('required', False)]
            core_deps_count = len(core_deps)
            
            if core_deps_count > 0:
                compatible_count = sum(1 for d in core_deps if d.get('compatible', False))
                compatibility_rate = compatible_count / core_deps_count
                
                if compatibility_rate >= 0.9:
                    deps_status = 'Excellent'
                    deps_color = '#4CAF50'
                elif compatibility_rate >= 0.7:
                    deps_status = 'Good'
                    deps_color = '#8BC34A'
                else:
                    deps_status = 'Issues'
                    deps_color = '#F44336'
        
        return f"""
        <!-- Enhanced Quick Actions Section with Dynamic CSS -->
        <style>
        .performance-status {{ color: {performance_color} !important; }}
        .presets-count {{ color: {presets_color} !important; }}
        .dependencies-count {{ color: {deps_color} !important; }}
        
        .quick-action-card {{
            position: relative;
            overflow: hidden;
        }}
        
        .quick-action-card::before {{
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, transparent 30%, {performance_color}22 50%, transparent 70%);
            z-index: -1;
            transition: opacity 0.3s ease;
            opacity: 0;
            border-radius: 10px;
        }}
        
        .quick-action-card:hover::before {{
            opacity: 1;
        }}
        
        .quick-actions-section .metric-value.performance-status::after {{
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {performance_color};
            margin-left: 8px;
            animation: statusPulse 2s infinite;
        }}
        
        .quick-actions-section .metric-value.presets-count::after {{
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {presets_color};
            margin-left: 8px;
            animation: statusPulse 2s infinite;
        }}
        
        .quick-actions-section .metric-value.dependencies-count::after {{
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {deps_color};
            margin-left: 8px;
            animation: statusPulse 2s infinite;
        }}
        </style>
        
        <div class="analysis-section quick-actions-section">
            <h2>🚀 Quick Actions</h2>
            <p class="tab-description">Fast access to key system functions and recent initialization history</p>
            
            <!-- Main Action Cards -->
            <div class="metric-grid quick-actions-grid">
                <div class="metric-card quick-action-card" data-tab="history" 
                    style="cursor: pointer; border-left-color: #2196F3;" 
                    onclick="{tab_function}('history')"
                    onmouseover="this.style.transform='translateY(-5px)'"
                    onmouseout="this.style.transform='translateY(0)'">
                    <div class="quick-action-shortcut">
                        <kbd>H</kbd>
                    </div>
                    <div class="metric-title">📈 View Initialization History</div>
                    <div class="metric-value" style="color: #2196F3;">{dashboard_json_file_data['metadata']['total_entries']}</div>
                    <div class="metric-description">
                        <span>Total initialization entries available</span>
                        <div style="margin-top: 8px; font-size: 0.85em; color: #1976D2; font-weight: 600;">
                            Track system performance over time
                        </div>
                    </div>
                </div>
                
                <div class="metric-card quick-action-card" data-tab="performance" 
                    style="cursor: pointer; border-left-color: {performance_color};" 
                    onclick="{tab_function}('performance')"
                    onmouseover="this.style.transform='translateY(-5px)'"
                    onmouseout="this.style.transform='translateY(0)'">
                    <div class="quick-action-shortcut">
                        <kbd>P</kbd>
                    </div>
                    <div class="metric-title">⚡ Performance Benchmarks</div>
                    <div class="metric-value performance-status">{performance_status}</div>
                    <div class="metric-description">
                        <span>{performance_message}</span>
                        <div style="margin-top: 8px; font-size: 0.85em; color: {performance_color}; font-weight: 600;">
                            {'Review system capabilities' if performance_available else 'Run performance tests'}
                        </div>
                    </div>
                </div>
                
                <div class="metric-card quick-action-card" data-tab="configuration" 
                    style="cursor: pointer; border-left-color: {presets_color};" 
                    onclick="{tab_function}('configuration')"
                    onmouseover="this.style.transform='translateY(-5px)'"
                    onmouseout="this.style.transform='translateY(0)'">
                    <div class="quick-action-shortcut">
                        <kbd>C</kbd>
                    </div>
                    <div class="metric-title">⚙️ Configuration Manager</div>
                    <div class="metric-value presets-count">{available_presets_count}</div>
                    <div class="metric-description">
                        <span>Available configuration presets</span>
                        <div style="margin-top: 8px; font-size: 0.85em; color: {presets_color}; font-weight: 600;">
                            {'Manage system configurations' if available_presets_count > 0 else 'Setup configuration presets'}
                        </div>
                    </div>
                </div>
                
                <div class="metric-card quick-action-card" data-tab="dependencies" 
                    style="cursor: pointer; border-left-color: {deps_color};" 
                    onclick="{tab_function}('dependencies')"
                    onmouseover="this.style.transform='translateY(-5px)'"
                    onmouseout="this.style.transform='translateY(0)'">
                    <div class="quick-action-shortcut">
                        <kbd>D</kbd>
                    </div>
                    <div class="metric-title">📦 Dependency Status</div>
                    <div class="metric-value dependencies-count">{core_deps_count}</div>
                    <div class="metric-description">
                        <span>Core dependencies - {deps_status}</span>
                        <div style="margin-top: 8px; font-size: 0.85em; color: {deps_color}; font-weight: 600;">
                            Check framework compatibility
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Export Actions -->
            <div style="margin: 25px 0; padding: 20px; background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%); border-radius: 8px; border-left: 4px solid #2196f3;">
                <h3 style="margin-top: 0; color: #1976d2; display: flex; align-items: center; gap: 10px;">
                    📤 Export Actions
                </h3>
                <div style="display: flex; gap: 15px; flex-wrap: wrap; align-items: center;">
                    <button class="quick-export-btn" onclick="{export_data_function}()" 
                            title="Export complete dashboard data including all historical entries">
                        <span class="button-icon">📊</span>
                        Export All Data
                    </button>
                    <button class="quick-export-btn" onclick="window.print()" 
                            style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);"
                            title="Print or save current dashboard view as PDF">
                        <span class="button-icon">🖨️</span>
                        Print Dashboard
                    </button>
                    <div style="margin-left: 20px; font-size: 0.9em; color: #666;">
                        Export includes all {dashboard_json_file_data['metadata']['total_entries']} initialization entries and system status
                    </div>
                </div>
            </div>
            
            <!-- Recent Initializations -->
            <div style="margin-top: 30px;">
                <h3 style="color: #e65100; display: flex; align-items: center; gap: 10px;">
                    🕒 Recent Initializations
                    <span style="font-size: 0.7em; background: rgba(255, 152, 0, 0.1); padding: 4px 8px; border-radius: 12px; font-weight: normal;">
                        Click for details
                    </span>
                </h3>
                <div class="metric-grid recent-initializations-grid">
                    {recent_cards_html}
                </div>
            </div>
        </div>
        """

    def _generate_history_tab_content(self, dashboard_json_file_data: Dict, system_health_score: float, compact_status: Dict) -> str:
        """
        Generate the history tab content HTML.
        Updated to use appropriate JavaScript calls based on module availability and enhanced CSS harmony.
        """
        # Calculate statistics
        all_entries = dashboard_json_file_data.get('entries', [])
        total_entries = len(all_entries)
        successful_entries = len([e for e in all_entries if e.get('data', {}).get('status', '').lower() == 'success'])
        
        if all_entries:
            avg_health = sum(e.get('data', {}).get('health_score', 0) for e in all_entries) / len(all_entries)
            avg_duration = sum(e.get('data', {}).get('duration_seconds', 0) for e in all_entries) / len(all_entries)
            avg_models = sum(e.get('data', {}).get('model_variants', 0) for e in all_entries) / len(all_entries)
        else:
            avg_health = system_health_score
            avg_duration = compact_status['duration_seconds']
            avg_models = compact_status['model_variants']
        
        # Determine which JavaScript functions to use
        view_details_function = "window.DashboardJS.viewDetails" if self.javascript_module_available else "viewInitializationDetails"
        export_data_function = "window.DashboardJS.exportData" if self.javascript_module_available else "exportDashboardData"
        
        # Generate history items
        history_items_html = ""
        if all_entries:
            for i, entry in enumerate(reversed(all_entries)):
                entry_data = entry.get('data', {})
                entry_timestamp = entry.get('timestamp', 'Unknown time')
                
                try:
                    parsed_time = datetime.fromisoformat(entry_timestamp.replace('Z', '+00:00'))
                    display_timestamp = parsed_time.strftime('%Y-%m-%d %H:%M:%S')
                    relative_time = self._get_relative_time(parsed_time)
                except:
                    display_timestamp = entry_timestamp
                    relative_time = 'Unknown time ago'
                
                entry_status = entry_data.get('status', 'unknown').lower()
                entry_health = entry_data.get('health_score', 0)
                entry_duration = entry_data.get('duration_seconds', 0)
                entry_models = entry_data.get('model_variants', 0)
                entry_cuda = entry_data.get('cuda_available', False)
                entry_repro_score = entry_data.get('reproducibility_score', 0)
                entry_logging_score = entry_data.get('logging_compliance', 0)
                entry_perf_monitor = entry_data.get('performance_monitoring', False)
                entry_mem_mgmt = entry_data.get('memory_management', False)
                
                # Determine status styling
                if entry_status == 'success':
                    status_class = 'history-status-success'
                    status_display = 'SUCCESS'
                    status_icon = '✅'
                elif entry_status in ['warning', 'partial']:
                    status_class = 'history-status-warning'
                    status_display = 'WARNING'
                    status_icon = '⚠️'
                else:
                    status_class = 'history-status-error'
                    status_display = 'ERROR'
                    status_icon = '❌'
                
                health_color = '#4CAF50' if entry_health > 80 else '#FF9800' if entry_health > 60 else '#F44336'
                cuda_status = 'Available' if entry_cuda else 'Not Available'
                cuda_color = '#4CAF50' if entry_cuda else '#F44336'
                
                # Performance indicators
                performance_badges = []
                if entry_perf_monitor:
                    performance_badges.append('<span style="background: #4CAF50; color: white; padding: 2px 6px; border-radius: 10px; font-size: 0.75em;">📊 Monitoring</span>')
                if entry_mem_mgmt:
                    performance_badges.append('<span style="background: #2196F3; color: white; padding: 2px 6px; border-radius: 10px; font-size: 0.75em;">💾 Memory</span>')
                if entry_cuda:
                    performance_badges.append('<span style="background: #FF9800; color: white; padding: 2px 6px; border-radius: 10px; font-size: 0.75em;">🚀 GPU</span>')
                
                performance_badges_html = ' '.join(performance_badges) if performance_badges else '<span style="color: #999;">No special features</span>'
                
                history_items_html += f"""
                <div class="history-item" onclick="{view_details_function}({len(all_entries) - 1 - i})" 
                    style="cursor: pointer; transition: all 0.3s ease;"
                    onmouseover="this.style.backgroundColor='rgba(33, 150, 243, 0.02)'"
                    onmouseout="this.style.backgroundColor='transparent'">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px;">
                        <div>
                            <span class="history-time" style="font-weight: bold; color: #1976d2; font-size: 1.1em; display: flex; align-items: center; gap: 8px;">
                                🕐 {display_timestamp}
                                <span style="font-size: 0.8em; color: #666; font-weight: normal;">({relative_time})</span>
                            </span>
                        </div>
                        <span class="history-status {status_class}" style="display: flex; align-items: center; gap: 5px;">
                            {status_icon} {status_display}
                        </span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 15px 0;">
                        <div>
                            <h5 style="margin: 0 0 8px 0; color: #1976d2; font-size: 1em;">Performance Metrics</h5>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.9em;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="color: #666;">Health Score:</span>
                                    <span style="font-weight: 600; color: {health_color};">{entry_health:.1f}%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="color: #666;">Duration:</span>
                                    <span style="font-weight: 600;">{entry_duration:.2f}s</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="color: #666;">Models:</span>
                                    <span style="font-weight: 600;">{entry_models}</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="color: #666;">CUDA:</span>
                                    <span style="font-weight: 600; color: {cuda_color};">{cuda_status}</span>
                                </div>
                            </div>
                        </div>
                        
                        <div>
                            <h5 style="margin: 0 0 8px 0; color: #1976d2; font-size: 1em;">Configuration Scores</h5>
                            <div style="display: grid; grid-template-columns: 1fr; gap: 8px; font-size: 0.9em;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="color: #666;">Reproducibility:</span>
                                    <span style="font-weight: 600; color: {health_color};">{entry_repro_score:.1f}%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="color: #666;">Logging Compliance:</span>
                                    <span style="font-weight: 600; color: {health_color};">{entry_logging_score:.1f}%</span>
                                </div>
                                <div style="margin-top: 10px;">
                                    <span style="color: #666; font-size: 0.85em;">Features:</span>
                                    <div style="margin-top: 4px;">
                                        {performance_badges_html}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-size: 0.85em; color: #666;">
                            Entry #{len(all_entries) - i} • Click for complete details and raw data
                        </div>
                        <button class="details-button" style="padding: 6px 12px; font-size: 0.85em;">
                            📋 View Details
                        </button>
                    </div>
                </div>
                """
        else:
            history_items_html = """
            <div class="history-item" style="text-align: center; padding: 40px; color: #666;">
                <div style="font-size: 1.5em; margin-bottom: 10px;">📊</div>
                <h4 style="color: #999; margin: 10px 0;">No Initialization History Available</h4>
                <p>Previous initialization entries will appear here after running the system multiple times.</p>
                <div style="margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%); border-radius: 8px; color: #1976d2;">
                    <strong>Tip:</strong> Each system initialization creates an entry that tracks performance, configuration, and system status over time.
                </div>
            </div>
            """
        
        # Generate statistics cards with enhanced styling
        if all_entries:
            success_rate = successful_entries / total_entries * 100
            success_color = '#4CAF50' if success_rate > 80 else '#FF9800' if success_rate > 60 else '#F44336'
            health_color = '#4CAF50' if avg_health > 80 else '#FF9800' if avg_health > 60 else '#F44336'
            
            # Calculate trends if we have enough data
            trend_analysis = ""
            if len(all_entries) >= 2:
                recent_half = all_entries[len(all_entries)//2:]
                older_half = all_entries[:len(all_entries)//2]
                
                recent_avg_health = sum(e.get('data', {}).get('health_score', 0) for e in recent_half) / len(recent_half)
                older_avg_health = sum(e.get('data', {}).get('health_score', 0) for e in older_half) / len(older_half)
                health_trend = recent_avg_health - older_avg_health
                
                recent_avg_duration = sum(e.get('data', {}).get('duration_seconds', 0) for e in recent_half) / len(recent_half)
                older_avg_duration = sum(e.get('data', {}).get('duration_seconds', 0) for e in older_half) / len(older_half)
                duration_trend = recent_avg_duration - older_avg_duration
                
                health_trend_text = '📈 Improving' if health_trend > 1 else '📉 Declining' if health_trend < -1 else '📊 Stable'
                health_trend_color = '#4CAF50' if health_trend > 0 else '#F44336' if health_trend < 0 else '#666'
                
                duration_trend_text = '⚡ Getting Faster' if duration_trend < -0.1 else '⏳ Getting Slower' if duration_trend > 0.1 else '⚖️ Consistent'
                duration_trend_color = '#4CAF50' if duration_trend < 0 else '#F44336' if duration_trend > 0 else '#666'
                
                trend_analysis = f"""
                <div style="margin: 20px 0; padding: 20px; background: linear-gradient(135deg, #f9f9f9 0%, #f0f0f0 100%); border-radius: 8px; border-left: 4px solid #2196f3;">
                    <h4 style="margin-top: 0; color: #1976d2;">📊 Performance Trends</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div style="text-align: center; padding: 10px;">
                            <div style="font-size: 1.1em; font-weight: 600; color: {health_trend_color}; margin-bottom: 5px;">
                                {health_trend_text}
                            </div>
                            <div style="font-size: 0.9em; color: #666;">Health Score Trend</div>
                            <div style="font-size: 0.8em; color: #999; margin-top: 3px;">
                                {health_trend:+.1f}% change
                            </div>
                        </div>
                        <div style="text-align: center; padding: 10px;">
                            <div style="font-size: 1.1em; font-weight: 600; color: {duration_trend_color}; margin-bottom: 5px;">
                                {duration_trend_text}
                            </div>
                            <div style="font-size: 0.9em; color: #666;">Duration Trend</div>
                            <div style="font-size: 0.8em; color: #999; margin-top: 3px;">
                                {duration_trend:+.2f}s change
                            </div>
                        </div>
                    </div>
                </div>
                """
            
            statistics_cards = f"""
            <div class="metric-card" style="border-left-color: {success_color};">
                <div class="metric-title">Success Rate</div>
                <div class="metric-value" style="color: {success_color};">{success_rate:.1f}%</div>
                <div class="metric-description">
                    <span>{successful_entries} out of {total_entries} successful</span>
                </div>
                <div style="margin-top: 10px; padding: 8px; background: rgba(76, 175, 80, 0.1); border-radius: 4px;">
                    <span style="font-size: 0.85em; color: #2e7d32;">
                        {'Excellent reliability' if success_rate > 90 else 'Good performance' if success_rate > 70 else 'Needs attention'}
                    </span>
                </div>
            </div>
            
            <div class="metric-card" style="border-left-color: #2196F3;">
                <div class="metric-title">Average Duration</div>
                <div class="metric-value">{avg_duration:.2f}s</div>
                <div class="metric-description">
                    <span>Average initialization time</span>
                </div>
                <div style="margin-top: 10px; padding: 8px; background: rgba(33, 150, 243, 0.1); border-radius: 4px;">
                    <span style="font-size: 0.85em; color: #1565c0;">
                        {'Fast' if avg_duration < 2 else 'Moderate' if avg_duration < 5 else 'Slow'} initialization
                    </span>
                </div>
            </div>
            
            <div class="metric-card" style="border-left-color: {health_color};">
                <div class="metric-title">Average Health Score</div>
                <div class="metric-value" style="color: {health_color};">{avg_health:.1f}%</div>
                <div class="metric-description">
                    <span>Overall system health</span>
                </div>
                <div style="margin-top: 10px; padding: 8px; background: rgba(76, 175, 80, 0.1); border-radius: 4px;">
                    <span style="font-size: 0.85em; color: #2e7d32;">
                        {'Excellent' if avg_health > 80 else 'Good' if avg_health > 60 else 'Needs improvement'} condition
                    </span>
                </div>
            </div>
            
            <div class="metric-card" style="border-left-color: #FF9800;">
                <div class="metric-title">Average Models</div>
                <div class="metric-value">{avg_models:.1f}</div>
                <div class="metric-description">
                    <span>Model variants per initialization</span>
                </div>
                <div style="margin-top: 10px; padding: 8px; background: rgba(255, 152, 0, 0.1); border-radius: 4px;">
                    <span style="font-size: 0.85em; color: #f57c00;">
                        {'High variety' if avg_models > 5 else 'Moderate variety' if avg_models > 2 else 'Limited variety'}
                    </span>
                </div>
            </div>
            """
        else:
            statistics_cards = f"""
            <div class="metric-card" style="border-left-color: #4CAF50;">
                <div class="metric-title">Current Success Rate</div>
                <div class="metric-value">100%</div>
                <div class="metric-description">Current initialization successful</div>
            </div>
            
            <div class="metric-card" style="border-left-color: #2196F3;">
                <div class="metric-title">Current Duration</div>
                <div class="metric-value">{compact_status['duration_seconds']:.2f}s</div>
                <div class="metric-description">Current initialization time</div>
            </div>
            
            <div class="metric-card" style="border-left-color: #FF9800;">
                <div class="metric-title">Current Health Score</div>
                <div class="metric-value">{system_health_score:.1f}%</div>
                <div class="metric-description">Current system health</div>
            </div>
            
            <div class="metric-card" style="border-left-color: #9C27B0;">
                <div class="metric-title">Current Models</div>
                <div class="metric-value">{compact_status['model_variants']}</div>
                <div class="metric-description">Current model variants</div>
            </div>
            """
            trend_analysis = """
            <div style="margin: 20px 0; padding: 20px; background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); border-radius: 8px; border-left: 4px solid #ff9800;">
                <h4 style="margin-top: 0; color: #e65100;">📊 Trend Analysis</h4>
                <p style="margin: 0; color: #666;">
                    Run the system multiple times to generate trend analysis and performance comparisons. 
                    Historical data will show patterns in initialization time, system health, and configuration changes.
                </p>
            </div>
            """
        
        return f"""
        <div id="history" class="tab-content" role="tabpanel" aria-labelledby="history-tab">
            <div class="tab-header">
                <h3>Initialization History</h3>
                <p class="tab-description">Complete history of system initializations with performance tracking and trend analysis</p>
            </div>
            
            <!-- Summary Statistics -->
            <div class="metric-grid">
                {statistics_cards}
            </div>
            
            {trend_analysis}
            
            <!-- History Viewer -->
            <div class="history-viewer">
                <div class="history-header">
                    <span style="display: flex; align-items: center; gap: 10px;">
                        📋 Initialization History
                        <span style="background: rgba(33, 150, 243, 0.1); padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: normal;">
                            {total_entries} {'entry' if total_entries == 1 else 'entries'}
                        </span>
                    </span>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <button class="details-button" onclick="{export_data_function}()" 
                                style="padding: 8px 16px; font-size: 0.9em;">
                            📤 Export Data
                        </button>
                    </div>
                </div>
                <div class="history-content">
                    {history_items_html}
                </div>
            </div>
            
            <!-- Footer Actions -->
            <div style="margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%); border-radius: 8px; text-align: center;">
                <h4 style="color: #1976d2; margin-top: 0;">💡 History Management Tips</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: left;">
                    <div>
                        <strong>📊 Export Data:</strong> Use the export button to download complete history data for external analysis
                    </div>
                    <div>
                        <strong>🔍 View Details:</strong> Click any entry to see complete initialization details and raw system data
                    </div>
                    <div>
                        <strong>📈 Track Trends:</strong> Monitor health scores and duration changes over time to identify patterns
                    </div>
                    <div>
                        <strong>⚡ Performance:</strong> Compare CUDA availability and model variant counts across initializations
                    </div>
                </div>
            </div>
        </div>
        """

    def _get_relative_time(self, past_time: datetime) -> str:
        """Calculate relative time from past datetime to now."""
        try:
            now = datetime.now(past_time.tzinfo) if past_time.tzinfo else datetime.now()
            diff = now - past_time
            
            if diff.days > 0:
                return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
            elif diff.seconds > 3600:
                hours = diff.seconds // 3600
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif diff.seconds > 60:
                minutes = diff.seconds // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            else:
                return "Just now"
        except:
            return "Unknown time ago"

    def render_dashboard(self, 
                        enhanced_status: Dict[str, Any], 
                        compact_status: Dict[str, Any],
                        system_health_score: float,
                        dashboard_json_file_data: Dict[str, Any],
                        consolidated_files: Dict[str, Path],
                        report_data: Dict[str, Any],
                        status_data: Dict[str, Any],
                        diagnostics_data: Dict[str, Any],
                        timestamp_obj: datetime) -> str:
        """
        Render the complete dashboard HTML using templates.
        Updated to work with external JavaScript module (dashboard_template.js) and harmonized with dashboard_styles.css.
        
        Args:
            enhanced_status: Enhanced system status dictionary
            compact_status: Compact status data
            system_health_score: Calculated system health score
            dashboard_json_file_data: Dashboard JSON data
            consolidated_files: Dictionary of consolidated file paths
            report_data: Report data dictionary
            status_data: Status data dictionary  
            diagnostics_data: Diagnostics data dictionary
            timestamp_obj: Timestamp object
            
        Returns:
            Complete HTML content as string
        """
        try:
            # Load the main template
            main_template_content = self._load_template("dashboard_base.html")
            template = Template(main_template_content)
            
            # Extract data sections safely
            init_info = enhanced_status.get('initialization', {})
            sys_info = enhanced_status.get('system', {})
            detailed_hw = enhanced_status.get('detailed_hardware', {})
            config_info = enhanced_status.get('config', {})
            model_info = enhanced_status.get('models', {})
            performance = enhanced_status.get('performance', {})
            detailed_versions = enhanced_status.get('detailed_versions', {})
            
            # Extract hardware info safely
            cpu_info = detailed_hw.get('cpu_cores', {})
            ram_info = detailed_hw.get('system_ram', {})
            disk_info = detailed_hw.get('disk_space', {})
            cuda_info = detailed_hw.get('cuda', {})
            
            # Generate component HTML sections
            gpu_info_html = self._generate_gpu_info_html(cuda_info)
            performance_html = self._generate_performance_html(performance)
            core_deps_html, optional_deps_html = self._generate_dependencies_tables(detailed_versions)
            quick_actions_html = self._generate_quick_actions_html(
                dashboard_json_file_data, performance, config_info, detailed_versions
            )
            history_tab_content = self._generate_history_tab_content(
                dashboard_json_file_data, system_health_score, compact_status
            )
            
            # Prepare JavaScript data based on module availability
            if self.javascript_module_available:
                javascript_data = self._prepare_javascript_data_external(dashboard_json_file_data, compact_status)
                logger.info("Using external JavaScript module for enhanced dashboard functionality")
            else:
                javascript_data = self._prepare_javascript_data_inline_fallback(dashboard_json_file_data, compact_status)
                logger.warning("Using inline JavaScript fallback - external module not available")
            
            # Determine health score class with enhanced granularity
            if system_health_score > 90:
                health_score_class = 'excellent'
            elif system_health_score > 80:
                health_score_class = 'excellent'
            elif system_health_score > 70:
                health_score_class = 'good'
            elif system_health_score > 60:
                health_score_class = 'good'
            elif system_health_score > 40:
                health_score_class = 'fair'
            else:
                health_score_class = 'poor'
            
            # Determine status classes with enhanced logic
            init_status_class = ('status-success' if compact_status['status'] == 'success' 
                            else 'status-warning' if compact_status['status'] == 'warning' 
                            else 'status-error')
            
            cuda_status_class = 'status-success' if compact_status['cuda_available'] else 'status-error'
            perf_monitor_class = 'status-success' if compact_status.get('performance_monitoring') else 'status-error'
            mem_mgmt_class = 'status-success' if compact_status.get('memory_management') else 'status-warning'
            
            # Calculate storage usage percentage safely
            storage_total = disk_info.get('total_gb', 0)
            storage_free = disk_info.get('free_gb', 0)
            storage_used_percent = ((storage_total - storage_free) / storage_total * 100) if storage_total > 0 else 0

            # Generate variant status table rows with enhanced styling
            variant_status_rows = ""
            variant_statuses = model_info.get('variant_status', {})
            if variant_statuses:
                for name, status in variant_statuses.items():
                    status_class = 'status-success' if status == 'available' else 'status-error'
                    status_text = '✅ Available' if status == 'available' else '❌ Missing'
                    status_icon = '🟢' if status == 'available' else '🔴'
                    variant_status_rows += f"""
                    <tr style="transition: background-color 0.2s ease;" 
                        onmouseover="this.style.backgroundColor='rgba(33, 150, 243, 0.05)'" 
                        onmouseout="this.style.backgroundColor='transparent'">
                        <td style="font-weight: 600;">{name}</td>
                        <td><span class='{status_class}' style="display: flex; align-items: center; gap: 5px;">{status_icon} {status_text}</span></td>
                    </tr>
                    """
            else:
                variant_status_rows = "<tr><td colspan='2' style='text-align: center; color: #666; font-style: italic; padding: 20px;'>No model variant information available</td></tr>"

            # Enhanced preset information
            available_presets = config_info.get('available_presets', [])
            available_presets_display = ', '.join(available_presets) if available_presets else 'No presets configured'
            
            # Prepare template context with comprehensive data
            context = {
                'javascript_data': javascript_data,
                'date_str': timestamp_obj.strftime('%Y%m%d'),
                'time_str': timestamp_obj.strftime('%H%M%S'),
                'formatted_timestamp': timestamp_obj.strftime('%Y-%m-%d %H:%M:%S'),
                'total_entries': dashboard_json_file_data['metadata']['total_entries'],
                'quick_actions_html': quick_actions_html,
                'system_health_score': f"{system_health_score:.1f}",
                'health_score_class': health_score_class,
                
                # Status information
                'init_status': compact_status['status'].upper(),
                'init_status_class': init_status_class,
                'init_duration': f"{compact_status['duration_seconds']:.2f}",
                'init_method': init_info.get('method', 'standard'),
                
                # CUDA information
                'cuda_status': 'CUDA Available' if compact_status['cuda_available'] else 'CUDA Not Available',
                'cuda_status_class': cuda_status_class,
                'gpu_count': cuda_info.get('gpu_count', 0) if cuda_info.get('available') else 0,
                'model_variants': compact_status['model_variants'],
                
                # Configuration scores (ensuring numeric values only)
                'reproducibility_score': f"{compact_status.get('reproducibility_score', 0):.1f}",
                'logging_compliance': f"{compact_status.get('logging_compliance', 0):.1f}",
                'performance_monitoring_status': 'Available' if compact_status.get('performance_monitoring') else 'Not Available',
                'performance_monitoring_class': perf_monitor_class,
                'memory_management_status': 'Comprehensive' if compact_status.get('memory_management') else 'Basic',
                'memory_management_class': mem_mgmt_class,
                
                # System resources
                'cpu_logical_cores': cpu_info.get('logical_cores', 'Unknown'),
                'cpu_physical_cores': cpu_info.get('physical_cores', 'Unknown'),
                'ram_total_gb': f"{ram_info.get('ram_total_gb', 0):.1f}",
                'ram_available_gb': f"{ram_info.get('ram_available_gb', 0):.1f}",
                'ram_percent': f"{ram_info.get('ram_percent', 0):.1f}",
                
                # Hardware details
                'cpu_frequency': str(cpu_info.get('capacity', {}).get('frequency_ghz', 'N/A')),
                'storage_total_gb': f"{storage_total:.1f}",
                'storage_free_gb': f"{storage_free:.1f}",
                'storage_used_percent': f"{storage_used_percent:.1f}",
                
                # GPU information
                'gpu_info_html': gpu_info_html,
                
                # Configuration details
                'preset_name': config_info.get('preset_name', 'Default Configuration'),
                'validation_status': config_info.get('validation_status', 'Not Validated'),
                'config_file': config_info.get('config_file', 'config.yaml'),
                'available_presets_count': len(available_presets),
                'available_presets': available_presets_display,
                'variant_status_rows': variant_status_rows,
                
                # Performance metrics
                'performance_html': performance_html,
                
                # Dependencies
                'python_version': sys_info.get('python_version', 'Unknown'),
                'pytorch_version': sys_info.get('pytorch_version', 'Unknown'),
                'platform': sys_info.get('platform', platform.platform()),
                'core_deps_html': core_deps_html,
                'optional_deps_html': optional_deps_html,
                
                # History tab
                'history_tab_content': history_tab_content,
                
                # Report files with safe access
                'report_filename': consolidated_files.get('report', Path('unknown')).name,
                'report_entries_count': report_data.get('metadata', {}).get('total_reports', 0),
                'summary_filename': consolidated_files.get('summary', Path('unknown')).name,
                'status_filename': consolidated_files.get('status', Path('unknown')).name,
                'status_entries_count': status_data.get('metadata', {}).get('total_entries', 0),
                'diagnostics_filename': consolidated_files.get('diagnostics', Path('unknown')).name,
                'diagnostics_entries_count': diagnostics_data.get('metadata', {}).get('total_entries', 0),
                'dashboard_json_filename': consolidated_files.get('dashboard_json', Path('unknown')).name,
                'dashboard_entries_count': dashboard_json_file_data['metadata']['total_entries'],
                
                'type': 'dashboard',
                'tabName': 'dashboard',
                'section_type': 'main'
            }
            
            # Render the template
            rendered_html = template.substitute(context)
            
            # Add comprehensive metadata comment
            architecture_info = f"""
            <!-- 
            =================================================================
            Deep Learning System Dashboard - Template-Based Architecture v3.0
            =================================================================
            JavaScript Module: {'External (dashboard_template.js)' if self.javascript_module_available else 'Inline Fallback'}
            CSS Framework: {'External (dashboard_styles.css)' if self.css_available else 'Inline Fallback'}
            Generated: {timestamp_obj.strftime('%Y-%m-%d %H:%M:%S')}
            Total Entries: {dashboard_json_file_data['metadata']['total_entries']}
            System Health: {system_health_score:.1f}%
            =================================================================
            -->
            """
            rendered_html = rendered_html.replace('<body>', f'<body>\n{architecture_info}')
            
            logger.info(f"Dashboard rendered successfully with {'external' if self.javascript_module_available else 'inline'} JavaScript")
            return rendered_html
            
        except KeyError as e:
            logger.error(f"Missing template variable in main dashboard: {e}")
            return self._generate_error_fallback(f"Missing template variable: {e}", timestamp_obj)
        except Exception as e:
            logger.error(f"Failed to render dashboard template: {e}")
            logger.debug("Template rendering error details:", exc_info=True)
            return self._generate_error_fallback(str(e), timestamp_obj)
    
    def _generate_error_fallback(self, error_message: str, timestamp_obj: datetime) -> str:
        """Generate a comprehensive error fallback HTML page with enhanced styling."""
        
        # Fix the backslash issue by using string replacement instead of f-string
        import json
        template_dir_escaped = str(self.template_dir).replace('\\', '\\\\')

        # Precompute a JS-safe string (quotes + backslashes escaped)
        error_message_js = json.dumps(str(error_message))
        
        return f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard Rendering Error - Deep Learning System</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
                color: #333;
                line-height: 1.6;
                min-height: 100vh;
            }}
            .error-container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
                border-left: 5px solid #f44336;
                position: relative;
                overflow: hidden;
            }}
            .error-container::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #f44336 0%, #d32f2f 50%, #f44336 100%);
                background-size: 200% 100%;
                animation: errorGradientShift 3s ease-in-out infinite;
            }}
            @keyframes errorGradientShift {{
                0%, 100% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
            }}
            .error-header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #f44336;
            }}
            .error-title {{
                color: #d32f2f;
                font-size: 2.2em;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 15px;
                animation: errorPulse 2s infinite;
            }}
            @keyframes errorPulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.02); }}
            }}
            .error-subtitle {{
                color: #666;
                font-size: 1.1em;
                font-weight: 300;
            }}
            .error-message {{
                background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #f44336;
                margin: 20px 0;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 0.95em;
                overflow-x: auto;
                word-break: break-word;
                white-space: pre-wrap;
            }}
            .error-message::before {{
                content: '⚠️ ';
                font-size: 1.2em;
                margin-right: 5px;
            }}
            .status-section {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .status-card {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #2196f3;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }}
            .status-card::before {{
                content: '';
                position: absolute;
                top: 0;
                right: 0;
                width: 30px;
                height: 30px;
                background: linear-gradient(135deg, transparent 50%, #2196f3 50%);
                opacity: 0.1;
            }}
            .status-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                border-left-color: #1976d2;
            }}
            .status-card:hover::before {{
                opacity: 0.2;
            }}
            .status-title {{
                font-weight: 600;
                color: #2196f3;
                margin-bottom: 10px;
                font-size: 1.1em;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .status-value {{
                font-size: 1.3em;
                font-weight: bold;
                color: #333;
            }}
            .status-available {{ 
                color: #4CAF50; 
            }}
            .status-unavailable {{ 
                color: #f44336; 
            }}
            .timestamp {{
                color: #666;
                text-align: center;
                margin-top: 30px;
                font-size: 0.95em;
                padding-top: 20px;
                border-top: 1px solid #e0e0e0;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            }}
            .recovery-suggestions {{
                background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%);
                padding: 25px;
                border-radius: 8px;
                border-left: 4px solid #2196f3;
                margin: 25px 0;
                position: relative;
                overflow: hidden;
            }}
            .recovery-suggestions::before {{
                content: '';
                position: absolute;
                top: 0;
                right: 0;
                width: 100px;
                height: 100px;
                background: radial-gradient(circle, rgba(33, 150, 243, 0.1) 0%, transparent 70%);
                border-radius: 50%;
                transform: translate(30%, -30%);
            }}
            .recovery-title {{
                color: #1976d2;
                font-size: 1.3em;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
                position: relative;
                z-index: 1;
            }}
            .recovery-list {{
                list-style: none;
                padding: 0;
                position: relative;
                z-index: 1;
            }}
            .recovery-list li {{
                padding: 10px 0;
                border-bottom: 1px solid rgba(33, 150, 243, 0.1);
                display: flex;
                align-items: flex-start;
                gap: 12px;
                transition: all 0.2s ease;
            }}
            .recovery-list li:hover {{
                background: rgba(33, 150, 243, 0.05);
                transform: translateX(5px);
            }}
            .recovery-list li:last-child {{
                border-bottom: none;
            }}
            .recovery-list li::before {{
                content: '🔧';
                font-size: 1.1em;
                flex-shrink: 0;
                margin-top: 2px;
            }}
            .recovery-step {{
                flex: 1;
            }}
            .recovery-step strong {{
                color: #1976d2;
                font-weight: 600;
            }}
            .error-details {{
                background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #666;
                margin: 20px 0;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 0.9em;
                overflow-x: auto;
            }}
            .error-details-title {{
                color: #666;
                font-weight: 600;
                margin-bottom: 10px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .error-details-title::before {{
                content: '🔍';
                font-size: 1.1em;
            }}
            .system-info {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .info-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 0;
                border-bottom: 1px solid #eee;
                font-size: 0.9em;
            }}
            .info-item:last-child {{
                border-bottom: none;
            }}
            .info-label {{
                font-weight: 600;
                color: #666;
            }}
            .info-value {{
                color: #2196f3;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 0.85em;
            }}
            .emergency-actions {{
                background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #ff9800;
                margin: 25px 0;
                text-align: center;
            }}
            .emergency-actions h4 {{
                color: #e65100;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
            }}
            .emergency-actions h4::before {{
                content: '🆘';
                font-size: 1.2em;
            }}
            .refresh-btn {{
                background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3);
                position: relative;
                overflow: hidden;
                margin: 0 10px;
            }}
            .refresh-btn::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                transition: left 0.5s ease;
            }}
            .refresh-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(33, 150, 243, 0.4);
            }}
            .refresh-btn:hover::before {{
                left: 100%;
            }}
            .fallback-btn {{
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
            }}
            .fallback-btn:hover {{
                box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
            }}
            .footer-error {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 8px;
                color: #666;
                font-size: 0.9em;
            }}
            .footer-error strong {{
                color: #1976d2;
            }}
            @media (max-width: 768px) {{
                body {{
                    margin: 20px;
                }}
                .error-container {{
                    padding: 25px;
                }}
                .error-title {{
                    font-size: 1.8em;
                    flex-direction: column;
                    gap: 10px;
                }}
                .status-section,
                .system-info {{
                    grid-template-columns: 1fr;
                    gap: 15px;
                }}
                .recovery-list li {{
                    flex-direction: column;
                    gap: 8px;
                }}
                .refresh-btn {{
                    display: block;
                    width: 100%;
                    margin: 5px 0;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="error-container">
            <div class="error-header">
                <h1 class="error-title">
                    🚨 Dashboard Rendering Failed
                </h1>
                <p class="error-subtitle">Deep Learning System Dashboard encountered a critical error</p>
            </div>
            
            <div class="error-message">
    {error_message}
            </div>
            
            <div class="status-section">
                <div class="status-card">
                    <div class="status-title">🎨 CSS Framework Status</div>
                    <div class="status-value {'status-available' if self.css_available else 'status-unavailable'}">
                        {'Available' if self.css_available else 'Not Available'}
                    </div>
                    <p style="margin: 10px 0 0 0; color: #666; font-size: 0.9em;">
                        {'External dashboard_styles.css loaded successfully' if self.css_available else 'Using fallback styling - external CSS not found'}
                    </p>
                </div>
                
                <div class="status-card">
                    <div class="status-title">⚙️ JavaScript Module Status</div>
                    <div class="status-value {'status-available' if self.javascript_module_available else 'status-unavailable'}">
                        {'Available' if self.javascript_module_available else 'Not Available'}
                    </div>
                    <p style="margin: 10px 0 0 0; color: #666; font-size: 0.9em;">
                        {'External dashboard_template.js ready for use' if self.javascript_module_available else 'Fallback inline JavaScript will be used'}
                    </p>
                </div>
                
                <div class="status-card">
                    <div class="status-title">📁 Template Directory</div>
                    <div class="status-value status-available">
                        {self.template_dir.name if self.template_dir.exists() else 'Unknown'}
                    </div>
                    <p style="margin: 10px 0 0 0; color: #666; font-size: 0.9em; font-family: monospace;">
                        {str(self.template_dir)}
                    </p>
                </div>
                
                <div class="status-card">
                    <div class="status-title">🕐 Error Timestamp</div>
                    <div class="status-value">
                        {timestamp_obj.strftime('%H:%M:%S')}
                    </div>
                    <p style="margin: 10px 0 0 0; color: #666; font-size: 0.9em;">
                        {timestamp_obj.strftime('%Y-%m-%d')}
                    </p>
                </div>
            </div>
            
            <div class="recovery-suggestions">
                <div class="recovery-title">
                    🛠️ Recovery Suggestions
                </div>
                <ul class="recovery-list">
                    <li>
                        <div class="recovery-step">
                            <strong>Check Template Files:</strong> Ensure all template files (dashboard_base.html, _quick_actions.html) exist in the templates directory
                        </div>
                    </li>
                    <li>
                        <div class="recovery-step">
                            <strong>Verify CSS Framework:</strong> Confirm dashboard_styles.css is present and accessible in the templates directory
                        </div>
                    </li>
                    <li>
                        <div class="recovery-step">
                            <strong>JavaScript Module:</strong> Check if dashboard_template.js exists for enhanced functionality (fallback will work without it)
                        </div>
                    </li>
                    <li>
                        <div class="recovery-step">
                            <strong>File Permissions:</strong> Verify read permissions on all template files and the templates directory
                        </div>
                    </li>
                    <li>
                        <div class="recovery-step">
                            <strong>Template Syntax:</strong> Review template files for correct placeholder syntax (${{variable_name}})
                        </div>
                    </li>
                    <li>
                        <div class="recovery-step">
                            <strong>Data Structure:</strong> Ensure all required data is properly structured and accessible
                        </div>
                    </li>
                    <li>
                        <div class="recovery-step">
                            <strong>System Resources:</strong> Check available memory and disk space for template processing
                        </div>
                    </li>
                </ul>
            </div>
            
            <div class="error-details">
                <div class="error-details-title">Technical Error Details</div>
                <div class="system-info">
                    <div class="info-item">
                        <span class="info-label">Python Version:</span>
                        <span class="info-value">{sys.version.split()[0]}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Platform:</span>
                        <span class="info-value">{platform.platform()}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Architecture:</span>
                        <span class="info-value">Template-Based v3.0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Template Directory:</span>
                        <span class="info-value">{'Exists' if self.template_dir.exists() else 'Missing'}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">CSS Framework:</span>
                        <span class="info-value">{'External' if self.css_available else 'Fallback'}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">JS Module:</span>
                        <span class="info-value">{'External' if self.javascript_module_available else 'Inline'}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Error Time:</span>
                        <span class="info-value">{timestamp_obj.strftime('%Y-%m-%d %H:%M:%S')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Fallback Mode:</span>
                        <span class="info-value">Active</span>
                    </div>
                </div>
            </div>
            
            <div class="emergency-actions">
                <h4>Emergency Actions</h4>
                <p style="margin-bottom: 20px; color: #666;">Try these quick fixes to restore dashboard functionality:</p>
                <button class="refresh-btn" onclick="window.location.reload();">
                    🔄 Refresh Page
                </button>
                <button class="refresh-btn fallback-btn" onclick="alert('This is the fallback error page. Check the logs for detailed error information and ensure all template files are present.');">
                    📋 Show Help
                </button>
            </div>
            
            <div class="footer-error">
                <p><strong>Deep Learning System Dashboard</strong> - Template-Based Architecture v3.0</p>
                <p>Error occurred at {timestamp_obj.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>For support, check the system logs and verify template file integrity.</p>
            </div>
        </div>
        
        <script>
            // Enhanced error page functionality
            console.error('Dashboard rendering failed:', {error_message_js});
            
            // Auto-refresh functionality
            let refreshTimer;
            function startAutoRefresh() {{
                refreshTimer = setTimeout(() => {{
                    if (confirm('Auto-refresh in 30 seconds. Continue?')) {{
                        window.location.reload();
                    }} else {{
                        clearTimeout(refreshTimer);
                    }}
                }}, 30000);
            }}
            
            // Keyboard shortcuts
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'F5' || (event.ctrlKey && event.key === 'r')) {{
                    event.preventDefault();
                    window.location.reload();
                }} else if (event.key === 'F1' || event.key === 'h') {{
                    alert('Dashboard Error Help:\\n\\n' +
                        '1. Check template files exist\\n' +
                        '2. Verify file permissions\\n' +
                        '3. Review system logs\\n' +
                        '4. Restart the application\\n\\n' +
                        'Press F5 to refresh the page.');
                }}
            }});
            
            // Start auto-refresh timer
            startAutoRefresh();
            
            // Log system information
            console.info('System Information:', {{
                userAgent: navigator.userAgent,
                timestamp: new Date().toISOString(),
                cssAvailable: {str(self.css_available).lower()},
                jsModuleAvailable: {str(self.javascript_module_available).lower()},
                templateDir: '{template_dir_escaped}'
            }});
        </script>
    </body>
    </html>"""