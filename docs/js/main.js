// Mobile Navigation Toggle
document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');

    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
        });

        // Close menu when clicking on a link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                navMenu.classList.remove('active');
            });
        });
    }

    // Search functionality
    initializeSearch();

    // Smooth scrolling for anchor links
    initializeSmoothScrolling();

    // Intersection Observer for animations
    initializeAnimations();

    // Module filtering
    initializeModuleFiltering();
});

// Search functionality
function initializeSearch() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    const searchBtn = document.querySelector('.search-btn');

    if (!searchInput || !searchResults) return;

    // Sample search data - in a real implementation, this would come from a backend or static JSON
    const searchData = [
        {
            title: 'Transformer 架構',
            description: '深入理解 Transformer 的自注意力機制和架構設計',
            url: 'modules/fundamentals.html#transformer',
            module: 'LLM 基礎理論',
            tags: ['transformer', 'attention', 'architecture']
        },
        {
            title: 'LoRA 低秩適應',
            description: '參數高效微調技術，大幅減少訓練參數',
            url: 'modules/training.html#lora',
            module: '核心訓練技術',
            tags: ['lora', 'peft', 'fine-tuning']
        },
        {
            title: 'vLLM 推理引擎',
            description: '高效的大語言模型推理服務框架',
            url: 'modules/inference.html#vllm',
            module: '高效推理與服務',
            tags: ['vllm', 'inference', 'serving']
        },
        {
            title: 'GPTQ 量化',
            description: '後訓練量化技術，壓縮模型大小',
            url: 'modules/compression.html#gptq',
            module: '模型壓縮',
            tags: ['gptq', 'quantization', 'compression']
        },
        {
            title: 'OpenCompass 評估',
            description: '大語言模型綜合評估平台',
            url: 'modules/evaluation.html#opencompass',
            module: '評估與數據工程',
            tags: ['opencompass', 'evaluation', 'benchmark']
        },
        {
            title: 'DeepSpeed ZeRO',
            description: '分散式訓練記憶體優化技術',
            url: 'modules/training.html#deepspeed',
            module: '核心訓練技術',
            tags: ['deepspeed', 'zero', 'distributed']
        },
        {
            title: 'DPO 直接偏好優化',
            description: '無需獎勵模型的對齊技術',
            url: 'modules/training.html#dpo',
            module: '核心訓練技術',
            tags: ['dpo', 'alignment', 'preference']
        },
        {
            title: '知識蒸餾',
            description: '將大模型知識轉移到小模型',
            url: 'modules/compression.html#distillation',
            module: '模型壓縮',
            tags: ['distillation', 'knowledge', 'transfer']
        },
        {
            title: 'Triton Server',
            description: '企業級模型推理服務平台',
            url: 'modules/inference.html#triton',
            module: '高效推理與服務',
            tags: ['triton', 'serving', 'enterprise']
        },
        {
            title: 'MMLU 評估基準',
            description: '大規模多任務語言理解評估',
            url: 'modules/evaluation.html#mmlu',
            module: '評估與數據工程',
            tags: ['mmlu', 'benchmark', 'evaluation']
        }
    ];

    let searchTimeout;

    searchInput.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        const query = this.value.trim().toLowerCase();

        if (query.length < 2) {
            searchResults.style.display = 'none';
            return;
        }

        searchTimeout = setTimeout(() => {
            performSearch(query, searchData, searchResults);
        }, 300);
    });

    searchBtn.addEventListener('click', function() {
        const query = searchInput.value.trim().toLowerCase();
        if (query.length >= 2) {
            performSearch(query, searchData, searchResults);
        }
    });

    // Close search results when clicking outside
    document.addEventListener('click', function(e) {
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.style.display = 'none';
        }
    });
}

function performSearch(query, data, resultsContainer) {
    const results = data.filter(item => {
        return item.title.toLowerCase().includes(query) ||
               item.description.toLowerCase().includes(query) ||
               item.module.toLowerCase().includes(query) ||
               item.tags.some(tag => tag.toLowerCase().includes(query));
    });

    displaySearchResults(results, resultsContainer);
}

function displaySearchResults(results, container) {
    if (results.length === 0) {
        container.innerHTML = '<div style="padding: 1rem; text-align: center; color: #64748b;">找不到相關內容</div>';
        container.style.display = 'block';
        return;
    }

    const html = results.map(result => `
        <div class="search-result-item" style="padding: 1rem; border-bottom: 1px solid #e2e8f0; cursor: pointer;"
             onclick="window.location.href='${result.url}'">
            <h4 style="margin: 0 0 0.5rem 0; color: #2d3748; font-size: 1rem;">${result.title}</h4>
            <p style="margin: 0 0 0.5rem 0; color: #64748b; font-size: 0.9rem;">${result.description}</p>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="background: #e2e8f0; color: #64748b; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem;">
                    ${result.module}
                </span>
                ${result.tags.slice(0, 2).map(tag =>
                    `<span style="background: #f1f5f9; color: #64748b; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem;">
                        ${tag}
                    </span>`
                ).join('')}
            </div>
        </div>
    `).join('');

    container.innerHTML = html;
    container.style.display = 'block';

    // Add hover effects
    container.querySelectorAll('.search-result-item').forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.style.backgroundColor = '#f8fafc';
        });
        item.addEventListener('mouseleave', function() {
            this.style.backgroundColor = 'white';
        });
    });
}

// Smooth scrolling for anchor links
function initializeSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const offsetTop = target.offsetTop - 100; // Account for fixed navbar
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Intersection Observer for animations
function initializeAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all cards and sections
    document.querySelectorAll('.overview-card, .module-card, .resource-card, .step').forEach(el => {
        observer.observe(el);
    });
}

// Module filtering functionality
function initializeModuleFiltering() {
    // This could be expanded to include filtering by difficulty, technology, etc.
    const moduleCards = document.querySelectorAll('.module-card');

    // Add click tracking for analytics
    moduleCards.forEach(card => {
        card.querySelector('.module-btn')?.addEventListener('click', function(e) {
            const module = card.dataset.module;
            // Here you could send analytics data
            console.log(`Module clicked: ${module}`);
        });
    });
}

// Utility functions
function debounce(func, wait) {
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

// Progress tracking (could be expanded for learning progress)
function trackProgress(section) {
    const progress = JSON.parse(localStorage.getItem('llm-course-progress') || '{}');
    progress[section] = progress[section] || {};
    progress[section].lastVisited = new Date().toISOString();
    progress[section].visits = (progress[section].visits || 0) + 1;
    localStorage.setItem('llm-course-progress', JSON.stringify(progress));
}

// Theme toggle (for future dark mode support)
function initializeThemeToggle() {
    const theme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', theme);

    // Create theme toggle button (could be added to nav)
    const themeToggle = document.createElement('button');
    themeToggle.innerHTML = theme === 'dark' ? '☀️' : '🌙';
    themeToggle.style.cssText = 'background: none; border: none; font-size: 1.2rem; cursor: pointer; margin-left: 1rem;';

    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        themeToggle.innerHTML = newTheme === 'dark' ? '☀️' : '🌙';
    });

    // Optionally add to navigation
    // document.querySelector('.nav-menu')?.appendChild(themeToggle);
}

// Performance monitoring
function initializePerformanceMonitoring() {
    // Monitor page load time
    window.addEventListener('load', () => {
        const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
        console.log(`Page load time: ${loadTime}ms`);

        // Could send to analytics service
    });

    // Monitor Core Web Vitals
    if ('web-vital' in window) {
        // This would require the web-vitals library
        // import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';
    }
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    // Could send to error reporting service
});

// Service Worker registration (for future PWA support)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

// Initialize additional features
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme toggle
    initializeThemeToggle();

    // Initialize performance monitoring
    initializePerformanceMonitoring();

    // Track page visit
    trackProgress(window.location.pathname);
});

// Export functions for potential module use
window.LLMCourse = {
    search: performSearch,
    trackProgress: trackProgress,
    debounce: debounce
};