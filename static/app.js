const socket = io("https://logus2k.com", {
    path: "/cxray/socket.io/"
});

let currentConfig = null;
let currentDimension = null;
let currentPage = 0;
let totalImages = 0;
let filteredImages = 0;
let isLoadingConfig = false;

const IMAGES_PER_PAGE = 18;

// DU metric ranges and filters
let metricRanges = {};  // Stores min/max/outlier data from server
let duFilters = {};     // Active filters: {metric: [min, max]}
let duSlidersInitialized = false;

socket.on('connect', () => { console.log('Connected'); showStatus('Connected to server', false); loadAvailableConfigs(); });

socket.on('disconnect', () => showStatus('Disconnected', true));

socket.on('available_configs', (configs) => { populateConfigList(configs); });

socket.on('config_loaded', (c) => {
    console.log('=== CONFIG_LOADED EVENT ===');
    console.log('Received config with dimensions:', Object.keys(c.dimensions || {}).length);
    currentConfig = c;
    currentDimension = null;
    isLoadingConfig = false;
    const dataset = document.getElementById('dataset').value;
    populateDimensionList(c, dataset);
    showStatus('Config loaded', false);
    console.log('=== CONFIG_LOADED COMPLETE ===');
});

socket.on('config_saved', () => showStatus('Config saved!', false));

socket.on('metric_ranges', (data) => {
    console.log('Received metric ranges:', data);
    metricRanges = data.ranges;
    
    // Initialize DU sliders with received ranges
    if (!duSlidersInitialized) {
        createDUSliders();
        duSlidersInitialized = true;
    }
});

socket.on('images_loaded', (d) => {
    console.log('images_loaded event:', d);
    totalImages = d.total_available || 0;
    filteredImages = d.filtered_count || totalImages;
    
    document.getElementById('info-box').style.display = 'block';
    
    // Show filtered count if different from total
    let countText = `<b>${totalImages}</b> image(s)`;
    if (filteredImages < totalImages) {
        countText = `<b>${totalImages}</b> image(s) (<b>${filteredImages}</b> match filters)`;
    }
    
    document.getElementById('dimension-info').innerHTML = 
        `Dataset: <b>${d.dataset.toUpperCase()}</b>&nbsp; | &nbsp;` +
        `Dimension: <b>${d.dimension}</b>&nbsp; | &nbsp;` +
        `Total: ${countText}`;
    
    ensureGridExists();
    updatePaginationButtons();
    
    window.expectedImageCount = d.count;
    window.receivedImageCount = 0;
});

socket.on('image_data', (d) => {
    const index = d.index;
    
    // Render image directly
    displayImage(index, d.data, d.filename, d.du || {});
    
    window.receivedImageCount++;
    
    if (window.receivedImageCount === window.expectedImageCount) {
        // Mark empty cells
        for (let i = window.expectedImageCount; i < IMAGES_PER_PAGE; i++) {
            const cell = document.getElementById(`cell-${i}`);
            if (cell) {
                const placeholder = cell.querySelector('.placeholder');
                if (placeholder && placeholder.textContent === 'Loading...') {
                    placeholder.textContent = 'Empty';
                    placeholder.style.color = '#ccc';
                }
            }
        }
    }
});

socket.on('error', (d) => showStatus(`Error: ${d.message}`, true));


const duTooltip = document.getElementById("du-tooltip");

// Listen for hover on image cells
document.addEventListener("mousemove", (event) => {
    const cell = event.target.closest(".image-cell");

    if (!cell || !cell.dataset.du) {
        duTooltip.style.display = "none";
        return;
    }

    // Parse DU metrics
    let metrics = {};
    try {
        metrics = JSON.parse(cell.dataset.du);
    } catch (e) {
        duTooltip.style.display = "none";
        return;
    }

    // Build tooltip text (compact formatting)
    let text = "";
    for (const [key, value] of Object.entries(metrics)) {
        text += `${key}: ${value}\n`;
    }

    duTooltip.textContent = text.trim();
    duTooltip.style.display = "block";

    // Position tooltip near cursor
    duTooltip.style.left = event.pageX + 15 + "px";
    duTooltip.style.top = event.pageY + 15 + "px";
});

// Hide tooltip on mouse leave
document.addEventListener("mouseleave", () => {
    duTooltip.style.display = "none";
});


function showStatus(m, e = false) { const s = document.getElementById('status'); s.textContent = m; s.className = 'status' + (e ? ' error' : ''); s.style.display = 'block'; setTimeout(() => s.style.display = 'none', 3000); }

function loadAvailableConfigs() {
    socket.emit('get_available_configs');
}

function populateConfigList(configs) {
    const s = document.getElementById('config-file');
    s.innerHTML = '';

    if (configs && configs.length > 0) {
        configs.forEach(filename => {
            const o = document.createElement('option');
            o.value = filename;
            o.textContent = filename;
            s.appendChild(o);
        });

        s.value = configs[0];
        loadSelectedConfig();
    } else {
        s.innerHTML = '<option value="">No configs found</option>';
    }
}

function loadSelectedConfig() {
    const configFile = document.getElementById('config-file').value;
    console.log('=== SWITCHING CONFIG ===');
    console.log('New config file:', configFile);
    console.log('Old currentDimension:', currentDimension);
    if (configFile) {
        isLoadingConfig = true;
        currentDimension = null;
        currentPage = 0;
        totalImages = 0;

        console.log('Cleared currentDimension:', currentDimension);

        const grid = document.getElementById('image-grid');
        grid.innerHTML = '';

        document.getElementById('info-box').style.display = 'none';

        console.log('Emitting load_config for:', configFile);
        socket.emit('load_config', configFile);
    }
}

function populateDimensionList(c, dataset = 'train') {
    const s = document.getElementById('dimension');
    s.innerHTML = '';

    if (c && c.dimensions) {
        const dimensions = Object.keys(c.dimensions);
        const validDimensions = [];

        dimensions.forEach(d => {
            const dimConfig = c.dimensions[d];
            const allPaths = dimConfig.image_paths || [];

            const datasetDir = dataset === 'train' ? 'chest_xray/train' : 'chest_xray/test';
            const datasetPaths = allPaths.filter(p => p.includes(datasetDir));
            const count = datasetPaths.length;

            if (count > 0) {
                const o = document.createElement('option');
                o.value = d;
                o.textContent = `${d} (${count})`;
                s.appendChild(o);
                validDimensions.push(d);
            }
        });

        if (validDimensions.length > 0) {
            s.value = validDimensions[0];
            currentDimension = validDimensions[0];
            onDimensionChange();
        } else {
            showStatus('No dimensions found for this dataset', true);
        }
    }
}

function loadConfig() { socket.emit('load_config'); }

function loadDimensionParams(d) {
    console.log('loadDimensionParams called for dimension:', d);
    if (!currentConfig || !currentConfig.dimensions || !currentConfig.dimensions[d]) {
        console.log('Dimension not found in config, returning');
        return;
    }
    const p = currentConfig.dimensions[d];
    console.log('Loading params from config:', p);

    const hCrop = document.getElementById('h_crop');
    const vCrop = document.getElementById('v_crop');
    const clahe = document.getElementById('clahe');
    const zoomIn = document.getElementById('zoom_in');

    const oldHOnchange = hCrop.onchange;
    const oldVOnchange = vCrop.onchange;
    const oldClaheOnchange = clahe.onchange;
    const oldZoomOnchange = zoomIn.onchange;

    hCrop.onchange = null;
    vCrop.onchange = null;
    clahe.onchange = null;
    zoomIn.onchange = null;

    hCrop.value = p.h_crop || 0;
    vCrop.value = p.v_crop || 0;
    clahe.value = p.clahe || 0;
    zoomIn.value = p.zoom_in || 0;

    console.log('Updated input values to:', { h_crop: hCrop.value, v_crop: vCrop.value, clahe: clahe.value, zoom_in: zoomIn.value });

    hCrop.onchange = oldHOnchange;
    vCrop.onchange = oldVOnchange;
    clahe.onchange = oldClaheOnchange;
    zoomIn.onchange = oldZoomOnchange;
}

function onDimensionChange() {
    console.log('onDimensionChange called');
    const d = document.getElementById('dimension').value;
    console.log('Selected dimension:', d);
    
    if (d && currentConfig && currentConfig.dimensions && currentConfig.dimensions[d]) {
        currentDimension = d;
        currentPage = 0;
        duFilters = {};
        duSlidersInitialized = false;
        
        console.log('About to load dimension params');
        loadDimensionParams(d);
        
        // Request metric ranges from server
        const configFile = document.getElementById('config-file').value;
        const dataset = document.getElementById('dataset').value;
        
        console.log('Requesting metric ranges for dimension:', d);
        socket.emit('get_metric_ranges', {
            config_file: configFile,
            dimension: d,
            dataset: dataset
        });
        
        console.log('About to ensure grid and load images');
        ensureGridExists();
        loadImagesPage();
    } else if (d) {
        console.warn('Dimension not found in current config:', d);
    }
}

function loadImages() {
    if (!currentConfig) {
        showStatus('Config not loaded yet', true);
        return;
    }
    const dataset = document.getElementById('dataset').value;
    populateDimensionList(currentConfig, dataset);
}

function ensureGridExists() {
    const g = document.getElementById('image-grid');
    if (g.children.length === 0) {
        for (let i = 0; i < 18; i++) {
            const c = document.createElement('div');
            c.className = 'image-cell';
            c.id = `cell-${i}`;
            c.innerHTML = '<div class="filename"></div><div class="image-wrapper"><span class="placeholder">Empty</span></div>';
            g.appendChild(c);
        }
    }
}

function loadImagesPage() {
    console.log('loadImagesPage called - isLoadingConfig:', isLoadingConfig, 'currentDimension:', currentDimension);

    if (isLoadingConfig) {
        console.log('Config loading in progress, skipping image load');
        return;
    }

    if (!currentDimension) {
        console.log('No currentDimension set, returning');
        return;
    }

    if (!currentConfig || !currentConfig.dimensions || !currentConfig.dimensions[currentDimension]) {
        console.error('Current dimension not found in config:', currentDimension);
        console.error('Available dimensions:', currentConfig ? Object.keys(currentConfig.dimensions || {}) : 'no config');
        showStatus('Invalid dimension selected', true);
        return;
    }

    console.log('Loading images for dimension:', currentDimension);

    ensureGridExists();

    for (let i = 0; i < IMAGES_PER_PAGE; i++) {
        const cell = document.getElementById(`cell-${i}`);
        if (cell) {
            cell.innerHTML = '<div class="filename"></div><div class="image-wrapper"><span class="placeholder">Loading...</span></div>';
        }
    }

    const p = getCurrentParams();
    const classFilter = document.getElementById('class-filter').value;
    const configFile = document.getElementById('config-file').value;
    
    console.log('Requesting images: page', currentPage, 'offset', currentPage * IMAGES_PER_PAGE, 'params', p, 'class', classFilter, 'du_filters', duFilters);
    
    socket.emit('get_dimension_images', {
        config_file: configFile,
        dimension: currentDimension,
        dataset: document.getElementById('dataset').value,
        class_filter: classFilter,
        params: p,
        du_filters: duFilters,
        offset: currentPage * IMAGES_PER_PAGE,
        limit: IMAGES_PER_PAGE
    });
    
    document.getElementById('loading').style.display = 'block';
}

function getCurrentParams() { return { h_crop: parseFloat(document.getElementById('h_crop').value) || 0, v_crop: parseFloat(document.getElementById('v_crop').value) || 0, clahe: parseFloat(document.getElementById('clahe').value) || 0, zoom_in: parseFloat(document.getElementById('zoom_in').value) || 0 }; }

function updatePreview() { if (!currentDimension) { showStatus('Load images first', true); return; } loadImagesPage(); }

function displayImage(i, d, filename, du = {}) {

    const c = document.getElementById(`cell-${i}`);

    if (!c) {
        console.log('Cell not found:', `cell-${i}`);
        return;
    }

    const blob = new Blob([d], { type: 'image/png' });
    const url = URL.createObjectURL(blob);

    const filenameDiv = c.querySelector('.filename');
    const imageWrapper = c.querySelector('.image-wrapper');

    // Update filename
    if (filenameDiv && filename) {
        filenameDiv.textContent = filename;
    }

    // Update image element
    if (imageWrapper) {
        let img = imageWrapper.querySelector('img');
        if (img) {
            img.src = url;
        } else {
            imageWrapper.innerHTML = `<img src="${url}">`;
        }
    }

    // Attach DU metrics to DOM element for later use
    c.dataset.du = JSON.stringify(du);

    document.getElementById('loading').style.display = 'none';
}

function saveToConfig() {
    const d = document.getElementById('dimension').value;
    const configFile = document.getElementById('config-file').value;
    if (!d || !currentConfig) { showStatus('Select dimension', true); return; }
    if (!configFile) { showStatus('No config file selected', true); return; }
    const p = getCurrentParams();
    currentConfig.dimensions[d] = { ...currentConfig.dimensions[d], ...p };
    socket.emit('save_config_data', { filename: configFile, config: currentConfig });
}

function updatePaginationButtons() {
    const tp = Math.ceil(filteredImages / IMAGES_PER_PAGE) || 1;
    console.log('updatePaginationButtons: currentPage=', currentPage, 'filteredImages=', filteredImages, 'totalPages=', tp);
    document.getElementById('prev-btn').disabled = currentPage === 0;
    document.getElementById('next-btn').disabled = currentPage >= tp - 1 || tp <= 1;
    document.getElementById('page-input').value = currentPage + 1;
    document.getElementById('page-input').max = tp;
    document.getElementById('total-pages').textContent = tp;
}

function previousPage() { 
    console.log('previousPage clicked'); 
    if (currentPage > 0) { 
        currentPage--; 
        loadImagesPage(); 
    } 
}

function nextPage() { 
    console.log('nextPage clicked'); 
    const tp = Math.ceil(filteredImages / IMAGES_PER_PAGE); 
    if (currentPage < tp - 1) { 
        currentPage++; 
        loadImagesPage(); 
    } 
}

function goToPage() {
    const pageInput = document.getElementById('page-input');
    const targetPage = parseInt(pageInput.value) - 1;
    const tp = Math.ceil(filteredImages / IMAGES_PER_PAGE);

    if (isNaN(targetPage) || targetPage < 0 || targetPage >= tp) {
        pageInput.value = currentPage + 1;
        return;
    }

    currentPage = targetPage;
    loadImagesPage();
}

// Create DU Filter Sliders with dual-knob range controls
function createDUSliders() {
    const panel = document.getElementById("du-slider-panel");
    panel.innerHTML = "";

    if (!metricRanges || Object.keys(metricRanges).length === 0) {
        console.log('No metric ranges available yet');
        return;
    }

    Object.entries(metricRanges).forEach(([metricKey, config]) => {
        const container = document.createElement("div");
        container.className = "du-slider-container";
        container.style.width = "160px";

        const label = document.createElement("div");
        label.className = "du-slider-label";
        label.textContent = config.label;
        label.style.marginBottom = "8px";

        const sliderDiv = document.createElement("div");
        sliderDiv.className = "du-slider";
        sliderDiv.id = `slider-${metricKey}`;

        const valueDisplay = document.createElement("div");
        valueDisplay.style.fontSize = "11px";
        valueDisplay.style.color = "#333";
        valueDisplay.style.marginTop = "4px";
        valueDisplay.style.textAlign = "center";
        valueDisplay.textContent = `${config.min.toFixed(config.precision)} - ${config.max.toFixed(config.precision)}`;

        container.appendChild(label);
        container.appendChild(sliderDiv);
        container.appendChild(valueDisplay);
        panel.appendChild(container);

        // Initialize noUiSlider
        noUiSlider.create(sliderDiv, {
            start: [config.min, config.max],
            connect: true,
            range: {
                'min': config.min,
                'max': config.max
            },
            step: (config.max - config.min) / 1000,
            format: {
                to: (value) => value.toFixed(config.precision),
                from: (value) => parseFloat(value)
            }
        });

        // Add outlier zone visual indicators
        const sliderBase = sliderDiv.querySelector('.noUi-base');
        if (sliderBase) {
            // Calculate outlier zone positions as percentages
            const range = config.max - config.min;
            const lowZoneEnd = Math.max(0, ((config.outlier_low - config.min) / range) * 100);
            const highZoneStart = Math.min(100, ((config.outlier_high - config.min) / range) * 100);

            // Left outlier zone
            if (lowZoneEnd > 0) {
                const leftZone = document.createElement('div');
                leftZone.style.position = 'absolute';
                leftZone.style.left = '0';
                leftZone.style.width = `${lowZoneEnd}%`;
                leftZone.style.height = '100%';
                leftZone.style.background = 'rgba(255, 0, 0, 0.45)';
                leftZone.style.pointerEvents = 'none';
                leftZone.style.zIndex = '0';
                sliderBase.appendChild(leftZone);
            }

            // Right outlier zone
            if (highZoneStart < 100) {
                const rightZone = document.createElement('div');
                rightZone.style.position = 'absolute';
                rightZone.style.left = `${highZoneStart}%`;
                rightZone.style.width = `${100 - highZoneStart}%`;
                rightZone.style.height = '100%';
                rightZone.style.background = 'rgba(255, 0, 0, 0.45)';
                rightZone.style.pointerEvents = 'none';
                rightZone.style.zIndex = '0';
                sliderBase.appendChild(rightZone);
            }
        }

        // Handle slider changes
        sliderDiv.noUiSlider.on('change', (values) => {
            const minVal = parseFloat(values[0]);
            const maxVal = parseFloat(values[1]);
            
            valueDisplay.textContent = `${minVal.toFixed(config.precision)} - ${maxVal.toFixed(config.precision)}`;

            // Update filters
            if (minVal === config.min && maxVal === config.max) {
                // Reset to full range - remove filter
                delete duFilters[metricKey];
            } else {
                duFilters[metricKey] = [minVal, maxVal];
            }

            // Reset to first page and reload
            currentPage = 0;
            loadImagesPage();
        });
    });
}

// Config loaded automatically on connect
window.onload = () => { };
