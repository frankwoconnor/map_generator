// ============================================================================
// Map Generator - Client-Side JavaScript
// ============================================================================

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);

function init() {
    const data = loadInitialData();

    // Initialize components
    initDataSourceToggle();
    initPBFSelection();
    initPBFScanner();
    initRangeSliders();
    initBuildingStyleMode(data);
    loadPalettes(data);
    initPresetManagement();
}

// ============================================================================
// Data Loading
// ============================================================================

function loadInitialData() {
    const dataEl = document.getElementById('initial-data');
    return dataEl ? JSON.parse(dataEl.textContent) : {
        palettes: {},
        buildingCategories: [],
        buildingMode: 'manual',
        autoSizePalette: '',
        autoDistancePalette: ''
    };
}

// ============================================================================
// Data Source Toggle (Remote/Local PBF)
// ============================================================================

function initDataSourceToggle() {
    const remoteRadio = document.getElementById('data_remote');
    const localRadio = document.getElementById('data_local');

    function updateVisibility() {
        const isLocal = localRadio && localRadio.checked;
        document.querySelectorAll('.pbf-only').forEach(el => {
            el.style.display = isLocal ? 'grid' : 'none';
        });
        if (!isLocal) {
            document.querySelectorAll('.pbf-manual').forEach(el => {
                el.style.display = 'none';
            });
        }
    }

    if (remoteRadio) remoteRadio.addEventListener('change', updateVisibility);
    if (localRadio) localRadio.addEventListener('change', updateVisibility);
    updateVisibility();
}

// ============================================================================
// PBF File Selection
// ============================================================================

function initPBFSelection() {
    const pbfDropdown = document.getElementById('location_pbf_file_selection');
    const manualPathGroup = document.querySelector('.pbf-manual');

    if (!pbfDropdown || !manualPathGroup) return;

    function toggleManual() {
        manualPathGroup.style.display =
            pbfDropdown.value === 'manual' ? 'grid' : 'none';
    }

    pbfDropdown.addEventListener('change', toggleManual);
    toggleManual();
}

// ============================================================================
// PBF Folder Scanner
// ============================================================================

function initPBFScanner() {
    const scanBtn = document.getElementById('scan_pbf_folder_btn');
    const folderInput = document.getElementById('location_pbf_folder');
    const pbfSelect = document.getElementById('location_pbf_file_selection');
    const statusEl = document.getElementById('pbf_scan_status');

    if (!scanBtn || !folderInput || !pbfSelect || !statusEl) return;

    scanBtn.addEventListener('click', async () => {
        const folderPath = folderInput.value.trim();
        if (!folderPath) {
            statusEl.textContent = 'Please enter a folder path.';
            statusEl.className = 'field-hint';
            return;
        }

        statusEl.textContent = 'Scanning...';
        statusEl.className = 'field-hint';

        try {
            const response = await fetch('/scan-pbf-folder', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ folder_path: folderPath })
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Unknown error');
            }

            // Clear and repopulate dropdown
            pbfSelect.innerHTML = '<option value="">-- Select a file --</option>';

            if (result.pbf_files && result.pbf_files.length > 0) {
                result.pbf_files.forEach(file => {
                    const opt = document.createElement('option');
                    opt.value = file.path;
                    opt.textContent = file.name;
                    pbfSelect.appendChild(opt);
                });
                statusEl.textContent = `Found ${result.pbf_files.length} PBF file(s)`;
            } else {
                statusEl.textContent = 'No PBF files found';
            }

            // Add manual entry option
            const manualOpt = document.createElement('option');
            manualOpt.value = 'manual';
            manualOpt.textContent = '-- Enter path manually --';
            pbfSelect.appendChild(manualOpt);

        } catch (error) {
            statusEl.textContent = `Error: ${error.message}`;
        }
    });
}

// ============================================================================
// Range Sliders (Opacity/Alpha)
// ============================================================================

function initRangeSliders() {
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        const output = slider.nextElementSibling;
        if (output && output.tagName === 'OUTPUT') {
            slider.addEventListener('input', () => {
                output.textContent = parseFloat(slider.value).toFixed(2);
            });
        }
    });
}

// ============================================================================
// Building Style Mode
// ============================================================================

function initBuildingStyleMode(data) {
    const modeSelect = document.getElementById('building_styling_mode');
    if (!modeSelect) return;

    function updateSections() {
        const mode = modeSelect.value;

        document.querySelectorAll('.color-mode-section').forEach(section => {
            section.style.display = 'none';
        });

        const sectionMap = {
            'manual': 'manual-color-section',
            'auto_distance': 'auto-distance-section',
            'auto_size': 'auto-size-section',
            'manual_floorsize': 'manual-floorsize-section'
        };

        const targetSection = document.getElementById(sectionMap[mode]);
        if (targetSection) {
            targetSection.style.display = 'block';
        }
    }

    modeSelect.addEventListener('change', updateSections);
    updateSections();

    // Initialize categories if in manual_floorsize mode
    if (data.buildingMode === 'manual_floorsize') {
        initBuildingCategories(data.buildingCategories);
    }
}

// ============================================================================
// Building Size Categories
// ============================================================================

let categoryIndex = 0;

function initBuildingCategories(initialCategories) {
    const addBtn = document.getElementById('addCategoryBtn');
    const listEl = document.getElementById('categories-list');

    if (!addBtn || !listEl) return;

    // Add existing categories
    initialCategories.forEach(cat => {
        listEl.appendChild(createCategoryItem(categoryIndex++, cat));
    });

    // Add button handler
    addBtn.addEventListener('click', () => {
        listEl.appendChild(createCategoryItem(categoryIndex++));
    });
}

function createCategoryItem(index, data = {}) {
    const div = document.createElement('div');
    div.className = 'category-item';
    div.dataset.index = index;

    div.innerHTML = `
        <h5>Category ${index + 1}</h5>
        <div class="form-row">
            <div class="form-field">
                <label>Name</label>
                <input type="text" name="buildings_size_category_${index}_name"
                       value="${data.name || ''}">
            </div>
            <div class="form-field">
                <label>Min Area (m²)</label>
                <input type="number" step="0.01"
                       name="buildings_size_category_${index}_min_area"
                       value="${data.min_area !== undefined ? data.min_area : ''}">
            </div>
            <div class="form-field">
                <label>Max Area (m²)</label>
                <input type="number" step="0.01"
                       name="buildings_size_category_${index}_max_area"
                       value="${data.max_area !== undefined ? data.max_area : ''}">
            </div>
            <div class="form-field">
                <label>Color</label>
                <input type="color"
                       name="buildings_size_category_${index}_facecolor"
                       value="${data.facecolor || '#000000'}">
            </div>
        </div>
        <button type="button" class="remove-category-btn">Remove</button>
    `;

    div.querySelector('.remove-category-btn').addEventListener('click', () => {
        div.remove();
    });

    return div;
}

// ============================================================================
// Palette Loading
// ============================================================================

async function loadPalettes(data) {
    try {
        const response = await fetch('/api/palettes');
        if (!response.ok) throw new Error('Failed to load palettes');

        const result = await response.json();
        if (!result.success) throw new Error('Palette API error');

        const palettes = result.palettes;

        // Populate selects
        populatePaletteSelect('auto_size_palette', 'auto_size_palette_preview',
                            palettes, data.autoSizePalette);
        populatePaletteSelect('auto_distance_palette', 'auto_distance_palette_preview',
                            palettes, data.autoDistancePalette);

    } catch (error) {
        console.error('Error loading palettes:', error);
        // Fallback to server-rendered data
        if (data.palettes && Object.keys(data.palettes).length > 0) {
            populatePaletteSelect('auto_size_palette', 'auto_size_palette_preview',
                                data.palettes, data.autoSizePalette);
            populatePaletteSelect('auto_distance_palette', 'auto_distance_palette_preview',
                                data.palettes, data.autoDistancePalette);
        }
    }
}

function populatePaletteSelect(selectId, previewId, palettes, selectedPalette) {
    const select = document.getElementById(selectId);
    const preview = document.getElementById(previewId);

    if (!select) return;

    // Clear and populate
    select.innerHTML = '<option value="">-- Select Palette --</option>';

    Object.keys(palettes).sort().forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        if (name === selectedPalette) opt.selected = true;
        select.appendChild(opt);
    });

    // Render preview
    function renderPreview(paletteName) {
        if (!preview) return;
        preview.innerHTML = '';
        const colors = palettes[paletteName] || [];
        colors.forEach(color => {
            const span = document.createElement('span');
            span.style.backgroundColor = color;
            span.title = color;
            preview.appendChild(span);
        });
    }

    select.addEventListener('change', () => renderPreview(select.value));
    if (selectedPalette) renderPreview(selectedPalette);
}

// ============================================================================
// Preset Management
// ============================================================================

function initPresetManagement() {
    const loadBtn = document.getElementById('load-preset-btn');
    const saveBtn = document.getElementById('save-preset-btn');
    const deleteBtn = document.getElementById('delete-preset-btn');

    if (loadBtn) {
        loadBtn.addEventListener('click', handleLoadPreset);
    }
    if (saveBtn) {
        saveBtn.addEventListener('click', handleSavePreset);
    }
    if (deleteBtn) {
        deleteBtn.addEventListener('click', handleDeletePreset);
    }

    // Initial load of presets
    loadPresetsIntoSelect();
}

async function handleSavePreset() {
    const nameInput = document.getElementById('new-preset-name');
    const presetName = nameInput.value.trim();
    if (!presetName) {
        alert('Please enter a name for the preset.');
        return;
    }

    const style = getStyleFromForm();

    try {
        const response = await fetch('/api/presets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: presetName, style: style })
        });
        const data = await response.json();
        if (data.success) {
            alert(data.message);
            nameInput.value = '';
            window.location.reload(); // Reload to see the new preset
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error('Error saving preset:', error);
        alert('An error occurred while saving the preset.');
    }
}

async function handleLoadPreset() {
    const select = document.getElementById('preset-select');
    const presetName = select.value;
    if (!presetName) {
        alert('Please select a preset to load.');
        return;
    }

    try {
        const response = await fetch(`/api/presets/${presetName}`,
        {
            method: 'PUT'
        });
        const data = await response.json();
        if (data.success) {
            alert(data.message);
            window.location.reload();
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error('Error loading preset:', error);
        alert('An error occurred while loading the preset.');
    }
}

async function handleDeletePreset() {
    const select = document.getElementById('preset-select');
    const presetName = select.value;
    if (!presetName) {
        alert('Please select a preset to delete.');
        return;
    }

    if (!confirm(`Are you sure you want to delete the preset "${presetName}"?`)) {
        return;
    }

    try {
        const response = await fetch(`/api/presets/${presetName}`, {
            method: 'DELETE'
        });
        const data = await response.json();
        if (data.success) {
            alert(data.message);
            window.location.reload(); // Reload to update the list
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error('Error deleting preset:', error);
        alert('An error occurred while deleting the preset.');
    }
}

function getStyleFromForm() {
    const form = document.getElementById('map-form');
    const formData = new FormData(form);
    const data = {};
    for (const [key, value] of formData.entries()) {
        data[key] = value;
    }

    const style = {
        location: {
            query: `${data.location_latitude} ${data.location_longitude}`,
            distance: parseFloat(data.location_distance),
            data_source: data.location_data_source,
            pbf_folder: data.location_pbf_folder,
            pbf_path: data.location_pbf_path,
            bbox: null
        },
        output: {
            filename_prefix: data.filename_prefix,
            separate_layers: formData.has('separate_layers'),
            figure_size: [parseFloat(data.figure_size_width), parseFloat(data.figure_size_height)],
            figure_dpi: parseInt(data.figure_dpi),
            background_color: data.background_color,
            transparent_background: formData.has('transparent_background'),
            margin: parseFloat(data.margin),
            enable_debug_legends: formData.has('enable_debug_legends'),
            preview_type: 'embedded'
        },
        layers: {}
    };

    const layerNames = ['streets', 'water', 'green', 'buildings'];
    layerNames.forEach(layerName => {
        if (formData.has(`${layerName}_enabled`)) {
            style.layers[layerName] = {
                enabled: true,
                facecolor: data[`${layerName}_facecolor`],
                edgecolor: data[`${layerName}_edgecolor`],
                linewidth: parseFloat(data[`${layerName}_linewidth`]),
                alpha: parseFloat(data[`${layerName}_alpha`]),
                zorder: parseInt(data[`${layerName}_zorder`]),
                hatch: data[`${layerName}_hatch`] === 'null' ? null : data[`${layerName}_hatch`],
                simplify_tolerance: parseFloat(data[`${layerName}_simplify_tolerance`]),
                min_size_threshold: parseFloat(data[`${layerName}_min_size_threshold`]),
                filters: {}
            };
        }
    });

    // Handle complex building settings
    if (style.layers.buildings) {
        style.layers.buildings.auto_style_mode = data.building_styling_mode;
        style.layers.buildings.manual_color_settings = {
            facecolor: data.buildings_manual_color_facecolor
        };
        style.layers.buildings.auto_size_palette = data.auto_size_palette;
        style.layers.buildings.auto_distance_palette = data.auto_distance_palette;
        style.layers.buildings.size_categories = [];
        
        const categoryIndexes = new Set();
        for (const key of formData.keys()) {
            if (key.startsWith('buildings_size_category_')) {
                const parts = key.split('_');
                categoryIndexes.add(parts[3]);
            }
        }

        categoryIndexes.forEach(index => {
            style.layers.buildings.size_categories.push({
                name: data[`buildings_size_category_${index}_name`],
                min_area: parseFloat(data[`buildings_size_category_${index}_min_area`]),
                max_area: parseFloat(data[`buildings_size_category_${index}_max_area`]),
                facecolor: data[`buildings_size_category_${index}_facecolor`]
            });
        });
    }

    return style;
}
