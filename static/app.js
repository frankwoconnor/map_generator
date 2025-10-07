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
