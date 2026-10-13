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
    initStreetStyleMode(data);
    initPaletteGallery(data);
    loadPalettes(data);
    initPresetGallery();
    initLayerChips();
    initRunGallery();
    initDirtyState();
    initAsyncGeneration();
    initRegionPicker();
    initDataPanel();
    initMapPicker();
}

// ============================================================================
// Map picker (Phase C4) - Leaflet via CDN with an offline fallback
// ============================================================================

function initMapPicker() {
    const wrap = document.getElementById('map-canvas-wrap');
    const canvas = document.getElementById('map-canvas');
    const fallback = document.getElementById('map-fallback');
    const status = document.getElementById('map-status');
    if (!wrap || !canvas || !fallback) return;

    const cssEl = document.createElement('link');
    cssEl.rel = 'stylesheet';
    cssEl.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
    document.head.appendChild(cssEl);

    const scriptEl = document.createElement('script');
    scriptEl.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
    let done = false;
    const fail = () => {
        if (done) return;
        done = true;
        wrap.hidden = true;
        fallback.hidden = false;
        if (status) status.textContent = 'offline';
    };
    scriptEl.onload = () => {
        if (done) return;
        try {
            wrap.hidden = false;
            setupLeafletMap(canvas);
            if (status) status.textContent = 'active';
            done = true;
        } catch (err) {
            fail();
        }
    };
    scriptEl.onerror = fail;
    setTimeout(fail, 6000);
    document.head.appendChild(scriptEl);
}

function setupLeafletMap(canvas) {
    const readNum = (id, fb) => {
        const v = parseFloat((document.getElementById(id) || {}).value);
        return Number.isFinite(v) ? v : fb;
    };
    let lat = readNum('location_latitude', 53.3498);
    let lon = readNum('location_longitude', -6.2603);
    let dist = Math.min(Math.max(readNum('location_distance', 2000), 200), 20000);

    const map = L.map(canvas, { center: [lat, lon], zoom: 12 });
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).addTo(map);

    const marker = L.marker([lat, lon], { draggable: true }).addTo(map);
    const circle = L.circle([lat, lon], {
        radius: dist, color: '#6f5bff', weight: 1, fillOpacity: 0.12,
    }).addTo(map);
    const handle = L.marker([lat + dist / 111320, lon], {
        icon: L.divIcon({
            className: 'map-radius-handle',
            html: '<span>⟲</span>',
            iconSize: [18, 18],
            iconAnchor: [9, 9],
        }),
        draggable: true,
        keyboard: false,
    }).addTo(map);

    const setInput = (id, value) => {
        const el = document.getElementById(id);
        if (el && String(el.value) !== String(value)) {
            el.value = value;
            markFormDirty();
        }
    };
    const updateRadiusControls = (m) => {
        const slider = document.getElementById('map_radius');
        const out = document.getElementById('map_radius_out');
        const distInput = document.getElementById('location_distance');
        if (slider) slider.value = m;
        if (out) out.textContent = m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${m} m`;
        if (distInput) setInput('location_distance', m);
    };
    const setCentre = (latlng, opts) => {
        marker.setLatLng(latlng);
        circle.setLatLng(latlng);
        handle.setLatLng(L.latLng(latlng.lat + dist / 111320, latlng.lng));
        if (!opts || !opts.silent) {
            setInput('location_latitude', latlng.lat.toFixed(5));
            setInput('location_longitude', latlng.lng.toFixed(5));
        }
    };
    const setRadius = (m) => {
        dist = Math.min(Math.max(Math.round(m), 200), 20000);
        updateRadiusControls(dist);
        circle.setRadius(dist);
        const c = marker.getLatLng();
        handle.setLatLng(L.latLng(c.lat + dist / 111320, c.lng));
    };

    map.on('click', (e) => { setCentre(e.latlng); markFormDirty(); });
    marker.on('dragend', () => { setCentre(marker.getLatLng()); markFormDirty(); });
    handle.on('drag', () => {
        const m = haversineMetres(marker.getLatLng(), handle.getLatLng());
        dist = Math.min(Math.max(Math.round(m), 200), 20000);
        circle.setRadius(dist);
        updateRadiusControls(dist);
    });

    const slider = document.getElementById('map_radius');
    if (slider) slider.addEventListener('input', () => setRadius(parseFloat(slider.value)));

    const syncCentreFromForm = () => {
        const a = parseFloat(document.getElementById('location_latitude').value);
        const b = parseFloat(document.getElementById('location_longitude').value);
        if (Number.isFinite(a) && Number.isFinite(b)) setCentre(L.latLng(a, b), { silent: true });
    };
    const syncDistanceFromForm = () => {
        const v = parseFloat(document.getElementById('location_distance').value);
        if (Number.isFinite(v)) setRadius(v);
    };
    document.getElementById('location_latitude').addEventListener('change', syncCentreFromForm);
    document.getElementById('location_longitude').addEventListener('change', syncCentreFromForm);
    document.getElementById('location_distance').addEventListener('change', syncDistanceFromForm);

    window.mapPicker = {
        ready: true,
        invalidate() {
            setTimeout(() => map.invalidateSize(), 0);
        },
        syncFromForm() {
            syncCentreFromForm();
            syncDistanceFromForm();
        },
    };

    // The section starts collapsed (display:none); Leaflet measures the canvas
    // as 0x0 then, so resize it whenever the section opens.
    const section = document.getElementById('map-picker-section');
    if (section) {
        section.addEventListener('toggle', () => {
            if (section.open) window.mapPicker.invalidate();
        });
    }
    map.invalidateSize();
}

function haversineMetres(a, b) {
    const R = 6371000;
    const dLat = (b.lat - a.lat) * Math.PI / 180;
    const dLon = (b.lng - a.lng) * Math.PI / 180;
    const s = Math.sin(dLat / 2) ** 2
        + Math.cos(a.lat * Math.PI / 180) * Math.cos(b.lat * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(s), Math.sqrt(1 - s));
}

function markFormDirty() {
    const indicator = document.getElementById('status-indicator');
    if (indicator) {
        indicator.textContent = 'Unsaved changes';
        indicator.classList.add('dirty');
    }
}

// ============================================================================
// Region quick-pick (fills location settings from the catalog)
// ============================================================================

function initRegionPicker() {
    const pick = document.getElementById('region_pick');
    if (!pick) return;
    const hint = document.getElementById('region-info');

    pick.addEventListener('change', () => {
        const opt = pick.options[pick.selectedIndex];
        if (opt && opt.value) applyRegionOption(opt);
    });
    initRegionExtractor();
}

function applyRegionOption(opt) {
    const hint = document.getElementById('region-info');
    if (!opt || !opt.value) return;

    const localRadio = document.querySelector('#data_local');
    if (localRadio) localRadio.checked = true;
    const set = (id, value) => {
        const el = document.getElementById(id);
        if (el && value !== undefined && value !== '') el.value = value;
    };
    set('location_pbf_path', opt.dataset.path);
    set('location_latitude', opt.dataset.lat);
    set('location_longitude', opt.dataset.lon);
    set('location_distance', opt.dataset.radius);

    const indicator = document.getElementById('status-indicator');
    if (indicator) {
        indicator.textContent = 'Unsaved changes';
        indicator.classList.add('dirty');
    }
    if (hint) {
        hint.textContent = `${opt.dataset.path} · centre ${opt.dataset.lat}, ${opt.dataset.lon}`;
    }

    if (window.mapPicker && window.mapPicker.ready) window.mapPicker.syncFromForm();

    fetch(`/api/pbf/info?path=${encodeURIComponent(opt.dataset.path)}`)
        .then((resp) => (resp.ok ? resp.json() : null))
        .then((data) => {
            if (!data || !data.success || !hint) return;
            const i = data.info;
            let extra = `${i.size_mb.toFixed(1)} MB · `;
            if (typeof i.nodes === 'number') {
                extra += `${formatCount(i.nodes)} nodes / ${formatCount(i.ways)} ways / ${formatCount(i.relations)} relations`;
            }
            if (i.bbox) {
                extra += ` · bbox ${i.bbox.map((v) => (+v).toFixed(3)).join(', ')}`;
            }
            hint.textContent = `${String(i.path).split('/').pop()} · ${extra}`;
        })
        .catch(() => { /* keep the static hint */ });
}

// ============================================================================
// New region extraction (C3) - clip from a source as a background job
// ============================================================================

function initRegionExtractor() {
    const toggle = document.getElementById('xr_toggle');
    const form = document.getElementById('xr_form');
    const sourceEl = document.getElementById('xr_source');
    const goBtn = document.getElementById('xr_go');
    if (!form || !goBtn) return;

    const showForm = (visible) => { form.hidden = !visible; };
    if (toggle) toggle.addEventListener('click', () => showForm(form.hidden));

    // Fill source options from /api/sources
    fetch('/api/sources')
        .then((resp) => (resp.ok ? resp.json() : null))
        .then((data) => {
            if (!data || !data.success) {
                sourceEl.innerHTML = '';
                const opt = new Option('No source PBFs found', '');
                sourceEl.add(opt);
                return;
            }
            sourceEl.innerHTML = '';
            if (!data.sources.length) {
                sourceEl.add(new Option('No source PBFs found', ''));
                return;
            }
            const preferred = '/data/sources/ireland';
            let selected = false;
            data.sources.forEach((s) => {
                const opt = new Option(`${s.name} (${s.size_mb.toFixed(1)} MB)`, s.path);
                if (!selected && s.path.includes(preferred)) {
                    opt.selected = true;
                    selected = true;
                }
                if (data.sources.length === 1) opt.selected = true;
                sourceEl.add(opt);
            });
            updateBboxPreview();
        })
        .catch(() => {
            sourceEl.innerHTML = '';
            sourceEl.add(new Option('Could not load sources', ''));
        });

    const fillBtn = document.getElementById('xr_fill');
    if (fillBtn) {
        fillBtn.addEventListener('click', () => {
            const grab = (id) => (document.getElementById(id) || {}).value || '';
            document.getElementById('xr_lat').value = grab('location_latitude');
            document.getElementById('xr_lon').value = grab('location_longitude');
            document.getElementById('xr_dist').value = grab('location_distance') || '4000';
            updateBboxPreview();
        });
    }

    ['xr_lat', 'xr_lon', 'xr_dist'].forEach((id) => {
        (document.getElementById(id) || { addEventListener: () => {} })
            .addEventListener('input', updateBboxPreview);
    });

    goBtn.addEventListener('click', async () => {
        const name = (document.getElementById('xr_name') || {}).value || '';
        const slug = slugifyName(name);
        const source = (sourceEl.options[sourceEl.selectedIndex] || {}).value || '';
        const lat = parseFloat((document.getElementById('xr_lat') || {}).value);
        const lon = parseFloat((document.getElementById('xr_lon') || {}).value);
        const distM = parseFloat((document.getElementById('xr_dist') || {}).value);

        const errors = [];
        if (!name) errors.push('A region name is required.');
        else if (!/^[a-zA-Z0-9_\- ]+$/.test(name)) {
            errors.push('Name may contain only letters, numbers, spaces, dashes, underscores.');
        }
        if (!source) errors.push('A source PBF is required.');
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
            errors.push('Latitude and longitude are required.');
        } else if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
            errors.push('Coordinates out of range.');
        }
        if (!Number.isFinite(distM) || distM < 50 || distM > 100000) {
            errors.push('Half-width (m) must be between 50 and 100000.');
        }
        if (errors.length) { showInlineErrors(errors); return; }
        clearAlerts();

        goBtn.disabled = true;
        const fd = new FormData();
        fd.append('source', source);
        fd.append('name', name.trim());
        fd.append('lat', String(lat));
        fd.append('lon', String(lon));
        fd.append('dist_m', String(distM));
        try {
            const resp = await fetch('/api/regions/extract', { method: 'POST', body: fd });
            const data = await resp.json();
            if (!resp.ok || !data.success) {
                showInlineErrors(data.errors || [data.error || 'Extraction failed to start.']);
                goBtn.disabled = false;
                return;
            }
            showJobPanel(`Extracting “${name.trim()}”…`);
            setStatus('Extracting…');
            await trackJob(data.job_id, (msg) => handleExtractionTerminal(msg, data.slug));
        } catch (err) {
            showInlineErrors([String(err)]);
            goBtn.disabled = false;
        }
    });
}

function updateBboxPreview() {
    const lat = parseFloat((document.getElementById('xr_lat') || {}).value);
    const lon = parseFloat((document.getElementById('xr_lon') || {}).value);
    const distM = parseFloat((document.getElementById('xr_dist') || {}).value);
    const out = document.getElementById('xr_bbox');
    if (!out) return;
    if (![lat, lon, distM].every(Number.isFinite) || distM <= 0) {
        out.textContent = '';
        return;
    }
    const M = 111320;
    const dLon = distM / (M * Math.cos(lat * Math.PI / 180));
    const dLat = distM / M;
    out.textContent = `Clipping bbox: ${(lon - dLon).toFixed(4)}..${(lon + dLon).toFixed(4)} ` +
        `(lon), ${(lat - dLat).toFixed(4)}..${(lat + dLat).toFixed(4)} (lat)`;
}

function slugifyName(name) {
    return name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'region';
}

async function handleExtractionTerminal(msg, slug) {
    const goBtn = document.getElementById('xr_go');
    if (goBtn) goBtn.disabled = false;
    setJobStatusLabel(msg.status);
    if (msg.status === 'done') {
        appendJobLine('✓ Region extracted.');
        setStatus('Region ready');
        await refreshRegionPickOptions(slug);
    } else {
        setStatus(msg.status === 'error' ? 'Error' : 'Cancelled');
        if (msg.error) showJobError(msg.error);
    }
}

async function refreshRegionPickOptions(selectSlug) {
    try {
        const resp = await fetch('/api/regions');
        const data = await resp.json();
        if (!resp.ok || !data.success) return;
        rebuildRegionOptions(data.regions || [], selectSlug);
    } catch (err) { /* non-fatal */ }
}

function rebuildRegionOptions(regions, selectSlug) {
    const pick = document.getElementById('region_pick');
    if (!pick) return;
    while (pick.options.length > 1) pick.remove(1); // keep the placeholder
    const count = document.getElementById('region-count');
    if (count) count.textContent = `${regions.length} region${regions.length === 1 ? '' : 's'}`;

    let want = null;
    regions.forEach((r) => {
        const opt = document.createElement('option');
        opt.value = r.slug;
        opt.dataset.path = r.form_path || '';
        if (r.bbox && r.bbox.length === 4) {
            const [w, s, e, n] = r.bbox;
            opt.dataset.lat = ((s + n) / 2).toFixed(5);
            opt.dataset.lon = ((w + e) / 2).toFixed(5);
            opt.dataset.radius = Math.round((n - s) / 2 * 111320);
        }
        const size = r.size_mb != null ? ` · ${r.size_mb.toFixed(1)} MB` : '';
        opt.textContent = `${r.name}${size}`;
        if (r.slug === selectSlug) want = opt;
        pick.appendChild(opt);
    });

    if (want) {
        pick.value = selectSlug;
        applyRegionOption(want);
        const xr = document.getElementById('xr_form');
        if (xr) xr.hidden = true;
        const log = getJobLog();
        if (log) log.scrollTop = log.scrollHeight;
    }
}

// ============================================================================
// Data & sources panel
// ============================================================================

async function initDataPanel() {
    const rows = document.getElementById('data-rows');
    if (!rows) return;

    const makeRow = (label, value, extra) => {
        const div = document.createElement('div');
        div.className = 'data-row';
        const l = document.createElement('span');
        l.className = 'data-label';
        l.textContent = label;
        const v = document.createElement('span');
        v.className = 'data-value';
        v.textContent = value;
        div.append(l, v);
        if (extra) {
            const x = document.createElement('div');
            x.className = 'data-extra';
            x.textContent = extra;
            div.appendChild(x);
        }
        return div;
    };
    const fmtBytes = (b) => (b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB`
        : b >= 1024 ? `${(b / 1024).toFixed(0)} KB` : `${b} B`);

    try {
        const [envRes, srcRes] = await Promise.all([fetch('/api/env'), fetch('/api/sources')]);
        const env = (await envRes.json()).env;
        const sources = (await srcRes.json()).sources || [];
        if (!env) throw new Error('env missing');
        const free = env.free_space_bytes >= 1073741824
            ? `${(env.free_space_bytes / 1073741824).toFixed(1)} GB`
            : `${(env.free_space_bytes / 1048576).toFixed(0)} MB`;
        const d = env.dirs || {};
        rows.innerHTML = '';
        rows.append(
            makeRow('Data root', env.data_root),
            makeRow('Free space', free, env.osmium_available ? 'osmium available ✔' : '⚠ osmium not installed'),
            makeRow('Sources', `${d.sources ? d.sources.files : 0} file(s) · ${fmtBytes(d.sources ? d.sources.size_bytes : 0)}`),
            makeRow('Regions', `${d.regions ? d.regions.files : 0} file(s) · ${fmtBytes(d.regions ? d.regions.size_bytes : 0)}`),
            makeRow('Output', `${d.output ? d.output.files : 0} file(s) · ${fmtBytes(d.output ? d.output.size_bytes : 0)}`)
        );
        if (sources.length) {
            const head = document.createElement('div');
            head.className = 'data-subhead';
            head.textContent = 'Sources';
            rows.appendChild(head);
            sources.forEach((s) => rows.appendChild(makeRow(s.name, s.size_mb.toFixed(1) + ' MB')));
        }
    } catch (err) {
        rows.innerHTML = '';
        rows.appendChild(makeRow('Data', 'unavailable', String(err)));
    }
}

function formatCount(n) {
    return new Intl.NumberFormat().format(n);
}

// ============================================================================
// Run Gallery (click a past run to preview it without reloading)
// ============================================================================

function initRunGallery() {
    const grid = document.getElementById('run-grid');
    if (!grid) return;

    grid.addEventListener('click', (event) => {
        const card = event.target.closest('.run-card');
        if (!card) return;
        const svg = card.dataset.combinedSvg;
        if (!svg) return; // nothing to preview for this run
        event.preventDefault();

        grid.querySelectorAll('.run-card.active').forEach(c => c.classList.remove('active'));
        card.classList.add('active');

        let preview = document.getElementById('preview-image');
        const empty = document.getElementById('preview-empty');
        if (!preview) {
            preview = document.createElement('img');
            preview.id = 'preview-image';
            preview.className = 'preview-image';
            preview.alt = 'Generated map';
            if (empty) {
                empty.replaceWith(preview);
            } else {
                card.closest('.section-content').prepend(preview);
            }
        }
        preview.src = svg;

        const dlSvg = document.getElementById('download-svg');
        const dlPng = document.getElementById('download-png');
        if (dlSvg) dlSvg.href = svg;
        if (dlPng) {
            const png = card.dataset.combinedPng;
            dlPng.href = png || '#';
            dlPng.style.display = png ? '' : 'none';
        }

        const meta = document.querySelector('.preview-meta');
        if (meta) meta.textContent = card.dataset.meta || '';
    });
}

// ============================================================================
// Dirty State (reflect unsaved edits in the status chip)
// ============================================================================

function initDirtyState() {
    const form = document.getElementById('map-form');
    const indicator = document.getElementById('status-indicator');
    if (!form || !indicator) return;

    const markDirty = () => {
        indicator.textContent = 'Unsaved changes';
        indicator.classList.add('dirty');
    };
    form.addEventListener('input', markDirty);
    form.addEventListener('change', markDirty);
    form.addEventListener('submit', () => {
        indicator.textContent = 'Generating…';
        indicator.classList.remove('dirty');
    });
}

// ============================================================================
// Async Generation (never block the page - SSE progress + in-place result)
// ============================================================================

function initAsyncGeneration() {
    const form = document.getElementById('map-form');
    const cancelBtn = document.getElementById('job-cancel');
    if (!form) return;
    let currentJobId = null;

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const button = document.getElementById('generate-btn');

        const errors = validateLocationInputs(form);
        if (errors.length) {
            showInlineErrors(errors);
            setStatus('Check inputs', 'dirty');
            return;
        }
        clearAlerts();
        if (button) button.disabled = true;
        showJobPanel('Queued…');
        setStatus('Starting…');

        try {
            const resp = await fetch('/api/generate', {
                method: 'POST',
                body: new FormData(form),
            });
            const data = await resp.json();
            if (!resp.ok || !data.success) {
                showInlineErrors(data.errors || [data.error || 'Failed to start generation.']);
                setStatus('Error');
                if (button) button.disabled = false;
                return;
            }
            currentJobId = data.job_id;
            setStatus('Running');
            await trackJob(data.job_id);
        } catch (err) {
            showInlineErrors([String(err)]);
            setStatus('Error');
            if (button) button.disabled = false;
        }
    });

    if (cancelBtn) {
        cancelBtn.addEventListener('click', async () => {
            if (!currentJobId) return;
            cancelBtn.disabled = true;
            setStatus('Cancelling…');
            try {
                await fetch(`/api/jobs/${currentJobId}/cancel`, { method: 'POST' });
            } catch (err) { /* server will notice the job end either way */ }
        });
    }

    const refreshBtn = document.getElementById('job-refresh');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', async () => {
            if (!currentJobId) return;
            refreshBtn.disabled = true;
            refreshBtn.textContent = '⟳ Checking…';
            try {
                // Fetch latest job status
                const resp = await fetch(`/api/jobs/${currentJobId}`);
                const data = await resp.json();
                if (resp.ok && data.success) {
                    const job = data.job;
                    setJobStatusLabel(job.status);
                    if (job.status === 'done') {
                        await refreshResultAndGallery(true);
                        refreshBtn.hidden = true;
                    } else if (job.status === 'error' || job.status === 'cancelled') {
                        refreshBtn.textContent = '⟳ Refresh results';
                        refreshBtn.disabled = false;
                    } else {
                        // Still running
                        refreshBtn.textContent = '⟳ Refresh results';
                        refreshBtn.disabled = false;
                    }
                }
            } catch (err) {
                showAlert('Refresh failed: ' + err.message);
                refreshBtn.textContent = '⟳ Refresh results';
                refreshBtn.disabled = false;
            }
        });
    }
}

function validateLocationInputs(form) {
    const errs = [];
    const lat = parseFloat((form.querySelector('[name="location_latitude"]') || {}).value);
    const lon = parseFloat((form.querySelector('[name="location_longitude"]') || {}).value);
    const latRaw = ((form.querySelector('[name="location_latitude"]') || {}).value || '').trim();
    const lonRaw = ((form.querySelector('[name="location_longitude"]') || {}).value || '').trim();

    if (!latRaw || !lonRaw || !Number.isFinite(lat) || !Number.isFinite(lon)) {
        errs.push('Latitude and longitude (numeric) are required.');
    } else {
        if (lat < -90 || lat > 90) errs.push('Latitude must be between -90 and 90.');
        if (lon < -180 || lon > 180) errs.push('Longitude must be between -180 and 180.');
    }
    const dist = parseFloat((form.querySelector('[name="location_distance"]') || {}).value || '');
    if (Number.isFinite(dist) && (dist < 10 || dist > 20000)) {
        errs.push('Distance must be between 10 and 20000 metres.');
    }
    const source = (form.querySelector('[name="location_data_source"]:checked') || {}).value;
    const pbfPath = ((form.querySelector('[name="location_pbf_path"]') || {}).value || '').trim();
    if (source === 'local' && !pbfPath) {
        errs.push('A PBF file path is required when data source is "Local PBF files".');
    }
    return errs;
}

async function trackJob(jobId, handleTerminal) {
    const terminalHandler = handleTerminal || handleJobTerminal;
    const log = getJobLog();
    let lastSeq = 0;
    let pollTimer = null;
    let sseClosed = false;
    let terminalCalled = false;

    const callTerminal = (msg) => {
        if (terminalCalled) return;
        terminalCalled = true;
        if (pollTimer) clearInterval(pollTimer);
        terminalHandler(msg);
    };

    // Initial status fetch
    try {
        const resp = await fetch(`/api/jobs/${jobId}`);
        const data = await resp.json();
        if (resp.ok && data.success) {
            const job = data.job;
            setJobStatusLabel(job.status);
            log.textContent = job.lines.join('\n');
            lastSeq = job.total_lines || 0;
            if (job.error) showJobError(job.error);
            if (job.status !== 'running' && job.status !== 'queued') {
                // Already finished before we subscribed.
                callTerminal({ job_id: jobId, status: job.status, error: job.error });
                return;
            }
        }
    } catch (err) { /* fall through to SSE */ }

    // SSE connection with fallback polling
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    let sseConnected = false;

    es.addEventListener('open', () => {
        sseConnected = true;
        // If we already have a poll timer, clear it since SSE is working
        if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
    });

    es.addEventListener('message', (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'line' && msg.seq > lastSeq) {
            appendJobLine(msg.text);
            lastSeq = msg.seq;
        }
    });

    es.addEventListener('terminal', (event) => {
        sseClosed = true;
        es.close();
        callTerminal(JSON.parse(event.data));
    });

    // SSE error/close handling - start polling fallback
    es.onerror = () => {
        if (sseClosed) return;
        sseClosed = true;
        es.close();
        // Start polling fallback if not already terminal
        if (!terminalCalled) {
            startPollingFallback();
        }
    };

    function startPollingFallback() {
        if (pollTimer) return;
        pollTimer = setInterval(async () => {
            if (terminalCalled) {
                clearInterval(pollTimer);
                pollTimer = null;
                return;
            }
            try {
                const resp = await fetch(`/api/jobs/${jobId}`);
                const data = await resp.json();
                if (resp.ok && data.success) {
                    const job = data.job;
                    setJobStatusLabel(job.status);
                    // Append any new log lines
                    if (job.lines && job.total_lines > lastSeq) {
                        const newLines = job.lines.slice(-(job.total_lines - lastSeq));
                        newLines.forEach(line => appendJobLine(line));
                        lastSeq = job.total_lines;
                    }
                    if (job.error && !log.textContent.includes(job.error)) {
                        showJobError(job.error);
                    }
                    if (job.status !== 'running' && job.status !== 'queued') {
                        callTerminal({ job_id: jobId, status: job.status, error: job.error });
                    }
                }
            } catch (err) {
                // Polling failed, will retry
            }
        }, 2000); // Poll every 2 seconds
    }

    // Also start a delayed polling as safety net in case SSE silently fails
    setTimeout(() => {
        if (!sseConnected && !terminalCalled && !pollTimer) {
            startPollingFallback();
        }
    }, 5000);
}

async function handleJobTerminal(msg) {
    msg = msg || {};
    const status = msg.status || 'error';
    setJobStatusLabel(status);
    if (status === 'done') {
        setStatus('Done');
        appendJobLine('✓ Finished.');
        // If the result panel could not exist yet (first run on this page),
        // fall back to a reload so the server renders the full result view.
        if (!document.getElementById('download-svg')) {
            window.location.reload();
            return;
        }
        await refreshResultAndGallery();
        // Show refresh button in case the user wants to manually refresh
        const refreshBtn = document.getElementById('job-refresh');
        if (refreshBtn) {
            refreshBtn.hidden = false;
        }
    } else {
        setStatus(status.charAt(0).toUpperCase() + status.slice(1));
        if (msg.error) showJobError(msg.error);
    }
    const button = document.getElementById('generate-btn');
    if (button) button.disabled = false;
    const cancelBtn = document.getElementById('job-cancel');
    if (cancelBtn) cancelBtn.disabled = false;
}

async function refreshResultAndGallery(showErrors = false) {
    try {
        const resp = await fetch('/api/runs?limit=12');
        const data = await resp.json();
        if (!resp.ok || !data.success || !data.runs || !data.runs.length) {
            if (showErrors) showAlert('No runs found after generation.');
            return;
        }
        rebuildRunGrid(data.runs);
        applyRunToPreview(data.runs[0]);
    } catch (err) {
        if (showErrors) showAlert('Failed to refresh results: ' + err.message);
    }
}

function rebuildRunGrid(runs) {
    const grid = document.getElementById('run-grid');
    if (!grid) return;
    grid.innerHTML = '';
    const fragment = document.createDocumentFragment();
    runs.forEach((run, i) => fragment.appendChild(buildRunCard(run, i === 0)));
    grid.appendChild(fragment);
}

function buildRunCard(run, active) {
    const card = document.createElement('a');
    card.className = 'run-card' + (active ? ' active' : '');
    card.href = '#';
    card.dataset.combinedSvg = run.combined_svg ? `/output/${run.combined_svg}` : '';
    card.dataset.combinedPng = run.combined_png ? `/output/${run.combined_png}` : '';
    card.dataset.meta = `${run.region || run.prefix} · ${run.timestamp_display}`;

    const thumb = run.thumbnail
        ? (() => { const img = document.createElement('img');
                   img.className = 'run-thumb'; img.src = `/output/${run.thumbnail}`;
                   img.alt = run.id; img.loading = 'lazy'; return img; })()
        : (() => { const div = document.createElement('div');
                   div.className = 'run-thumb run-thumb-empty'; div.textContent = '—'; return div; })();
    card.appendChild(thumb);

    const meta = document.createElement('div');
    meta.className = 'run-card-meta';
    const title = document.createElement('span');
    title.className = 'run-card-title';
    title.textContent = run.region || run.prefix;
    const time = document.createElement('span');
    time.className = 'run-card-time';
    time.textContent = run.timestamp_display;
    meta.append(title, time);
    card.appendChild(meta);
    return card;
}

function applyRunToPreview(run) {
    if (!run || !run.combined_svg) return;
    let preview = document.getElementById('preview-image');
    const empty = document.getElementById('preview-empty');
    if (!preview) {
        preview = document.createElement('img');
        preview.id = 'preview-image';
        preview.className = 'preview-image';
        preview.alt = 'Generated map';
        if (empty) {
            empty.replaceWith(preview);
        } else {
            const content = document.querySelector('.preview-panel .section-content');
            if (content) content.prepend(preview);
        }
    }
    preview.src = `/output/${run.combined_svg}`;

    const dlSvg = document.getElementById('download-svg');
    const dlPng = document.getElementById('download-png');
    if (dlSvg) dlSvg.href = `/output/${run.combined_svg}`;
    if (dlPng) {
        if (run.combined_png) {
            dlPng.href = `/output/${run.combined_png}`;
            dlPng.style.display = '';
        } else {
            dlPng.href = '#';
            dlPng.style.display = 'none';
        }
    }
    const meta = document.querySelector('.preview-meta');
    if (meta) meta.textContent = `${run.region || run.prefix} · ${run.timestamp_display}`;
}

// -- small DOM helpers -------------------------------------------------------

function getJobLog() {
    const log = document.getElementById('job-log');
    if (!log) {
        const panel = document.getElementById('job-panel');
        const el = document.createElement('pre');
        el.id = 'job-log';
        el.className = 'job-log';
        if (panel) panel.querySelector('.section-content').prepend(el);
        return el;
    }
    return log;
}

function showJobPanel(title) {
    const panel = document.getElementById('job-panel');
    if (!panel) return;
    panel.hidden = false;
    const t = document.getElementById('job-title');
    if (t) t.textContent = title || 'Generating…';
    getJobLog().textContent = '';
    const cancelBtn = document.getElementById('job-cancel');
    if (cancelBtn) cancelBtn.disabled = false;
}

function setJobStatusLabel(status) {
    const label = document.getElementById('job-status-label');
    if (!label) return;
    label.textContent = status;
    label.className = `badge job-status-${status}`;
}

function appendJobLine(text) {
    const log = getJobLog();
    const parts = log.textContent.split('\n');
    parts.push(text);
    if (parts.length > 300) parts.splice(0, parts.length - 300);
    log.textContent = parts.join('\n');
    log.scrollTop = log.scrollHeight;
}

function showJobError(text) {
    appendJobLine('');
    appendJobLine('⚠ ' + text);
}

function setStatus(text, extraClass) {
    const indicator = document.getElementById('status-indicator');
    if (!indicator) return;
    indicator.textContent = text;
    indicator.className = 'status-indicator';
    if (extraClass) indicator.classList.add(extraClass);
}

function showInlineErrors(errors) {
    const box = document.getElementById('client-alerts');
    if (!box) return;
    box.innerHTML = '';
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'
    }[c]));
    errors.forEach((e) => {
        const div = document.createElement('div');
        div.className = 'alert alert-error';
        div.innerHTML = `<strong>Error:</strong> ${esc(e)}`;
        box.appendChild(div);
    });
    box.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function clearAlerts() {
    const box = document.getElementById('client-alerts');
    if (box) box.innerHTML = '';
}

function showAlert(message, type = 'error') {
    const box = document.getElementById('client-alerts');
    if (!box) return;
    const div = document.createElement('div');
    div.className = `alert alert-${type}`;
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'
    }[c]));
    div.innerHTML = `<strong>${esc(type)}:</strong> ${esc(message)}`;
    box.appendChild(div);
    box.scrollIntoView({ behavior: 'smooth', block: 'start' });
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

    // Ensure categories are initialized once, regardless of initial mode
    // This will set up the add button listener and render initial categories
    initBuildingCategories(data.buildingCategories);
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
// Street Style Mode (By Road Class / Uniform)
// ============================================================================

function initStreetStyleMode(data) {
    const modeSelect = document.getElementById('streets_style_mode');
    if (!modeSelect) return;

    const byClassSection = document.getElementById('streets-by-class-section');
    const paletteSelect = document.getElementById('streets_street_palette');

    function updateVisibility() {
        const byClass = modeSelect.value === 'by_class';
        if (byClassSection) byClassSection.style.display = byClass ? 'block' : 'none';
    }

    // Picking a palette re-fills every class color picker with that theme
    // (darkest on Motorways; classes beyond the palette keep the lightest
    // color). The individual pickers can then be tweaked as overrides.
    function assignPaletteToClasses() {
        if (!paletteSelect) return;
        const colors = (cachedPalettes[paletteSelect.value] || []).slice().reverse();
        if (!colors.length) return;
        document.querySelectorAll('[data-street-class]').forEach((row, idx) => {
            const input = document.getElementById('streets_class_color_' + row.dataset.streetClass);
            if (input) {
                input.value = colors[Math.min(idx, colors.length - 1)];
            }
        });
    }

    modeSelect.addEventListener('change', updateVisibility);
    if (paletteSelect) paletteSelect.addEventListener('change', assignPaletteToClasses);
    updateVisibility();
}

// ============================================================================
// Palette Loading
// ============================================================================

let cachedPalettes = {};

async function loadPalettes(data) {
    try {
        const response = await fetch('/api/palettes');
        if (!response.ok) throw new Error('Failed to load palettes');

        const result = await response.json();
        if (!result.success) throw new Error('Palette API error');

        cachedPalettes = result.palettes;
        repopulatePaletteSelects(result.palettes, data);
        renderPaletteGallery(result.palettes);

    } catch (error) {
        console.error('Error loading palettes:', error);
        // Fallback to server-rendered data
        if (data.palettes && Object.keys(data.palettes).length > 0) {
            cachedPalettes = data.palettes;
            repopulatePaletteSelects(data.palettes, data);
            renderPaletteGallery(data.palettes);
        }
    }
}

// Re-fetch palettes from the API and refresh the selects + gallery.
// Used after create / edit / rename / delete so a change is visible everywhere.
async function refreshPalettes() {
    const response = await fetch('/api/palettes');
    if (!response.ok) throw new Error('Failed to load palettes');
    const result = await response.json();
    if (!result.success) throw new Error('Palette API error');
    cachedPalettes = result.palettes;
    repopulatePaletteSelects(result.palettes);
    renderPaletteGallery(result.palettes);
}

// Populate the three palette selects. Keeps whatever the user has currently
// selected, falling back to the server-rendered selection on first load.
function repopulatePaletteSelects(palettes, defaults = {}) {
    const targets = [
        ['auto_size_palette', defaults.autoSizePalette || ''],
        ['auto_distance_palette', defaults.autoDistancePalette || ''],
        ['streets_street_palette', defaults.streetPalette || ''],
    ];
    targets.forEach(([selectId, fallback]) => {
        const select = document.getElementById(selectId);
        const current = select && select.value ? select.value : fallback;
        const previewId = selectId.replace('_palette', '_palette_preview');
        populatePaletteSelect(selectId, previewId, palettes, current);
    });
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

    if (select._palettePreviewHandler) {
        select.removeEventListener('change', select._palettePreviewHandler);
    }
    const handler = () => renderPreview(select.value);
    select._palettePreviewHandler = handler;
    select.addEventListener('change', handler);
    if (selectedPalette) renderPreview(selectedPalette);
}

// ============================================================================
// Palette Gallery & Editor
// ============================================================================

const PALETTE_SELECT_IDS = ['streets_street_palette', 'auto_size_palette', 'auto_distance_palette'];

function initPaletteGallery(data) {
    const gallery = document.getElementById('palette-gallery');
    const createBtn = document.getElementById('create-palette-btn');
    if (!createBtn) return;

    renderColorEditorRow('new-palette-colors-row', ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']);

    createBtn.addEventListener('click', () => {
        const name = document.getElementById('new-palette-name').value.trim();
        const colors = readColorEditorRow('new-palette-colors-row');
        if (!name) return showAlert('Please enter a palette name.');
        if (colors.length < 2) return showAlert('A palette needs at least two colors.');
        if (cachedPalettes[name] && !confirm(`Palette "${name}" already exists. Overwrite it?`)) return;
        apiSavePalette(name, colors).then(res => {
            if (!res) return;
            document.getElementById('new-palette-name').value = '';
            refreshPalettes().catch(err => console.error('Error refreshing palettes:', err));
        });
    });

    if (gallery) gallery.addEventListener('click', onGalleryClick);

    if (data.palettes && Object.keys(data.palettes).length) {
        renderPaletteGallery(data.palettes);
    }
}

function renderPaletteGallery(palettes) {
    const gallery = document.getElementById('palette-gallery');
    if (!gallery) return;

    const applied = appliedPaletteNames(); // { paletteName: [layerLabels...] }

    gallery.innerHTML = '';
    Object.keys(palettes).sort().forEach(name => {
        const colors = palettes[name] || [];
        const card = document.createElement('div');
        card.className = 'palette-card';
        card.dataset.paletteName = name;

        const header = document.createElement('div');
        header.className = 'palette-card-header';

        const title = document.createElement('span');
        title.className = 'palette-card-name';
        title.textContent = name;
        header.appendChild(title);

        // Show "in use by" hint
        const inUseBy = applied[name];
        if (inUseBy && inUseBy.length > 0) {
            const usageSpan = document.createElement('span');
            usageSpan.className = 'badge palette-inuse';
            usageSpan.textContent = 'in use: ' + inUseBy.join(', ');
            usageSpan.style.fontSize = '10px';
            usageSpan.style.padding = '2px 6px';
            header.appendChild(usageSpan);
        }

        const actions = document.createElement('div');
        actions.className = 'palette-card-actions';
        actions.appendChild(makeGalleryButton('Streets', 'apply', 'Apply to streets (By Road Class mode)'));
        actions.appendChild(makeGalleryButton('Buildings', 'applyb', 'Apply to buildings colour modes'));
        actions.appendChild(makeGalleryButton('Edit', 'edit', 'Edit colors'));
        actions.appendChild(makeGalleryButton('Rename', 'rename', 'Rename palette'));
        actions.appendChild(makeGalleryButton('Delete', 'delete', 'Delete palette'));
        header.appendChild(actions);
        card.appendChild(header);

        const strip = document.createElement('div');
        strip.className = 'palette-preview';
        colors.forEach(color => {
            const span = document.createElement('span');
            span.style.backgroundColor = color;
            span.title = color;
            strip.appendChild(span);
        });
        card.appendChild(strip);

        const editor = document.createElement('div');
        editor.className = 'palette-card-editor';

        const nameInput = document.createElement('input');
        nameInput.type = 'text';
        nameInput.className = 'palette-editor-name';
        nameInput.value = name;

        const colorRow = document.createElement('div');
        colorRow.className = 'palette-editor-colors';
        colors.forEach(color => colorRow.appendChild(createColorInput(color)));
        colorRow.appendChild(createAddColorButton());

        const saveBtn = makeGalleryButton('Save', 'save', 'Save changes');
        const cancelBtn = makeGalleryButton('Cancel', 'cancel', 'Discard changes');
        editor.append(nameInput, colorRow, saveBtn, cancelBtn);
        card.appendChild(editor);

        gallery.appendChild(card);
    });
}

function makeGalleryButton(label, action, title) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-secondary btn-sm';
    btn.dataset.action = action;
    btn.textContent = label;
    btn.title = title || label;
    return btn;
}

function makeInUseBadge() {
    const badge = document.createElement('span');
    badge.className = 'badge palette-inuse';
    badge.textContent = 'in use';
    return badge;
}

function onGalleryClick(event) {
    const btn = event.target.closest('button[data-action]');
    if (!btn) return;
    const card = btn.closest('.palette-card');
    if (!card) return;
    const name = card.dataset.paletteName;
    const action = btn.dataset.action;

    if (action === 'apply') {
        applyPaletteToStreets(name);
        updateInUseBadges();
    } else if (action === 'applyb') {
        applyPaletteToBuildings(name);
        updateInUseBadges();
    } else if (action === 'edit' || action === 'rename') {
        showCardEditor(card, action === 'rename');
    } else if (action === 'cancel') {
        hideCardEditor(card);
    } else if (action === 'save') {
        savePalette(card, name);
    } else if (action === 'delete') {
        deletePalette(name);
    }
}

function showCardEditor(card, selectName) {
    const editor = card.querySelector('.palette-card-editor');
    if (!editor) return;
    editor.style.display = 'block';
    if (selectName) {
        const nameInput = editor.querySelector('.palette-editor-name');
        nameInput.focus();
        nameInput.select();
    }
}

function hideCardEditor(card) {
    const editor = card.querySelector('.palette-card-editor');
    if (editor) editor.style.display = 'none';
}

function savePalette(card, originalName) {
    const editor = card.querySelector('.palette-card-editor');
    const nameInput = editor.querySelector('.palette-editor-name');
    const newName = nameInput.value.trim();
    const colors = readColorEditorRowFrom(editor.querySelector('.palette-editor-colors'));
    if (!newName) return showAlert('Please enter a palette name.');
    if (colors.length < 2) return showAlert('A palette needs at least two colors.');
    if (newName !== originalName && cachedPalettes[newName]) {
        if (!confirm(`Palette "${newName}" already exists. Overwrite it?`)) return;
    }
    apiSavePalette(newName, colors).then(res => {
        if (!res) return;
        if (newName !== originalName) {
            apiDeletePalette(originalName).then(() => refreshPalettes().catch(console.error));
        } else {
            refreshPalettes().catch(console.error);
        }
    });
}

function deletePalette(name) {
    if (!confirm(`Delete palette "${name}"?\n\nMaps already generated with these colors are unaffected.`)) return;
    apiDeletePalette(name).then(res => {
        if (res) refreshPalettes().catch(console.error);
    });
}

async function apiSavePalette(name, colors) {
    try {
        const response = await fetch('/api/palettes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, colors })
        });
        const result = await response.json();
        if (!result.success) throw new Error(result.error || 'Failed to save palette');
        return result;
    } catch (error) {
        showAlert(error.message);
        return null;
    }
}

async function apiDeletePalette(name) {
    try {
        const response = await fetch(`/api/palettes/${encodeURIComponent(name)}`, { method: 'DELETE' });
        const result = await response.json();
        if (!result.success) throw new Error(result.error || 'Failed to delete palette');
        return result;
    } catch (error) {
        showAlert(error.message);
        return null;
    }
}

// --- Apply actions ----------------------------------------------------------

function applyPaletteToStreets(name) {
    const select = document.getElementById('streets_street_palette');
    if (select) {
        select.value = name;
        // Re-fills the class-color pickers from the theme + updates the preview.
        select.dispatchEvent(new Event('change'));
    }
    const mode = document.getElementById('streets_style_mode');
    if (mode && mode.value !== 'by_class') {
        mode.value = 'by_class';
        mode.dispatchEvent(new Event('change')); // reveal the class controls
    }
}

function applyPaletteToBuildings(name) {
    ['auto_size_palette', 'auto_distance_palette'].forEach(id => {
        const select = document.getElementById(id);
        if (select) {
            select.value = name;
            select.dispatchEvent(new Event('change'));
        }
    });
}

function appliedPaletteNames() {
    // Returns object: { paletteName: ['streets', 'buildings-size', 'buildings-distance'] }
    const usage = {};
    const selectMap = {
        'streets_street_palette': 'streets (by class)',
        'auto_size_palette': 'buildings (by size)',
        'auto_distance_palette': 'buildings (by distance)',
    };
    Object.entries(selectMap).forEach(([id, label]) => {
        const select = document.getElementById(id);
        if (select && select.value) {
            if (!usage[select.value]) usage[select.value] = [];
            usage[select.value].push(label);
        }
    });
    return usage;
}

// Lightweight badge refresh after Apply (keeps any open editors intact).
function updateInUseBadges() {
    const applied = appliedPaletteNames(); // { paletteName: [layerLabels...] }
    document.querySelectorAll('#palette-gallery .palette-card').forEach(card => {
        const name = card.dataset.paletteName;
        const inUseBy = applied[name];
        // Update or create the usage badge
        let usageBadge = card.querySelector('.palette-inuse');
        if (inUseBy && inUseBy.length > 0) {
            if (!usageBadge) {
                usageBadge = document.createElement('span');
                usageBadge.className = 'badge palette-inuse';
                usageBadge.style.fontSize = '10px';
                usageBadge.style.padding = '2px 6px';
                card.querySelector('.palette-card-header').appendChild(usageBadge);
            }
            usageBadge.textContent = 'in use: ' + inUseBy.join(', ');
        } else if (usageBadge) {
            usageBadge.remove();
        }
    });
}

// --- Color-row helpers ------------------------------------------------------

function createColorInput(value) {
    const input = document.createElement('input');
    input.type = 'color';
    input.className = 'palette-color-input';
    input.value = value;
    return input;
}

function createAddColorButton() {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-secondary btn-sm palette-add-color';
    btn.textContent = '+';
    btn.title = 'Add color';
    btn.addEventListener('click', () => {
        btn.parentElement.insertBefore(createColorInput('#969696'), btn);
    });
    return btn;
}

function renderColorEditorRow(rowId, colors) {
    const row = document.getElementById(rowId);
    if (!row) return;
    row.innerHTML = '';
    colors.forEach(color => row.appendChild(createColorInput(color)));
    row.appendChild(createAddColorButton());
}

function readColorEditorRow(rowId) {
    const row = document.getElementById(rowId);
    return row ? readColorEditorRowFrom(row) : [];
}

function readColorEditorRowFrom(element) {
    return [...element.querySelectorAll('input[type="color"]')].map(input => input.value);
}

// ============================================================================
// Preset Gallery (D2) - cards with swatches, in-place load, "Surprise me"
// ============================================================================

let cachedPresetCards = [];

function initPresetGallery() {
    const gallery = document.getElementById('preset-gallery');
    const saveBtn = document.getElementById('save-preset-btn');
    const surpriseBtn = document.getElementById('surprise-btn');

    if (saveBtn) saveBtn.addEventListener('click', handleSavePreset);
    if (surpriseBtn) surpriseBtn.addEventListener('click', surpriseMe);
    if (gallery) gallery.addEventListener('click', onPresetGalleryClick);

    initCreativeControls();
    loadPresetCards().catch(err => console.error('Error loading presets:', err));
}

async function loadPresetCards() {
    const response = await fetch('/api/presets');
    const data = await response.json();
    cachedPresetCards = data.success ? (data.presets || []) : [];
    renderPresetGallery(cachedPresetCards);
}

function renderPresetGallery(cards) {
    const gallery = document.getElementById('preset-gallery');
    if (!gallery) return;
    gallery.innerHTML = '';
    if (!cards || !cards.length) {
        const empty = document.createElement('p');
        empty.className = 'preset-empty';
        empty.textContent = 'No presets yet — tune the style and press Save above.';
        gallery.appendChild(empty);
        return;
    }
    cards.forEach(card => gallery.appendChild(buildPresetCard(card)));
}

function layerLabel(name) {
    return { streets: 'Streets', buildings: 'Buildings', water: 'Water', green: 'Green' }[name] || name;
}

function buildPresetCard(card) {
    const el = document.createElement('div');
    el.className = 'preset-card';
    el.dataset.presetName = card.name;

    const header = document.createElement('div');
    header.className = 'preset-card-header';

    const name = document.createElement('span');
    name.className = 'preset-card-name';
    name.textContent = card.name;
    header.appendChild(name);

    const actions = document.createElement('div');
    actions.className = 'preset-card-actions';
    actions.appendChild(makePresetBtn('Load', 'btn-secondary', 'load', 'Apply this preset to the form'));
    actions.appendChild(makePresetBtn('Delete', 'btn-danger', 'delete', 'Delete this preset'));
    header.appendChild(actions);
    el.appendChild(header);

    if (card.colors && card.colors.length) {
        const sw = document.createElement('div');
        sw.className = 'preset-card-swatches';
        card.colors.forEach(color => {
            const span = document.createElement('span');
            span.style.backgroundColor = color;
            span.title = color;
            sw.appendChild(span);
        });
        el.appendChild(sw);
    }

    const meta = document.createElement('div');
    meta.className = 'preset-card-meta';
    const bits = [];
    if (card.region) bits.push(`📍 ${card.region.trim()}`);
    if (card.distance) bits.push(`${(card.distance / 1000).toFixed(1)} km`);
    if (card.enabled && card.enabled.length) bits.push(card.enabled.map(layerLabel).join(' · '));
    if (card.palettes && card.palettes.length) bits.push(`🎨 ${card.palettes.join(', ')}`);
    (bits.length ? bits : ['empty preset']).forEach(text => {
        const b = document.createElement('span');
        b.className = 'preset-card-badge';
        b.textContent = text;
        meta.appendChild(b);
    });
    el.appendChild(meta);
    return el;
}

function makePresetBtn(label, styleClass, action, title) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `btn ${styleClass} btn-sm`;
    btn.dataset.action = action;
    btn.textContent = label;
    btn.title = title;
    return btn;
}

function onPresetGalleryClick(event) {
    const btn = event.target.closest('button[data-action]');
    if (!btn) return;
    const card = btn.closest('.preset-card');
    if (!card) return;
    const name = card.dataset.presetName;
    const action = btn.dataset.action;
    if (action === 'load') loadPresetIntoForm(name);
    else if (action === 'delete') deletePresetCard(name);
}

async function loadPresetIntoForm(name) {
    try {
        const response = await fetch(`/api/presets/${encodeURIComponent(name)}`, { method: 'PUT' });
        const data = await response.json();
        if (!data.success) return showAlert(`Could not load preset: ${data.error || 'unknown error'}`);
        if (data.style) {
            populateFormFromStyle(data.style);
            await refreshPalettes();   // re-populates palette selects, keeps selection
            updateLayerChipsFromForm(); // sync on/off chips with the loaded style
            showAlert(`Preset "${name}" applied.`, 'success');
            markFormDirty();
            schedulePreviewRefresh();
        } else {
            showAlert(data.message, 'success');
        }
    } catch (error) {
        console.error('Error loading preset:', error);
        showAlert('An error occurred while loading the preset.');
    }
}

async function deletePresetCard(name) {
    if (!confirm(`Delete preset "${name}"?`)) return;
    try {
        const response = await fetch(`/api/presets/${encodeURIComponent(name)}`, { method: 'DELETE' });
        const data = await response.json();
        if (!data.success) return showAlert(data.error || 'Delete failed');
        showAlert(data.message, 'success');
        await loadPresetCards();
    } catch (error) {
        console.error('Error deleting preset:', error);
        showAlert('An error occurred while deleting the preset.');
    }
}

async function handleSavePreset() {
    const nameInput = document.getElementById('new-preset-name');
    const presetName = nameInput.value.trim();
    if (!presetName) return showAlert('Please enter a name for the preset.');

    try {
        const response = await fetch('/api/presets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: presetName, style: getStyleFromForm() })
        });
        const data = await response.json();
        if (!data.success) return showAlert(`Error: ${data.error}`);
        nameInput.value = '';
        showAlert(data.message, 'success');
        await loadPresetCards();
    } catch (error) {
        console.error('Error saving preset:', error);
        showAlert('An error occurred while saving the preset.');
    }
}

function surpriseMe() {
    const names = Object.keys(cachedPalettes || {});
    const palette = names.length ? names[Math.floor(Math.random() * names.length)] : '';
    if (palette) applyPaletteToStreets(palette);

    // Randomize lane widths so "Surprise me" visibly changes the map.
    document.querySelectorAll('[data-street-class]').forEach((row, idx) => {
        const w = document.getElementById(`streets_category_width_${row.dataset.streetClass}`);
        if (w) w.value = (0.3 + Math.random() * 1.4 * (1.6 - idx * 0.12)).toFixed(2);
    });

    const bm = document.getElementById('building_styling_mode');
    if (bm && names.length) {
        const modeName = Math.random() < 0.5 ? 'auto_size' : 'auto_distance';
        bm.value = modeName;
        const sel = document.getElementById(modeName === 'auto_size' ? 'auto_size_palette' : 'auto_distance_palette');
        if (sel) sel.value = palette;
        bm.dispatchEvent(new Event('change'));
    }

    // Randomize a crop mask + title occasionally.
    const mask = document.getElementById('output_mask');
    if (mask && Math.random() < 0.5) {
        const shapes = ['circle', 'hexagon', 'rounded'];
        mask.value = shapes[Math.floor(Math.random() * shapes.length)];
        mask.dispatchEvent(new Event('change'));
    }
    const title = document.getElementById('output_title');
    if (title && Math.random() < 0.4 && !title.value) {
        title.value = Math.random() < 0.5 ? 'Cork {coords}' : 'Little map {coords}';
    }

    showAlert(palette ? `Surprise! Using palette "${palette}".` : 'Surprise! Randomised widths.', 'success');
    markFormDirty();
    schedulePreviewRefresh();
}

// ============================================================================
// Form hydration (D2) - set every form control from a parsed style object
// ============================================================================

function populateFormFromStyle(style) {
    if (!style) return;
    const form = document.getElementById('map-form');
    if (!form) return;

    function setValue(name, value) {
        const el = form.querySelector(`[name="${name}"]`);
        if (!el) return;
        if (el.type === 'checkbox') {
            el.checked = !!value;
        } else if (el.type === 'radio') {
            el.checked = String(el.value) === String(value);
        } else {
            el.value = (value === undefined || value === null) ? '' : String(value);
        }
    }

    const loc = style.location || {};
    const parts = String(loc.query || '').replace(',', ' ').split(/\s+/);
    setValue('location_latitude', parts[0]);
    setValue('location_longitude', parts[1]);
    setValue('location_distance', loc.distance);
    setValue('location_data_source', loc.data_source === 'local' ? 'local' : 'remote');
    setValue('location_pbf_folder', loc.pbf_folder || '');
    setValue('location_pbf_path', loc.pbf_path || '');

    const out = style.output || {};
    const fig = Array.isArray(out.figure_size) ? out.figure_size : [10, 10];
    setValue('filename_prefix', out.filename_prefix || '');
    setValue('figure_size_width', fig[0]);
    setValue('figure_size_height', fig[1]);
    setValue('figure_dpi', out.figure_dpi || 100);
    setValue('background_color', out.background_color || '#ffffff');
    setValue('margin', out.margin || 0);
    setValue('separate_layers', out.separate_layers);
    setValue('transparent_background', out.transparent_background);
    setValue('enable_debug_legends', out.enable_debug_legends);

    // Creative output (D4)
    setValue('output_mask', out.mask || 'none');
    const mo = out.mask_options || {};
    setValue('output_mask_inset', mo.inset != null ? mo.inset : 0);
    setValue('output_mask_radius', mo.rounded_radius != null ? mo.rounded_radius : 0.15);
    const frame = out.frame || {};
    setValue('output_frame_width', frame.width != null ? frame.width : 0);
    setValue('output_frame_color', frame.color || '#000000');
    setValue('output_title', out.title || '');
    setValue('output_subtitle', out.subtitle || '');
    setValue('output_title_font_size', out.title_font_size || 16);
    setValue('output_subtitle_font_size', out.subtitle_font_size || 11);
    setValue('output_title_color', out.title_color || '#000000');
    setValue('output_subtitle_color', out.subtitle_color || '#444444');

    const layers = style.layers || {};
    ['streets', 'water', 'green', 'buildings'].forEach(name => {
        const cfg = layers[name] || {};
        setValue(`${name}_enabled`, cfg.enabled);
        if (name !== 'streets' && name !== 'buildings') {
            setValue(`${name}_facecolor`, cfg.facecolor || '#000000');
        }
        setValue(`${name}_edgecolor`, cfg.edgecolor || '#000000');
        setValue(`${name}_linewidth`, cfg.linewidth != null ? cfg.linewidth : '');
        setValue(`${name}_alpha`, cfg.alpha != null ? cfg.alpha : 1);
        if (cfg.zorder != null) setValue(`${name}_zorder`, cfg.zorder);
        if (cfg.simplify_tolerance != null) setValue(`${name}_simplify_tolerance`, cfg.simplify_tolerance);
        if (cfg.min_size_threshold != null) setValue(`${name}_min_size_threshold`, cfg.min_size_threshold);
    });

    // Streets specials
    const streets = layers.streets || {};
    setValue('streets_style_mode', streets.style_mode || 'uniform');
    setValue('streets_variable_widths', streets.variable_widths !== false);
    setValue('streets_street_palette', streets.street_palette || '');
    document.querySelectorAll('[data-street-class]').forEach(row => {
        const catId = row.dataset.streetClass;
        const cc = (streets.class_colors || {})[catId];
        if (cc) setValue(`streets_class_color_${catId}`, cc);
        const cw = (streets.category_widths || {})[catId];
        setValue(`streets_category_width_${catId}`, cw != null ? cw : '');
    });

    // Buildings specials
    const buildings = layers.buildings || {};
    setValue('building_styling_mode', buildings.auto_style_mode || 'manual');
    setValue('buildings_manual_color_facecolor', (buildings.manual_color_settings || {}).facecolor || '#000000');
    setValue('auto_size_palette', buildings.auto_size_palette || '');
    setValue('auto_distance_palette', buildings.auto_distance_palette || '');

    // Rebuild the size-category rows
    const listEl = document.getElementById('categories-list');
    if (listEl) {
        listEl.innerHTML = '';
        categoryIndex = 0;
        (buildings.size_categories || []).forEach(cat => {
            listEl.appendChild(createCategoryItem(categoryIndex++, cat));
        });
    }

    // Re-apply dependent visibility (buildings mode sections, by-class section)
    ['building_styling_mode', 'streets_style_mode'].forEach(name => {
        const el = form.querySelector(`[name="${name}"]`);
        if (el) el.dispatchEvent(new Event('change', { bubbles: true }));
    });

    // Feature-category checkboxes, if present in the style
    const filters = style.layer_filters || {};
    Object.keys(filters).forEach(catId => {
        const el = form.querySelector(`[name="feature_${catId.replace(/\./g, '__')}"]`);
        if (el) el.checked = !!filters[catId];
    });

    syncMaskUi(); // Creative-section visibility depends on the mask select
}

// ============================================================================
// Layer quick-toggles + cheap live preview (D3)
// ============================================================================

let previewRefreshTimer = null;

function initLayerChips() {
    const bar = document.getElementById('layer-chips');
    if (!bar) return;

    document.querySelectorAll('.layer-chip').forEach(chip => {
        const dot = chip.querySelector('.layer-chip-dot');
        if (dot && chip.dataset.color) dot.style.backgroundColor = chip.dataset.color;
    });

    bar.addEventListener('click', event => {
        const chip = event.target.closest('.layer-chip');
        if (!chip) return;
        const name = chip.dataset.layer;
        const checkbox = document.querySelector(`[name="${name}_enabled"]`);
        if (checkbox) checkbox.checked = !checkbox.checked;
        syncLayerChipsFromForm();
        schedulePreviewRefresh();
    });

    // Keep chips in sync when a layer is enabled/disabled in the form.
    ['streets', 'buildings', 'water', 'green'].forEach(name => {
        const checkbox = document.querySelector(`[name="${name}_enabled"]`);
        if (checkbox) checkbox.addEventListener('change', syncLayerChipsFromForm);
    });

    // Show the chips bar once there is something to preview.
    const previewImage = document.getElementById('preview-image');
    if (previewImage) {
        bar.hidden = false;
    }
}

function syncLayerChipsFromForm() {
    document.querySelectorAll('.layer-chip').forEach(chip => {
        const name = chip.dataset.layer;
        const checkbox = document.querySelector(`[name="${name}_enabled"]`);
        chip.classList.toggle('on', !!checkbox && checkbox.checked);
    });
}

function updateLayerChipsFromForm() {
    syncLayerChipsFromForm();
}

function schedulePreviewRefresh(delay = 500) {
    if (previewRefreshTimer) clearTimeout(previewRefreshTimer);
    previewRefreshTimer = setTimeout(refreshCheapPreview, delay);
}

async function refreshCheapPreview() {
    const form = document.getElementById('map-form');
    const bar = document.getElementById('layer-chips');
    if (!form || !bar) return;

    // Skip while a full generation is in flight so we don't steal CPU.
    const jobPanel = document.getElementById('job-panel');
    if (jobPanel && !jobPanel.hidden) return;

    const statusEl = document.getElementById('preview-hint');
    if (statusEl) {
        statusEl.textContent = 'Rendering preview…';
        statusEl.style.display = 'inline';
    }

    try {
        const body = new URLSearchParams(new FormData(form)).toString();
        const response = await fetch('/api/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: body
        });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'Preview failed');

        let preview = document.getElementById('preview-image');
        if (!preview) {
            preview = document.createElement('img');
            preview.id = 'preview-image';
            preview.className = 'preview-image';
            preview.alt = 'Live style preview';
            const empty = document.getElementById('preview-empty');
            if (empty) empty.replaceWith(preview);
            else bar.closest('.section-content').prepend(preview);
        }
        preview.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(data.svg);
        bar.hidden = false;
        if (statusEl) {
            statusEl.textContent = 'Live preview';
            statusEl.style.display = 'none';
        }
        updateLayerChipsFromForm();
    } catch (error) {
        console.error('Preview error:', error);
        if (statusEl) {
            statusEl.textContent = 'Preview unavailable';
            statusEl.style.display = 'inline';
        }
    } finally {
        previewRefreshTimer = null;
    }
}

// ============================================================================
// Creative output controls (D4)
// ============================================================================

function initCreativeControls() {
    const mask = document.getElementById('output_mask');
    if (!mask) return;

    mask.addEventListener('change', syncMaskUi);

    const showRangeValue = (rangeId, outputId) => {
        const range = document.getElementById(rangeId);
        const output = document.getElementById(outputId);
        if (range && output) {
            output.textContent = range.value;
            range.addEventListener('input', () => { output.textContent = range.value; });
        }
    };
    showRangeValue('output_mask_inset', 'output_mask_inset_out');
    showRangeValue('output_mask_radius', 'output_mask_radius_out');

    syncMaskUi();
}

function syncMaskUi() {
    const maskEl = document.getElementById('output_mask');
    const cornerRow = document.querySelector('.mask-only-rounded');
    if (!maskEl || !cornerRow) return;
    cornerRow.style.display = maskEl.value === 'rounded' ? 'grid' : 'none';
}

// ============================================================================

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
            preview_type: 'embedded',
            mask: data.output_mask || 'none',
            mask_options: {
                inset: parseFloat(data.output_mask_inset) || 0,
                rounded_radius: parseFloat(data.output_mask_radius) || 0.15
            },
            frame: {
                width: parseFloat(data.output_frame_width) || 0,
                color: data.output_frame_color || '#000000'
            },
            title: data.output_title || '',
            subtitle: data.output_subtitle || '',
            title_font_size: parseFloat(data.output_title_font_size) || 16,
            subtitle_font_size: parseFloat(data.output_subtitle_font_size) || 11,
            title_color: data.output_title_color || '#000000',
            subtitle_color: data.output_subtitle_color || '#444444'
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

    // Handle complex street settings
    if (style.layers.streets) {
        style.layers.streets.style_mode = data.streets_style_mode || 'uniform';
        style.layers.streets.variable_widths = formData.has('streets_variable_widths');
        style.layers.streets.street_palette = data.streets_street_palette || '';
        style.layers.streets.class_colors = {};
        style.layers.streets.category_widths = {};
        document.querySelectorAll('[data-street-class]').forEach(row => {
            const catId = row.dataset.streetClass;
            style.layers.streets.class_colors[catId] = data[`streets_class_color_${catId}`];
            const widthRaw = data[`streets_category_width_${catId}`];
            if (widthRaw && widthRaw.trim() !== '') {
                style.layers.streets.category_widths[catId] = parseFloat(widthRaw);
            }
        });
    }

    return style;
}
