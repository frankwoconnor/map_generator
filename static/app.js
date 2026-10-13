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
    loadPalettes(data);
    initPresetManagement();
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
                terminalHandler({ job_id: jobId, status: job.status, error: job.error });
                return;
            }
        }
    } catch (err) { /* fall through to SSE */ }

    const es = new EventSource(`/api/jobs/${jobId}/events`);
    es.addEventListener('message', (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'line' && msg.seq > lastSeq) {
            appendJobLine(msg.text);
            lastSeq = msg.seq;
        }
    });
    es.addEventListener('terminal', (event) => {
        es.close();
        terminalHandler(JSON.parse(event.data));
    });
}

async function handleJobTerminal(jobId, msg) {
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
    } else {
        setStatus(status.charAt(0).toUpperCase() + status.slice(1));
        if (msg.error) showJobError(msg.error);
    }
    const button = document.getElementById('generate-btn');
    if (button) button.disabled = false;
    const cancelBtn = document.getElementById('job-cancel');
    if (cancelBtn) cancelBtn.disabled = false;
}

async function refreshResultAndGallery() {
    try {
        const resp = await fetch('/api/runs?limit=12');
        const data = await resp.json();
        if (!resp.ok || !data.success || !data.runs || !data.runs.length) return;
        rebuildRunGrid(data.runs);
        applyRunToPreview(data.runs[0]);
    } catch (err) { /* non-fatal */ }
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

        const palettes = result.palettes;
        cachedPalettes = palettes;

        // Populate selects
        populatePaletteSelect('auto_size_palette', 'auto_size_palette_preview',
                            palettes, data.autoSizePalette);
        populatePaletteSelect('auto_distance_palette', 'auto_distance_palette_preview',
                            palettes, data.autoDistancePalette);
        populatePaletteSelect('streets_street_palette', 'streets_street_palette_preview',
                            palettes, data.streetPalette);

    } catch (error) {
        console.error('Error loading palettes:', error);
        // Fallback to server-rendered data
        if (data.palettes && Object.keys(data.palettes).length > 0) {
            cachedPalettes = data.palettes;
            populatePaletteSelect('auto_size_palette', 'auto_size_palette_preview',
                                data.palettes, data.autoSizePalette);
            populatePaletteSelect('auto_distance_palette', 'auto_distance_palette_preview',
                                data.palettes, data.autoDistancePalette);
            populatePaletteSelect('streets_street_palette', 'streets_street_palette_preview',
                                data.palettes, data.streetPalette);
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
