const selectedGames = new Map();
const dislikedGames = new Map();
let currentDataset = null;
let searchSequence = 0, lookupSequence = 0;
let searchController, lookupController, lookupTimer, toastTimer;
let hasResults = false;
let demoCatalog = false;
const byId = id => document.getElementById(id);

async function responseData(response) {
    try { return await response.json(); }
    catch { throw new Error('The server could not complete this request. Please try again.'); }
}
function notify(message) {
    clearTimeout(toastTimer);
    byId('toast').textContent = message;
    byId('toast').hidden = false;
    toastTimer = setTimeout(() => { byId('toast').hidden = true; }, 4000);
}
function showError(message) {
    byId('error').textContent = message;
    byId('error').hidden = false;
}
function closeLookup() { byId('game-options').hidden = true; }
function refreshSelections() {
    for (const [targetId, games, negative] of [['selected-games', selectedGames, false], ['disliked-games', dislikedGames, true]]) {
        const target = byId(targetId);
        target.replaceChildren();
        games.forEach((name, id) => {
            const button = document.createElement('button');
            button.type = 'button'; button.className = 'selection-chip';
            button.append(document.createTextNode(name));
            const cross = document.createElement('span');
            cross.textContent = '×'; cross.setAttribute('aria-hidden', 'true'); button.append(cross);
            button.setAttribute('aria-label', 'Remove ' + (negative ? 'disliked game ' : '') + name);
            button.addEventListener('click', () => { games.delete(id); refreshSelections(); });
            target.append(button);
        });
    }
    byId('selection-count').textContent = selectedGames.size ? selectedGames.size + ' / 10' : 'Optional';
    byId('seed-mode-field').hidden = selectedGames.size < 2;
    byId('disliked-section').hidden = !dislikedGames.size;
    document.querySelectorAll('[data-taste-id]').forEach(button => {
        const selected = selectedGames.has(Number(button.dataset.tasteId));
        button.setAttribute('aria-pressed', String(selected));
        button.textContent = selected ? '✓ In your taste' : '+ Add to taste';
    });
}
function selectGame(id, name) {
    if (selectedGames.size >= 10 && !selectedGames.has(id)) { showError('Choose up to 10 favorites. Remove one to add another.'); return false; }
    selectedGames.set(id, name); dislikedGames.delete(id);
    refreshSelections(); byId('error').hidden = true; return true;
}
function lookupMessage(text) {
    const target = byId('game-options'); target.replaceChildren();
    const message = document.createElement('p'); message.textContent = text;
    target.append(message); target.hidden = false;
}
async function lookupGames() {
    const sequence = ++lookupSequence, q = byId('game-lookup').value.trim();
    if (lookupController) lookupController.abort();
    lookupController = new AbortController();
    if (q.length < 2) { closeLookup(); return; }
    lookupMessage('Searching game titles…');
    try {
        const params = new URLSearchParams({q});
        if (currentDataset) params.set('dataset', currentDataset);
        const response = await fetch(API_BASE + '/games?' + params, {signal: lookupController.signal});
        if (!response.ok) throw new Error('Game lookup is unavailable. Please try again.');
        const data = await responseData(response);
        if (sequence !== lookupSequence) return;
        currentDataset = data.dataset;
        const target = byId('game-options'); target.replaceChildren();
        data.games.forEach(game => {
            const button = document.createElement('button');
            button.type = 'button'; button.textContent = game.name;
            button.addEventListener('click', () => {
                if (selectGame(game.app_id, game.name)) {
                    byId('game-lookup').value = ''; ++lookupSequence; closeLookup();
                    notify(game.name + ' added. Find games to update your picks.');
                    byId('game-lookup').focus();
                }
            });
            target.append(button);
        });
        if (!data.games.length) lookupMessage('No matching games in this catalog. Try another title.');
    } catch (error) { if (error.name !== 'AbortError' && sequence === lookupSequence) lookupMessage(error.message); }
}
function updateFilterCount() {
    const count = ['co-op', 'min_year', 'max_year'].filter(id => byId(id).value).length;
    byId('filter-count').textContent = count + ' active'; byId('filter-count').hidden = !count;
}
function clearFilters() {
    for (const id of ['co-op', 'min_year', 'max_year']) byId(id).value = '';
    document.querySelector('input[name=preference][value=balanced]').checked = true;
    updateFilterCount();
}
function resetSearch() {
    ++searchSequence; ++lookupSequence;
    if (searchController) searchController.abort();
    if (lookupController) lookupController.abort();
    clearTimeout(lookupTimer); clearTimeout(toastTimer);
    byId('discovery-form').reset(); selectedGames.clear(); dislikedGames.clear(); currentDataset = null;
    refreshSelections(); updateFilterCount(); closeLookup(); hasResults = false;
    byId('main-content').hidden = true; byId('welcome').hidden = false; byId('loading').hidden = true;
    byId('error').hidden = true; byId('toast').hidden = true; byId('search-button').disabled = false;
    byId('search-button-label').textContent = 'Find my next game'; byId('query').focus();
}
async function search() {
    const query = byId('query').value.trim();
    if (!query && !selectedGames.size) { showError('Describe what you want, or choose a game you love to get started.'); byId('query').focus(); return; }
    if (!byId('discovery-form').reportValidity()) return;
    const filters = {};
    if (byId('co-op').value) filters.co_op = byId('co-op').value === 'yes';
    for (const key of ['min_year', 'max_year']) { if (byId(key).value) filters[key] = Number(byId(key).value); }
    if (filters.min_year != null && filters.max_year != null && filters.min_year > filters.max_year) {
        showError('The start year must be earlier than or equal to the end year.');
        document.querySelector('.filter-details').open = true; byId('min_year').focus(); return;
    }
    const sequence = ++searchSequence, seedNames = [...selectedGames.values()];
    if (searchController) searchController.abort();
    searchController = new AbortController();
    byId('error').hidden = true; byId('welcome').hidden = true; byId('main-content').hidden = true;
    byId('loading').hidden = false; byId('search-button').disabled = true;
    byId('search-button-label').textContent = 'Finding games…'; closeLookup();
    try {
        const response = await fetch(API_BASE + '/recommend', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, signal: searchController.signal,
            body: JSON.stringify({text: query, dataset: currentDataset, hard_filters: filters,
                seed_app_ids: [...selectedGames.keys()], disliked_app_ids: [...dislikedGames.keys()],
                seed_mode: byId('seed-mode').value, preference: document.querySelector('input[name=preference]:checked').value})
        });
        const data = await responseData(response);
        if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check your search options and try again.');
        if (sequence !== searchSequence) return;
        currentDataset = data.dataset; renderResults(data, query, seedNames); hasResults = true;
        byId('main-content').hidden = false;
        if (window.matchMedia('(max-width: 850px)').matches) byId('main-content').scrollIntoView({behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start'});
    } catch (error) {
        if (error.name === 'AbortError' || sequence !== searchSequence) return;
        showError(error.message); byId('main-content').hidden = !hasResults; byId('welcome').hidden = hasResults;
    } finally {
        if (sequence === searchSequence) { byId('loading').hidden = true; byId('search-button').disabled = false; byId('search-button-label').textContent = 'Find my next game'; }
    }
}
function snippet(text, limit = 210) {
    const value = String(text || '').replace(/\s+/g, ' ').trim();
    if (value.length <= limit) return value;
    const end = value.lastIndexOf(' ', limit);
    return value.slice(0, end > limit / 2 ? end : limit) + '…';
}
function gameCard(game, query, tagsForFeedback, dataset, rank, sponsored = false) {
    const id = Number(game.app_id), card = document.createElement('article');
    card.className = 'game-card'; card.setAttribute('aria-label', game.name);
    const matched = new Set((game.matched_tags || []).map(t => t.toLowerCase()));
    const tags = (game.tags || []).slice(0, 5).map(t => '<span class="tag ' + (matched.has(t.toLowerCase()) ? 'matched' : '') + '">' + escapeHTML(t) + '</span>').join('');
    const evidence = game.evidence || 'No description available.';
    const sourceLabel = demoCatalog ? 'From the fictional demo catalog' : {description: 'From the Steam description', review: 'From a player review', metadata: 'From listed game tags'}[game.evidence_kind] || 'Matching game information';
    const sequels = game.sequel_group || [];
    const storeUrl = 'https://store.steampowered.com/app/' + id + '/';
    const imageUrl = game.header_image || 'https://cdn.cloudflare.steamstatic.com/steam/apps/' + id + '/header.jpg';
    card.innerHTML =
        (demoCatalog ? '<div class="game-art">' : '<a class="game-art" href="' + storeUrl + '" target="_blank" rel="noopener noreferrer" aria-label="View ' + escapeHTML(game.name) + ' on Steam">') + '<span class="art-fallback"' + (demoCatalog ? '' : ' hidden') + '>' + escapeHTML(game.name) + '</span>' + (demoCatalog ? '' : '<img class="game-image" alt="" loading="lazy" src="' + escapeHTML(imageUrl) + '">') + '<span class="rank-badge">' + (sponsored ? 'Sponsored' : 'PICK ' + String(rank).padStart(2, '0')) + '</span>' + (demoCatalog ? '</div>' : '</a>') +
        '<div class="game-card-content"><div class="game-header">' + (demoCatalog ? '<h3 class="game-title">' + escapeHTML(game.name) + '</h3>' : '<a class="game-title-link" href="https://store.steampowered.com/app/' + id + '/" target="_blank" rel="noopener noreferrer"><h3 class="game-title">' + escapeHTML(game.name) + '</h3></a><span class="store-arrow" aria-hidden="true">↗</span>') + '</div>' +
        '<div class="review-line">' + (demoCatalog ? 'Illustrative demo rating' : game.rating_pct == null ? 'Steam reviews unavailable' : '<strong>' + Number(game.rating_pct).toFixed(0) + '% positive</strong> <span>on Steam</span>') + '</div>' +
        (demoCatalog ? '' : '<a class="store-link" href="' + storeUrl + '" target="_blank" rel="noopener noreferrer">View on Steam ↗</a>') +
        '<div class="tags-container">' + (game.play_modes || []).map(mode => '<span class="tag play-mode">' + escapeHTML(mode) + '</span>').join('') + tags + '</div><p class="game-description">' + escapeHTML(snippet(evidence)) + '</p>' +
        '<details class="game-evidence"><summary>Read the matching passage</summary><p class="evidence-source">' + sourceLabel + '</p><p>' + escapeHTML(evidence) + '</p></details>' +
        (sequels.length ? '<details class="series"><summary class="sequel-badge">' + sequels.length + (sequels.length === 1 ? ' related game' : ' related games') + '</summary><div class="sequel-list">' + sequels.map(s => '<div class="sequel-item">' + (demoCatalog ? escapeHTML(s.name) : '<a href="https://store.steampowered.com/app/' + Number(s.app_id) + '/" target="_blank" rel="noopener noreferrer">' + escapeHTML(s.name) + '</a>') + '</div>').join('') + '</div></details>' : '') +
        '<div class="game-actions"></div></div>';
    const artwork = card.querySelector('img');
    artwork?.addEventListener('error', () => {
        const fallback = 'https://cdn.cloudflare.steamstatic.com/steam/apps/' + id + '/header.jpg';
        if (!artwork.dataset.retried && artwork.src !== fallback) { artwork.dataset.retried = 'true'; artwork.src = fallback; return; }
        artwork.hidden = true; card.querySelector('.art-fallback').hidden = false;
    });
    if (sponsored) return card;
    const actions = card.querySelector('.game-actions'), taste = document.createElement('button'), less = document.createElement('button');
    taste.type = 'button'; taste.className = 'taste-button'; taste.dataset.tasteId = id;
    taste.textContent = '+ Add to taste'; taste.setAttribute('aria-pressed', 'false');
    taste.setAttribute('aria-label', 'Add or remove ' + game.name + ' from your taste');
    taste.addEventListener('click', () => {
        if (selectedGames.has(id)) { selectedGames.delete(id); refreshSelections(); notify(game.name + ' removed from your taste.'); }
        else if (selectGame(id, game.name)) notify(game.name + ' added. Find games to update your picks.');
    });
    less.type = 'button'; less.className = 'less-button'; less.textContent = 'Less like this';
    less.setAttribute('aria-label', 'Find fewer games like ' + game.name);
    less.addEventListener('click', () => {
        if (dislikedGames.size >= 10 && !dislikedGames.has(id)) { showError('Choose up to 10 disliked games. Remove one to choose another.'); return; }
        dislikedGames.set(id, game.name); selectedGames.delete(id); refreshSelections(); search();
    });
    actions.append(taste, less);
    const feedback = document.createElement('details'); feedback.className = 'card-feedback';
    feedback.innerHTML = '<summary>Was this suggestion helpful?</summary><div class="feedback-buttons"></div>';
    [true, false].forEach(accepted => {
        const button = document.createElement('button'); button.type = 'button'; button.className = 'feedback-btn feedback-vote';
        button.textContent = accepted ? 'Yes, good fit' : 'Not for me';
        button.addEventListener('click', () => sendFeedback(id, accepted, query, tagsForFeedback, dataset, button));
        feedback.querySelector('.feedback-buttons').append(button);
    });
    card.querySelector('.game-card-content').append(feedback); return card;
}
function renderResults(data, query, seedNames) {
    const target = byId('results'); target.replaceChildren();
    byId('results-count').textContent = data.results.length + ' picks';
    const pieces = [];
    if (query) pieces.push('“' + snippet(query, 100) + '”');
    if (seedNames.length) pieces.push('Inspired by ' + seedNames.slice(0, 2).join(' + ') + (seedNames.length > 2 ? ' and ' + (seedNames.length - 2) + ' more' : ''));
    byId('results-summary').textContent = pieces.join(' · ');
    const interests = byId('active-interests'); interests.replaceChildren();
    (data.tags || []).slice(0, 5).forEach(tag => { const chip = document.createElement('span'); chip.className = 'interest-chip'; chip.textContent = tag; interests.append(chip); });
    if (!data.results.length) {
        const empty = document.createElement('div'); empty.className = 'empty-state';
        const filtered = ['co-op', 'min_year', 'max_year'].some(id => byId(id).value) || document.querySelector('input[name=preference]:checked').value !== 'balanced';
        empty.innerHTML = '<h3>A little too specific?</h3><p>Try a wider release range, fewer exclusions, or a different favorite.</p><button type="button">' + (filtered ? 'Clear filters and try again' : 'Edit my search') + '</button>';
        empty.querySelector('button').addEventListener('click', () => { if (filtered) { clearFilters(); search(); } else { byId('query').focus(); } }); target.append(empty);
    }
    data.results.forEach((game, i) => target.append(gameCard(game, query, data.tags, data.dataset, i + 1)));
    if (data.sponsored) { const label = document.createElement('h3'); label.className = 'sponsored-label'; label.textContent = 'Sponsored suggestion'; target.append(label, gameCard(data.sponsored, query, data.tags, data.dataset, 0, true)); }
    refreshSelections(); byId('search-details').hidden = !data.debug_log;
    if (data.debug_log) renderLogs(data.debug_log);
}
function renderLogs(log) {
    const target = byId('logs-content'); target.replaceChildren();
    const entry = document.createElement('div'); entry.className = 'log-entry';
    entry.innerHTML = '<div class="log-query">' + escapeHTML(log.query || 'Based on your selected games') + '</div><div class="log-section-title">Interests used in this search</div><div class="log-tags">' +
        (log.extracted_tags || []).slice(0, 8).map(t => '<span class="log-tag">' + escapeHTML(t.name) + '</span>').join(' · ') +
        '</div><div class="log-section-title">Avoiding</div><div class="log-tags">' +
        ((log.excluded_tags || []).map(t => '<span class="log-tag excluded">' + escapeHTML(t) + '</span>').join(' · ') || 'No genre exclusions') +
        '</div><div class="log-time">Search completed in ' + Number(log.processing_time_ms).toFixed(0) + ' ms</div>';
    target.append(entry);
}
async function sendFeedback(appId, accepted, query, tags, dataset, button) {
    const buttons = button.parentElement.querySelectorAll('.feedback-vote');
    buttons.forEach(b => { b.disabled = true; });
    try {
        const response = await fetch(API_BASE + '/feedback', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query_text: query, tags, app_id: appId, accepted, dataset})});
        if (!response.ok) throw new Error('Feedback could not be saved. Please try again.');
        button.classList.add(accepted ? 'active-good' : 'active-bad'); button.textContent = '✓ Saved';
    } catch (error) { buttons.forEach(b => { b.disabled = false; }); showError(error.message); }
}
byId('discovery-form').addEventListener('submit', event => { event.preventDefault(); search(); });
byId('discovery-form').addEventListener('invalid', event => { if (event.target.closest('.filter-details')) document.querySelector('.filter-details').open = true; }, true);
byId('reset-search').addEventListener('click', resetSearch);
byId('query').addEventListener('keydown', event => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); search(); } });
byId('game-lookup').addEventListener('input', () => {
    ++lookupSequence; clearTimeout(lookupTimer); if (lookupController) lookupController.abort();
    closeLookup(); lookupTimer = setTimeout(lookupGames, 250);
});
byId('game-lookup').addEventListener('keydown', event => {
    if (event.key === 'Escape') { ++lookupSequence; clearTimeout(lookupTimer); if (lookupController) lookupController.abort(); closeLookup(); }
    if (['ArrowDown', 'Enter'].includes(event.key)) { event.preventDefault(); if (!byId('game-options').hidden) byId('game-options').querySelector('button')?.focus(); }
});
document.addEventListener('click', event => { if (!event.target.closest('.taste-block')) closeLookup(); });
for (const id of ['co-op', 'min_year', 'max_year']) byId(id).addEventListener('input', updateFilterCount);
document.querySelectorAll('[data-prompt]').forEach(button => { button.addEventListener('click', () => { byId('query').value = button.dataset.prompt; search(); }); });
document.querySelectorAll('[data-seed]').forEach(button => {
    button.addEventListener('click', () => { if (selectGame(Number(button.dataset.seed), button.dataset.name)) notify(button.dataset.name + ' added. Find games to discover similar picks.'); });
});
fetch(API_BASE + '/health').then(responseData).then(data => {
    demoCatalog = !!data.demo_catalog;
    byId('demo-notice').hidden = !demoCatalog && !data.catalog_notice;
    if (!demoCatalog && data.catalog_notice) byId('demo-notice').textContent = data.catalog_notice;
    if (demoCatalog) {
        document.querySelector('.starter-section').hidden = true;
        document.querySelector('.site-footer span:last-child').textContent = 'Fictional sample catalog for this demo.';
    }
    byId('history-link').hidden = !(data.history_enabled && (data.history_access_mode === 'token' || ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)));
}).catch(() => {});
