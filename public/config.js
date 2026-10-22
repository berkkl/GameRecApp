// Set window.GAMEREC_API_BASE before loading this file for a separate API host.
const API_BASE = window.GAMEREC_API_BASE ?? new URL('.', document.baseURI).pathname.replace(/\/$/, '');

function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}
