document.addEventListener("DOMContentLoaded", () => {
    const textInput = document.getElementById("text-input");
    const analyzeBtn = document.getElementById("analyze-btn");
    const spinner = document.getElementById("spinner");
    const backendUrlInput = document.getElementById("backend-url");
    const apiStatusBadge = document.getElementById("api-status-badge");
    const resultsContainer = document.getElementById("results-container");

    // Presets
    const presets = {
        "preset-pos": "What an amazing product, I absolutely love it!",
        "preset-neg": "Terrible experience, very bad quality and waste of money.",
        "preset-neu": "Just got back from work, eating lunch.",
        "preset-cpx": "I thought it would be bad, but it turned out surprisingly decent."
    };

    Object.keys(presets).forEach(id => {
        const btn = document.getElementById(id);
        if (btn) {
            btn.addEventListener("click", () => {
                textInput.value = presets[id];
                analyzeSentiment();
            });
        }
    });

    // Check Backend API Health
    async function checkHealth() {
        const baseUrl = backendUrlInput.value.trim().replace(/\/$/, "");
        try {
            const res = await fetch(`${baseUrl}/health`, { method: "GET" });
            if (res.ok) {
                const data = await res.json();
                if (data.model_loaded) {
                    apiStatusBadge.textContent = "🟢 API Online (Model Ready)";
                    apiStatusBadge.className = "status-tag status-success";
                } else {
                    apiStatusBadge.textContent = "🟡 API Online (Model Loading)";
                    apiStatusBadge.className = "status-tag status-unknown";
                }
            } else {
                throw new Error("HTTP error " + res.status);
            }
        } catch (e) {
            apiStatusBadge.textContent = "🔴 API Offline / Connecting...";
            apiStatusBadge.className = "status-tag status-error";
        }
    }

    backendUrlInput.addEventListener("change", checkHealth);
    checkHealth();

    // Analyze Sentiment Function
    async function analyzeSentiment() {
        const text = textInput.value.strip ? textInput.value.strip() : textInput.value.trim();
        if (!text) return;

        const baseUrl = backendUrlInput.value.trim().replace(/\/$/, "");
        spinner.classList.remove("hidden");
        analyzeBtn.disabled = true;

        try {
            const res = await fetch(`${baseUrl}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text })
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Server returned error ${res.status}`);
            }

            const data = await res.json();
            renderResults(data);
            resultsContainer.classList.remove("hidden");
        } catch (e) {
            alert(`Error: ${e.message}`);
        } finally {
            spinner.classList.add("hidden");
            analyzeBtn.disabled = false;
        }
    }

    analyzeBtn.addEventListener("click", analyzeSentiment);

    // Render Results
    function renderResults(data) {
        const label = data.prediction.toLowerCase();
        const confidence = (data.confidence * 100).toFixed(1);

        const badgeEl = document.getElementById("sentiment-badge");
        const iconEl = document.getElementById("badge-icon");
        const labelEl = document.getElementById("badge-label");
        const confEl = document.getElementById("badge-confidence");

        badgeEl.className = "sentiment-badge";
        if (label === "positive") {
            badgeEl.classList.add("badge-positive");
            iconEl.textContent = "😊";
            labelEl.textContent = "POSITIVE";
        } else if (label === "negative") {
            badgeEl.classList.add("badge-negative");
            iconEl.textContent = "😡";
            labelEl.textContent = "NEGATIVE";
        } else {
            badgeEl.classList.add("badge-neutral");
            iconEl.textContent = "😐";
            labelEl.textContent = "NEUTRAL";
        }
        confEl.textContent = `${confidence}%`;

        // Render Probabilities
        const probs = data.probabilities;
        const setProb = (key, fillId, valId) => {
            const val = ((probs[key] || 0) * 100).toFixed(1);
            document.getElementById(valId).textContent = `${val}%`;
            document.getElementById(fillId).style.width = `${val}%`;
        };
        setProb("positive", "prob-bar-positive", "prob-val-positive");
        setProb("neutral", "prob-bar-neutral", "prob-val-neutral");
        setProb("negative", "prob-bar-negative", "prob-val-negative");

        // Render Tokens
        const nonPadTokens = data.tokens.filter(t => t !== "<PAD>");
        const padTokens = data.tokens.filter(t => t === "<PAD>");
        document.getElementById("token-count").textContent = nonPadTokens.length;
        document.getElementById("pad-count").textContent = padTokens.length;

        const chipsContainer = document.getElementById("token-chips");
        chipsContainer.innerHTML = "";
        data.timestep_details.forEach(item => {
            if (item.word !== "<PAD>") {
                const chip = document.createElement("span");
                chip.className = "token-chip";
                chip.textContent = `t${item.timestep}: ${item.word}`;
                chipsContainer.appendChild(chip);
            }
        });

        // Render Heatmap Matrix
        renderHeatmap(data.timestep_details);
    }

    // Render RNN Timestep Heatmap Matrix
    function renderHeatmap(timestepDetails) {
        const grid = document.getElementById("heatmap-grid");
        grid.innerHTML = "";

        const validSteps = timestepDetails.filter(item => item.word !== "<PAD>");

        validSteps.forEach(item => {
            const row = document.createElement("div");
            row.className = "heatmap-row";

            const label = document.createElement("div");
            label.className = "word-label";
            label.textContent = `t${item.timestep}: ${item.word}`;
            row.appendChild(label);

            const cellsFlex = document.createElement("div");
            cellsFlex.className = "cells-flex";

            item.hidden_state.forEach((val, idx) => {
                const cell = document.createElement("div");
                cell.className = "heatmap-cell";
                cell.style.backgroundColor = getHeatmapColor(val);
                cell.title = `Word: "${item.word}" | Neuron ${idx+1}: ${val}`;
                cell.textContent = (val >= 0 ? "+" : "") + val.toFixed(2);
                cellsFlex.appendChild(cell);
            });

            row.appendChild(cellsFlex);
            grid.appendChild(row);
        });
    }

    // Map activation value [-1.0, 1.0] to color gradient
    function getHeatmapColor(val) {
        // Clamp val between -1 and 1
        const v = Math.max(-1, Math.min(1, val));
        
        if (v < 0) {
            // Negative: Indigo to Teal (-1 -> 0)
            const ratio = Math.abs(v);
            const r = Math.round(49 * (1 - ratio) + 99 * ratio);
            const g = Math.round(46 * (1 - ratio) + 102 * ratio);
            const b = Math.round(129 * (1 - ratio) + 241 * ratio);
            return `rgba(${r}, ${g}, ${b}, 0.85)`;
        } else {
            // Positive: Green/Amber to Emerald (0 -> 1)
            const ratio = v;
            const r = Math.round(16 * ratio + 49 * (1 - ratio));
            const g = Math.round(185 * ratio + 102 * (1 - ratio));
            const b = Math.round(129 * ratio + 129 * (1 - ratio));
            return `rgba(${r}, ${g}, ${b}, 0.85)`;
        }
    }
});
