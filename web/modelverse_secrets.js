import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const cssUrl = new URL("./modelverse_secrets.css", import.meta.url);
if (!document.querySelector(`link[href="${cssUrl}"]`)) {
    document.head.appendChild(Object.assign(document.createElement("link"), { rel: "stylesheet", href: cssUrl }));
}

const NODE_NAME = "UCloud ModelVerse Secret Client";
const WIDGET_NAME = "secret";
const ENDPOINT = "/modelverse-secrets";

let cachedSecretNames = [];

function el(tag, props = {}, ...children) {
    const node = Object.assign(document.createElement(tag), props);
    node.append(...children);
    return node;
}

async function fetchSecrets() {
    const response = await api.fetchApi(ENDPOINT, { method: "GET" });
    return await response.json();
}

async function refreshSecretNamesCache() {
    const secrets = await fetchSecrets();
    cachedSecretNames = Object.keys(secrets);
    return cachedSecretNames;
}

async function refreshSecretDropdowns(preferredValue) {
    const names = await refreshSecretNamesCache();
    for (const node of app.graph._nodes ?? []) {
        if (node.comfyClass !== NODE_NAME && node.type !== NODE_NAME) {
            continue;
        }
        const widget = node.widgets?.find((item) => item.name === WIDGET_NAME);
        if (!widget) {
            continue;
        }
        widget.options.values = names.slice();
        if (preferredValue && names.includes(preferredValue)) {
            widget.value = preferredValue;
        } else if (names.length && !names.includes(widget.value)) {
            widget.value = names[0];
        } else if (!names.length) {
            widget.value = "";
        }
        if (typeof widget.callback === "function") {
            try { widget.callback(widget.value); } catch (e) { /* noop */ }
        }
        node.setDirtyCanvas?.(true, true);
    }
    app.graph?.setDirtyCanvas?.(true, true);
}

function buildSecretRow(name = "", value = "", isNew = false, onSaved) {
    let originalName = name;
    const nameInput = el("input", { className: "mv-secret-input", placeholder: "Secret name", value: name });
    const valueInput = el("input", { className: "mv-secret-input", placeholder: "API key", value, type: "password", autocomplete: "off" });

    const saveButton = el("button", { className: "mv-secret-button mv-secret-save", textContent: isNew ? "Add" : "Save" });
    const deleteButton = el("button", { className: "mv-secret-button mv-secret-delete", textContent: "Delete" });
    const row = el("div", { className: "mv-secret-row" }, nameInput, valueInput, saveButton, deleteButton);

    saveButton.onclick = async () => {
        const key = nameInput.value.trim();
        if (!key) {
            nameInput.focus();
            return;
        }

        const response = await api.fetchApi(ENDPOINT, {
            method: "POST",
            body: JSON.stringify({ key, value: valueInput.value }),
        });
        if (!response.ok) {
            return;
        }

        if (!isNew && originalName && originalName !== key) {
            await api.fetchApi(`${ENDPOINT}/${encodeURIComponent(originalName)}`, { method: "DELETE" });
        }

        originalName = key;
        await refreshSecretDropdowns(key);

        if (typeof onSaved === "function") {
            onSaved();
        }

        if (isNew) {
            row.replaceWith(buildSecretRow(key, valueInput.value, false, onSaved));
            return;
        }

        saveButton.textContent = "Saved";
        setTimeout(() => {
            saveButton.textContent = "Save";
        }, 1200);
    };

    deleteButton.onclick = async () => {
        if (!originalName) {
            row.remove();
            return;
        }
        const response = await api.fetchApi(`${ENDPOINT}/${encodeURIComponent(originalName)}`, { method: "DELETE" });
        if (response.ok) {
            row.remove();
            await refreshSecretDropdowns();
        }
    };

    return row;
}

function createSecretsModal(secrets) {
    const saveNotice = el("div", { className: "mv-secret-save-notice" });
    const showSaveNotice = () => {
        saveNotice.textContent = "Once added, there is a 10-second delay before the item shows up in the dropdown.";
        saveNotice.classList.remove("mv-secret-save-notice-fade");
        void saveNotice.offsetWidth;
        saveNotice.classList.add("mv-secret-save-notice-visible");
        clearTimeout(saveNotice._hideTimer);
        saveNotice._hideTimer = setTimeout(() => {
            saveNotice.classList.add("mv-secret-save-notice-fade");
        }, 10000);
    };
    const rows = el(
        "div",
        { className: "mv-secret-rows" },
        ...Object.entries(secrets).map(([name, value]) => buildSecretRow(name, value, false, showSaveNotice)),
    );
    const addButton = el("button", {
        className: "mv-secret-add",
        textContent: "+ Add Secret",
        onclick: () => {
            const row = buildSecretRow("", "", true, showSaveNotice);
            rows.appendChild(row);
            row.querySelector("input")?.focus();
        },
    });
    const closeButton = el("button", { className: "mv-secret-close", textContent: "x" });

    const overlay = el("div", { className: "mv-secret-overlay" },
        el("div", { className: "mv-secret-dialog" },
            el("div", { className: "mv-secret-header" },
                el("h3", { className: "mv-secret-title", textContent: "Modelverse Secrets" }),
                closeButton,
            ),
            rows,
            addButton,
            saveNotice,
        ),
    );

    closeButton.onclick = () => overlay.remove();
    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) {
            overlay.remove();
        }
    });

    return overlay;
}

app.registerExtension({
    name: "compshare.modelverse_secrets",
    async nodeCreated(node) {
        if (node.comfyClass !== NODE_NAME && node.type !== NODE_NAME) {
            return;
        }

        const names = await refreshSecretNamesCache();
        const index = node.widgets?.findIndex((item) => item.name === WIDGET_NAME) ?? -1;
        const savedValue = index >= 0 ? node.widgets[index].value : undefined;
        if (index >= 0) {
            node.widgets.splice(index, 1);
        }

        const defaultValue = savedValue && names.includes(savedValue) ? savedValue : (names[0] ?? "");
        const comboOptions = {
            get values() {
                return cachedSecretNames.slice();
            },
            set values(v) { /* ignore external writes; cache is the source of truth */ },
        };
        node.addWidget("combo", WIDGET_NAME, defaultValue, () => {}, comboOptions);
        node.addWidget("button", "Edit Secrets", null, async () => {
            await refreshSecretNamesCache();
            document.body.appendChild(createSecretsModal(await fetchSecrets()));
        });
    },
});
