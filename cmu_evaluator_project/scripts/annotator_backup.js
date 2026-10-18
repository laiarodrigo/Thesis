(() => {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const fragment = new URLSearchParams(window.location.hash.slice(1));
  const campaignId = params.get("campaign_id") || "";
  const userId = params.get("user_id") || "";
  const backupToken = fragment.get("backup_token") || "";
  const safeId = /^[A-Za-z0-9._-]{1,128}$/;
  const safeToken = /^[A-Za-z0-9_-]{43}$/;

  function completedHeading() {
    return Array.from(document.querySelectorAll("#output_div h2")).find((heading) =>
      heading.textContent.includes("All done")
    );
  }

  function filenameFrom(response) {
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([A-Za-z0-9._-]+\.zip)"/);
    return match ? match[1] : `${userId}_annotation_backup.zip`;
  }

  async function download(button, message) {
    button.disabled = true;
    button.textContent = "Preparing backup…";
    message.textContent = "";
    try {
      const response = await fetch("/annotator-backup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "omit",
        cache: "no-store",
        body: JSON.stringify({
          campaign_id: campaignId,
          user_id: userId,
          backup_token: backupToken,
        }),
      });
      if (!response.ok) {
        const reason = (await response.text()).trim();
        throw new Error(reason || `Backup request failed (${response.status})`);
      }
      const objectUrl = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = filenameFrom(response);
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
      button.textContent = "Download backup again";
      message.textContent =
        "Downloaded. Keep this copy if you wish; the researcher’s server log remains authoritative.";
    } catch (error) {
      button.textContent = "Download backup";
      message.textContent = `Backup unavailable: ${error.message}`;
    } finally {
      button.disabled = false;
    }
  }

  function addCompletionBackup() {
    const heading = completedHeading();
    if (!heading || document.getElementById("annotator_backup")) return;

    const container = document.createElement("div");
    container.id = "annotator_backup";
    container.style.marginTop = "1.25rem";

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Download backup";
    button.style.padding = "0.7rem 1rem";
    button.style.fontSize = "11pt";
    button.style.cursor = "pointer";

    const message = document.createElement("p");
    message.style.maxWidth = "42rem";
    message.style.fontSize = "10pt";
    message.style.color = "#555";
    message.textContent =
      "Optional: download your own annotations and progress. The researcher’s server log remains authoritative.";

    if (
      !safeId.test(campaignId) ||
      !safeId.test(userId) ||
      !safeToken.test(backupToken)
    ) {
      button.disabled = true;
      message.textContent =
        "This link does not contain a valid personal backup capability. Ask the researcher for the original annotation link.";
    } else {
      button.addEventListener("click", () => void download(button, message));
    }

    container.append(button, message);
    (heading.closest(".white-box") || heading.parentElement).appendChild(container);
  }

  const observer = new MutationObserver(addCompletionBackup);
  const start = () => {
    const output = document.getElementById("output_div");
    if (!output) return;
    observer.observe(output, { childList: true, subtree: true });
    addCompletionBackup();
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
