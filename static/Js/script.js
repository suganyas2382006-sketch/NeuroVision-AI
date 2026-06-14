// Page loaded
document.addEventListener("DOMContentLoaded", function () {
    console.log("NeuroVision AI Loaded");
});

// Preview MRI image before upload
function previewImage(event) {
    const preview = document.getElementById("preview");

    if (preview && event.target.files.length > 0) {
        preview.src = URL.createObjectURL(event.target.files[0]);
        preview.style.display = "block";
    }
}

// Confirm report download
function confirmDownload() {
    return confirm("Download Medical Report?");
}