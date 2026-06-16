document.addEventListener("DOMContentLoaded", function () {
    console.log("NeuroVision AI Loaded");
});

function previewImage(event) {
    const preview = document.getElementById("preview");
    const selectedFile = event.target.files[0];

    if (preview && selectedFile) {
        // Safe memory management: clear old blob pointer strings
        if (preview.src && preview.src.startsWith("blob:")) {
            URL.revokeObjectURL(preview.src);
        }

        preview.src = URL.createObjectURL(selectedFile);
        preview.style.display = "block";
    }
}

function confirmDownload() {
    return confirm("Download Medical Report?");
}
