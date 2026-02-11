async function upload() {
    const fileInput = document.getElementById("file");
    if (!fileInput.files.length) {
        alert("Please select an image");
        return;
    }

    const userId = document.body.dataset.userId;
    if (!userId) {
        alert("User not detected. Please login again.");
        return;
    }

    const fd = new FormData();
    fd.append("file", fileInput.files[0]);

    const res = await fetch(`/predict/${userId}`, {
        method: "POST",
        body: fd
    });

    if (!res.ok) {
        alert("Prediction failed");
        return;
    }

    const data = await res.json();
    document.getElementById("result").innerHTML =
        `<b>Result:</b> ${data.result}<br><b>Confidence:</b> ${data.confidence}`;
}
