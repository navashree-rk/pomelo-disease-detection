const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const preview = document.getElementById('preview');
const previewContainer = document.getElementById('preview-container');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => webcam.srcObject = stream)
  .catch(err => alert("Camera access denied."));

function captureImage() {
  ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
  const dataURL = canvas.toDataURL('image/jpeg');
  preview.src = dataURL;
  previewContainer.style.display = 'block';  // show captured image and label
}

function dataURLtoBlob(dataurl) {
  const arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
  for (let i = 0; i < n; i++) u8arr[i] = bstr.charCodeAt(i);
  return new Blob([u8arr], { type: mime });
}

function submitImage() {
  const dataURL = canvas.toDataURL('image/jpeg');
  const blob = dataURLtoBlob(dataURL);
  const formData = new FormData();
  formData.append("image", blob, "captured.jpg");

  fetch("/predict", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById("result").innerHTML = `
      <h3>🧠 Prediction: ${data.class} (${data.confidence}%)</h3>
      <p>🌾 Fertilizer Suggestion: ${data.suggestion}</p>
    `;
  });
}
