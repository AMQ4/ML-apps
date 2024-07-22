const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const status = document.getElementById('status');
const predictForm = document.getElementById('predictForm');
const predictButton = document.getElementById('predictButton');

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', handleFileSelect);
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
        resetStatus();
        uploadFile(file);
    } else {
        setStatus('Please select a valid CSV file.', 'error');
    }
}

function handleDragOver(event) {
    event.preventDefault();
    dropZone.classList.add('hover');
    dropZone.innerHTML = '<p>Release to drop the file</p>';
}

function handleDragLeave(event) {
    dropZone.classList.remove('hover');
    dropZone.innerHTML = '<p>Drag and drop your CSV file here or click to select</p>';
}

function handleDrop(event) {
    event.preventDefault();
    dropZone.classList.remove('hover');
    dropZone.innerHTML = '<p>Drag and drop your CSV file here or click to select</p>';
    const file = event.dataTransfer.files[0];
    if (file && file.type === 'text/csv') {
        resetStatus();
        uploadFile(file);
    } else {
        setStatus('Please drop a valid CSV file.', 'error');
    }
}

function resetStatus() {
    status.textContent = '';
    status.className = '';
    status.style.opacity = '1';
    predictForm.style.display = 'none';
    predictForm.style.textAlign = 'center';
    status.classList.remove('fade-out');
}

function setStatus(message, type) {
    status.textContent = message;
    status.className = type;
    if (type === 'success') {
        predictForm.style.display = 'block';
        predictForm.style.textAlign = 'center';
        status.classList.add('fade-out');
    } else {
        predictForm.style.display = 'none';
        status.style.opacity = '1';
    }
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        if (data.includes('successfully')) {
            setStatus(data, 'success');
        } else {
            setStatus(data, 'error');
        }
    })
    .catch(error => {
        setStatus('Error uploading file', 'error');
    });
}

predictForm.addEventListener('submit', function(event) {
    event.preventDefault();
    predictButton.disabled = true;
    predictButton.textContent = 'Processing...';

    fetch('/predict', {
        method: 'POST'
    })
    .then(response => response.text())
    .then(data => {
        setStatus(data, 'success');
        predictButton.disabled = false;
        predictButton.textContent = 'Next Month Prediction';
    })
    .catch(error => {
        setStatus('Error making prediction', 'error');
        predictButton.disabled = false;
        predictButton.textContent = 'Next Month Prediction';
    });
});
