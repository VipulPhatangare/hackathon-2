// phone input script
document.addEventListener('DOMContentLoaded', () => {
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const statusEl = document.getElementById('status');
  const video = document.getElementById('phoneVideo');
  const canvas = document.getElementById('phoneCanvas');
  const intervalInput = document.getElementById('intervalSec');

  let socket = null;
  let stream = null;
  let timer = null;

  function setStatus(txt) { statusEl.textContent = txt; }

  async function startStreaming() {
    try {
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: { ideal: "environment" } // ask for back camera
        },
        audio: false
      };
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      video.play();

      // connect socket
      socket = io();

      socket.on('connect', () => {
        setStatus('Connected to server (socket id: ' + socket.id + ')');
      });

      socket.on('ack', (a) => {
        // optional
      });

      socket.on('prediction', (data) => {
        console.log('Prediction arrived (phone):', data);
      });

      // start capture loop
      scheduleCapture();

      startBtn.disabled = true;
      stopBtn.disabled = false;
      setStatus('Streaming started');
    } catch (err) {
      console.error(err);
      setStatus('Error: ' + err.message);
    }
  }

  function stopStreaming() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    if (socket) {
      socket.disconnect();
      socket = null;
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
    setStatus('Stopped');
  }

  function scheduleCapture() {
    const seconds = Math.max(1, parseInt(intervalInput.value || '5', 10));
    const intervalMs = seconds * 1000;

    // ensure canvas size is sync with video natural size once available
    timer = setInterval(() => {
      if (!video || video.readyState < 2) return;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // send as jpeg base64 with quality to reduce size
      const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
      const base64 = dataUrl.split(',')[1];

      if (socket && socket.connected) {
        socket.emit('frame', { imageBase64: base64 });
        setStatus('Frame sent at ' + new Date().toLocaleTimeString());
      }
    }, intervalMs);
  }

  startBtn.addEventListener('click', startStreaming);
  stopBtn.addEventListener('click', stopStreaming);
});
