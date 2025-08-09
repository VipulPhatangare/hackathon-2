
document.addEventListener('DOMContentLoaded', () => {
  const annotatedImg = document.getElementById('annotatedImg');
  const countDisplay = document.getElementById('countDisplay');
  const predList = document.getElementById('predList');

  const socket = io();

  socket.on('connect', () => {
    console.log('Connected monitor socket', socket.id);
  });


  socket.on('prediction', (data) => {
  console.log('Prediction received:', data);
  if (!data) return;
  if (data.success) {
    if (data.annotatedImage) {
      annotatedImg.src = 'data:image/png;base64,' + data.annotatedImage;
    } else {
      annotatedImg.src = '';
    }
    countDisplay.textContent = 'Count: ' + (data.count ?? 0);

    if (data.predictions && data.predictions.length) {
      predList.innerHTML = '';
      const sorted = [...data.predictions].sort((a,b) => (b.confidence||0) - (a.confidence||0));
      sorted.forEach(p => {
        const div = document.createElement('div');
        div.textContent = `${p.class || 'obj'} — ${(Math.round((p.confidence||0)*100))}%`;
        predList.appendChild(div);
      });
    }
  } else {
    console.warn('Prediction error', data.error);
  }
});
});
