require('dotenv').config();
const express = require('express');
const path = require('path');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const http = require('http');
const { Server } = require('socket.io');
const { createCanvas, loadImage } = require('@napi-rs/canvas');
const sharp = require('sharp');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

const PORT = process.env.PORT || 3000;
const ROBOFLOW_ENDPOINT = process.env.ROBOFLOW_ENDPOINT;
const ROBOFLOW_API_KEY = process.env.ROBOFLOW_API_KEY;

if (!ROBOFLOW_ENDPOINT || !ROBOFLOW_API_KEY) {
  console.warn('WARNING: ROBOFLOW_ENDPOINT or ROBOFLOW_API_KEY missing in .env');
}

// Alert threshold system
let ALERT_THRESHOLD = 5;
const ALERT_COOLDOWN = 5 * 60 * 1000;
let lastAlertTime = 0;

// views + static
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));

// ensure uploads dir exists
const uploadDir = path.join(__dirname, 'public', 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

// multer for single uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const safe = file.originalname.replace(/[^\w.-]/g, '_');
    cb(null, `${Date.now()}-${safe}`);
  }
});
const upload = multer({ storage, limits: { fileSize: 20 * 1024 * 1024 } });

// ROUTES
app.get('/input', (req, res) => {
  res.render('input', { title: 'Crowd Count - Input' });
});

app.get('/', (req, res) => {
  res.render('monitor', { title: 'Crowd Count - Monitor' });
});

app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ success: false, error: 'No image uploaded' });

    const buf = fs.readFileSync(req.file.path);
    const base64 = buf.toString('base64');

    const resp = await axios.post(
      ROBOFLOW_ENDPOINT,
      { api_key: ROBOFLOW_API_KEY, inputs: { image: { type: 'base64', value: base64 }, confidence: 0.3} },
      { headers: { 'Content-Type': 'application/json' }, timeout: 30000 }
    );

    const output = resp.data.outputs?.[0] ?? {};
    const count = output.count_objects ?? 0;
    
    // Check threshold for API responses
    if (count >= ALERT_THRESHOLD && Date.now() - lastAlertTime > ALERT_COOLDOWN) {
      lastAlertTime = Date.now();
      console.log(`ALERT: Threshold exceeded (${count} people)`);
    }
    
    // Process images
    const processedImage = await drawDotsOnImage(buf, output.predictions?.predictions || []);
    const heatmapImage = await generateHeatmap(buf, output.predictions?.predictions || []);
    
    res.json({
      success: true,
      originalImage: `/uploads/${req.file.filename}`,
      count: count,
      annotatedImage: processedImage.toString('base64'),
      heatmapImage: heatmapImage.toString('base64'),
      predictions: output.predictions?.predictions || []
    });
  } catch (err) {
    console.error('Predict error:', err?.response?.data || err.message);
    res.status(500).json({ success: false, error: err.message || 'Prediction failed' });
  }
});

// -------------- Socket.IO real-time ----------------

const MAX_CONCURRENT = Number(process.env.MAX_CONCURRENT || 2);
let inFlight = 0;
const queue = [];

async function callRoboflowWithBase64(base64) {
  const body = {
    api_key: ROBOFLOW_API_KEY,
    inputs: { image: { type: 'base64', value: base64 }, confidence: 0.3 }
  };
  const resp = await axios.post(ROBOFLOW_ENDPOINT, body, { 
    headers: { 'Content-Type': 'application/json' }, 
    timeout: 30000 
  });
  return resp.data;
}

async function processFrameAndEmit(clientSocket, frameBase64) {
  if (inFlight >= MAX_CONCURRENT) {
    queue.push({ socket: clientSocket, frameBase64 });
    return;
  }
  inFlight++;
  try {
    const buf = Buffer.from(frameBase64, 'base64');
    const rf = await callRoboflowWithBase64(frameBase64);
    
    const output = rf.outputs?.[0] || {};
    const predictions = output.predictions?.predictions || [];
    const count = output.count_objects || predictions.length;
    
    // Check threshold for real-time alerts
    if (count >= ALERT_THRESHOLD && Date.now() - lastAlertTime > ALERT_COOLDOWN) {
      lastAlertTime = Date.now();
      console.log(`ALERT: Threshold exceeded (${count} people)`);
      io.emit('alert', { count, threshold: ALERT_THRESHOLD });
    }
    
    // Process images
    const processedImage = await drawDotsOnImage(buf, predictions);
    const heatmapImage = await generateHeatmap(buf, predictions);
    
    const payload = {
      success: true,
      count: count,
      annotatedImage: processedImage.toString('base64'),
      heatmapImage: heatmapImage.toString('base64'),
      predictions: predictions,
      threshold: ALERT_THRESHOLD
    };
    
    io.emit('prediction', payload);
  } catch (err) {
    console.error('Roboflow processing error:', err);
    io.emit('prediction', { 
      success: false, 
      error: err.message || 'Inference error' 
    });
  } finally {
    inFlight--;
    if (queue.length > 0 && inFlight < MAX_CONCURRENT) {
      const next = queue.shift();
      processFrameAndEmit(next.socket, next.frameBase64);
    }
  }
}

// Image processing functions
async function drawDotsOnImage(imageBuffer, predictions) {
  try {
    if (!Array.isArray(predictions)) {
      console.warn('Predictions is not an array, using empty array');
      predictions = [];
    }

    const image = await sharp(imageBuffer).toBuffer();
    const img = await loadImage(image);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    
    ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
    predictions.forEach(pred => {
      try {
        const x = pred.x;
        const y = pred.y;
        const dotSize = Math.min(20, Math.max(5, Math.sqrt(pred.width * pred.height) / 5));
        
        if (x && y) {
          ctx.beginPath();
          ctx.arc(x, y, dotSize, 0, Math.PI * 2);
          ctx.fill();
          
          ctx.fillStyle = 'white';
          ctx.font = '12px Arial';
          ctx.fillText(`${Math.round(pred.confidence * 100)}%`, x + dotSize + 2, y);
          ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        }
      } catch (err) {
        console.warn('Error drawing prediction:', err);
      }
    });
    
    return canvas.toBuffer('image/jpeg', { quality: 0.9 });
  } catch (err) {
    console.error('Error in drawDotsOnImage:', err);
    throw err;
  }
}

async function generateHeatmap(imageBuffer, predictions) {
  try {
    if (!Array.isArray(predictions)) {
      predictions = [];
    }

    const image = await sharp(imageBuffer).toBuffer();
    const img = await loadImage(image);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    
    ctx.globalAlpha = 0.7;
    ctx.drawImage(img, 0, 0);
    ctx.globalAlpha = 1.0;
    
    predictions.forEach(pred => {
      if (pred.x && pred.y) {
        const radius = Math.min(img.width, img.height) * 0.1;
        const grd = ctx.createRadialGradient(
          pred.x, pred.y, 0, 
          pred.x, pred.y, radius
        );
        
        grd.addColorStop(0, 'rgba(255, 0, 0, 0.8)');
        grd.addColorStop(0.5, 'rgba(255, 255, 0, 0.4)');
        grd.addColorStop(1, 'rgba(0, 0, 255, 0)');
        
        ctx.fillStyle = grd;
        ctx.beginPath();
        ctx.arc(pred.x, pred.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    
    return canvas.toBuffer('image/jpeg', { quality: 0.9 });
  } catch (err) {
    console.error('Error generating heatmap:', err);
    throw err;
  }
}

// Socket.IO Events
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  // Live frame processing
  socket.on('frame', (data) => {
    if (!data || !data.imageBase64) {
      return socket.emit('prediction', { success: false, error: 'No frame provided' });
    }

    if (queue.length > 60) {
      return socket.emit('prediction', { success: false, error: 'Server busy, frame dropped' });
    }
    
    socket.emit('ack', { received: true });
    processFrameAndEmit(socket, data.imageBase64);
  });

  // Model testing
  socket.on('testModel', async (data) => {
    try {
      if (!data || !data.imageBase64) {
        return socket.emit('testPrediction', { success: false, error: 'No test data provided' });
      }

      const buf = Buffer.from(data.imageBase64, 'base64');
      const rf = await callRoboflowWithBase64(data.imageBase64);
      
      const output = rf.outputs?.[0] || {};
      const predictions = output.predictions?.predictions || [];
      const count = output.count_objects || predictions.length;
      
      // Process images
      const processedImage = await drawDotsOnImage(buf, predictions);
      
      socket.emit('testPrediction', {
        success: true,
        testType: data.type,
        count: count,
        annotatedImage: processedImage.toString('base64'),
        predictions: predictions
      });
    } catch (err) {
      console.error('Test model error:', err);
      socket.emit('testPrediction', {
        success: false,
        error: err.message || 'Test failed'
      });
    }
  });

  // Threshold updates
  socket.on('updateThreshold', (data) => {
    if (data && data.threshold) {
      ALERT_THRESHOLD = parseInt(data.threshold);
      console.log(`Threshold updated to ${ALERT_THRESHOLD}`);
      io.emit('thresholdUpdated', { threshold: ALERT_THRESHOLD });
    }
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Error handling
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
server.listen(PORT, () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
});