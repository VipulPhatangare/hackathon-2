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
const ffmpeg = require('fluent-ffmpeg');
const ffmpegPath = require('ffmpeg-static');
ffmpeg.setFfmpegPath(ffmpegPath);

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
let ALERT_THRESHOLD = 15;
const ALERT_COOLDOWN = 5 * 60 * 1000;
let lastAlertTime = 0;

// Video processing variables
const FRAME_RATE = 25;
const MAX_CONCURRENT = Number(process.env.MAX_CONCURRENT || 2);
let inFlight = 0;
const queue = [];

// views + static
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));

// ensure uploads dir exists
const uploadDir = path.join(__dirname, 'public', 'uploads');
const processedDir = path.join(__dirname, 'public', 'processed');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });
if (!fs.existsSync(processedDir)) fs.mkdirSync(processedDir, { recursive: true });

// multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const safe = file.originalname.replace(/[^\w.-]/g, '_');
    cb(null, `${Date.now()}-${safe}`);
  }
});
const upload = multer({ storage, limits: { fileSize: 50 * 1024 * 1024 } });

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
      { api_key: ROBOFLOW_API_KEY, inputs: { image: { type: 'base64', value: base64 }, confidence: 0.33 }},
      { headers: { 'Content-Type': 'application/json' }, timeout: 30000 }
    );

    const output = resp.data.outputs?.[0] ?? {};
    const count = output.count_objects ?? 0;
    
    // Check threshold for API responses
    if (count >= ALERT_THRESHOLD && Date.now() - lastAlertTime > ALERT_COOLDOWN) {
      lastAlertTime = Date.now();
      console.log(`ALERT: Threshold exceeded (${count} people)`);
      io.emit('alert', { count, threshold: ALERT_THRESHOLD });
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

// Video processing endpoint
app.post('/process-video', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ success: false, error: 'No video uploaded' });

    const videoPath = req.file.path;
    const outputVideoPath = path.join(processedDir, `processed_${req.file.filename}`);
    
    // Process video frames
    const frameCounts = await processVideoFrames(videoPath);
    
    // Reconstruct video with annotations (simplified - in production you'd use ffmpeg)
    // For demo purposes, we'll just return the original video path
    // In a real implementation, you would:
    // 1. Combine all processed frames into a new video
    // 2. Return the path to the processed video
    
    res.json({
      success: true,
      videoUrl: `/uploads/${req.file.filename}`,
      processedVideoUrl: `/processed/processed_${req.file.filename}`,
      frameCounts: frameCounts
    });
  } catch (err) {
    console.error('Video processing error:', err);
    res.status(500).json({ success: false, error: err.message || 'Video processing failed' });
  }
});

// Helper Functions
async function callRoboflowWithBase64(base64) {
  const body = {
    api_key: ROBOFLOW_API_KEY,
    inputs: { image: { type: 'base64', value: base64 }, confidence: 0.33 }
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
    
    // Draw count at top left
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(10, 10, 150, 60);
    ctx.fillStyle = 'white';
    ctx.font = 'bold 24px Arial';
    ctx.fillText(`People: ${predictions.length}`, 20, 40);
    
    // Draw dots for each prediction
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

async function processVideoFrames(videoPath) {
  return new Promise((resolve, reject) => {
    const frameCounts = [];
    const tempDir = path.join(__dirname, 'temp_frames');
    if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir);
    
    // Extract frames from video
    ffmpeg(videoPath)
      .on('error', reject)
      .on('end', async () => {
        try {
          // Process each frame
          const frameFiles = fs.readdirSync(tempDir)
            .filter(file => file.endsWith('.jpg'))
            .sort((a, b) => parseInt(a.split('-')[1]) - parseInt(b.split('-')[1]));
          
          for (const file of frameFiles) {
            const framePath = path.join(tempDir, file);
            const buf = fs.readFileSync(framePath);
            const base64 = buf.toString('base64');
            
            try {
              const rf = await callRoboflowWithBase64(base64);
              const count = rf.outputs?.[0]?.count_objects || 0;
              frameCounts.push(count);
              
              // Emit progress to client
              io.emit('videoFrameProcessed', {
                frameIndex: frameCounts.length - 1,
                count: count,
                totalFrames: frameFiles.length
              });
              
              // Clean up frame file
              fs.unlinkSync(framePath);
            } catch (err) {
              console.error('Frame processing error:', err);
              frameCounts.push(0);
            }
          }
          
          // Clean up temp directory
          fs.rmdirSync(tempDir);
          resolve(frameCounts);
        } catch (err) {
          reject(err);
        }
      })
      .output(path.join(tempDir, 'frame-%04d.jpg'))
      .fps(FRAME_RATE)
      .run();
  });
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

  // Video frame processing
  socket.on('processVideoFrame', async (data) => {
    try {
      if (!data || !data.frameBase64) {
        return socket.emit('processedVideoFrame', { success: false, error: 'No frame data provided' });
      }

      const buf = Buffer.from(data.frameBase64, 'base64');
      const rf = await callRoboflowWithBase64(data.frameBase64);
      
      const output = rf.outputs?.[0] || {};
      const count = output.count_objects || 0;
      
      // Process image (optional - can skip to save processing time)
      const processedImage = await drawDotsOnImage(buf, output.predictions?.predictions || []);
      
      socket.emit('processedVideoFrame', {
        success: true,
        frameIndex: data.frameIndex,
        count: count,
        totalFrames: data.totalFrames,
        annotatedImage: processedImage.toString('base64')
      });
    } catch (err) {
      console.error('Video frame processing error:', err);
      socket.emit('processedVideoFrame', {
        success: false,
        frameIndex: data.frameIndex,
        error: err.message || 'Frame processing failed'
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