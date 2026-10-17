import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { countBoxesExpressHandler } from './routes/countBoxes.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

const app = express();
// Port must be 3000 for Nginx proxy compatibility
const port = 3000;

app.use(express.json({ limit: '25mb' }));
app.use(express.urlencoded({ extended: true, limit: '25mb' }));

// Set up Multer for handling multipart/form-data memory uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 },
});

// Middleware to catch Multer errors before handler
const uploadMiddleware = (req: express.Request, res: express.Response, next: express.NextFunction) => {
  upload.single('image')(req, res, (err) => {
    if (err) {
      if (err instanceof multer.MulterError) {
        return res.status(400).json({
          success: false,
          error: `Image upload limit exceeded or invalid: ${err.message}`,
        });
      }
      return res.status(400).json({
        success: false,
        error: `Upload processing failed: ${err.message || 'Unknown error'}`,
      });
    }
    next();
  });
};

// Primary Backend API Endpoint for Box Counting & Affine Cropping
app.post('/api/count-boxes', uploadMiddleware, countBoxesExpressHandler);

// Endpoint to check External Model configuration status
app.get('/api/model-status', (_req, res) => {
  const modelUrl = process.env.MODEL_API_URL || 'http://localhost:8001/predict';
  res.json({
    configuredModelUrl: modelUrl,
    status: 'ready',
    pipeline: '2D Affine Inverse Matrix -> Sharp Crop & Resize (400x300) -> External Model',
  });
});

// Setup Vite Dev Server or Production Static Serving
async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(rootDir, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(rootDir, 'dist', 'index.html'));
    });
  }

  // Global Error Handler - Always return JSON, never HTML
  app.use((err: any, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    console.error('Express Error Handler caught:', err);
    res.status(500).json({
      success: false,
      error: err?.message || 'Internal Server Error',
    });
  });

  app.listen(port, '0.0.0.0', () => {
    console.log(`Smart Box Counter server running at http://0.0.0.0:${port}`);
  });
}

startServer();
