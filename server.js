/**
 * server.js — SignBridge Backend
 * ─────────────────────────────────────────────────────────────────
 * Express server that:
 *   1. Serves the frontend (public/ folder)
 *   2. Keeps a single Python worker alive (recognize_worker.py)
 *   3. Sends each camera frame to Python via stdin, gets result via stdout
 *   4. Returns gesture + per-hand landmarks to the browser
 *
 * Camera access fix: run on localhost:5000 — browsers allow camera on localhost.
 * DO NOT use file:// protocol to open index.html directly.
 */

const express    = require('express');
const cors       = require('cors');
const bodyParser = require('body-parser');
const { spawn }  = require('child_process');
const path       = require('path');
const { execSync } = require('child_process');

// ── Optional MongoDB ──────────────────────────────────────────────
let mongoose = null;
let SignImage = null;
try {
    mongoose = require('mongoose');
    mongoose.connect('mongodb://localhost:27017/signbridge', {
        useNewUrlParser:    true,
        useUnifiedTopology: true,
        serverSelectionTimeoutMS: 3000,
    }).catch(() => {});
    mongoose.connection.once('open',  () => console.log('✅ MongoDB connected'));
    mongoose.connection.on('error',   () => {});

    const imageSchema = new mongoose.Schema({
        letter:      { type: String, required: true, unique: true },
        imageData:   { type: String, required: true },
        filename:    String,
        contentType: String,
        uploadedAt:  { type: Date, default: Date.now }
    });
    SignImage = mongoose.model('SignImage', imageSchema);
} catch (e) {
    console.log('[server] MongoDB optional — skipping.');
}

const app  = express();
const PORT = process.env.PORT || 5000;

// ── Middleware ────────────────────────────────────────────────────
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ── Auto-detect Python executable ────────────────────────────────
function findPython() {
    // Honour explicit env var first
    if (process.env.PYTHON_PATH) return process.env.PYTHON_PATH;

    const candidates = ['python3', 'python'];
    for (const cmd of candidates) {
        try {
            const out = execSync(`${cmd} --version 2>&1`).toString();
            if (out.toLowerCase().includes('python')) {
                console.log(`[server] Using Python: ${cmd} (${out.trim()})`);
                return cmd;
            }
        } catch (_) {}
    }

    // Windows fallback common paths
    const winPaths = [
        'C:\\Python311\\python.exe',
        'C:\\Python310\\python.exe',
        'C:\\Python39\\python.exe',
        `${process.env.LOCALAPPDATA}\\Programs\\Python\\Python311\\python.exe`,
        `${process.env.LOCALAPPDATA}\\Programs\\Python\\Python310\\python.exe`,
        `${process.env.USERPROFILE}\\AppData\\Local\\Programs\\Python\\Python311\\python.exe`,
    ];
    for (const p of winPaths) {
        try {
            execSync(`"${p}" --version 2>&1`);
            console.log(`[server] Using Python: ${p}`);
            return p;
        } catch (_) {}
    }

    console.warn('[server] WARNING: Could not auto-detect Python. Set PYTHON_PATH env var.');
    return 'python';
}

const PYTHON_EXE = findPython();

// ── Python worker (persistent process) ────────────────────────────
let pyProcess = null;
let pyQueue   = [];
let pyBuffer  = '';

function getPython() {
    if (pyProcess && !pyProcess.killed) return pyProcess;

    console.log('[server] Starting Python worker...');
    const script = path.join(__dirname, 'recognize_worker.py');
    pyProcess = spawn(PYTHON_EXE, [script], { cwd: __dirname });

    pyProcess.stdout.setEncoding('utf8');
    pyProcess.stdout.on('data', chunk => {
        pyBuffer += chunk;
        let nl;
        while ((nl = pyBuffer.indexOf('\n')) !== -1) {
            const line = pyBuffer.slice(0, nl).trim();
            pyBuffer   = pyBuffer.slice(nl + 1);
            if (!line) continue;
            const cb = pyQueue.shift();
            if (!cb)  continue;
            try   { cb.resolve(JSON.parse(line)); }
            catch { cb.reject(new Error('Bad JSON from Python: ' + line)); }
        }
    });

    pyProcess.stderr.on('data', d => process.stderr.write('[python] ' + d));

    pyProcess.on('exit', code => {
        console.warn(`[server] Python worker exited (code ${code}) — will restart on next request`);
        pyProcess = null;
        pyQueue.forEach(cb => cb.reject(new Error('Python worker exited')));
        pyQueue = [];
    });

    return pyProcess;
}

function recognizeFrame(imageDataUrl) {
    return new Promise((resolve, reject) => {
        const py = getPython();
        pyQueue.push({ resolve, reject });
        py.stdin.write(JSON.stringify({ image: imageDataUrl }) + '\n');
    });
}

// ── Routes ────────────────────────────────────────────────────────

app.post('/api/recognize', async (req, res) => {
    const { image } = req.body;
    if (!image) return res.status(400).json({ error: 'No image provided' });

    try {
        const result = await recognizeFrame(image);
        // Tag the source so the frontend's isAi gate allows confirmed gestures
        // to be added to the sentence builder.
        if (!result.source) {
            result.source = result.num_hands > 0 ? 'rule-based' : 'none';
        }
        res.json({ success: true, ...result });
    } catch (err) {
        console.error('[recognize]', err.message);
        res.status(500).json({
            success:    false,
            error:      err.message,
            gesture:    '',
            confidence: 0,
            hands:      [],
            landmarks:  [],
            confirmed:  false,
            num_hands:  0
        });
    }
});

// Health check
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'ok', 
        python: PYTHON_EXE,
        worker: pyProcess && !pyProcess.killed ? 'running' : 'stopped'
    });
});

if (SignImage) {
    app.post('/api/sign-images', async (req, res) => {
        try {
            const { letter, imageData, filename, contentType } = req.body;
            if (await SignImage.findOne({ letter }))
                return res.status(400).json({ error: 'Letter already exists' });
            await new SignImage({ letter, imageData, filename, contentType }).save();
            res.json({ message: `Image for "${letter}" saved` });
        } catch (err) { res.status(500).json({ error: err.message }); }
    });

    app.get('/api/sign-images', async (req, res) => {
        try { res.json(await SignImage.find({}, 'letter filename uploadedAt')); }
        catch (err) { res.status(500).json({ error: err.message }); }
    });
}

// Homepage
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'landing.html'));
});
// App
app.get('/app', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});
// Catch-all
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'landing.html'));
});

// ── Start ────────────────────────────────────────────────────────
app.listen(PORT, () => {
    console.log(`\n🚀 SignBridge running → http://localhost:${PORT}`);
    console.log(`   Camera will work because we're on localhost ✅\n`);
    getPython();
});