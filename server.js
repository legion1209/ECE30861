const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8000;
const STATIC_DIR = path.join(__dirname, 'public');

// --- Create the 'public' directory if it doesn't exist ---
if (!fs.existsSync(STATIC_DIR)) {
    fs.mkdirSync(STATIC_DIR);
    console.log(`Created directory: ${STATIC_DIR}`);
}
// --- Create a default index.html if it doesn't exist ---
const INDEX_FILE = path.join(STATIC_DIR, 'index.html');
if (!fs.existsSync(INDEX_FILE)) {
    fs.writeFileSync(INDEX_FILE, 
        `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Node.js File Server</title>
</head>
<body>
    <h1>Success!</h1>
    <p>This is the default <strong>index.html</strong> served from the 'public' directory.</p>
    <p>Try accessing a file like <code>/index.html</code></p>
</body>
</html>`);
    console.log('Created default public/index.html');
}

// Map file extensions to MIME types
const MIME_TYPES = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.ico': 'image/x-icon',
    '.svg': 'image/svg+xml',
    '.txt': 'text/plain'
};

const server = http.createServer((req, res) => {
    // 1. Sanitize the requested path (prevents path traversal)
    let filePath = path.join(STATIC_DIR, req.url);
    
    // Check for the root path and append index.html
    if (filePath === STATIC_DIR || filePath.endsWith(path.sep)) {
        filePath = path.join(filePath, 'index.html');
    }

    // 2. Check if the path is trying to escape the public directory
    if (!filePath.startsWith(STATIC_DIR)) {
        res.writeHead(403); // Forbidden
        res.end('403: Forbidden - Path Traversal Attempt');
        console.log(`403 ${req.url}`);
        return;
    }

    // 3. Read and serve the file
    fs.readFile(filePath, (err, content) => {
        if (err) {
            if (err.code === 'ENOENT') {
                // File not found (404)
                res.writeHead(404, { 'Content-Type': 'text/plain' });
                res.end('404: File Not Found');
                console.log(`404 ${req.url}`);
            } else {
                // Server error (500)
                res.writeHead(500);
                res.end(`500: Server Error: ${err.code}`);
                console.log(`500 ${req.url}`);
            }
        } else {
            // Success (200)
            const ext = path.extname(filePath).toLowerCase();
            const contentType = MIME_TYPES[ext] || 'application/octet-stream';
            
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
            console.log(`200 ${req.url}`);
        }
    });
});

server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}/`);
    console.log(`Serving files from: ${STATIC_DIR}`);
});