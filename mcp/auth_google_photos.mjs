#!/usr/bin/env node
/**
 * Interactive Google Photos OAuth2 Authentication Helper
 * Opens the Google consent page in your browser, receives the authorization code,
 * retrieves the refresh token, and updates mcp_config.json automatically.
 */

import http from 'node:http';
import https from 'node:https';
import fs from 'node:fs';
import { exec } from 'node:child_process';
import readline from 'node:readline';

const CONFIG_PATH = 'C:\\Users\\Alok Agarwal\\.gemini\\config\\mcp_config.json';
const REDIRECT_PORT = 8085;
const REDIRECT_URI = `http://localhost:${REDIRECT_PORT}/oauth2callback`;
const SCOPES = [
  'https://www.googleapis.com/auth/photoslibrary',
  'https://www.googleapis.com/auth/photoslibrary.sharing',
  'https://www.googleapis.com/auth/photoslibrary.appendonly'
].join(' ');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function ask(query) {
  return new Promise((resolve) => rl.question(query, resolve));
}

function exchangeCodeForTokens(code, clientId, clientSecret) {
  return new Promise((resolve, reject) => {
    const postData = new URLSearchParams({
      code,
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uri: REDIRECT_URI,
      grant_type: 'authorization_code'
    }).toString();

    const req = https.request({
      hostname: 'oauth2.googleapis.com',
      path: '/token',
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Content-Length': Buffer.byteLength(postData)
      }
    }, (res) => {
      let data = '';
      res.on('data', chunk => { data += chunk; });
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (parsed.error) {
            reject(new Error(parsed.error_description || parsed.error));
          } else {
            resolve(parsed);
          }
        } catch (e) {
          reject(e);
        }
      });
    });

    req.on('error', reject);
    req.write(postData);
    req.end();
  });
}

function updateMcpConfig(clientId, clientSecret, refreshToken, accessToken) {
  if (!fs.existsSync(CONFIG_PATH)) {
    console.error(`mcp_config.json not found at ${CONFIG_PATH}`);
    return false;
  }
  try {
    const raw = fs.readFileSync(CONFIG_PATH, 'utf-8');
    const config = JSON.parse(raw);
    if (!config.mcpServers) config.mcpServers = {};
    if (!config.mcpServers['google-photos']) {
      config.mcpServers['google-photos'] = {
        command: 'node',
        args: ['D:\\scan\\mcp\\google_photos_mcp_server.mjs'],
        env: {}
      };
    }
    config.mcpServers['google-photos'].env = {
      GOOGLE_PHOTOS_CLIENT_ID: clientId,
      GOOGLE_PHOTOS_CLIENT_SECRET: clientSecret,
      GOOGLE_PHOTOS_REFRESH_TOKEN: refreshToken || '',
      GOOGLE_PHOTOS_ACCESS_TOKEN: accessToken || ''
    };
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2), 'utf-8');
    console.log(`\nSuccessfully updated ${CONFIG_PATH}!`);
    return true;
  } catch (err) {
    console.error(`Failed to update config: ${err.message}`);
    return false;
  }
}

async function main() {
  console.log('================================================================');
  console.log('       Google Photos OAuth2 Authentication Helper');
  console.log('================================================================\n');

  // Check existing config
  let defaultClientId = '';
  let defaultClientSecret = '';
  if (fs.existsSync(CONFIG_PATH)) {
    try {
      const cfg = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8'));
      const env = cfg?.mcpServers?.['google-photos']?.env || {};
      defaultClientId = env.GOOGLE_PHOTOS_CLIENT_ID || '';
      defaultClientSecret = env.GOOGLE_PHOTOS_CLIENT_SECRET || '';
    } catch {}
  }

  const clientId = (await ask(`Enter Google OAuth Client ID [${defaultClientId || 'none'}]: `)).trim() || defaultClientId;
  const clientSecret = (await ask(`Enter Google OAuth Client Secret [${defaultClientSecret || 'none'}]: `)).trim() || defaultClientSecret;

  if (!clientId || !clientSecret) {
    console.error('\nError: Both Client ID and Client Secret are required.');
    console.log('Obtain them from: https://console.cloud.google.com/apis/credentials');
    rl.close();
    process.exit(1);
  }

  const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?` + new URLSearchParams({
    client_id: clientId,
    redirect_uri: REDIRECT_URI,
    response_type: 'code',
    scope: SCOPES,
    access_type: 'offline',
    prompt: 'consent'
  }).toString();

  // Create temporary local HTTP server to receive OAuth callback
  const server = http.createServer(async (req, res) => {
    const reqUrl = new URL(req.url, `http://localhost:${REDIRECT_PORT}`);
    if (reqUrl.pathname === '/oauth2callback') {
      const code = reqUrl.searchParams.get('code');
      const error = reqUrl.searchParams.get('error');

      if (error) {
        res.writeHead(400, { 'Content-Type': 'text/html' });
        res.end(`<h2>Authentication Failed</h2><p>${error}</p>`);
        console.error(`\nOAuth authorization failed: ${error}`);
        server.close();
        rl.close();
        return;
      }

      if (code) {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(`
          <div style="font-family: sans-serif; text-align: center; padding: 40px;">
            <h1 style="color: #1a73e8;">Authentication Successful!</h1>
            <p>You can close this tab and return to your terminal.</p>
          </div>
        `);

        try {
          console.log('\nReceived authorization code. Exchanging for tokens...');
          const tokens = await exchangeCodeForTokens(code, clientId, clientSecret);
          console.log('Access Token acquired!');
          if (tokens.refresh_token) {
            console.log('Refresh Token acquired!');
          }

          updateMcpConfig(clientId, clientSecret, tokens.refresh_token, tokens.access_token);
          console.log('\nGoogle Photos MCP server is now fully authenticated and ready to use!');
        } catch (tokenErr) {
          console.error(`\nError exchanging tokens: ${tokenErr.message}`);
        } finally {
          server.close();
          rl.close();
        }
      }
    }
  });

  server.listen(REDIRECT_PORT, () => {
    console.log(`\n1. Listening for callback on ${REDIRECT_URI}`);
    console.log('2. Opening Google sign-in page in your browser...');
    console.log(`\nIf the browser does not open automatically, visit this URL:\n${authUrl}\n`);
    exec(`start "" "${authUrl}"`);
  });
}

main().catch((e) => {
  console.error('Unexpected error:', e);
  rl.close();
});
