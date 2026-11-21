#!/usr/bin/env node
/**
 * Google Photos API MCP Server for Antigravity
 * Implements Model Context Protocol (MCP) JSON-RPC 2.0 over Standard I/O.
 * Provides tools to list albums, search media, upload photos, and create albums in Google Photos.
 * Zero external dependencies: uses Node.js standard library (node:https, node:readline, node:fs).
 */

import readline from 'node:readline';
import https from 'node:https';
import fs from 'node:fs';
import path from 'node:path';

const CLIENT_ID = process.env.GOOGLE_PHOTOS_CLIENT_ID || '';
const CLIENT_SECRET = process.env.GOOGLE_PHOTOS_CLIENT_SECRET || '';
const REFRESH_TOKEN = process.env.GOOGLE_PHOTOS_REFRESH_TOKEN || '';
let ACCESS_TOKEN = process.env.GOOGLE_PHOTOS_ACCESS_TOKEN || '';

const API_BASE = 'photoslibrary.googleapis.com';

function httpRequest(options, postData = null, headers = {}) {
  return new Promise((resolve, reject) => {
    const reqOptions = {
      hostname: options.hostname || API_BASE,
      port: 443,
      path: options.path,
      method: options.method || 'GET',
      headers: {
        'User-Agent': 'Antigravity-GooglePhotos-MCP/1.0',
        ...headers
      }
    };

    if (ACCESS_TOKEN && !headers['Authorization']) {
      reqOptions.headers['Authorization'] = `Bearer ${ACCESS_TOKEN}`;
    }

    const req = https.request(reqOptions, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try {
          const parsed = data ? JSON.parse(data) : {};
          resolve({ status: res.statusCode, headers: res.headers, data: parsed, raw: data });
        } catch {
          resolve({ status: res.statusCode, headers: res.headers, data: null, raw: data });
        }
      });
    });

    req.on('error', (err) => {
      reject(err);
    });

    if (postData) {
      if (Buffer.isBuffer(postData)) {
        req.write(postData);
      } else if (typeof postData === 'object') {
        req.write(JSON.stringify(postData));
      } else {
        req.write(postData);
      }
    }
    req.end();
  });
}

async function refreshAccessToken() {
  if (!REFRESH_TOKEN || !CLIENT_ID || !CLIENT_SECRET) {
    return false;
  }
  try {
    const postData = new URLSearchParams({
      client_id: CLIENT_ID,
      client_secret: CLIENT_SECRET,
      refresh_token: REFRESH_TOKEN,
      grant_type: 'refresh_token'
    }).toString();

    const res = await httpRequest({
      hostname: 'oauth2.googleapis.com',
      path: '/token',
      method: 'POST'
    }, postData, {
      'Content-Type': 'application/x-www-form-urlencoded'
    });

    if (res.status === 200 && res.data?.access_token) {
      ACCESS_TOKEN = res.data.access_token;
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

const TOOLS = [
  {
    name: 'google_photos_status',
    description: 'Check Google Photos API authentication status and configuration.',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  {
    name: 'google_photos_list_albums',
    description: 'List albums in your Google Photos library with album title, item count, and cover photo URL.',
    inputSchema: {
      type: 'object',
      properties: {
        pageSize: {
          type: 'number',
          description: 'Maximum number of albums to return (1-50, default: 20).'
        },
        pageToken: {
          type: 'string',
          description: 'Optional pagination token from a previous call.'
        }
      }
    }
  },
  {
    name: 'google_photos_search_media',
    description: 'Search media items (photos/videos) in Google Photos by album ID or date filters.',
    inputSchema: {
      type: 'object',
      properties: {
        albumId: {
          type: 'string',
          description: 'Specific Google Photos album ID to search within.'
        },
        pageSize: {
          type: 'number',
          description: 'Maximum number of media items to return (1-100, default: 25).'
        },
        pageToken: {
          type: 'string',
          description: 'Pagination token.'
        }
      }
    }
  },
  {
    name: 'google_photos_create_album',
    description: 'Create a new album in Google Photos (e.g. for a celebrity or face cluster).',
    inputSchema: {
      type: 'object',
      properties: {
        title: {
          type: 'string',
          description: 'Title of the new album (e.g. "CelebA - Keanu Reeves").'
        }
      },
      required: ['title']
    }
  },
  {
    name: 'google_photos_upload_image',
    description: 'Upload a local image file (e.g. from D:\\scan\\img_align_celeba) to Google Photos library.',
    inputSchema: {
      type: 'object',
      properties: {
        filePath: {
          type: 'string',
          description: 'Absolute local path to the image file (e.g. D:\\scan\\img_align_celeba\\001553.png).'
        },
        albumId: {
          type: 'string',
          description: 'Optional album ID to add the uploaded photo into.'
        },
        description: {
          type: 'string',
          description: 'Optional caption or description for the photo.'
        }
      },
      required: ['filePath']
    }
  }
];

async function handleToolCall(name, args = {}) {
  if (name === 'google_photos_status') {
    const isConfigured = Boolean(ACCESS_TOKEN || (CLIENT_ID && REFRESH_TOKEN));
    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          configured: isConfigured,
          authenticated: Boolean(ACCESS_TOKEN),
          clientIdSet: Boolean(CLIENT_ID),
          clientSecretSet: Boolean(CLIENT_SECRET),
          refreshTokenSet: Boolean(REFRESH_TOKEN),
          instructions: isConfigured
            ? 'Google Photos API credentials loaded.'
            : 'To connect your Google Photos account, set GOOGLE_PHOTOS_CLIENT_ID, GOOGLE_PHOTOS_CLIENT_SECRET, and GOOGLE_PHOTOS_REFRESH_TOKEN (or GOOGLE_PHOTOS_ACCESS_TOKEN) in mcp_config.json.'
        }, null, 2)
      }]
    };
  }

  // Ensure valid access token
  if (!ACCESS_TOKEN && REFRESH_TOKEN) {
    await refreshAccessToken();
  }

  if (!ACCESS_TOKEN) {
    return {
      isError: true,
      content: [{
        type: 'text',
        text: 'Google Photos API access token not configured. Please supply GOOGLE_PHOTOS_ACCESS_TOKEN or (CLIENT_ID + CLIENT_SECRET + REFRESH_TOKEN) in mcp_config.json.'
      }]
    };
  }

  try {
    if (name === 'google_photos_list_albums') {
      const pageSize = Math.min(args.pageSize || 20, 50);
      let queryPath = `/v1/albums?pageSize=${pageSize}`;
      if (args.pageToken) queryPath += `&pageToken=${encodeURIComponent(args.pageToken)}`;

      const res = await httpRequest({ path: queryPath, method: 'GET' });
      return { content: [{ type: 'text', text: JSON.stringify(res.data, null, 2) }] };
    }

    if (name === 'google_photos_search_media') {
      const payload = {
        pageSize: Math.min(args.pageSize || 25, 100)
      };
      if (args.albumId) payload.albumId = args.albumId;
      if (args.pageToken) payload.pageToken = args.pageToken;

      const res = await httpRequest({ path: '/v1/mediaItems:search', method: 'POST' }, payload, {
        'Content-Type': 'application/json'
      });
      return { content: [{ type: 'text', text: JSON.stringify(res.data, null, 2) }] };
    }

    if (name === 'google_photos_create_album') {
      const payload = {
        album: { title: args.title }
      };
      const res = await httpRequest({ path: '/v1/albums', method: 'POST' }, payload, {
        'Content-Type': 'application/json'
      });
      return { content: [{ type: 'text', text: JSON.stringify(res.data, null, 2) }] };
    }

    if (name === 'google_photos_upload_image') {
      const resolvedPath = path.resolve(args.filePath);
      if (!fs.existsSync(resolvedPath)) {
        return { isError: true, content: [{ type: 'text', text: `File not found: ${resolvedPath}` }] };
      }

      const fileBuffer = fs.readFileSync(resolvedPath);
      const fileName = path.basename(resolvedPath);

      // Step 1: Upload binary bytes to get an upload token
      const uploadRes = await httpRequest({
        path: '/v1/uploads',
        method: 'POST'
      }, fileBuffer, {
        'Content-type': 'application/octet-stream',
        'X-Goog-Upload-Content-Type': 'image/jpeg',
        'X-Goog-Upload-Protocol': 'raw'
      });

      const uploadToken = uploadRes.raw?.trim();
      if (!uploadToken || uploadRes.status !== 200) {
        return { isError: true, content: [{ type: 'text', text: `Failed to upload bytes: ${uploadRes.raw}` }] };
      }

      // Step 2: Batch create media item from upload token
      const createPayload = {
        newMediaItems: [
          {
            description: args.description || `CelebA Face: ${fileName}`,
            simpleMediaItem: {
              uploadToken: uploadToken,
              fileName: fileName
            }
          }
        ]
      };
      if (args.albumId) createPayload.albumId = args.albumId;

      const batchRes = await httpRequest({
        path: '/v1/mediaItems:batchCreate',
        method: 'POST'
      }, createPayload, {
        'Content-Type': 'application/json'
      });

      return { content: [{ type: 'text', text: JSON.stringify(batchRes.data, null, 2) }] };
    }
  } catch (err) {
    return { isError: true, content: [{ type: 'text', text: `Google Photos API Error: ${err.message}` }] };
  }

  return { isError: true, content: [{ type: 'text', text: `Unknown tool: ${name}` }] };
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on('line', async (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;

  let req;
  try {
    req = JSON.parse(trimmed);
  } catch {
    return;
  }

  const { id, method, params } = req;

  if (method === 'initialize') {
    const resp = {
      jsonrpc: '2.0',
      id,
      result: {
        protocolVersion: '2024-11-05',
        capabilities: { tools: {} },
        serverInfo: {
          name: 'google-photos-mcp',
          version: '1.0.0'
        }
      }
    };
    process.stdout.write(JSON.stringify(resp) + '\n');
    return;
  }

  if (method === 'notifications/initialized') {
    return;
  }

  if (method === 'tools/list') {
    const resp = {
      jsonrpc: '2.0',
      id,
      result: { tools: TOOLS }
    };
    process.stdout.write(JSON.stringify(resp) + '\n');
    return;
  }

  if (method === 'tools/call') {
    const result = await handleToolCall(params?.name, params?.arguments);
    const resp = {
      jsonrpc: '2.0',
      id,
      result
    };
    process.stdout.write(JSON.stringify(resp) + '\n');
    return;
  }

  if (method === 'ping') {
    const resp = { jsonrpc: '2.0', id, result: {} };
    process.stdout.write(JSON.stringify(resp) + '\n');
    return;
  }

  if (id !== undefined && id !== null) {
    const resp = {
      jsonrpc: '2.0',
      id,
      error: { code: -32601, message: `Method not found: ${method}` }
    };
    process.stdout.write(JSON.stringify(resp) + '\n');
  }
});
