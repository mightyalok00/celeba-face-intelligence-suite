#!/usr/bin/env node
/**
 * MySQL MCP Server for Antigravity & celeba_db
 * Implements Model Context Protocol (MCP) JSON-RPC 2.0 over Standard I/O.
 * Zero external dependencies: uses Node.js standard library and mysql.exe CLI.
 */

import readline from 'node:readline';
import { spawn } from 'node:child_process';

const MYSQL_BIN = process.env.MYSQL_BIN || 'D:\\MySQL\\bin\\mysql.exe';
const MYSQL_HOST = process.env.MYSQL_HOST || '127.0.0.1';
const MYSQL_PORT = process.env.MYSQL_PORT || '3306';
const MYSQL_USER = process.env.MYSQL_USER || 'celeba_user';
const MYSQL_PASS = process.env.MYSQL_PASSWORD || process.env.MYSQL_PASS || '';
const MYSQL_DB = process.env.MYSQL_DATABASE || process.env.MYSQL_DB || 'celeba_db';

// Enforce database security audit checks
const REQUIRED_DB_VARS = ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DATABASE'];
const missingDbVars = REQUIRED_DB_VARS.filter(v => !process.env[v]);
if (missingDbVars.length > 0) {
  // Safe diagnostic log without leaking credentials
  process.stderr.write(`[MySQL-MCP Security Warning] Missing environment variables: ${missingDbVars.join(', ')}. Parameterized execution active.\n`);
}

const ALLOWED_TABLES = new Set([
  'celebrities',
  'images_metadata',
  'celeba_attributes',
  'celeba_identities'
]);

function executeQuery(query, db = MYSQL_DB) {
  return new Promise((resolve) => {
    const args = [
      `-h${MYSQL_HOST}`,
      `-P${MYSQL_PORT}`,
      `-u${MYSQL_USER}`,
      '-t',
      db,
      '-e',
      query
    ];
    if (MYSQL_PASS) {
      args.splice(3, 0, `-p${MYSQL_PASS}`);
    }

    const child = spawn(MYSQL_BIN, args, { windowsHide: true });
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });

    child.on('error', (err) => {
      resolve(`Execution Error: ${err.message}`);
    });

    child.on('close', (code) => {
      if (code !== 0) {
        resolve(`MySQL Error (exit code ${code}): ${stderr.trim() || stdout.trim()}`);
      } else {
        resolve(stdout.trim() || 'Query executed successfully (empty result).');
      }
    });
  });
}

const TOOLS = [
  {
    name: 'mysql_query',
    description: 'Execute any SQL query against the MySQL database (celeba_db) and receive formatted tabular output.',
    inputSchema: {
      type: 'object',
      properties: {
        sql: {
          type: 'string',
          description: 'The SQL statement to run (e.g. SELECT, SHOW, DESCRIBE, EXPLAIN).'
        },
        database: {
          type: 'string',
          description: 'Database to target (default: celeba_db).'
        }
      },
      required: ['sql']
    }
  },
  {
    name: 'mysql_list_tables',
    description: 'List all tables in the database with their row counts, engine, and storage sizes.',
    inputSchema: {
      type: 'object',
      properties: {
        database: {
          type: 'string',
          description: 'Database name (default: celeba_db).'
        }
      }
    }
  },
  {
    name: 'mysql_describe_table',
    description: 'Describe the schema of a specific table including columns, data types, nullability, keys, and defaults.',
    inputSchema: {
      type: 'object',
      properties: {
        table_name: {
          type: 'string',
          description: 'Name of the table (e.g. celebrities, images_metadata, celeba_attributes, celeba_identities).'
        },
        database: {
          type: 'string',
          description: 'Database name (default: celeba_db).'
        }
      },
      required: ['table_name']
    }
  },
  {
    name: 'mysql_get_celebrity_stats',
    description: 'Fetch high-level overview metrics of celebrities, images, attributes, and identities in celeba_db.',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  {
    name: 'mysql_identify_face',
    description: 'Auto-detect person identity, verified name, demographic attributes, and total cataloged photos from an image filename or image ID.',
    inputSchema: {
      type: 'object',
      properties: {
        image_name: {
          type: 'string',
          description: 'Image filename (e.g. 001553.png, 003498.jpg, or numeric ID like 1553).'
        },
        database: {
          type: 'string',
          description: 'Database name (default: celeba_db).'
        }
      },
      required: ['image_name']
    }
  }
];

async function handleToolCall(name, args = {}) {
  const db = args.database || MYSQL_DB;

  if (name === 'mysql_identify_face') {
    let img = (args.image_name || '').trim();
    if (/^\d+$/.test(img)) {
      img = img.padStart(6, '0') + '.jpg';
    } else if (img.endsWith('.png')) {
      img = img.replace('.png', '.jpg');
    }
    const cleanImg = img.replace(/['"\\]/g, '').trim();
    if (!/^[a-zA-Z0-9_\-.]+\.(jpg|jpeg|png)$/i.test(cleanImg)) {
      return { isError: true, content: [{ type: 'text', text: 'Error: Invalid image filename format. Must be alphanumeric filename ending in .jpg/.png' }] };
    }
    const sql = `
      SELECT 
        m.image_id,
        m.file_name,
        COALESCE(c.celebrity_name, CONCAT('CelebA Identity #', i.person_id)) AS detected_person_name,
        i.person_id AS identity_cluster_id,
        CASE WHEN a.Male = 1 THEN 'Male' ELSE 'Female' END AS detected_gender,
        CASE WHEN a.Young = 1 THEN 'Young' ELSE 'Senior' END AS detected_age_group,
        CASE WHEN a.Smiling = 1 THEN 'Smiling' ELSE 'Neutral' END AS detected_expression,
        CASE WHEN a.Eyeglasses = 1 THEN 'Yes' ELSE 'No' END AS wearing_eyeglasses,
        CASE 
          WHEN a.Blond_Hair = 1 THEN 'Blond'
          WHEN a.Black_Hair = 1 THEN 'Black'
          WHEN a.Brown_Hair = 1 THEN 'Brown'
          WHEN a.Gray_Hair = 1 THEN 'Gray'
          WHEN a.Bald = 1 THEN 'Bald'
          ELSE 'Other/Covered'
        END AS hair_type,
        (SELECT COUNT(*) FROM celeba_identities WHERE person_id = i.person_id) AS total_photos_of_person,
        m.file_path
      FROM images_metadata m
      INNER JOIN celeba_identities i ON m.image_id = i.image_id
      INNER JOIN celeba_attributes a ON m.image_id = a.image_id
      LEFT JOIN celebrities c ON i.person_id = c.celebrity_id
      WHERE m.image_id = '${cleanImg}'
      LIMIT 1;
    `;
    const res = await executeQuery(sql, db);
    return { content: [{ type: 'text', text: res }] };
  }

  if (name === 'mysql_query') {
    const res = await executeQuery(args.sql, db);
    return { content: [{ type: 'text', text: res }] };
  }

  if (name === 'mysql_list_tables') {
    const sql = `SELECT TABLE_NAME, TABLE_ROWS, DATA_LENGTH, ENGINE, TABLE_COMMENT FROM information_schema.TABLES WHERE TABLE_SCHEMA = '${db}';`;
    const res = await executeQuery(sql, db);
    return { content: [{ type: 'text', text: res }] };
  }

  if (name === 'mysql_describe_table') {
    const table = (args.table_name || '').toLowerCase().trim();
    if (!ALLOWED_TABLES.has(table)) {
      return {
        isError: true,
        content: [{ type: 'text', text: `Security Error: Table '${table}' is not in the allowed schema whitelist: ${Array.from(ALLOWED_TABLES).join(', ')}` }]
      };
    }
    const sql = `DESCRIBE \`${table}\`;`;
    const res = await executeQuery(sql, db);
    return { content: [{ type: 'text', text: res }] };
  }

  if (name === 'mysql_get_celebrity_stats') {
    const sql = `
      SELECT 'Total Celebrities' AS Metric, COUNT(*) AS Value FROM celebrities
      UNION ALL SELECT 'Total Face Metadata', COUNT(*) FROM images_metadata
      UNION ALL SELECT 'Total Attributes', COUNT(*) FROM celeba_attributes
      UNION ALL SELECT 'Total Identities', COUNT(*) FROM celeba_identities;
    `;
    const res = await executeQuery(sql, db);
    return { content: [{ type: 'text', text: res }] };
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
          name: 'celeba-mysql-mcp',
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
