export interface StoredPhotoRecord {
  id: string;
  blob: Blob;
  filename: string;
  mimeType: string;
  sizeBytes: number;
  width: number | null;
  height: number | null;
  createdAt: number;
  timestamp: string; // YYYY-MM-DD
  monthKey: string; // YYYY-MM
  yearKey: string; // YYYY
  isFavorite: boolean;
  album: string;
  attributes: string[];
  sha256: string | null;
  dhash: string | null;
  isScreenshot: boolean;
  isUserUpload: true;
}

export interface GalleryPreferences {
  userAlbums: string[];
  sampleFavorites: Record<string, boolean>;
  sampleAlbums: Record<string, string>;
  hiddenSampleIds: string[];
}

export interface SaveVaultResult {
  ok: boolean;
  quotaExceeded?: boolean;
  message?: string;
}

export interface VaultStorageEstimate {
  localBytes: number;
  browserUsageBytes: number | null;
  browserQuotaBytes: number | null;
  supported: boolean;
}

export interface HashablePhotoLike {
  id: string;
  sha256?: string | null;
  dhash?: string | null;
}

export interface DuplicateAnalysis {
  exactPhotoIds: Set<string>;
  similarPhotoIds: Set<string>;
  exactDuplicateExtraCount: number;
  similarPairCount: number;
  exactGroupSizeByPhotoId: Map<string, number>;
  similarMatchesByPhotoId: Map<string, Array<{ id: string; distance: number }>>;
}

export type TimelineGranularity = "day" | "month" | "year";

export interface TimelineGroup<T> {
  key: string;
  label: string;
  items: T[];
}

const DB_NAME = "celeba_photo_vault_v1";
const DB_VERSION = 1;
const STORE_PHOTOS = "photos";
const STORE_PREFS = "preferences";
const PREFS_KEY = "gallery_preferences_v1";

export const DEFAULT_PREFERENCES: GalleryPreferences = {
  userAlbums: [],
  sampleFavorites: {},
  sampleAlbums: {},
  hiddenSampleIds: [],
};

function hasIndexedDb(): boolean {
  return typeof indexedDB !== "undefined";
}

function isQuotaError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { name?: string; code?: number; message?: string };
  return (
    e.name === "QuotaExceededError" ||
    e.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
    e.code === 22 ||
    e.code === 1014 ||
    (typeof e.message === "string" && e.message.toLowerCase().includes("quota"))
  );
}

export function openPhotoVault(): Promise<IDBDatabase> {
  if (!hasIndexedDb()) {
    return Promise.reject(new Error("IndexedDB is unavailable in this browser context."));
  }
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_PHOTOS)) {
        const store = db.createObjectStore(STORE_PHOTOS, { keyPath: "id" });
        store.createIndex("createdAt", "createdAt", { unique: false });
        store.createIndex("album", "album", { unique: false });
        store.createIndex("sha256", "sha256", { unique: false });
      }
      if (!db.objectStoreNames.contains(STORE_PREFS)) {
        db.createObjectStore(STORE_PREFS, { keyPath: "key" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("Failed to open IndexedDB photo vault."));
  });
}

export async function loadAllStoredPhotos(): Promise<StoredPhotoRecord[]> {
  if (!hasIndexedDb()) return [];
  const db = await openPhotoVault();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_PHOTOS, "readonly");
    const store = tx.objectStore(STORE_PHOTOS);
    const req = store.getAll();
    req.onsuccess = () => {
      const records = (req.result as StoredPhotoRecord[] | undefined) ?? [];
      records.sort((a, b) => b.createdAt - a.createdAt);
      resolve(records);
    };
    req.onerror = () => reject(req.error ?? new Error("Failed to read stored photos."));
    tx.oncomplete = () => db.close();
  });
}

export async function savePhotoToVault(record: StoredPhotoRecord): Promise<SaveVaultResult> {
  if (!hasIndexedDb()) {
    return {
      ok: false,
      quotaExceeded: false,
      message: "IndexedDB unavailable; photo kept in active session memory only.",
    };
  }
  try {
    const db = await openPhotoVault();
    return await new Promise<SaveVaultResult>((resolve) => {
      const tx = db.transaction(STORE_PHOTOS, "readwrite");
      const store = tx.objectStore(STORE_PHOTOS);
      store.put(record);
      tx.oncomplete = () => {
        db.close();
        resolve({ ok: true });
      };
      tx.onerror = () => {
        const err = tx.error;
        db.close();
        if (isQuotaError(err)) {
          resolve({
            ok: false,
            quotaExceeded: true,
            message:
              "Local browser storage quota exceeded. Remove unused photos or duplicates to persist new uploads.",
          });
        } else {
          resolve({
            ok: false,
            quotaExceeded: false,
            message: err?.message || "Failed to persist photo to local IndexedDB.",
          });
        }
      };
      tx.onabort = () => {
        const err = tx.error;
        db.close();
        if (isQuotaError(err)) {
          resolve({
            ok: false,
            quotaExceeded: true,
            message:
              "Local browser storage quota exceeded. Remove unused photos or duplicates to persist new uploads.",
          });
        } else {
          resolve({
            ok: false,
            quotaExceeded: false,
            message: err?.message || "Storage transaction aborted.",
          });
        }
      };
    });
  } catch (err) {
    if (isQuotaError(err)) {
      return {
        ok: false,
        quotaExceeded: true,
        message: "Local browser storage quota exceeded.",
      };
    }
    return {
      ok: false,
      quotaExceeded: false,
      message: err instanceof Error ? err.message : "Failed to save photo locally.",
    };
  }
}

export async function updatePhotoMetadata(
  id: string,
  patch: Partial<Pick<StoredPhotoRecord, "isFavorite" | "album" | "attributes" | "filename">>
): Promise<boolean> {
  if (!hasIndexedDb()) return false;
  try {
    const db = await openPhotoVault();
    return await new Promise<boolean>((resolve) => {
      const tx = db.transaction(STORE_PHOTOS, "readwrite");
      const store = tx.objectStore(STORE_PHOTOS);
      const getReq = store.get(id);
      getReq.onsuccess = () => {
        const existing = getReq.result as StoredPhotoRecord | undefined;
        if (!existing) {
          resolve(false);
          return;
        }
        store.put({ ...existing, ...patch });
      };
      tx.oncomplete = () => {
        db.close();
        resolve(true);
      };
      tx.onerror = () => {
        db.close();
        resolve(false);
      };
    });
  } catch {
    return false;
  }
}

export async function deletePhotosFromVault(ids: string[]): Promise<boolean> {
  if (!hasIndexedDb() || ids.length === 0) return false;
  try {
    const db = await openPhotoVault();
    return await new Promise<boolean>((resolve) => {
      const tx = db.transaction(STORE_PHOTOS, "readwrite");
      const store = tx.objectStore(STORE_PHOTOS);
      ids.forEach((id) => store.delete(id));
      tx.oncomplete = () => {
        db.close();
        resolve(true);
      };
      tx.onerror = () => {
        db.close();
        resolve(false);
      };
    });
  } catch {
    return false;
  }
}

export async function loadGalleryPreferences(): Promise<GalleryPreferences> {
  if (!hasIndexedDb()) return { ...DEFAULT_PREFERENCES };
  try {
    const db = await openPhotoVault();
    return await new Promise<GalleryPreferences>((resolve) => {
      const tx = db.transaction(STORE_PREFS, "readonly");
      const store = tx.objectStore(STORE_PREFS);
      const req = store.get(PREFS_KEY);
      req.onsuccess = () => {
        const result = req.result as { key: string; value: GalleryPreferences } | undefined;
        resolve(result?.value ? { ...DEFAULT_PREFERENCES, ...result.value } : { ...DEFAULT_PREFERENCES });
      };
      req.onerror = () => resolve({ ...DEFAULT_PREFERENCES });
      tx.oncomplete = () => db.close();
    });
  } catch {
    return { ...DEFAULT_PREFERENCES };
  }
}

export async function saveGalleryPreferences(prefs: GalleryPreferences): Promise<boolean> {
  if (!hasIndexedDb()) return false;
  try {
    const db = await openPhotoVault();
    return await new Promise<boolean>((resolve) => {
      const tx = db.transaction(STORE_PREFS, "readwrite");
      const store = tx.objectStore(STORE_PREFS);
      store.put({ key: PREFS_KEY, value: prefs });
      tx.oncomplete = () => {
        db.close();
        resolve(true);
      };
      tx.onerror = () => {
        db.close();
        resolve(false);
      };
    });
  } catch {
    return false;
  }
}

export async function estimateVaultStorage(localBytes: number): Promise<VaultStorageEstimate> {
  if (typeof navigator !== "undefined" && navigator.storage && typeof navigator.storage.estimate === "function") {
    try {
      const est = await navigator.storage.estimate();
      return {
        localBytes,
        browserUsageBytes: typeof est.usage === "number" ? est.usage : null,
        browserQuotaBytes: typeof est.quota === "number" ? est.quota : null,
        supported: true,
      };
    } catch {
      return { localBytes, browserUsageBytes: null, browserQuotaBytes: null, supported: false };
    }
  }
  return { localBytes, browserUsageBytes: null, browserQuotaBytes: null, supported: false };
}

/**
 * Pure utilities for hashing comparison, duplicate analysis, deduplication, and timeline grouping.
 */

const POPCOUNT_NIBBLE = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

export function hammingDistanceHex(hexA: string, hexB: string): number {
  if (!hexA || !hexB || hexA.length !== hexB.length) return Number.POSITIVE_INFINITY;
  let dist = 0;
  for (let i = 0; i < hexA.length; i += 1) {
    const a = parseInt(hexA[i], 16);
    const b = parseInt(hexB[i], 16);
    if (Number.isNaN(a) || Number.isNaN(b)) return Number.POSITIVE_INFINITY;
    dist += POPCOUNT_NIBBLE[a ^ b];
  }
  return dist;
}

export function findDuplicateAnalysis(
  items: HashablePhotoLike[],
  maxHammingDistance = 10
): DuplicateAnalysis {
  const exactPhotoIds = new Set<string>();
  const similarPhotoIds = new Set<string>();
  const exactGroupSizeByPhotoId = new Map<string, number>();
  const similarMatchesByPhotoId = new Map<string, Array<{ id: string; distance: number }>>();

  // 1. Exact SHA-256 duplicates
  const bySha = new Map<string, string[]>();
  for (const item of items) {
    if (!item.sha256) continue;
    const list = bySha.get(item.sha256) ?? [];
    list.push(item.id);
    bySha.set(item.sha256, list);
  }

  let exactDuplicateExtraCount = 0;
  bySha.forEach((ids) => {
    if (ids.length > 1) {
      exactDuplicateExtraCount += ids.length - 1;
      ids.forEach((id) => {
        exactPhotoIds.add(id);
        exactGroupSizeByPhotoId.set(id, ids.length);
      });
    }
  });

  // 2. Visually similar perceptual dHash pairs (excluding pairs already sharing the exact SHA-256)
  const hashed = items.filter((i): i is HashablePhotoLike & { dhash: string } => Boolean(i.dhash));
  let similarPairCount = 0;

  for (let i = 0; i < hashed.length; i += 1) {
    const a = hashed[i];
    for (let j = i + 1; j < hashed.length; j += 1) {
      const b = hashed[j];
      if (a.sha256 && b.sha256 && a.sha256 === b.sha256) continue;
      const dist = hammingDistanceHex(a.dhash, b.dhash);
      if (dist <= maxHammingDistance) {
        similarPairCount += 1;
        similarPhotoIds.add(a.id);
        similarPhotoIds.add(b.id);
        const listA = similarMatchesByPhotoId.get(a.id) ?? [];
        listA.push({ id: b.id, distance: dist });
        similarMatchesByPhotoId.set(a.id, listA);

        const listB = similarMatchesByPhotoId.get(b.id) ?? [];
        listB.push({ id: a.id, distance: dist });
        similarMatchesByPhotoId.set(b.id, listB);
      }
    }
  }

  return {
    exactPhotoIds,
    similarPhotoIds,
    exactDuplicateExtraCount,
    similarPairCount,
    exactGroupSizeByPhotoId,
    similarMatchesByPhotoId,
  };
}

export function ensureUniqueFilename(filename: string, existingNames: Iterable<string>): string {
  const clean = filename.trim() || "untitled-photo.png";
  const taken = new Set<string>();
  for (const name of existingNames) {
    taken.add(name.toLowerCase());
  }
  if (!taken.has(clean.toLowerCase())) return clean;

  const dotIdx = clean.lastIndexOf(".");
  const base = dotIdx > 0 ? clean.slice(0, dotIdx) : clean;
  const ext = dotIdx > 0 ? clean.slice(dotIdx) : "";

  let counter = 1;
  while (counter < 10000) {
    const candidate = `${base} (${counter})${ext}`;
    if (!taken.has(candidate.toLowerCase())) return candidate;
    counter += 1;
  }
  return `${base}-${Date.now()}${ext}`;
}

export function detectScreenshotHeuristic(
  filename: string,
  mimeType: string,
  width: number | null,
  height: number | null
): boolean {
  const lower = filename.toLowerCase();
  if (/(screenshot|screen[\s_-]?shot|screen[\s_-]?capture|snip|clean[\s_-]?shot)/.test(lower)) {
    return true;
  }
  if (mimeType === "image/png" && width && height && width >= 800 && height >= 450) {
    const ratio = width / height;
    const isDesktopRatio = ratio >= 1.55 && ratio <= 2.4;
    const isMobileRatio = ratio >= 0.42 && ratio <= 0.62;
    if (isDesktopRatio || isMobileRatio) {
      return true;
    }
  }
  return false;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let val = bytes;
  let idx = 0;
  while (val >= 1024 && idx < units.length - 1) {
    val /= 1024;
    idx += 1;
  }
  return `${val >= 10 || idx === 0 ? val.toFixed(0) : val.toFixed(1)} ${units[idx]}`;
}

export function formatDateHeading(key: string, granularity: TimelineGranularity): string {
  if (granularity === "year") return key;
  if (granularity === "month") {
    const [y, m] = key.split("-");
    const date = new Date(Number(y), Number(m) - 1, 1);
    if (Number.isNaN(date.getTime())) return key;
    return date.toLocaleDateString("en-US", { month: "long", year: "numeric" });
  }
  const [y, m, d] = key.split("-");
  const date = new Date(Number(y), Number(m) - 1, Number(d));
  if (Number.isNaN(date.getTime())) return key;
  return date.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function groupPhotosByTimeline<T extends { timestamp: string }>(
  items: T[],
  granularity: TimelineGranularity
): TimelineGroup<T>[] {
  const map = new Map<string, T[]>();
  for (const item of items) {
    const raw = item.timestamp || "Unknown";
    let key = raw;
    if (granularity === "month" && raw.length >= 7) {
      key = raw.slice(0, 7);
    } else if (granularity === "year" && raw.length >= 4) {
      key = raw.slice(0, 4);
    }
    const group = map.get(key) ?? [];
    group.push(item);
    map.set(key, group);
  }

  return Array.from(map.entries()).map(([key, groupItems]) => ({
    key,
    label: formatDateHeading(key, granularity),
    items: groupItems,
  }));
}

export function dataUrlToBlob(dataUrl: string): Blob {
  const [header, base64] = dataUrl.split(",");
  const mimeMatch = /data:([^;]+);base64/.exec(header || "");
  const mime = mimeMatch ? mimeMatch[1] : "image/jpeg";
  const binary = atob(base64 || "");
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mime });
}
