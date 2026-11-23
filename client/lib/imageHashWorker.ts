export interface ImageFingerprint {
  sha256: string | null;
  dhash: string | null;
  width: number | null;
  height: number | null;
  computedInWorker: boolean;
}

interface WorkerResponse {
  id: number;
  sha256: string | null;
  dhash: string | null;
  width: number | null;
  height: number | null;
}

const WORKER_SOURCE = `
self.onmessage = async (event) => {
  const { id, buffer, mimeType } = event.data;
  let sha256 = null;
  let dhash = null;
  let width = null;
  let height = null;

  try {
    if (self.crypto && self.crypto.subtle) {
      const digest = await self.crypto.subtle.digest("SHA-256", buffer);
      const bytes = new Uint8Array(digest);
      let hex = "";
      for (let i = 0; i < bytes.length; i++) {
        hex += bytes[i].toString(16).padStart(2, "0");
      }
      sha256 = hex;
    }
  } catch (_) {}

  try {
    if (typeof createImageBitmap === "function" && typeof OffscreenCanvas !== "undefined") {
      const blob = new Blob([buffer], { type: mimeType || "image/jpeg" });
      const bitmap = await createImageBitmap(blob);
      width = bitmap.width;
      height = bitmap.height;

      const canvas = new OffscreenCanvas(9, 8);
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (ctx) {
        ctx.drawImage(bitmap, 0, 0, 9, 8);
        const data = ctx.getImageData(0, 0, 9, 8).data;
        let bits = "";
        for (let y = 0; y < 8; y++) {
          for (let x = 0; x < 8; x++) {
            const idxL = (y * 9 + x) * 4;
            const idxR = (y * 9 + (x + 1)) * 4;
            const lumL = data[idxL] * 0.299 + data[idxL + 1] * 0.587 + data[idxL + 2] * 0.114;
            const lumR = data[idxR] * 0.299 + data[idxR + 1] * 0.587 + data[idxR + 2] * 0.114;
            bits += lumL > lumR ? "1" : "0";
          }
        }
        let hexHash = "";
        for (let i = 0; i < 64; i += 4) {
          hexHash += parseInt(bits.slice(i, i + 4), 2).toString(16);
        }
        dhash = hexHash;
      }
      if (typeof bitmap.close === "function") bitmap.close();
    }
  } catch (_) {}

  self.postMessage({ id, sha256, dhash, width, height });
};
`;

let sharedWorker: Worker | null = null;
let workerUnavailable = false;
let nextRequestId = 1;
const pendingRequests = new Map<number, (res: WorkerResponse) => void>();

function getOrCreateWorker(): Worker | null {
  if (workerUnavailable || typeof Worker === "undefined" || typeof Blob === "undefined") {
    return null;
  }
  if (sharedWorker) return sharedWorker;

  try {
    const blob = new Blob([WORKER_SOURCE], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    const worker = new Worker(url);
    URL.revokeObjectURL(url);

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const payload = event.data;
      const resolver = pendingRequests.get(payload.id);
      if (resolver) {
        pendingRequests.delete(payload.id);
        resolver(payload);
      }
    };
    worker.onerror = () => {
      workerUnavailable = true;
      pendingRequests.forEach((resolve, id) => {
        resolve({ id, sha256: null, dhash: null, width: null, height: null });
      });
      pendingRequests.clear();
      sharedWorker = null;
    };

    sharedWorker = worker;
    return worker;
  } catch {
    workerUnavailable = true;
    return null;
  }
}

async function computeSha256MainThread(buffer: ArrayBuffer): Promise<string | null> {
  try {
    if (typeof crypto === "undefined" || !crypto.subtle) return null;
    const digest = await crypto.subtle.digest("SHA-256", buffer);
    return Array.from(new Uint8Array(digest))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  } catch {
    return null;
  }
}

function computeDhashAndDimensionsMainThread(
  blob: Blob
): Promise<{ dhash: string | null; width: number | null; height: number | null }> {
  if (typeof document === "undefined" || typeof Image === "undefined") {
    return Promise.resolve({ dhash: null, width: null, height: null });
  }
  return new Promise((resolve) => {
    const objectUrl = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      try {
        const width = img.naturalWidth || null;
        const height = img.naturalHeight || null;
        const canvas = document.createElement("canvas");
        canvas.width = 9;
        canvas.height = 8;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) {
          URL.revokeObjectURL(objectUrl);
          resolve({ dhash: null, width, height });
          return;
        }
        ctx.drawImage(img, 0, 0, 9, 8);
        const data = ctx.getImageData(0, 0, 9, 8).data;
        let bits = "";
        for (let y = 0; y < 8; y += 1) {
          for (let x = 0; x < 8; x += 1) {
            const idxL = (y * 9 + x) * 4;
            const idxR = (y * 9 + (x + 1)) * 4;
            const lumL = data[idxL] * 0.299 + data[idxL + 1] * 0.587 + data[idxL + 2] * 0.114;
            const lumR = data[idxR] * 0.299 + data[idxR + 1] * 0.587 + data[idxR + 2] * 0.114;
            bits += lumL > lumR ? "1" : "0";
          }
        }
        let hexHash = "";
        for (let i = 0; i < 64; i += 4) {
          hexHash += parseInt(bits.slice(i, i + 4), 2).toString(16);
        }
        URL.revokeObjectURL(objectUrl);
        resolve({ dhash: hexHash, width, height });
      } catch {
        URL.revokeObjectURL(objectUrl);
        resolve({ dhash: null, width: null, height: null });
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      resolve({ dhash: null, width: null, height: null });
    };
    img.src = objectUrl;
  });
}

export async function computeImageFingerprint(blob: Blob): Promise<ImageFingerprint> {
  const buffer = await blob.arrayBuffer();
  const worker = getOrCreateWorker();

  if (worker) {
    const id = nextRequestId++;
    const workerRes = await new Promise<WorkerResponse>((resolve) => {
      const timeout = setTimeout(() => {
        pendingRequests.delete(id);
        resolve({ id, sha256: null, dhash: null, width: null, height: null });
      }, 10000);

      pendingRequests.set(id, (res) => {
        clearTimeout(timeout);
        resolve(res);
      });

      const copy = buffer.slice(0);
      worker.postMessage({ id, buffer: copy, mimeType: blob.type }, [copy]);
    });

    if (workerRes.sha256 || workerRes.dhash) {
      // Fill dHash/dimensions on main thread only if OffscreenCanvas wasn't supported in worker
      if (!workerRes.dhash) {
        const extra = await computeDhashAndDimensionsMainThread(blob);
        return {
          sha256: workerRes.sha256 ?? (await computeSha256MainThread(buffer)),
          dhash: extra.dhash,
          width: workerRes.width ?? extra.width,
          height: workerRes.height ?? extra.height,
          computedInWorker: true,
        };
      }
      return {
        sha256: workerRes.sha256,
        dhash: workerRes.dhash,
        width: workerRes.width,
        height: workerRes.height,
        computedInWorker: true,
      };
    }
  }

  const [sha256, visual] = await Promise.all([
    computeSha256MainThread(buffer),
    computeDhashAndDimensionsMainThread(blob),
  ]);

  return {
    sha256,
    dhash: visual.dhash,
    width: visual.width,
    height: visual.height,
    computedInWorker: false,
  };
}
