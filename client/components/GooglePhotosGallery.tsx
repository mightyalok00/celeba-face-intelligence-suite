import {
  memo,
  useEffect,
  useState,
  useMemo,
  useRef,
  useCallback,
  type DragEvent,
  type ChangeEvent,
  type MouseEvent,
  type TouchEvent,
} from "react";
import {
  Search,
  Upload,
  Heart,
  FolderPlus,
  CheckSquare,
  Square,
  Trash2,
  Download,
  ZoomIn,
  ZoomOut,
  ChevronLeft,
  ChevronRight,
  X,
  Sparkles,
  Grid,
  Calendar,
  Edit3,
  Clock,
  Monitor,
  Copy,
  HardDrive,
  FolderInput,
  AlertTriangle,
  Lock,
  Layers,
  ShieldCheck,
  Loader2,
  Plus,
} from "lucide-react";
import type { GallerySample } from "../types";
import { PhotoEditorModal } from "./PhotoEditorModal";
import {
  DEFAULT_PREFERENCES,
  dataUrlToBlob,
  deletePhotosFromVault,
  detectScreenshotHeuristic,
  ensureUniqueFilename,
  estimateVaultStorage,
  findDuplicateAnalysis,
  formatBytes,
  groupPhotosByTimeline,
  loadAllStoredPhotos,
  loadGalleryPreferences,
  saveGalleryPreferences,
  savePhotoToVault,
  updatePhotoMetadata,
  type GalleryPreferences,
  type StoredPhotoRecord,
  type TimelineGranularity,
  type VaultStorageEstimate,
} from "../lib/photoVaultDb";
import { computeImageFingerprint } from "../lib/imageHashWorker";

interface GooglePhotosGalleryProps {
  samples: GallerySample[];
  isOffline?: boolean;
  onSelectForAnalysis?: (imageUrl: string) => void;
}

interface GalleryPhotoItem {
  id: string;
  url: string;
  filename: string;
  mimeType: string;
  sizeBytes: number | null;
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
  isUserUpload: boolean;
}

type SmartCollectionKey =
  | "All Photos"
  | "Recently Added"
  | "Favorites"
  | "Screenshots"
  | "Similar Photos"
  | string;

type SortMode = "newest" | "oldest" | "filename" | "size" | "favorites";
type MimeFilter = "all" | "image/jpeg" | "image/png" | "image/webp" | "other";

const RECENT_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;
const INITIAL_WINDOW_SIZE = 36;
const WINDOW_STEP = 36;
const MAX_LOCAL_FILE_BYTES = 25 * 1024 * 1024;

function inferMimeFromFilename(filename: string, url: string): string {
  const lower = `${filename} ${url}`.toLowerCase();
  if (lower.includes(".png") || lower.startsWith("data:image/png")) return "image/png";
  if (lower.includes(".webp") || lower.startsWith("data:image/webp")) return "image/webp";
  if (lower.includes(".svg") || lower.startsWith("data:image/svg")) return "image/svg+xml";
  return "image/jpeg";
}

function shortMimeLabel(mime: string): string {
  if (mime === "image/jpeg") return "JPEG";
  if (mime === "image/png") return "PNG";
  if (mime === "image/webp") return "WebP";
  if (mime === "image/svg+xml") return "SVG";
  return mime.replace("image/", "").toUpperCase() || "IMG";
}

interface PhotoGridCardProps {
  photo: GalleryPhotoItem;
  isSelected: boolean;
  isExactDup: boolean;
  isSimilar: boolean;
  bestDistance: number | null;
  isOffline: boolean;
  onOpen: (id: string) => void;
  onToggleSelect: (id: string, e: MouseEvent) => void;
  onEdit: (photo: GalleryPhotoItem) => void;
  onToggleFavorite: (photo: GalleryPhotoItem, e: MouseEvent) => void;
  onSelectForAnalysis?: (imageUrl: string) => void;
}

const PhotoGridCard = memo(function PhotoGridCard({
  photo,
  isSelected,
  isExactDup,
  isSimilar,
  bestDistance,
  isOffline,
  onOpen,
  onToggleSelect,
  onEdit,
  onToggleFavorite,
  onSelectForAnalysis,
}: PhotoGridCardProps) {
  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={`View ${photo.filename}`}
      onClick={() => onOpen(photo.id)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen(photo.id);
        }
      }}
      className={`cv-auto group relative rounded-xl border bg-slate-950/85 overflow-hidden cursor-pointer transition-all duration-300 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-sky-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
        isSelected
          ? "border-sky-400 ring-2 ring-sky-400/50 scale-[0.98]"
          : "border-white/10 hover:border-sky-400/50"
      }`}
    >
      <div className="relative aspect-square w-full bg-slate-900 overflow-hidden">
        <img
          src={photo.url}
          alt={photo.filename}
          loading="lazy"
          decoding="async"
          referrerPolicy="no-referrer"
          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
        />

        {/* Persistent top-left indicator badges (selection / duplicate / favorite) */}
        <div className="absolute top-2 left-2 right-2 flex items-center justify-between pointer-events-none">
          <div className="flex items-center gap-1">
            {isExactDup && (
              <span className="px-1.5 py-0.5 rounded bg-amber-500/90 text-slate-950 text-[10px] font-bold tracking-tight shadow">
                Exact Dup
              </span>
            )}
            {isSimilar && bestDistance !== null && (
              <span className="px-1.5 py-0.5 rounded bg-indigo-500/90 text-white text-[10px] font-semibold tracking-tight shadow">
                Similar · d={bestDistance}
              </span>
            )}
            {photo.isScreenshot && !isExactDup && !isSimilar && (
              <span className="px-1.5 py-0.5 rounded bg-slate-900/85 border border-white/15 text-slate-200 text-[10px] font-medium">
                Screen
              </span>
            )}
          </div>
          {photo.isFavorite && (
            <span className="p-1 rounded-md bg-slate-950/75 text-rose-400 group-hover:opacity-0 transition-opacity">
              <Heart className="w-3.5 h-3.5 fill-rose-400" />
            </span>
          )}
        </div>

        {/* Interactive hover overlay */}
        <div
          className={`absolute inset-0 bg-gradient-to-t from-slate-950/90 via-slate-950/20 to-slate-950/40 transition-opacity flex flex-col justify-between p-2.5 ${
            isSelected ? "opacity-100" : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100"
          }`}
        >
          <div className="flex items-center justify-between">
            <button
              type="button"
              aria-label={isSelected ? `Deselect ${photo.filename}` : `Select ${photo.filename}`}
              onClick={(e) => onToggleSelect(photo.id, e)}
              className="p-1.5 rounded-md bg-slate-900/85 text-slate-200 hover:text-sky-300 transition-colors"
              title={isSelected ? "Deselect" : "Select photo"}
            >
              {isSelected ? (
                <CheckSquare className="w-4 h-4 text-sky-400" />
              ) : (
                <Square className="w-4 h-4" />
              )}
            </button>

            <div className="flex items-center gap-1">
              <button
                type="button"
                aria-label={`Edit ${photo.filename}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onEdit(photo);
                }}
                className="p-1.5 rounded-md bg-slate-900/85 text-sky-400 hover:text-sky-300 transition-colors"
                title="Edit Photo (Non-Destructive · Offline Ready)"
              >
                <Edit3 className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                aria-label={photo.isFavorite ? `Unfavorite ${photo.filename}` : `Favorite ${photo.filename}`}
                onClick={(e) => onToggleFavorite(photo, e)}
                className="p-1.5 rounded-md bg-slate-900/85 text-slate-200 transition-colors"
                title={photo.isFavorite ? "Remove Favorite" : "Add Favorite"}
              >
                <Heart
                  className={`w-3.5 h-3.5 ${
                    photo.isFavorite
                      ? "fill-rose-400 text-rose-400"
                      : "text-slate-300 hover:text-rose-400"
                  }`}
                />
              </button>
            </div>
          </div>

          <div className="space-y-1">
            <div className="text-[11px] font-mono font-medium text-slate-100 truncate">
              {photo.filename}
            </div>
            <div className="flex items-center justify-between gap-1 text-[10px] text-slate-300 font-mono">
              <span className="truncate">
                {photo.width && photo.height
                  ? `${photo.width}×${photo.height}`
                  : shortMimeLabel(photo.mimeType)}
                {photo.sizeBytes ? ` · ${formatBytes(photo.sizeBytes)}` : ""}
              </span>
              {onSelectForAnalysis && (
                <button
                  type="button"
                  disabled={isOffline}
                  aria-label={`Analyze ${photo.filename}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!isOffline) onSelectForAnalysis(photo.url);
                  }}
                  className="p-1 rounded bg-sky-500 hover:bg-sky-400 disabled:bg-slate-700 disabled:text-slate-400 disabled:cursor-not-allowed text-slate-950 shrink-0"
                  title={
                    isOffline
                      ? "AI backend analysis disabled while offline"
                      : "Send to AI Attribute Analysis"
                  }
                >
                  <Sparkles className="w-3 h-3" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

export function GooglePhotosGallery({
  samples,
  isOffline = false,
  onSelectForAnalysis,
}: GooglePhotosGalleryProps) {
  const [storedPhotos, setStoredPhotos] = useState<GalleryPhotoItem[]>([]);
  const [preferences, setPreferences] = useState<GalleryPreferences>(DEFAULT_PREFERENCES);
  const [isHydratingVault, setIsHydratingVault] = useState(true);
  const [isProcessingUploads, setIsProcessingUploads] = useState(false);
  const [storageBanner, setStorageBanner] = useState<{
    tone: "warning" | "error" | "info";
    message: string;
  } | null>(null);
  const [storageEstimate, setStorageEstimate] = useState<VaultStorageEstimate>({
    localBytes: 0,
    browserUsageBytes: null,
    browserQuotaBytes: null,
    supported: false,
  });

  // Organization & Search states
  const [activeCollection, setActiveCollection] = useState<SmartCollectionKey>("All Photos");
  const [searchQuery, setSearchQuery] = useState("");
  const [dateFilter, setDateFilter] = useState<string>("all");
  const [mimeFilter, setMimeFilter] = useState<MimeFilter>("all");
  const [sortBy, setSortBy] = useState<SortMode>("newest");
  const [viewMode, setViewMode] = useState<"timeline" | "grid">("timeline");
  const [timelineGranularity, setTimelineGranularity] = useState<TimelineGranularity>("day");
  const [visibleCount, setVisibleCount] = useState<number>(INITIAL_WINDOW_SIZE);

  // Selection, Bulk actions, Modals
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [moveTargetAlbum, setMoveTargetAlbum] = useState<string>("User Uploads");
  const [showNewAlbumInput, setShowNewAlbumInput] = useState(false);
  const [newAlbumName, setNewAlbumName] = useState("");
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  // Lightbox & Photo Editor
  const [activePhotoIndex, setActivePhotoIndex] = useState<number | null>(null);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [editingPhoto, setEditingPhoto] = useState<GalleryPhotoItem | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const loadMoreSentinelRef = useRef<HTMLDivElement>(null);
  const managedObjectUrlsRef = useRef<Map<string, string>>(new Map());
  const downloadTimersRef = useRef<number[]>([]);
  const touchStartRef = useRef<{ x: number; y: number } | null>(null);

  // Clean up any scheduled bulk-download timers on unmount
  useEffect(() => {
    const timers = downloadTimersRef.current;
    return () => {
      timers.forEach((id) => window.clearTimeout(id));
      timers.length = 0;
    };
  }, []);

  // Hydrate persisted IndexedDB photos & preferences once on mount
  useEffect(() => {
    let active = true;
    const objectUrls = managedObjectUrlsRef.current;

    async function hydrate() {
      try {
        const [records, prefs] = await Promise.all([
          loadAllStoredPhotos(),
          loadGalleryPreferences(),
        ]);
        if (!active) return;

        const mapped: GalleryPhotoItem[] = records.map((rec) => {
          const url = URL.createObjectURL(rec.blob);
          objectUrls.set(rec.id, url);
          return {
            id: rec.id,
            url,
            filename: rec.filename,
            mimeType: rec.mimeType || "image/jpeg",
            sizeBytes: rec.sizeBytes,
            width: rec.width,
            height: rec.height,
            createdAt: rec.createdAt,
            timestamp: rec.timestamp,
            monthKey: rec.monthKey || rec.timestamp.slice(0, 7),
            yearKey: rec.yearKey || rec.timestamp.slice(0, 4),
            isFavorite: rec.isFavorite,
            album: rec.album || "User Uploads",
            attributes: rec.attributes || [],
            sha256: rec.sha256,
            dhash: rec.dhash,
            isScreenshot: Boolean(rec.isScreenshot),
            isUserUpload: true,
          };
        });

        setStoredPhotos(mapped);
        setPreferences(prefs);
      } catch {
        if (active) {
          setStorageBanner({
            tone: "warning",
            message:
              "IndexedDB persistence is restricted in this browser context. Uploaded photos will remain in local session memory only.",
          });
        }
      } finally {
        if (active) setIsHydratingVault(false);
      }
    }

    hydrate();

    return () => {
      active = false;
      objectUrls.forEach((url) => URL.revokeObjectURL(url));
      objectUrls.clear();
    };
  }, []);

  // Convert API benchmark samples with persisted preference overlays (favorites, album moves, hidden)
  const samplePhotos: GalleryPhotoItem[] = useMemo(() => {
    const hiddenSet = new Set(preferences.hiddenSampleIds);
    // Anchor deterministic sample dates cleanly so timeline grouping works without shifting per render
    const anchorDayMs = Math.floor(Date.now() / 86400000) * 86400000;
    return samples
      .filter((s) => !hiddenSet.has(`sample-${s.id}`))
      .map((s) => {
        const id = `sample-${s.id}`;
        const createdAt = anchorDayMs - s.id * 86400000;
        const isoDate = new Date(createdAt).toISOString().split("T")[0];
        const mimeType = inferMimeFromFilename(s.filename, s.thumbnail);
        const defaultAlbum = s.smiling ? "Smiling Faces" : "Curated Corpus";
        return {
          id,
          url: s.thumbnail,
          filename: s.filename,
          mimeType,
          sizeBytes: null,
          width: null,
          height: null,
          createdAt,
          timestamp: isoDate,
          monthKey: isoDate.slice(0, 7),
          yearKey: isoDate.slice(0, 4),
          isFavorite:
            id in preferences.sampleFavorites ? preferences.sampleFavorites[id] : Boolean(s.smiling),
          album: preferences.sampleAlbums[id] || defaultAlbum,
          attributes:
            s.attributes && s.attributes.length > 0
              ? s.attributes
              : [s.smiling ? "Smiling" : "Neutral", s.eyeglasses ? "Eyeglasses" : "Natural"],
          sha256: null,
          dhash: null,
          isScreenshot: detectScreenshotHeuristic(s.filename, mimeType, null, null),
          isUserUpload: false,
        };
      });
  }, [samples, preferences]);

  const allPhotos = useMemo(() => [...storedPhotos, ...samplePhotos], [storedPhotos, samplePhotos]);

  // Storage usage telemetry (local blobs + navigator.storage.estimate)
  const totalLocalBytes = useMemo(
    () => storedPhotos.reduce((acc, p) => acc + (p.sizeBytes ?? 0), 0),
    [storedPhotos]
  );

  useEffect(() => {
    let active = true;
    estimateVaultStorage(totalLocalBytes).then((est) => {
      if (active) setStorageEstimate(est);
    });
    return () => {
      active = false;
    };
  }, [totalLocalBytes, storedPhotos.length]);

  // Exact SHA-256 & Perceptual dHash Duplicate Analysis
  const duplicateAnalysis = useMemo(() => findDuplicateAnalysis(allPhotos, 10), [allPhotos]);

  // Dynamic Smart Time Album options (Months & Years present in gallery)
  const availableMonths = useMemo(() => {
    const months = new Set<string>();
    allPhotos.forEach((p) => {
      if (p.monthKey) months.add(p.monthKey);
    });
    return Array.from(months).sort((a, b) => b.localeCompare(a));
  }, [allPhotos]);

  const availableYears = useMemo(() => {
    const years = new Set<string>();
    allPhotos.forEach((p) => {
      if (p.yearKey) years.add(p.yearKey);
    });
    return Array.from(years).sort((a, b) => b.localeCompare(a));
  }, [allPhotos]);

  // User + System Albums
  const assignableAlbums = useMemo(() => {
    const set = new Set<string>([
      "User Uploads",
      "Edited Photos",
      "Curated Corpus",
      "Smiling Faces",
      ...preferences.userAlbums,
    ]);
    allPhotos.forEach((p) => {
      if (p.album) set.add(p.album);
    });
    return Array.from(set);
  }, [allPhotos, preferences.userAlbums]);

  // Smart Collections & User Album cards with auto-generated cover thumbnails
  const collectionCards = useMemo(() => {
    const now = Date.now();
    const recentlyAdded = allPhotos.filter(
      (p) => p.isUserUpload || now - p.createdAt <= RECENT_WINDOW_MS
    );
    const favorites = allPhotos.filter((p) => p.isFavorite);
    const screenshots = allPhotos.filter((p) => p.isScreenshot);
    const similar = allPhotos.filter(
      (p) => duplicateAnalysis.exactPhotoIds.has(p.id) || duplicateAnalysis.similarPhotoIds.has(p.id)
    );

    const smartCards = [
      {
        key: "All Photos",
        label: "All Photos",
        subtitle: "Complete local library",
        kind: "smart" as const,
        items: allPhotos,
        icon: Layers,
      },
      {
        key: "Recently Added",
        label: "Recently Added",
        subtitle: "Uploads & last 7 days",
        kind: "smart" as const,
        items: recentlyAdded,
        icon: Clock,
      },
      {
        key: "Favorites",
        label: "Favorites",
        subtitle: "Persisted locally",
        kind: "smart" as const,
        items: favorites,
        icon: Heart,
      },
      {
        key: "Screenshots",
        label: "Screenshots",
        subtitle: "Metadata & PNG heuristic",
        kind: "smart" as const,
        items: screenshots,
        icon: Monitor,
      },
      {
        key: "Similar Photos",
        label: "Similar Photos",
        subtitle: "SHA-256 & perceptual dHash",
        kind: "smart" as const,
        items: similar,
        icon: Copy,
      },
    ];

    const albumCards = assignableAlbums.map((albumName) => {
      const items = allPhotos.filter((p) => p.album === albumName);
      return {
        key: albumName,
        label: albumName,
        subtitle: preferences.userAlbums.includes(albumName) ? "User Album" : "Album",
        kind: "album" as const,
        items,
        icon: FolderPlus,
      };
    });

    return [...smartCards, ...albumCards];
  }, [allPhotos, assignableAlbums, duplicateAnalysis, preferences.userAlbums]);

  // Filtered & sorted photos
  const filteredPhotos = useMemo(() => {
    const now = Date.now();
    let result = allPhotos;

    // 1. Smart Collection or Album filter
    if (activeCollection === "Recently Added") {
      result = result.filter((p) => p.isUserUpload || now - p.createdAt <= RECENT_WINDOW_MS);
    } else if (activeCollection === "Favorites") {
      result = result.filter((p) => p.isFavorite);
    } else if (activeCollection === "Screenshots") {
      result = result.filter((p) => p.isScreenshot);
    } else if (activeCollection === "Similar Photos") {
      result = result.filter(
        (p) => duplicateAnalysis.exactPhotoIds.has(p.id) || duplicateAnalysis.similarPhotoIds.has(p.id)
      );
    } else if (activeCollection !== "All Photos") {
      result = result.filter((p) => p.album === activeCollection);
    }

    // 2. Date / Smart Time Album filter
    if (dateFilter === "last7") {
      result = result.filter((p) => now - p.createdAt <= 7 * 86400000);
    } else if (dateFilter === "last30") {
      result = result.filter((p) => now - p.createdAt <= 30 * 86400000);
    } else if (dateFilter.startsWith("month:")) {
      const m = dateFilter.slice(6);
      result = result.filter((p) => p.monthKey === m);
    } else if (dateFilter.startsWith("year:")) {
      const y = dateFilter.slice(5);
      result = result.filter((p) => p.yearKey === y);
    }

    // 3. MIME / Image Type filter
    if (mimeFilter !== "all") {
      if (mimeFilter === "other") {
        result = result.filter(
          (p) => p.mimeType !== "image/jpeg" && p.mimeType !== "image/png" && p.mimeType !== "image/webp"
        );
      } else {
        result = result.filter((p) => p.mimeType === mimeFilter);
      }
    }

    // 4. Advanced Metadata Search (filename, tags/attributes, album, dimensions, hash prefix, date)
    const q = searchQuery.trim().toLowerCase();
    if (q) {
      result = result.filter((p) => {
        const dims = p.width && p.height ? `${p.width}x${p.height}` : "";
        return (
          p.filename.toLowerCase().includes(q) ||
          p.album.toLowerCase().includes(q) ||
          p.timestamp.includes(q) ||
          shortMimeLabel(p.mimeType).toLowerCase().includes(q) ||
          dims.includes(q) ||
          (p.sha256 ? p.sha256.toLowerCase().startsWith(q) : false) ||
          p.attributes.some((a) => a.toLowerCase().includes(q))
        );
      });
    }

    // 5. Sort
    const sorted = [...result];
    sorted.sort((a, b) => {
      if (sortBy === "oldest") return a.createdAt - b.createdAt;
      if (sortBy === "filename") return a.filename.localeCompare(b.filename);
      if (sortBy === "size") return (b.sizeBytes ?? 0) - (a.sizeBytes ?? 0);
      if (sortBy === "favorites") {
        const favDiff = (b.isFavorite ? 1 : 0) - (a.isFavorite ? 1 : 0);
        return favDiff !== 0 ? favDiff : b.createdAt - a.createdAt;
      }
      return b.createdAt - a.createdAt;
    });

    return sorted;
  }, [allPhotos, activeCollection, dateFilter, mimeFilter, searchQuery, sortBy, duplicateAnalysis]);

  // Reset progressive window when filters change
  useEffect(() => {
    setVisibleCount(INITIAL_WINDOW_SIZE);
  }, [activeCollection, dateFilter, mimeFilter, searchQuery, sortBy]);

  // IntersectionObserver progressive virtualization sentinel for large collections
  useEffect(() => {
    const sentinel = loadMoreSentinelRef.current;
    if (!sentinel || visibleCount >= filteredPhotos.length) return;
    if (typeof IntersectionObserver === "undefined") return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisibleCount((prev) => Math.min(filteredPhotos.length, prev + WINDOW_STEP));
        }
      },
      { rootMargin: "320px 0px" }
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [visibleCount, filteredPhotos.length]);

  const visiblePhotos = useMemo(
    () => filteredPhotos.slice(0, visibleCount),
    [filteredPhotos, visibleCount]
  );

  const timelineGroups = useMemo(
    () => groupPhotosByTimeline(visiblePhotos, timelineGranularity),
    [visiblePhotos, timelineGranularity]
  );

  const photoIndexById = useMemo(() => {
    const map = new Map<string, number>();
    filteredPhotos.forEach((p, idx) => map.set(p.id, idx));
    return map;
  }, [filteredPhotos]);

  // Lightbox keyboard navigation
  useEffect(() => {
    if (activePhotoIndex === null) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setActivePhotoIndex(null);
      } else if (e.key === "ArrowLeft") {
        setActivePhotoIndex((idx) => (idx !== null && idx > 0 ? idx - 1 : idx));
        setZoomLevel(1);
      } else if (e.key === "ArrowRight") {
        setActivePhotoIndex((idx) =>
          idx !== null && idx < filteredPhotos.length - 1 ? idx + 1 : idx
        );
        setZoomLevel(1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activePhotoIndex, filteredPhotos.length]);

  // Persist preference updates helper
  const updatePreferences = useCallback((updater: (prev: GalleryPreferences) => GalleryPreferences) => {
    setPreferences((prev) => {
      const next = updater(prev);
      void saveGalleryPreferences(next);
      return next;
    });
  }, []);

  // Process & persist uploaded files using Web Worker SHA-256 + dHash + IndexedDB
  const processFiles = useCallback(
    async (fileList: FileList) => {
      const allIncoming = Array.from(fileList);
      const imageFiles = allIncoming.filter((f) => f.type.startsWith("image/"));
      const validFiles = imageFiles.filter((f) => f.size > 0 && f.size <= MAX_LOCAL_FILE_BYTES);
      const oversizedCount = imageFiles.length - validFiles.length;
      const skippedCount = allIncoming.length - imageFiles.length;

      if (validFiles.length === 0) {
        setStorageBanner({
          tone: "error",
          message:
            oversizedCount > 0
              ? `Skipped ${oversizedCount} image(s) exceeding the 25 MB local vault limit.`
              : "Only valid image files (JPEG, PNG, WebP) can be imported into the local photo vault.",
        });
        return;
      }

      setIsProcessingUploads(true);
      const existingNames = new Set(allPhotos.map((p) => p.filename));
      const newItems: GalleryPhotoItem[] = [];
      let quotaWarning: string | null = null;
      const targetAlbum =
        activeCollection !== "All Photos" &&
        activeCollection !== "Recently Added" &&
        activeCollection !== "Favorites" &&
        activeCollection !== "Screenshots" &&
        activeCollection !== "Similar Photos"
          ? activeCollection
          : "User Uploads";

      try {
        for (const file of validFiles) {
          const uniqueName = ensureUniqueFilename(file.name, existingNames);
          existingNames.add(uniqueName);

          const fp = await computeImageFingerprint(file);
          const createdAt = file.lastModified && file.lastModified > 0 ? file.lastModified : Date.now();
          const isoDate = new Date(createdAt).toISOString().split("T")[0];
          const isScreenshot = detectScreenshotHeuristic(uniqueName, file.type, fp.width, fp.height);
          const id = `upload-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

          const attrs = ["Local Upload", shortMimeLabel(file.type)];
          if (fp.width && fp.height) attrs.push(`${fp.width}x${fp.height}`);
          if (isScreenshot) attrs.push("Screenshot");

          const record: StoredPhotoRecord = {
            id,
            blob: file,
            filename: uniqueName,
            mimeType: file.type || "image/jpeg",
            sizeBytes: file.size,
            width: fp.width,
            height: fp.height,
            createdAt,
            timestamp: isoDate,
            monthKey: isoDate.slice(0, 7),
            yearKey: isoDate.slice(0, 4),
            isFavorite: false,
            album: targetAlbum,
            attributes: attrs,
            sha256: fp.sha256,
            dhash: fp.dhash,
            isScreenshot,
            isUserUpload: true,
          };

          const saveRes = await savePhotoToVault(record);
          if (!saveRes.ok && saveRes.message) {
            quotaWarning = saveRes.message;
          }

          const url = URL.createObjectURL(file);
          managedObjectUrlsRef.current.set(id, url);

          newItems.push({
            id,
            url,
            filename: uniqueName,
            mimeType: record.mimeType,
            sizeBytes: record.sizeBytes,
            width: record.width,
            height: record.height,
            createdAt: record.createdAt,
            timestamp: record.timestamp,
            monthKey: record.monthKey,
            yearKey: record.yearKey,
            isFavorite: record.isFavorite,
            album: record.album,
            attributes: record.attributes,
            sha256: record.sha256,
            dhash: record.dhash,
            isScreenshot: record.isScreenshot,
            isUserUpload: true,
          });
        }

        if (newItems.length > 0) {
          setStoredPhotos((prev) => [...newItems, ...prev]);
        }

        if (quotaWarning) {
          setStorageBanner({ tone: "error", message: quotaWarning });
        } else if (skippedCount > 0) {
          setStorageBanner({
            tone: "warning",
            message: `Imported ${newItems.length} image(s) locally. Skipped ${skippedCount} non-image file(s).`,
          });
        } else {
          setStorageBanner(null);
        }
      } finally {
        setIsProcessingUploads(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    },
    [allPhotos, activeCollection]
  );

  // Non-destructive save from PhotoEditorModal -> persisted into IndexedDB under "Edited Photos"
  const handleSaveEdited = useCallback(
    async (newUrl: string, newFilename: string) => {
      try {
        const blob = dataUrlToBlob(newUrl);
        const existingNames = allPhotos.map((p) => p.filename);
        const uniqueName = ensureUniqueFilename(newFilename, existingNames);
        const fp = await computeImageFingerprint(blob);
        const createdAt = Date.now();
        const isoDate = new Date(createdAt).toISOString().split("T")[0];
        const id = `edited-${createdAt}-${Math.random().toString(36).slice(2, 7)}`;

        const attrs = ["Edited", shortMimeLabel(blob.type)];
        if (fp.width && fp.height) attrs.push(`${fp.width}x${fp.height}`);

        const record: StoredPhotoRecord = {
          id,
          blob,
          filename: uniqueName,
          mimeType: blob.type || "image/jpeg",
          sizeBytes: blob.size,
          width: fp.width,
          height: fp.height,
          createdAt,
          timestamp: isoDate,
          monthKey: isoDate.slice(0, 7),
          yearKey: isoDate.slice(0, 4),
          isFavorite: true,
          album: "Edited Photos",
          attributes: attrs,
          sha256: fp.sha256,
          dhash: fp.dhash,
          isScreenshot: false,
          isUserUpload: true,
        };

        const saveRes = await savePhotoToVault(record);
        if (!saveRes.ok && saveRes.message) {
          setStorageBanner({ tone: "error", message: saveRes.message });
        }

        const objectUrl = URL.createObjectURL(blob);
        managedObjectUrlsRef.current.set(id, objectUrl);

        setStoredPhotos((prev) => [
          {
            id,
            url: objectUrl,
            filename: uniqueName,
            mimeType: record.mimeType,
            sizeBytes: record.sizeBytes,
            width: record.width,
            height: record.height,
            createdAt: record.createdAt,
            timestamp: record.timestamp,
            monthKey: record.monthKey,
            yearKey: record.yearKey,
            isFavorite: record.isFavorite,
            album: record.album,
            attributes: record.attributes,
            sha256: record.sha256,
            dhash: record.dhash,
            isScreenshot: false,
            isUserUpload: true,
          },
          ...prev,
        ]);
      } catch {
        setStorageBanner({
          tone: "error",
          message: "Could not save edited copy to local IndexedDB.",
        });
      } finally {
        setEditingPhoto(null);
      }
    },
    [allPhotos]
  );

  // Drag & Drop handlers
  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingOver(true);
  };

  const handleDragLeave = () => {
    setIsDraggingOver(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      void processFiles(e.dataTransfer.files);
    }
  };

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      void processFiles(e.target.files);
    }
  };

  // Favorites toggle (persisted to IndexedDB or preferences)
  const toggleFavorite = useCallback(
    (photo: GalleryPhotoItem, e: MouseEvent) => {
      e.stopPropagation();
      const nextFav = !photo.isFavorite;
      if (photo.isUserUpload) {
        setStoredPhotos((prev) =>
          prev.map((p) => (p.id === photo.id ? { ...p, isFavorite: nextFav } : p))
        );
        void updatePhotoMetadata(photo.id, { isFavorite: nextFav });
      } else {
        updatePreferences((prev) => ({
          ...prev,
          sampleFavorites: { ...prev.sampleFavorites, [photo.id]: nextFav },
        }));
      }
    },
    [updatePreferences]
  );

  // Selection helpers
  const toggleSelectAll = () => {
    if (selectedIds.size === filteredPhotos.length && filteredPhotos.length > 0) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filteredPhotos.map((p) => p.id)));
    }
  };

  const toggleSelectOne = useCallback((id: string, e: MouseEvent) => {
    e.stopPropagation();
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handleOpenPhoto = useCallback(
    (id: string) => {
      const idx = photoIndexById.get(id);
      if (idx !== undefined) {
        setActivePhotoIndex(idx);
        setZoomLevel(1);
      }
    },
    [photoIndexById]
  );

  const handleEditPhoto = useCallback((photo: GalleryPhotoItem) => {
    setEditingPhoto(photo);
  }, []);

  // Create new custom User Album
  const handleCreateUserAlbum = () => {
    const trimmed = newAlbumName.trim();
    if (!trimmed) return;
    if (!preferences.userAlbums.includes(trimmed)) {
      updatePreferences((prev) => ({
        ...prev,
        userAlbums: [...prev.userAlbums, trimmed],
      }));
    }
    setMoveTargetAlbum(trimmed);
    setActiveCollection(trimmed);
    setNewAlbumName("");
    setShowNewAlbumInput(false);
  };

  // Bulk Move to Album
  const handleBulkMoveToAlbum = () => {
    if (selectedIds.size === 0 || !moveTargetAlbum) return;
    const target = moveTargetAlbum;
    const localIds: string[] = [];
    const sampleIds: string[] = [];

    allPhotos.forEach((p) => {
      if (!selectedIds.has(p.id)) return;
      if (p.isUserUpload) localIds.push(p.id);
      else sampleIds.push(p.id);
    });

    if (localIds.length > 0) {
      setStoredPhotos((prev) =>
        prev.map((p) => (selectedIds.has(p.id) ? { ...p, album: target } : p))
      );
      localIds.forEach((id) => void updatePhotoMetadata(id, { album: target }));
    }

    if (sampleIds.length > 0) {
      updatePreferences((prev) => {
        const nextSampleAlbums = { ...prev.sampleAlbums };
        sampleIds.forEach((id) => {
          nextSampleAlbums[id] = target;
        });
        return { ...prev, sampleAlbums: nextSampleAlbums };
      });
    }

    setStorageBanner({
      tone: "info",
      message: `Moved ${selectedIds.size} photo(s) to "${target}".`,
    });
    setSelectedIds(new Set());
  };

  // Bulk Download selected images (tracked timers cleaned up on unmount)
  const handleBulkDownload = () => {
    const selectedItems = allPhotos.filter((p) => selectedIds.has(p.id));
    selectedItems.forEach((item, idx) => {
      const timerId = window.setTimeout(() => {
        const link = document.createElement("a");
        link.href = item.url;
        link.download = item.filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }, idx * 140);
      downloadTimersRef.current.push(timerId);
    });
  };

  // Confirmed Bulk Delete
  const handleConfirmDelete = async () => {
    const localIds: string[] = [];
    const sampleIds: string[] = [];

    allPhotos.forEach((p) => {
      if (!selectedIds.has(p.id)) return;
      if (p.isUserUpload) localIds.push(p.id);
      else sampleIds.push(p.id);
    });

    if (localIds.length > 0) {
      await deletePhotosFromVault(localIds);
      localIds.forEach((id) => {
        const url = managedObjectUrlsRef.current.get(id);
        if (url) {
          URL.revokeObjectURL(url);
          managedObjectUrlsRef.current.delete(id);
        }
      });
      setStoredPhotos((prev) => prev.filter((p) => !selectedIds.has(p.id)));
    }

    if (sampleIds.length > 0) {
      updatePreferences((prev) => ({
        ...prev,
        hiddenSampleIds: Array.from(new Set([...prev.hiddenSampleIds, ...sampleIds])),
      }));
    }

    setSelectedIds(new Set());
    setShowDeleteConfirm(false);
    setActivePhotoIndex(null);
  };

  const activePhoto = activePhotoIndex !== null ? filteredPhotos[activePhotoIndex] ?? null : null;

  // Count how many selected are local IndexedDB vs curated corpus for confirmation modal
  const selectedBreakdown = useMemo(() => {
    let localCount = 0;
    let corpusCount = 0;
    allPhotos.forEach((p) => {
      if (!selectedIds.has(p.id)) return;
      if (p.isUserUpload) localCount += 1;
      else corpusCount += 1;
    });
    return { localCount, corpusCount };
  }, [allPhotos, selectedIds]);

  const handleLightboxTouchStart = (e: TouchEvent<HTMLDivElement>) => {
    if (e.touches.length !== 1 || zoomLevel > 1) return;
    touchStartRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
  };

  const handleLightboxTouchEnd = (e: TouchEvent<HTMLDivElement>) => {
    const start = touchStartRef.current;
    touchStartRef.current = null;
    if (!start || e.changedTouches.length !== 1 || zoomLevel > 1) return;
    const dx = e.changedTouches[0].clientX - start.x;
    const dy = e.changedTouches[0].clientY - start.y;

    if (Math.abs(dx) > 55 && Math.abs(dx) > Math.abs(dy)) {
      if (dx < 0 && activePhotoIndex !== null && activePhotoIndex < filteredPhotos.length - 1) {
        setActivePhotoIndex(activePhotoIndex + 1);
        setZoomLevel(1);
      } else if (dx > 0 && activePhotoIndex !== null && activePhotoIndex > 0) {
        setActivePhotoIndex(activePhotoIndex - 1);
        setZoomLevel(1);
      }
    } else if (dy > 90 && Math.abs(dy) > Math.abs(dx)) {
      setActivePhotoIndex(null);
    }
  };

  const renderPhotoCard = (photo: GalleryPhotoItem) => {
    const isExactDup = duplicateAnalysis.exactPhotoIds.has(photo.id);
    const similarMatches = duplicateAnalysis.similarMatchesByPhotoId.get(photo.id);
    const isSimilar = !isExactDup && Boolean(similarMatches && similarMatches.length > 0);
    const bestDistance = similarMatches
      ? Math.min(...similarMatches.map((m) => m.distance))
      : null;

    return (
      <PhotoGridCard
        key={photo.id}
        photo={photo}
        isSelected={selectedIds.has(photo.id)}
        isExactDup={isExactDup}
        isSimilar={isSimilar}
        bestDistance={bestDistance}
        isOffline={isOffline}
        onOpen={handleOpenPhoto}
        onToggleSelect={toggleSelectOne}
        onEdit={handleEditPhoto}
        onToggleFavorite={toggleFavorite}
        onSelectForAnalysis={onSelectForAnalysis}
      />
    );
  };

  return (
    <div
      className="space-y-5 relative"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag & Drop Overlay */}
      {isDraggingOver && (
        <div className="absolute inset-0 z-50 bg-sky-950/85 backdrop-blur-md rounded-2xl border-2 border-dashed border-sky-400 flex flex-col items-center justify-center p-8 text-center animate-fade-in">
          <Upload className="w-14 h-14 text-sky-400 animate-bounce mb-4" />
          <h3 className="text-xl font-bold text-slate-50">Drop Images Into Local Photo Vault</h3>
          <p className="text-sm text-sky-200 mt-1 max-w-md">
            Stored locally in IndexedDB with Web Worker SHA-256 &amp; perceptual dHash deduplication.
            Zero automatic cloud uploads.
          </p>
        </div>
      )}

      {/* Storage, Privacy & Duplicate Management Strip */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        <div className="p-3.5 rounded-xl bg-slate-900/85 border border-white/10 flex items-center justify-between">
          <div>
            <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">
              Local Vault Library
            </div>
            <div className="text-lg font-bold text-slate-50 font-mono tabular-nums mt-0.5">
              {allPhotos.length}{" "}
              <span className="text-xs font-normal text-slate-400">
                ({storedPhotos.length} local · {samplePhotos.length} corpus)
              </span>
            </div>
          </div>
          <div className="p-2 rounded-lg bg-sky-500/10 text-sky-400">
            <Layers className="w-4 h-4" />
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-900/85 border border-white/10 flex items-center justify-between">
          <div className="min-w-0">
            <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">
              IndexedDB Storage
            </div>
            <div className="text-lg font-bold text-slate-50 font-mono tabular-nums mt-0.5 truncate">
              {formatBytes(storageEstimate.localBytes)}
              {storageEstimate.supported && storageEstimate.browserQuotaBytes ? (
                <span className="text-xs font-normal text-slate-400 ml-1.5">
                  / {formatBytes(storageEstimate.browserQuotaBytes)}
                </span>
              ) : null}
            </div>
          </div>
          <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 shrink-0">
            <HardDrive className="w-4 h-4" />
          </div>
        </div>

        <button
          type="button"
          onClick={() => setActiveCollection("Similar Photos")}
          className="p-3.5 rounded-xl bg-slate-900/85 border border-white/10 hover:border-amber-400/40 transition-colors flex items-center justify-between text-left"
        >
          <div>
            <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">
              Duplicates &amp; Similar
            </div>
            <div className="text-lg font-bold text-slate-50 font-mono tabular-nums mt-0.5">
              {duplicateAnalysis.exactDuplicateExtraCount}{" "}
              <span className="text-xs font-normal text-amber-300">exact</span>{" "}
              <span className="text-slate-600">·</span> {duplicateAnalysis.similarPairCount}{" "}
              <span className="text-xs font-normal text-indigo-300">similar</span>
            </div>
          </div>
          <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400 shrink-0">
            <Copy className="w-4 h-4" />
          </div>
        </button>

        <div className="p-3.5 rounded-xl bg-slate-900/85 border border-white/10 flex items-center justify-between">
          <div>
            <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">
              Privacy Mode
            </div>
            <div className="text-xs text-emerald-300 font-medium mt-1">
              100% Local · No Identity Recognition
            </div>
            <div className="text-[11px] text-slate-400 mt-0.5">
              Zero automatic cloud uploads
            </div>
          </div>
          <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 shrink-0">
            <ShieldCheck className="w-4 h-4" />
          </div>
        </div>
      </div>

      {/* Inline Warning / Quota / Info Banner */}
      {storageBanner && (
        <div
          className={`flex items-center justify-between gap-3 px-4 py-3 rounded-xl border text-xs animate-fade-in ${
            storageBanner.tone === "error"
              ? "bg-rose-950/60 border-rose-400/40 text-rose-200"
              : storageBanner.tone === "warning"
              ? "bg-amber-950/60 border-amber-400/40 text-amber-200"
              : "bg-sky-950/60 border-sky-400/40 text-sky-200"
          }`}
        >
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{storageBanner.message}</span>
          </div>
          <button
            type="button"
            onClick={() => setStorageBanner(null)}
            className="p-1 rounded hover:bg-white/10 transition-colors"
            aria-label="Dismiss notification"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Smart Collections & Auto-Generated Album Cover Shelf */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Smart Collections &amp; Albums
            </h3>
            <p className="text-xs text-slate-400">
              Auto-generated covers from local photos, SHA-256 duplicates, and persistent user albums
            </p>
          </div>

          <div className="flex items-center gap-2">
            {showNewAlbumInput ? (
              <div className="flex items-center gap-1.5 animate-fade-in">
                <input
                  type="text"
                  value={newAlbumName}
                  onChange={(e) => setNewAlbumName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleCreateUserAlbum();
                    if (e.key === "Escape") setShowNewAlbumInput(false);
                  }}
                  placeholder="Album title..."
                  autoFocus
                  className="px-2.5 py-1.5 rounded-lg bg-slate-950 border border-sky-400/50 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none w-40"
                />
                <button
                  type="button"
                  onClick={handleCreateUserAlbum}
                  className="px-2.5 py-1.5 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-semibold transition-colors"
                >
                  Create
                </button>
                <button
                  type="button"
                  onClick={() => setShowNewAlbumInput(false)}
                  className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setShowNewAlbumInput(true)}
                className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-white/15 text-xs text-slate-200 font-medium inline-flex items-center gap-1.5 transition-colors"
              >
                <Plus className="w-3.5 h-3.5 text-sky-400" />
                New Album
              </button>
            )}
          </div>
        </div>

        <div className="flex gap-3 overflow-x-auto pb-2">
          {collectionCards.map((card) => {
            const Icon = card.icon;
            const isActive = activeCollection === card.key;
            const coverPhotos = card.items.slice(0, 4);

            return (
              <button
                key={card.key}
                type="button"
                onClick={() => setActiveCollection(card.key)}
                className={`group shrink-0 w-40 rounded-xl border p-2 text-left transition-all duration-200 ${
                  isActive
                    ? "bg-sky-500/15 border-sky-400 ring-1 ring-sky-400/40"
                    : "bg-slate-900/80 border-white/10 hover:border-sky-400/40 hover:bg-slate-900"
                }`}
              >
                {/* Auto-generated album cover thumbnail / 2x2 collage */}
                <div className="relative h-20 w-full rounded-lg overflow-hidden bg-slate-950 border border-white/5 mb-2">
                  {coverPhotos.length === 0 ? (
                    <div className="w-full h-full flex items-center justify-center text-slate-600">
                      <Icon className="w-6 h-6" />
                    </div>
                  ) : coverPhotos.length === 1 ? (
                    <img
                      src={coverPhotos[0].url}
                      alt={card.label}
                      loading="lazy"
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                  ) : (
                    <div className="grid grid-cols-2 grid-rows-2 w-full h-full gap-px bg-slate-950">
                      {coverPhotos.map((cp) => (
                        <img
                          key={cp.id}
                          src={cp.url}
                          alt={cp.filename}
                          loading="lazy"
                          className="w-full h-full object-cover"
                        />
                      ))}
                    </div>
                  )}
                  <span className="absolute bottom-1.5 right-1.5 px-1.5 py-0.5 rounded bg-slate-950/85 text-[10px] font-mono font-semibold text-slate-100 tabular-nums">
                    {card.items.length}
                  </span>
                </div>

                <div className="flex items-center gap-1.5">
                  <Icon
                    className={`w-3.5 h-3.5 shrink-0 ${
                      isActive ? "text-sky-300" : "text-slate-400"
                    }`}
                  />
                  <span className="text-xs font-semibold text-slate-100 truncate">
                    {card.label}
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 truncate mt-0.5">{card.subtitle}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* AI Semantic Search — Explicitly Disabled (No Local Embedding Runtime Installed) */}
      <div className="p-3.5 rounded-xl bg-slate-950/80 border border-white/10 flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="flex items-start gap-2.5 min-w-0">
          <div className="p-1.5 rounded-lg bg-slate-800 text-slate-400 shrink-0 mt-0.5">
            <Lock className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-semibold text-slate-200">
                AI Natural-Language Semantic Search
              </span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-white/10 text-[10px] font-mono uppercase tracking-wider text-amber-300">
                Disabled · Missing Local Embedding Model
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Requires an installed local image-embedding runtime (e.g., CLIP / SigLIP ONNX) to search
              concepts like &ldquo;sunset&rdquo; or &ldquo;mountains&rdquo; without uploading private photos.
              Use Advanced Metadata Search below for instant local filtering.
            </p>
          </div>
        </div>
        <div className="relative shrink-0 w-full md:w-64 opacity-60">
          <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            disabled
            value=""
            readOnly
            placeholder='Disabled (e.g. "sunset", "mountains")'
            aria-label="AI Semantic Search disabled due to missing local embedding model"
            className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-slate-900 border border-white/10 text-xs text-slate-500 cursor-not-allowed"
          />
        </div>
      </div>

      {/* Advanced Search, Filters, View Mode & Upload Toolbar */}
      <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-3 p-4 rounded-xl bg-slate-900/90 border border-white/10 backdrop-blur-md">
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search filename, tags, album, size, hash..."
              className="pl-9 pr-3 py-2 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-400 w-64"
            />
          </div>

          {/* Smart Date / Month / Year Filter */}
          <select
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
            aria-label="Filter by date"
            className="px-2.5 py-2 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-400"
          >
            <option value="all">All Dates</option>
            <option value="last7">Last 7 Days</option>
            <option value="last30">Last 30 Days</option>
            {availableYears.length > 0 && (
              <optgroup label="By Year">
                {availableYears.map((y) => (
                  <option key={y} value={`year:${y}`}>
                    Year {y}
                  </option>
                ))}
              </optgroup>
            )}
            {availableMonths.length > 0 && (
              <optgroup label="By Month">
                {availableMonths.map((m) => (
                  <option key={m} value={`month:${m}`}>
                    Month {m}
                  </option>
                ))}
              </optgroup>
            )}
          </select>

          {/* Image MIME Type Filter */}
          <select
            value={mimeFilter}
            onChange={(e) => setMimeFilter(e.target.value as MimeFilter)}
            aria-label="Filter by image type"
            className="px-2.5 py-2 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-400"
          >
            <option value="all">All Formats</option>
            <option value="image/jpeg">JPEG</option>
            <option value="image/png">PNG</option>
            <option value="image/webp">WebP</option>
            <option value="other">Other</option>
          </select>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortMode)}
            aria-label="Sort photos"
            className="px-2.5 py-2 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-400"
          >
            <option value="newest">Sort: Newest</option>
            <option value="oldest">Sort: Oldest</option>
            <option value="filename">Sort: Filename</option>
            <option value="size">Sort: File Size</option>
            <option value="favorites">Sort: Favorites First</option>
          </select>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Timeline vs Grid Toggle */}
          <div className="inline-flex rounded-lg bg-slate-950 border border-white/15 p-0.5">
            <button
              type="button"
              onClick={() => setViewMode("timeline")}
              className={`px-2.5 py-1.5 rounded-md text-xs font-medium inline-flex items-center gap-1.5 transition-colors ${
                viewMode === "timeline"
                  ? "bg-sky-500/20 text-sky-300"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Calendar className="w-3.5 h-3.5" />
              Timeline
            </button>
            <button
              type="button"
              onClick={() => setViewMode("grid")}
              className={`px-2.5 py-1.5 rounded-md text-xs font-medium inline-flex items-center gap-1.5 transition-colors ${
                viewMode === "grid"
                  ? "bg-sky-500/20 text-sky-300"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Grid className="w-3.5 h-3.5" />
              Grid
            </button>
          </div>

          {viewMode === "timeline" && (
            <select
              value={timelineGranularity}
              onChange={(e) => setTimelineGranularity(e.target.value as TimelineGranularity)}
              aria-label="Timeline grouping"
              className="px-2.5 py-2 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-200 focus:outline-none"
            >
              <option value="day">Group: Day</option>
              <option value="month">Group: Month</option>
              <option value="year">Group: Year</option>
            </select>
          )}

          <button
            type="button"
            disabled={isProcessingUploads}
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-60 text-slate-950 font-semibold text-xs inline-flex items-center gap-2 transition-colors"
          >
            {isProcessingUploads ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Upload className="w-4 h-4" />
            )}
            {isProcessingUploads ? "Hashing & Saving..." : "Import Photos"}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
            onChange={handleFileSelect}
          />
        </div>
      </div>

      {/* Bulk Actions Bar */}
      {selectedIds.size > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-sky-950/70 border border-sky-400/40 text-xs text-sky-100 animate-fade-in">
          <div className="flex items-center gap-2 font-medium">
            <span className="px-2 py-0.5 rounded bg-sky-500 text-slate-950 font-bold font-mono tabular-nums">
              {selectedIds.size}
            </span>
            <span>selected</span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1.5">
              <select
                value={moveTargetAlbum}
                onChange={(e) => setMoveTargetAlbum(e.target.value)}
                aria-label="Target album"
                className="px-2.5 py-1.5 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-200"
              >
                {assignableAlbums.map((alb) => (
                  <option key={alb} value={alb}>
                    {alb}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={handleBulkMoveToAlbum}
                className="px-3 py-1.5 rounded-lg bg-sky-500/20 hover:bg-sky-500/30 text-sky-200 border border-sky-400/40 inline-flex items-center gap-1.5 transition-colors"
              >
                <FolderInput className="w-3.5 h-3.5" />
                Move to Album
              </button>
            </div>

            <button
              type="button"
              onClick={handleBulkDownload}
              className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 text-slate-100 inline-flex items-center gap-1.5 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Download
            </button>

            <button
              type="button"
              onClick={() => setShowDeleteConfirm(true)}
              className="px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border border-rose-400/40 inline-flex items-center gap-1.5 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Delete
            </button>

            <button
              type="button"
              onClick={() => setSelectedIds(new Set())}
              className="px-2.5 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 transition-colors"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {/* Results Header & Select All */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400 px-1">
        <div className="flex items-center gap-2">
          <span>
            Showing{" "}
            <strong className="text-slate-100 font-mono tabular-nums">{visiblePhotos.length}</strong>{" "}
            of{" "}
            <strong className="text-slate-100 font-mono tabular-nums">{filteredPhotos.length}</strong>{" "}
            photos in <span className="text-sky-300 font-medium">{activeCollection}</span>
          </span>
          {isHydratingVault && (
            <span className="inline-flex items-center gap-1 text-sky-400">
              <Loader2 className="w-3 h-3 animate-spin" />
              Loading IndexedDB vault...
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {(searchQuery || dateFilter !== "all" || mimeFilter !== "all" || activeCollection !== "All Photos") && (
            <button
              type="button"
              onClick={() => {
                setActiveCollection("All Photos");
                setSearchQuery("");
                setDateFilter("all");
                setMimeFilter("all");
              }}
              className="text-sky-400 hover:text-sky-300 transition-colors"
            >
              Reset Filters
            </button>
          )}
          <button
            type="button"
            onClick={toggleSelectAll}
            className="hover:text-slate-100 transition-colors inline-flex items-center gap-1"
          >
            {selectedIds.size === filteredPhotos.length && filteredPhotos.length > 0 ? (
              <CheckSquare className="w-4 h-4 text-sky-400" />
            ) : (
              <Square className="w-4 h-4" />
            )}
            Select All
          </button>
        </div>
      </div>

      {/* Main Photo Timeline or Compact Grid */}
      {filteredPhotos.length === 0 ? (
        <div className="text-center py-16 rounded-2xl border border-white/10 bg-slate-900/50 space-y-2">
          <p className="text-sm text-slate-300 font-medium">
            {activeCollection === "Similar Photos"
              ? "No exact SHA-256 duplicates or perceptually similar photos detected yet."
              : activeCollection === "Screenshots"
              ? "No screenshots detected in your local library."
              : "No photos match your current collection or filters."}
          </p>
          <p className="text-xs text-slate-500">
            Import local images via drag-and-drop to enable SHA-256 &amp; perceptual dHash duplicate detection.
          </p>
        </div>
      ) : viewMode === "timeline" ? (
        <div className="space-y-6">
          {timelineGroups.map((group) => (
            <section key={group.key} className="space-y-3">
              {/* Sticky Date Heading */}
              <div className="sticky top-0 z-20 flex items-center justify-between px-3 py-2 rounded-lg bg-[#090D16]/90 backdrop-blur-md border border-white/10">
                <div className="flex items-center gap-2">
                  <Calendar className="w-3.5 h-3.5 text-sky-400" />
                  <h4 className="text-xs font-semibold text-slate-100">{group.label}</h4>
                </div>
                <span className="text-[11px] font-mono text-slate-400 tabular-nums">
                  {group.items.length} {group.items.length === 1 ? "photo" : "photos"}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {group.items.map(renderPhotoCard)}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {visiblePhotos.map(renderPhotoCard)}
        </div>
      )}

      {/* Virtualized / Progressive Load Sentinel */}
      {visibleCount < filteredPhotos.length && (
        <div ref={loadMoreSentinelRef} className="flex justify-center pt-2">
          <button
            type="button"
            onClick={() => setVisibleCount((c) => Math.min(filteredPhotos.length, c + WINDOW_STEP))}
            className="px-4 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-white/15 text-xs text-slate-200 font-medium transition-colors"
          >
            Load More ({filteredPhotos.length - visibleCount} remaining)
          </button>
        </div>
      )}

      {/* Destructive Delete Confirmation Modal (Never uses window.confirm) */}
      {showDeleteConfirm && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in"
          onClick={() => setShowDeleteConfirm(false)}
        >
          <div
            className="w-full max-w-md rounded-2xl bg-slate-900 border border-white/15 p-6 space-y-4 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start gap-3">
              <div className="p-2.5 rounded-xl bg-rose-500/15 text-rose-400 shrink-0">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-slate-50">Confirm Photo Removal</h3>
                <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                  You are about to remove{" "}
                  <strong className="text-white">{selectedIds.size}</strong> selected photo(s):
                </p>
                <ul className="text-xs text-slate-400 mt-2 space-y-1 list-disc list-inside">
                  {selectedBreakdown.localCount > 0 && (
                    <li>
                      <strong className="text-rose-300">{selectedBreakdown.localCount}</strong>{" "}
                      local IndexedDB photo(s) will be permanently deleted from browser storage.
                    </li>
                  )}
                  {selectedBreakdown.corpusCount > 0 && (
                    <li>
                      <strong className="text-slate-200">{selectedBreakdown.corpusCount}</strong>{" "}
                      curated corpus sample(s) will be hidden from your gallery view.
                    </li>
                  )}
                </ul>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setShowDeleteConfirm(false)}
                className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-xs font-medium text-slate-200 transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void handleConfirmDelete()}
                className="px-4 py-2 rounded-lg bg-rose-500 hover:bg-rose-400 text-xs font-semibold text-slate-950 transition-colors"
              >
                Confirm Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Full-Screen Lightbox Viewer with Mobile Touch Swipe Gestures */}
      {activePhoto && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`Viewing ${activePhoto.filename}`}
          className="fixed inset-0 z-50 bg-black/95 backdrop-blur-xl flex flex-col items-center justify-between p-4 sm:p-8 animate-fade-in"
          onClick={() => setActivePhotoIndex(null)}
          onTouchStart={handleLightboxTouchStart}
          onTouchEnd={handleLightboxTouchEnd}
        >
          <div
            className="w-full max-w-6xl flex flex-wrap items-center justify-between gap-3 text-slate-300"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="space-y-0.5 min-w-0">
              <div className="text-sm font-bold text-slate-50 font-mono truncate">
                {activePhoto.filename}
              </div>
              <div className="text-xs text-slate-400 flex flex-wrap items-center gap-x-2 gap-y-0.5">
                <span>
                  Album: <span className="text-sky-300">{activePhoto.album}</span>
                </span>
                <span>· {activePhoto.timestamp}</span>
                <span>· {shortMimeLabel(activePhoto.mimeType)}</span>
                {activePhoto.width && activePhoto.height && (
                  <span className="font-mono">
                    · {activePhoto.width}×{activePhoto.height}
                  </span>
                )}
                {activePhoto.sizeBytes && (
                  <span className="font-mono">· {formatBytes(activePhoto.sizeBytes)}</span>
                )}
                {activePhoto.sha256 && (
                  <span className="font-mono text-slate-500">
                    · SHA-256: {activePhoto.sha256.slice(0, 12)}…
                  </span>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setZoomLevel((z) => Math.max(0.5, z - 0.25))}
                className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-slate-200 transition-colors"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <span className="text-xs font-mono text-slate-300 w-10 text-center tabular-nums">
                {Math.round(zoomLevel * 100)}%
              </span>
              <button
                type="button"
                onClick={() => setZoomLevel((z) => Math.min(3, z + 0.25))}
                className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-slate-200 transition-colors"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>

              <button
                type="button"
                onClick={(e) => toggleFavorite(activePhoto, e)}
                className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-slate-200 transition-colors"
                title={activePhoto.isFavorite ? "Unfavorite" : "Favorite"}
              >
                <Heart
                  className={`w-4 h-4 ${
                    activePhoto.isFavorite ? "fill-rose-400 text-rose-400" : ""
                  }`}
                />
              </button>

              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setEditingPhoto(activePhoto);
                }}
                className="px-3 py-1.5 rounded-lg bg-sky-500/20 hover:bg-sky-500/30 text-sky-300 border border-sky-400/40 transition-colors inline-flex items-center gap-1.5 text-xs font-medium"
                title="Edit Photo (Saves New Copy)"
              >
                <Edit3 className="w-3.5 h-3.5" />
                Edit
              </button>

              <a
                href={activePhoto.url}
                download={activePhoto.filename}
                onClick={(e) => e.stopPropagation()}
                className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-slate-200 transition-colors"
                title="Download Photo"
              >
                <Download className="w-4 h-4" />
              </a>

              <button
                type="button"
                onClick={() => setActivePhotoIndex(null)}
                className="p-2 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 transition-colors"
                title="Close Viewer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div
            className="relative flex-1 flex items-center justify-center w-full max-w-6xl my-4 overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {activePhotoIndex !== null && activePhotoIndex > 0 && (
              <button
                type="button"
                onClick={() => {
                  setActivePhotoIndex((idx) => (idx !== null ? idx - 1 : 0));
                  setZoomLevel(1);
                }}
                className="absolute left-4 z-10 p-3 rounded-full bg-slate-900/80 hover:bg-sky-500 hover:text-slate-950 text-slate-100 transition-colors shadow-lg"
              >
                <ChevronLeft className="w-6 h-6" />
              </button>
            )}

            <div className="overflow-auto max-h-[74vh] max-w-full flex items-center justify-center p-2">
              <img
                src={activePhoto.url}
                alt={activePhoto.filename}
                referrerPolicy="no-referrer"
                style={{ transform: `scale(${zoomLevel})`, transition: "transform 0.2s ease-out" }}
                className="max-h-[68vh] object-contain rounded-lg shadow-2xl"
              />
            </div>

            {activePhotoIndex !== null && activePhotoIndex < filteredPhotos.length - 1 && (
              <button
                type="button"
                onClick={() => {
                  setActivePhotoIndex((idx) => (idx !== null ? idx + 1 : 0));
                  setZoomLevel(1);
                }}
                className="absolute right-4 z-10 p-3 rounded-full bg-slate-900/80 hover:bg-sky-500 hover:text-slate-950 text-slate-100 transition-colors shadow-lg"
              >
                <ChevronRight className="w-6 h-6" />
              </button>
            )}
          </div>

          <div
            className="flex flex-wrap items-center justify-center gap-2 text-xs"
            onClick={(e) => e.stopPropagation()}
          >
            {duplicateAnalysis.exactPhotoIds.has(activePhoto.id) && (
              <span className="px-3 py-1 rounded-full bg-amber-500/20 border border-amber-400/40 text-amber-200 font-medium">
                Exact SHA-256 Duplicate (×
                {duplicateAnalysis.exactGroupSizeByPhotoId.get(activePhoto.id)})
              </span>
            )}
            {activePhoto.dhash && (
              <span className="px-3 py-1 rounded-full bg-indigo-500/15 border border-indigo-400/30 text-indigo-200 font-mono">
                dHash: {activePhoto.dhash}
              </span>
            )}
            {activePhoto.attributes.map((attr) => (
              <span key={attr} className="px-3 py-1 rounded-full bg-white/10 text-slate-200 font-medium">
                {attr}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Non-Destructive Photo Editor Modal */}
      {editingPhoto && (
        <PhotoEditorModal
          imageUrl={editingPhoto.url}
          filename={editingPhoto.filename}
          onClose={() => setEditingPhoto(null)}
          onSaveEdited={(url, fname) => {
            void handleSaveEdited(url, fname);
          }}
        />
      )}
    </div>
  );
}
