import { useState, useMemo, useRef, type DragEvent, type ChangeEvent } from "react";
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
  Maximize2,
  Sparkles,
  Filter,
  Grid,
  Calendar,
  Edit3,
} from "lucide-react";
import type { GallerySample } from "../types";
import { PhotoEditorModal } from "./PhotoEditorModal";


interface GooglePhotosGalleryProps {
  samples: GallerySample[];
  onSelectForAnalysis?: (imageUrl: string) => void;
}

interface LocalPhotoItem {
  id: string;
  url: string;
  filename: string;
  timestamp: string;
  isFavorite: boolean;
  album: string;
  attributes: string[];
  isUserUpload: boolean;
}

export function GooglePhotosGallery({ samples, onSelectForAnalysis }: GooglePhotosGalleryProps) {
  // Convert API samples to gallery items
  const initialPhotos: LocalPhotoItem[] = useMemo(
    () =>
      samples.map((s) => ({
        id: `sample-${s.id}`,
        url: s.thumbnail,
        filename: s.filename,
        timestamp: new Date(Date.now() - s.id * 86400000).toISOString().split("T")[0],
        isFavorite: s.smiling,
        album: s.smiling ? "Smiling Faces" : "Curated Corpus",
        attributes: s.attributes || [s.smiling ? "Smiling" : "Neutral", s.eyeglasses ? "Eyeglasses" : "Natural"],
        isUserUpload: false,
      })),
    [samples]
  );

  const [photos, setPhotos] = useState<LocalPhotoItem[]>(initialPhotos);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedAlbum, setSelectedAlbum] = useState<string>("All Photos");
  const [sortBy, setSortBy] = useState<"date" | "filename" | "favorites">("date");
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [activePhotoIndex, setActivePhotoIndex] = useState<number | null>(null);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [editingPhoto, setEditingPhoto] = useState<LocalPhotoItem | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSaveEdited = (newUrl: string, newFilename: string) => {
    const newItem: LocalPhotoItem = {
      id: `edited-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      url: newUrl,
      filename: newFilename,
      timestamp: new Date().toISOString().split("T")[0],
      isFavorite: true,
      album: "Edited Photos",
      attributes: ["Edited", "Enhanced", "High Resolution"],
      isUserUpload: true,
    };
    setPhotos((prev) => [newItem, ...prev]);
    setEditingPhoto(null);
  };


  // Sync initial photos when props change if photos state is empty or initial
  useMemo(() => {
    if (photos.length === 0 && samples.length > 0) {
      setPhotos(
        samples.map((s) => ({
          id: `sample-${s.id}`,
          url: s.thumbnail,
          filename: s.filename,
          timestamp: new Date().toISOString().split("T")[0],
          isFavorite: s.smiling,
          album: "Curated Corpus",
          attributes: s.attributes || [],
          isUserUpload: false,
        }))
      );
    }
  }, [samples]);

  const albums = useMemo(() => {
    const set = new Set<string>(["All Photos", "Curated Corpus", "Smiling Faces", "User Uploads", "Edited Photos"]);
    photos.forEach((p) => p.album && set.add(p.album));
    return Array.from(set);
  }, [photos]);

  const filteredPhotos = useMemo(() => {
    let result = [...photos];

    if (selectedAlbum !== "All Photos") {
      result = result.filter((p) => p.album === selectedAlbum);
    }

    if (favoritesOnly) {
      result = result.filter((p) => p.isFavorite);
    }

    const q = searchQuery.trim().toLowerCase();
    if (q) {
      result = result.filter(
        (p) =>
          p.filename.toLowerCase().includes(q) ||
          p.attributes.some((a) => a.toLowerCase().includes(q)) ||
          p.album.toLowerCase().includes(q)
      );
    }

    result.sort((a, b) => {
      if (sortBy === "favorites") {
        return (b.isFavorite ? 1 : 0) - (a.isFavorite ? 1 : 0);
      }
      if (sortBy === "filename") {
        return a.filename.localeCompare(b.filename);
      }
      return b.timestamp.localeCompare(a.timestamp);
    });

    return result;
  }, [photos, selectedAlbum, favoritesOnly, searchQuery, sortBy]);

  // Handle Drag and Drop local uploads
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
      processFiles(e.dataTransfer.files);
    }
  };

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
    }
  };

  const processFiles = (fileList: FileList) => {
    const newItems: LocalPhotoItem[] = [];
    Array.from(fileList).forEach((file) => {
      if (!file.type.startsWith("image/")) return;
      const url = URL.createObjectURL(file);
      newItems.push({
        id: `upload-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        url,
        filename: file.name,
        timestamp: new Date().toISOString().split("T")[0],
        isFavorite: true,
        album: "User Uploads",
        attributes: ["Local Upload", "High Resolution"],
        isUserUpload: true,
      });
    });
    if (newItems.length > 0) {
      setPhotos((prev) => [...newItems, ...prev]);
    }
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === filteredPhotos.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filteredPhotos.map((p) => p.id)));
    }
  };

  const toggleSelectOne = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const next = new Set(selectedIds);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    setSelectedIds(next);
  };

  const toggleFavorite = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setPhotos((prev) =>
      prev.map((p) => (p.id === id ? { ...p, isFavorite: !p.isFavorite } : p))
    );
  };

  const deleteSelected = () => {
    setPhotos((prev) => prev.filter((p) => !selectedIds.has(p.id)));
    setSelectedIds(new Set());
  };

  const activePhoto = activePhotoIndex !== null ? filteredPhotos[activePhotoIndex] : null;

  return (
    <div
      className="space-y-6 relative"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag & Drop Overlay */}
      {isDraggingOver && (
        <div className="absolute inset-0 z-50 bg-sky-950/80 backdrop-blur-md rounded-2xl border-2 border-dashed border-sky-400 flex flex-col items-center justify-center p-8 text-center animate-fade-in">
          <Upload className="w-16 h-16 text-sky-400 animate-bounce mb-4" />
          <h3 className="text-xl font-bold text-slate-50">Drop Local Images Here</h3>
          <p className="text-sm text-sky-200 mt-1">
            Processed instantly in browser memory. Zero cloud upload.
          </p>
        </div>
      )}

      {/* Gallery Header Toolbar */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/90 border border-white/10 backdrop-blur-md">
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search photos, attributes..."
              className="pl-9 pr-3 py-2 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-400 w-64"
            />
          </div>

          <select
            value={selectedAlbum}
            onChange={(e) => setSelectedAlbum(e.target.value)}
            className="px-3 py-2 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-400"
          >
            {albums.map((alb) => (
              <option key={alb} value={alb}>
                📁 {alb}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={() => setFavoritesOnly(!favoritesOnly)}
            className={`px-3 py-2 rounded-lg text-xs font-medium inline-flex items-center gap-1.5 transition-colors ${
              favoritesOnly
                ? "bg-rose-500/20 text-rose-300 border border-rose-400/40"
                : "bg-slate-950 text-slate-300 border border-white/10 hover:text-white"
            }`}
          >
            <Heart className={`w-3.5 h-3.5 ${favoritesOnly ? "fill-rose-400 text-rose-400" : ""}`} />
            Favorites
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <div className="flex items-center gap-1.5 text-xs text-slate-400 mr-2">
            <span>Sort:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="px-2.5 py-1.5 rounded-lg bg-slate-950 border border-white/15 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-400"
            >
              <option value="date">Date Added</option>
              <option value="filename">Filename</option>
              <option value="favorites">Favorites First</option>
            </select>
          </div>

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold text-xs inline-flex items-center gap-2 transition-colors"
          >
            <Upload className="w-4 h-4" />
            Upload Photos
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

      {/* Batch Actions Bar if selected */}
      {selectedIds.size > 0 && (
        <div className="flex items-center justify-between p-3 rounded-xl bg-sky-950/60 border border-sky-400/30 text-xs text-sky-200 animate-fade-in">
          <div className="flex items-center gap-2 font-medium">
            <span className="px-2 py-0.5 rounded bg-sky-500 text-slate-950 font-bold">
              {selectedIds.size}
            </span>
            photos selected
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={deleteSelected}
              className="px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-400/40 inline-flex items-center gap-1.5 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Remove Selected
            </button>
            <button
              type="button"
              onClick={() => setSelectedIds(new Set())}
              className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 text-slate-200 transition-colors"
            >
              Clear Selection
            </button>
          </div>
        </div>
      )}

      {/* Responsive Masonry / Grid */}
      <div className="space-y-4">
        <div className="flex items-center justify-between text-xs text-slate-400 px-1">
          <span>
            Showing <strong className="text-slate-100">{filteredPhotos.length}</strong> photos in{" "}
            <span className="text-sky-300">{selectedAlbum}</span>
          </span>
          <button
            type="button"
            onClick={toggleSelectAll}
            className="hover:text-slate-100 transition-colors inline-flex items-center gap-1"
          >
            {selectedIds.size === filteredPhotos.length ? (
              <CheckSquare className="w-4 h-4 text-sky-400" />
            ) : (
              <Square className="w-4 h-4" />
            )}
            Select All
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {filteredPhotos.map((photo, idx) => {
            const isSelected = selectedIds.has(photo.id);
            return (
              <div
                key={photo.id}
                onClick={() => {
                  setActivePhotoIndex(idx);
                  setZoomLevel(1);
                }}
                className={`group relative rounded-xl border bg-slate-950/80 overflow-hidden cursor-pointer transition-all duration-300 hover:scale-[1.02] hover:shadow-xl hover:shadow-sky-500/10 ${
                  isSelected ? "border-sky-400 ring-2 ring-sky-400/50" : "border-white/10 hover:border-sky-400/50"
                }`}
              >
                <div className="relative aspect-square w-full bg-slate-900 overflow-hidden">
                  <img
                    src={photo.url}
                    alt={photo.filename}
                    loading="lazy"
                    referrerPolicy="no-referrer"
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-between p-2.5">
                    <div className="flex items-center justify-between">
                      <button
                        type="button"
                        onClick={(e) => toggleSelectOne(photo.id, e)}
                        className="p-1 rounded-md bg-slate-900/80 text-slate-200 hover:text-sky-300 transition-colors"
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
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingPhoto(photo);
                          }}
                          className="p-1 rounded-md bg-slate-900/80 text-sky-400 hover:text-sky-300 transition-colors"
                          title="Edit Photo"
                        >
                          <Edit3 className="w-4 h-4" />
                        </button>
                        <button
                          type="button"
                          onClick={(e) => toggleFavorite(photo.id, e)}
                          className="p-1 rounded-md bg-slate-900/80 text-slate-200 transition-colors"
                        >
                          <Heart
                            className={`w-4 h-4 ${
                              photo.isFavorite ? "fill-rose-400 text-rose-400" : "text-slate-300 hover:text-rose-400"
                            }`}
                          />
                        </button>
                      </div>
                    </div>

                    <div className="space-y-1">
                      <div className="text-[11px] font-mono font-medium text-slate-100 truncate">
                        {photo.filename}
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-sky-300 bg-sky-950/80 px-1.5 py-0.5 rounded">
                          {photo.album}
                        </span>
                        {onSelectForAnalysis && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onSelectForAnalysis(photo.url);
                            }}
                            className="p-1 rounded bg-sky-500 hover:bg-sky-400 text-slate-950"
                            title="Analyze Face"
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
          })}
        </div>

        {filteredPhotos.length === 0 && (
          <div className="text-center py-16 rounded-2xl border border-white/10 bg-slate-900/50">
            <p className="text-sm text-slate-400">No photos found matching your search or album.</p>
          </div>
        )}
      </div>

      {/* Full-Screen Lightbox Viewer */}
      {activePhoto && (
        <div
          className="fixed inset-0 z-50 bg-black/95 backdrop-blur-xl flex flex-col items-center justify-between p-4 sm:p-8 animate-fade-in"
          onClick={() => setActivePhotoIndex(null)}
        >
          {/* Lightbox Header */}
          <div
            className="w-full max-w-6xl flex items-center justify-between text-slate-300"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="space-y-0.5">
              <div className="text-sm font-bold text-slate-50 font-mono">{activePhoto.filename}</div>
              <div className="text-xs text-slate-400">
                Album: <span className="text-sky-300">{activePhoto.album}</span> · Added: {activePhoto.timestamp}
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => setZoomLevel((z) => Math.max(0.5, z - 0.25))}
                className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-slate-200 transition-colors"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <span className="text-xs font-mono text-slate-300">{Math.round(zoomLevel * 100)}%</span>
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
                onClick={(e) => {
                  e.stopPropagation();
                  setEditingPhoto(activePhoto);
                }}
                className="px-3 py-1.5 rounded-lg bg-sky-500/20 hover:bg-sky-500/30 text-sky-300 border border-sky-400/40 transition-colors inline-flex items-center gap-1.5 text-xs font-medium"
                title="Edit Photo"
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

          {/* Lightbox Main Image & Navigation */}
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

            <div className="overflow-auto max-h-[75vh] max-w-full flex items-center justify-center p-2">
              <img
                src={activePhoto.url}
                alt={activePhoto.filename}
                referrerPolicy="no-referrer"
                style={{ transform: `scale(${zoomLevel})`, transition: "transform 0.2s ease-out" }}
                className="max-h-[70vh] object-contain rounded-lg shadow-2xl"
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

          {/* Lightbox Footer Attributes */}
          <div
            className="flex flex-wrap items-center justify-center gap-2 text-xs"
            onClick={(e) => e.stopPropagation()}
          >
            {activePhoto.attributes.map((attr) => (
              <span key={attr} className="px-3 py-1 rounded-full bg-white/10 text-slate-200 font-medium">
                {attr}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Photo Editor Modal */}
      {editingPhoto && (
        <PhotoEditorModal
          imageUrl={editingPhoto.url}
          filename={editingPhoto.filename}
          onClose={() => setEditingPhoto(null)}
          onSaveEdited={handleSaveEdited}
        />
      )}
    </div>
  );
}

