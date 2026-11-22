import type { ReactNode } from "react";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
}

export function GlassCard({ children, className = "" }: GlassCardProps) {
  return (
    <div
      className={`rounded-xl border border-white/10 bg-slate-900/70 backdrop-blur-md p-5 ${className}`}
    >
      {children}
    </div>
  );
}
